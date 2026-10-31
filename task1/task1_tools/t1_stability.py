#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_stability.py —— 任务一：跨迭代稳定性（时长层证据）

前置：t1_multistep_trace.py 已产出 *_multi5_kernel_trace.json

判据（三层）
------------
1) 迭代墙钟 CV = std/mean        < 3% 稳 / 3–8% 弱稳 / >8% 不稳
2) kernel 数恒定性                各步 n_kernel 必须完全相同
3) 逐 op 耗时一致性               按 launcher op 位置对齐，算每 op 的 CV 与跨步 Spearman

两条对齐路径（自动选择）
------------------------
A. op 级（首选）：用 trace 自带的 cpu_op → kernel 的 `External id` 关联，
   把每次 cuDNN 调用聚成一个"op 耗时"，再按出现顺序跨步对齐。
   这是与 G10"调度单元"同一层次的粒度。

B. kernel 名级（退化）：如果 A 路径一个 op 都匹配不到（torch 版本改了 op 名、
   或 External id 未关联），自动退化为按 kernel 名字聚合 count/耗时再跨步比较。
   粒度粗一些，但绝不空手而归。

为什么不用 G10 的 449 个调度单元直接比
--------------------------------------
G10 的 449 是"调度单元"（一次 cuDNN 调用），trace 里的 921 是真实 CUDA kernel，
两者不是一一对应。A 路径按 cuDNN 调用聚合，才是同一层次的可比值。

诊断
----
脚本会打印：窗口内 cpu_op 数、带 External id 的 op 数、成功关联 kernel 的 op 数、
以及出现最多的 op 名称。A 路径失效时用这几行定位原因。

用法
----
    python3 t1_stability.py --trace ./task1_out/resnet50_bs16_multi5_kernel_trace.json
    python3 t1_stability.py --trace ... --mode kernel      # 强制走退化路径
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


# 不同 torch 版本里卷积 op 的命名不完全一致，因此用关键词而非完整名判定
FWD_KEYS = ("convolution", "conv2d", "cudnn_conv")
BWD_KEY = "backward"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def cv(xs):
    m = mean(xs)
    if not m:
        return float("nan")
    return stdev(xs) / m


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    idx = (len(s) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    f = idx - lo
    return s[lo] * (1 - f) + s[hi] * f


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    ra, rb = ranks(a[:n]), ranks(b[:n])
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def fmt_pct(x, nd=2):
    if x is None or x != x:
        return "N/A"
    return f"{x * 100:.{nd}f}%"


def fmt(x, nd=3):
    if x is None or x != x:
        return "N/A"
    return f"{x:.{nd}f}"


def load_events(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def windows(events):
    """返回 [(name, t_start, t_end_or_None)]，按开始时刻排序。

    两个必须注意的点（否则切窗必错）：
    1) 同一个 `record_function("STEP#i")` 在 trace 里会产生 **两个** 事件：
         cat=user_annotation     （CPU 侧，tid=主线程）
         cat=gpu_user_annotation （GPU 侧，tid=某条 stream）
       两者是同一注释的两个投影，只取 CPU 侧，否则一步会被算成两步。
    2) 不能用 [ts, ts+dur] 作为一步的边界：dur 只覆盖 Python 侧的发射期，
       而 GPU kernel 是异步执行的，尾部 kernel 会落在注释结束之后，
       导致每步 kernel 数偏小且逐步抖动（实测 623/517/532/526 vs 应为 854）。
       正确做法是用 **相邻注释的起始时刻** 作边界：[ts_i, ts_{i+1})。
    """
    wins = [(e["name"], e["ts"]) for e in events
            if isinstance(e.get("name"), str) and e["name"].startswith("STEP#")
            and e.get("ph") == "X" and e.get("cat") == "user_annotation"]
    wins.sort(key=lambda x: x[1])
    out = []
    for i, (name, ts) in enumerate(wins):
        t1 = wins[i + 1][1] if i + 1 < len(wins) else None
        out.append((name, ts, t1))
    return out


def classify(name):
    low = (name or "").lower()
    if BWD_KEY in low:
        return "backward"
    if any(k in low for k in FWD_KEYS):
        return "forward"
    return "other"


def collect_by_op(ks, ops, t0, t1):
    """A 路径：External id 关联，按 **op 名** 分组。

    返回 ({op_name: [每次调用的耗时, ...]}, diag)。
    按 op 名分组而不是把 forward/backward 混在一起，原因有两个：
      1) 不同 torch 版本对卷积反向的命名不一致（convolution_backward /
         cudnn_convolution_backward / convolution_backward_overrideable）；
      2) 同一类 op 在不同步可能出现次数不同（某些层没有 kernel），
         按名分组后各自按顺序对齐，不会互相错位。
    """
    by_ext = defaultdict(float)
    for e in ks:
        ext = (e.get("args") or {}).get("External id")
        if ext is not None:
            by_ext[ext] += e.get("dur", 0.0)

    in_win = [e for e in ops if t0 <= e.get("ts", -1) < t1]
    with_ext = [e for e in in_win if (e.get("args") or {}).get("External id") is not None]
    matched = [e for e in with_ext if (e.get("args") or {})["External id"] in by_ext]

    per_name = defaultdict(list)
    for e in matched:
        ext = (e.get("args") or {})["External id"]
        per_name[e.get("name")].append(by_ext[ext])

    diag = {
        "n_cpu_op_in_window": len(in_win),
        "n_cpu_op_with_ext_id": len(with_ext),
        "n_cpu_op_matched_kernel": len(matched),
        "top_op_names": Counter(e.get("name") for e in in_win).most_common(8),
        "top_matched_op_names": Counter(e.get("name") for e in matched).most_common(12),
        "n_forward_ops": sum(1 for e in matched if classify(e.get("name")) == "forward"),
        "n_backward_ops": sum(1 for e in matched if classify(e.get("name")) == "backward"),
    }
    return dict(per_name), diag


def collect_by_kernel_name(ks):
    """B 路径：按 kernel 名字聚合。返回 {name: (count, total_us)}。"""
    agg = defaultdict(lambda: [0, 0.0])
    for e in ks:
        n = e.get("name", "?")
        agg[n][0] += 1
        agg[n][1] += e.get("dur", 0.0)
    return {k: (v[0], v[1]) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "op", "kernel"])
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.trace))
    os.makedirs(outdir, exist_ok=True)
    events = load_events(args.trace)
    wins = windows(events)

    if len(wins) < 2:
        sys.stderr.write(
            f"[ERROR] trace 里只找到 {len(wins)} 个 STEP# 标记，跨迭代稳定性至少需要 2 步。\n"
            "        请先用 t1_multistep_trace.py 采样。\n")
        sys.exit(1)

    kernels = [e for e in events if e.get("cat") == "kernel"]
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    kernel_end = max((e["ts"] + e.get("dur", 0.0) for e in kernels), default=0.0)

    per_step = []
    diag_all = None
    seq_all = []
    for name, t0, t1 in wins:
        if t1 is None:                      # 最后一个窗口：延伸到最后一个 kernel 结束
            t1 = kernel_end + 1.0
        ks = [e for e in kernels if t0 <= e.get("ts", -1) < t1]
        if not ks:
            continue
        span0 = min(e["ts"] for e in ks)
        span1 = max(e["ts"] + e.get("dur", 0.0) for e in ks)
        per_op, diag = collect_by_op(ks, cpu_ops, t0, t1)
        if diag_all is None:
            diag_all = diag
        seq_all.append(per_op)
        per_step.append({
            "step": name,
            "wall_us": t1 - t0,
            "kernel_span_us": span1 - span0,
            "n_kernel": len(ks),
            "sum_kernel_us": sum(e.get("dur", 0.0) for e in ks),
            "by_name": collect_by_kernel_name(ks),
        })

    if not per_step:
        sys.stderr.write("[ERROR] 所有 STEP# 窗口内都没有 kernel 事件，检查 trace 是否完整。\n")
        sys.exit(1)

    n_fwd = diag_all["n_forward_ops"] if diag_all else 0
    n_bwd = diag_all["n_backward_ops"] if diag_all else 0
    if args.mode == "auto":
        use_op = (n_fwd + n_bwd) > 0
    else:
        use_op = (args.mode == "op")
    if args.mode == "op" and (n_fwd + n_bwd) == 0:
        sys.stderr.write("[WARN] --mode op 但一个 op 都没匹配到，自动退化为 kernel 名级。\n")
        use_op = False

    walls = [p["wall_us"] for p in per_step]
    spans = [p["kernel_span_us"] for p in per_step]
    nks = [p["n_kernel"] for p in per_step]

    res = {
        "trace": os.path.abspath(args.trace),
        "mode_used": "op" if use_op else "kernel_name",
        "n_recorded_steps": len(per_step),
        "steps": [p["step"] for p in per_step],
        "per_step": [{k: v for k, v in p.items() if k != "by_name"} for p in per_step],
        "wall_cv": cv(walls),
        "span_cv": cv(spans),
        "wall_cv_ex_first": cv(walls[1:]) if len(walls) >= 3 else None,
        "span_cv_ex_first": cv(spans[1:]) if len(spans) >= 3 else None,
        "wall_min_max_us": [min(walls), max(walls)],
        "n_kernel_set": sorted(set(nks)),
        "kernel_count_stable": len(set(nks)) == 1,
        "diagnostics": diag_all,
    }

    if use_op:
        # 按 op 名对齐：只取在每一步都出现的 op 名，逐次调用按顺序配对
        all_names = set()
        for s in seq_all:
            all_names |= set(s.keys())
        common = [n for n in all_names if all(n in s for s in seq_all)]

        def total_us(n):
            return sum(sum(s[n]) for s in seq_all)

        common.sort(key=lambda n: -total_us(n))

        op_rows = []
        for n in common:
            series = [s[n] for s in seq_all]
            k = min(len(x) for x in series)
            if k == 0:
                continue
            cvs = [cv([s[i] for s in series]) for i in range(k)]
            op_rows.append({
                "name": n,
                "category": classify(n),
                "count_per_step": [len(x) for x in series],
                "n_aligned": k,
                "cv_median": pct(cvs, 50),
                "cv_max": max(cvs),
                "n_pos_gt5pct": sum(1 for c in cvs if c > 0.05),
                "mean_us": mean([x for s in series for x in s]),
            })

        def block(cat):
            rows = [r for r in op_rows
                    if r["category"] == cat and "conv" in r["name"].lower()]
            if not rows:
                return {"n_op_names": 0, "n_positions": 0,
                        "cv_median": None, "cv_max": None, "n_pos_gt5pct": None}
            return {
                "n_op_names": len(rows),
                "n_positions": sum(r["n_aligned"] for r in rows),
                "cv_median": pct([r["cv_median"] for r in rows], 50),
                "cv_max": max(r["cv_max"] for r in rows),
                "n_pos_gt5pct": sum(r["n_pos_gt5pct"] for r in rows),
            }

        res["op_level"] = {
            "forward_conv": block("forward"),
            "backward_conv": block("backward"),
            "rows": op_rows[:25],
        }

        # 跨步排名一致性：取总耗时前 15 的 op 名，比较各步的耗时排序
        top_names = [r["name"] for r in op_rows[:15]]
        series = [[sum(s.get(n, [])) for n in top_names] for s in seq_all]
        pairs = []
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                r = spearman(series[i], series[j])
                if r == r:
                    pairs.append(r)
        res["fwd_rank_spearman"] = {"n_pairs": len(pairs),
                                    "min": min(pairs) if pairs else None,
                                    "median": pct(pairs, 50) if pairs else None}
    else:
        names = set()
        for p in per_step:
            names |= set(p["by_name"].keys())
        common = [n for n in names if all(n in p["by_name"] for p in per_step)]
        common.sort(key=lambda n: -sum(p["by_name"][n][1] for p in per_step))
        top = common[:20]
        res["kernel_level"] = {
            "n_common_names": len(common),
            "top_names": top,
            "counts_per_step": {n: [p["by_name"][n][0] for p in per_step] for n in top},
            "time_cv": {n: cv([p["by_name"][n][1] for p in per_step]) for n in top},
        }
        series = [[p["by_name"][n][1] for n in top] for p in per_step]
        pairs = []
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                r = spearman(series[i], series[j])
                if r == r:
                    pairs.append(r)
        res["fwd_rank_spearman"] = {"n_pairs": len(pairs),
                                    "min": min(pairs) if pairs else None,
                                    "median": pct(pairs, 50) if pairs else None}

    def verdict(v, good, ok):
        if v is None or v != v:
            return "无法判定"
        if v < good:
            return "稳定"
        if v < ok:
            return "弱稳定"
        return "不稳定"

    res["verdict"] = {
        "wall_cv": verdict(res["wall_cv"], 0.03, 0.08),
        "kernel_count": "稳定" if res["kernel_count_stable"] else "不稳定",
        "rank": "稳定" if (res["fwd_rank_spearman"]["min"] if res["fwd_rank_spearman"]["min"] is not None else -1) > 0.9 else "需检查",
    }

    path = os.path.join(outdir, "stability.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    md = ["### 表 7 跨迭代稳定性：逐步原始值\n",
          "| step | 墙钟(µs) | kernel 跨度(µs) | kernel 数 | kernel 时间和(µs) |",
          "|---|---|---|---|---|"]
    for p in per_step:
        md.append(f"| {p['step']} | {p['wall_us']:.0f} | {p['kernel_span_us']:.0f} | "
                  f"{p['n_kernel']} | {p['sum_kernel_us']:.0f} |")
    md.append("")
    md.append(f"- 墙钟 CV = **{fmt_pct(res['wall_cv'])}**（判据 <3%）→ {res['verdict']['wall_cv']}")
    md.append(f"- 剔除首步后墙钟 CV = {fmt_pct(res['wall_cv_ex_first'])}"
              f"（首步通常含残余预热开销）")
    md.append(f"- kernel 跨度 CV = {fmt_pct(res['span_cv'])}"
              f"；剔除首步后 {fmt_pct(res['span_cv_ex_first'])}")
    md.append(f"- kernel 数集合 = {res['n_kernel_set']} → {res['verdict']['kernel_count']}")
    md.append(f"- 排名一致性（最小 Spearman）= {fmt(res['fwd_rank_spearman']['min'])}"
              f"（判据 >0.9）→ {res['verdict']['rank']}")
    md.append(f"- 对齐路径 = `{res['mode_used']}`")

    if use_op:
        fo, bo = res["op_level"]["forward_conv"], res["op_level"]["backward_conv"]
        md.append(f"\n**op 级（卷积前向）**：{fo['n_op_names']} 个 op 名、"
                  f"{fo['n_positions']} 个对齐位置，CV 中位数 {fmt_pct(fo['cv_median'])}，"
                  f"最大 {fmt_pct(fo['cv_max'])}，CV>5% 的位置 {fo['n_pos_gt5pct']}")
        md.append(f"**op 级（卷积反向）**：{bo['n_op_names']} 个 op 名、"
                  f"{bo['n_positions']} 个对齐位置，CV 中位数 {fmt_pct(bo['cv_median'])}，"
                  f"最大 {fmt_pct(bo['cv_max'])}，CV>5% 的位置 {bo['n_pos_gt5pct']}")
        md.append("\n| op 名 | 各步 count | 对齐位置 | CV 中位数 | CV>5% 位置 | 均值(µs) |")
        md.append("|---|---|---|---|---|---|")
        for r in res["op_level"]["rows"][:12]:
            short = r["name"] if len(r["name"]) <= 45 else r["name"][:42] + "..."
            md.append(f"| {short} | {r['count_per_step']} | {r['n_aligned']} | "
                      f"{fmt_pct(r['cv_median'])} | {r['n_pos_gt5pct']} | {r['mean_us']:.1f} |")
    else:
        kl = res["kernel_level"]
        md.append(f"\n**kernel 名级（退化路径）**：跨步共有 {kl['n_common_names']} 个名字，"
                  f"取耗时前 {len(kl['top_names'])} 个比较")
        md.append("\n| kernel 名 | 各步 count | 耗时 CV |")
        md.append("|---|---|---|")
        for n in kl["top_names"][:10]:
            short = n if len(n) <= 60 else n[:57] + "..."
            md.append(f"| {short} | {kl['counts_per_step'][n]} | {fmt_pct(kl['time_cv'][n])} |")

    if diag_all:
        md.append(f"\n**诊断**：窗内 cpu_op {diag_all['n_cpu_op_in_window']} 个，"
                  f"带 External id {diag_all['n_cpu_op_with_ext_id']} 个，"
                  f"成功关联 kernel {diag_all['n_cpu_op_matched_kernel']} 个")
        if diag_all["top_op_names"]:
            md.append(f"  最常见的 cpu_op：{diag_all['top_op_names'][:4]}")

    txt = "\n".join(md)
    with open(os.path.join(outdir, "stability.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    print(txt)
    print(f"\n[OK] {path}")


if __name__ == "__main__":
    main()
