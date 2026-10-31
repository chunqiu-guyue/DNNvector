#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t1_analyze.py —— 非活跃窗口的分布统计与形态判定。

输入  t1_extract.py 的产物：intervals.csv / kernels.csv / axis.json
输出  stats.json（全部数值）、tables.md（表格片段）

口径  剔除 is_looped=1（跨迭代环绕），保留全局权重窗口，共 484 条。
用法  python3 t1_analyze.py --indir ./task1_out
"""

import argparse
import csv
import json
import math
import os


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(indir):
    iv = read_csv(os.path.join(indir, "intervals.csv"))
    kk = read_csv(os.path.join(indir, "kernels.csv"))
    with open(os.path.join(indir, "axis.json"), "r", encoding="utf-8") as f:
        axis = json.load(f)
    for r in iv:
        for k in ("size_bytes", "is_global", "is_looped", "k_start", "k_end", "cross_phase"):
            r[k] = int(r[k])
        for k in ("est_raw", "dur_us", "t_start_us", "t_end_us"):
            r[k] = float(r[k])
    for r in kk:
        r["dur_us"] = float(r["dur_us"])
        r["kid"] = int(r["kid"])
    return iv, kk, axis


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def describe(vals):
    v = sorted(vals)
    n = len(v)
    d = {
        "n": n,
        "sum": sum(v),
        "mean": sum(v) / n,
        "min": v[0],
        "p25": pct(v, 25),
        "median": pct(v, 50),
        "p75": pct(v, 75),
        "p90": pct(v, 90),
        "p99": pct(v, 99),
        "max": v[-1],
    }
    d["p75_over_p25"] = d["p75"] / d["p25"] if d["p25"] > 0 else float("inf")
    return d


def topk_share(vals, ks=(0.01, 0.05, 0.10, 0.20)):
    desc = sorted(vals, reverse=True)
    tot = sum(desc)
    out = {}
    for k in ks:
        m = max(1, int(len(desc) * k))
        out[f"top{int(k * 100)}pct"] = {"count": m, "share": sum(desc[:m]) / tot}
    return out


def gini(vals):
    x = sorted(vals)
    n = len(x)
    s = sum(x)
    if n == 0 or s == 0:
        return float("nan")
    cum = sum((i + 1) * xi for i, xi in enumerate(x))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def linfit_r2(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def tail_fit(vals, tail_pct=90):
    """对 x >= P{t} 的尾部做 log-log 与 semi-log 拟合，用于形态判定。"""
    v = sorted(vals)
    thr = pct(v, tail_pct)
    n = len(v)
    xs, ys = [], []
    for i, x in enumerate(v):
        if x >= thr and x > 0:
            ccdf = 1.0 - (i + 1) / n
            if ccdf <= 0:      # 最大点的 CCDF 为 0，log10 无定义
                continue
            xs.append(x)
            ys.append(ccdf)
    if len(xs) < 4:
        return {"note": "尾部样本不足"}
    sl_ll, _, r2_ll = linfit_r2([math.log10(x) for x in xs], [math.log10(y) for y in ys])
    sl_sl, _, r2_sl = linfit_r2(xs, [math.log10(y) for y in ys])
    return {
        "tail_threshold_us": thr,
        "tail_n": len(xs),
        "tail_total": sum(1 for x in v if x >= thr and x > 0),
        "loglog_slope": sl_ll,
        "loglog_alpha": -sl_ll,
        "loglog_r2": r2_ll,
        "semilog_slope": sl_sl,
        "semilog_r2": r2_sl,
    }


def detect_modality(vals, bins=24, gap=(3.0, 4.0)):
    """log10 尺度直方图判峰谷；gap 为待检验的低谷区间（默认 1–10 ms）。"""
    pos = [v for v in vals if v > 0]
    if len(pos) < 20:
        return {"shape": "样本不足"}
    logs = [math.log10(v) for v in pos]
    lo, hi = min(logs), max(logs)
    step = (hi - lo) / bins
    hist = [0] * bins
    for x in logs:
        hist[min(bins - 1, int((x - lo) / step))] += 1
    peaks = [i for i in range(bins)
             if hist[i] > 0 and (i == 0 or hist[i] >= hist[i - 1])
             and (i == bins - 1 or hist[i] >= hist[i + 1])]
    peaks = [i for i in peaks if hist[i] >= 0.15 * max(hist)]
    n_gap = sum(1 for x in logs if gap[0] <= x < gap[1])
    out = {
        "hist": hist,
        "bin_width_log10": step,
        "log10_range": [lo, hi],
        "n_in_gap_1_10ms": n_gap,
        "gap_share_pct": 100.0 * n_gap / len(logs),
    }
    if len(peaks) < 2:
        out["shape"] = "单峰"
        return out
    peaks.sort(key=lambda i: -hist[i])
    a, b = sorted(peaks[:2])
    valley = min(hist[a:b + 1])
    out["shape"] = "双峰" if valley <= 0.6 * min(hist[a], hist[b]) else "单峰（次峰不明显）"
    out["peaks_log10"] = [round(lo + (i + 0.5) * step, 2) for i in peaks[:2]]
    out["valley"] = valley
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="./task1_out")
    ap.add_argument("--tail-pct", type=float, default=90.0)
    ap.add_argument("--no-plot", action="store_true", help="已无绘图，保留参数以兼容旧命令")
    args = ap.parse_args()

    iv, kk, axis = load(args.indir)

    n_all = len(iv)
    n_global = sum(1 for r in iv if r["is_global"])
    n_looped = sum(1 for r in iv if r["is_looped"])
    clean = [r for r in iv if not r["is_looped"]]

    durs = [r["dur_us"] for r in clean]
    d_all = describe(durs)
    tk = topk_share(durs)
    g = gini(durs)
    tail = tail_fit(durs, args.tail_pct)
    shape = detect_modality(durs)

    # 三组分层
    grp = {
        "weight": [r for r in clean if r["is_global"] == 1],
        "cross": [r for r in clean if r["is_global"] == 0 and r["cross_phase"] == 1],
        "inner": [r for r in clean if r["is_global"] == 0 and r["cross_phase"] == 0],
    }
    tot = sum(durs)
    groups = {}
    for key, sub in grp.items():
        if sub:
            s = describe([r["dur_us"] for r in sub])
            s["share_pct"] = 100.0 * s["sum"] / tot
            groups[key] = s

    # 与 G10 估算对账：est 为 table[end]-table[start]，不含末端 kernel
    pairs = [(r["est_raw"] / 1000.0, r["dur_us"], r["k_end"] - r["k_start"])
             for r in clean if r["dur_us"] > 0]
    ratio = None
    if pairs:
        buckets = {"span=1": [], "span=2-10": [], "span>10": []}
        for e, d, span in pairs:
            if d <= 0 or e <= 0:
                continue
            key = "span=1" if span <= 1 else ("span=2-10" if span <= 10 else "span>10")
            buckets[key].append(e / d)
        ratio = {
            "n": len(pairs),
            "median": pct(sorted(e / d for e, d, _ in pairs if d > 0), 50),
            "by_span": {k: {"n": len(v), "median": pct(sorted(v), 50) if v else None}
                        for k, v in buckets.items()},
        }

    kernel_desc = describe([k["dur_us"] for k in kk if k["dur_us"] > 0])
    top_kernels = sorted(kk, key=lambda k: -k["dur_us"])[:15]

    stats = {
        "source": {
            "n_intervals_all": n_all,
            "n_global_records": n_global,
            "n_looped_records": n_looped,
            "n_clean": len(clean),
            "n_intermediate": len(clean) - len(grp["weight"]),
            "step_total_us": axis["step_total_us"],
            "n_kernel": axis["n_kernel"],
            "fwd_end_kernel": axis["fwd_end_kernel"],
            "kernel_time_unit_detected": axis["kernel_time_unit_detected"],
        },
        "window_duration_us": d_all,
        "topk": tk,
        "gini": g,
        "tail_fit": tail,
        "shape": shape,
        "groups": groups,
        "est_vs_measured_ratio": ratio,
        "kernel_duration_us": kernel_desc,
        "top15_kernels": [{k: t[k] for k in ("kid", "name", "layer_id", "layer_name", "dur_us")}
                          for t in top_kernels],
    }
    with open(os.path.join(args.indir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    md = []
    md.append("### 剔除口径\n")
    md.append("| 项 | 条数 | 说明 |")
    md.append("|---|---|---|")
    md.append(f"| 全部窗口 | {n_all} | 全局权重记录 {n_global} 条 |")
    md.append(f"| 跨迭代环绕 | {n_looped} | 起始序号大于结束序号，剔除 |")
    md.append(f"| 迭代内窗口 | **{len(clean)}** | 中间向量 {len(clean) - len(grp['weight'])} 条、全局权重 {len(grp['weight'])} 条 |")

    md.append("\n### 表 1 描述统计（484 条迭代内窗口）\n")
    md.append("| N | 总时长 (µs) | 均值 | 最小 | P25 | 中位数 | P75 | P90 | P99 | 最大 | P75/P25 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    md.append(f"| {d_all['n']} | {d_all['sum']:.0f} | {d_all['mean']:.0f} | {d_all['min']:.3f} | "
              f"{d_all['p25']:.1f} | {d_all['median']:.1f} | {d_all['p75']:.1f} | "
              f"{d_all['p90']:.1f} | {d_all['p99']:.1f} | {d_all['max']:.1f} | "
              f"{d_all['p75_over_p25']:.0f}× |")

    md.append("\n### 表 2 长尾指标\n")
    md.append("| 指标 | 值 |")
    md.append("|---|---|")
    for k, v in tk.items():
        md.append(f"| {k} 贡献 | {100*v['share']:.1f}%（{v['count']} 个） |")
    md.append(f"| Gini | {g:.3f} |")
    if tail.get("loglog_alpha") is not None:
        md.append(f"| 尾指数 α（P90 以上 {tail['tail_total']} 个，拟合 {tail['tail_n']} 个） | "
                  f"{tail['loglog_alpha']:.2f}，R²={tail['loglog_r2']:.3f} |")
        md.append(f"| semi-log R² | {tail['semilog_r2']:.3f} |")
    md.append(f"| 形态判定 | **{shape.get('shape', '?')}**（1–10 ms 仅 {shape.get('n_in_gap_1_10ms', 0)} 个） |")

    md.append("\n### 表 3 分层（三组）\n")
    md.append("| 类别 | 窗口数 | 总时长 (µs) | 占比 | 中位数 (µs) | 最大 (µs) |")
    md.append("|---|---|---|---|---|---|")
    for key, lab in (("cross", "激活·跨前后向"), ("weight", "权重·迭代内"), ("inner", "激活·相位内")):
        s = groups.get(key)
        if s:
            md.append(f"| {lab} | {s['n']} | {s['sum']:.0f} | **{s['share_pct']:.2f}%** | "
                      f"{s['median']:.1f} | {s['max']:.1f} |")

    if ratio:
        md.append("\n### 表 4 G10 估算 vs 物理时长（按跨度分层）\n")
        md.append("| 窗口跨度 | N | median(est/1000 ÷ dur) |")
        md.append("|---|---|---|")
        for k, v in ratio["by_span"].items():
            if v["n"]:
                md.append(f"| {k} | {v['n']} | {v['median']:.4f} |")
        md.append(f"| 全部 | {ratio['n']} | {ratio['median']:.4f} |")

    md.append("\n### 表 5 Top-15 慢 kernel\n")
    md.append("| kid | 名称 | 层 | 耗时 (µs) | 占比 |")
    md.append("|---|---|---|---|---|")
    tot_k = kernel_desc["sum"]
    for t in top_kernels:
        md.append(f"| {t['kid']} | {t['name']} | {t['layer_id']} {t['layer_name']} | "
                  f"{t['dur_us']:.1f} | {100*t['dur_us']/tot_k:.2f}% |")

    txt = "\n".join(md)
    with open(os.path.join(args.indir, "tables.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    print(txt)
    print(f"\n[OK] stats.json / tables.md 已写入 {args.indir}")


if __name__ == "__main__":
    main()
