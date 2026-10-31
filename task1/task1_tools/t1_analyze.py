#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_analyze.py —— 任务一：非活跃窗口的分布特征 + 长尾形态 + 可卸载潜力

输入：t1_extract.py 的产物（task1_out/）
输出：
    task1_out/stats.json        全部数值，供报告引用
    task1_out/tables.md         Markdown 表格片段，可直接贴进报告
    task1_out/fig1_size_vs_time.pdf/.png     size–inactive time 散点（对齐论文 Figure）
    task1_out/fig2_ccdf.pdf/.png             窗口时长 CCDF + 尾部拟合
    task1_out/fig3_size_bucket.pdf/.png      按张量大小分桶的箱线图
    task1_out/fig4_idle_bytes.pdf/.png       空闲字节随时间曲线（扫描线）
    task1_out/fig5_kernel_cdf.pdf/.png       逐 kernel 耗时 CDF（Q1）

三个长尾指标（报告核心）
------------------------
Top-k 贡献率 : 降序累计到 k% 窗口时的时长占比
Gini         : 0=完全均匀，1=极度集中
CCDF 斜率 α  : 尾部 log-log 线性拟合，斜率 = -α；α 越小尾越重

形状判定不预设"长尾"，而是让数据说话：
    比较 log-log 与 semi-log 的拟合优度；两者都不好时判为双峰/多峰。
"""

import argparse
import csv
import json
import math
import os


# ---------------------------------------------------------------- 读数据
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


# ---------------------------------------------------------------- 统计
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
    return {
        "n": n,
        "sum": sum(v),
        "mean": sum(v) / n,
        "min": v[0],
        "p25": pct(v, 25),
        "median": pct(v, 50),
        "p75": pct(v, 75),
        "p90": pct(v, 90),
        "p95": pct(v, 95),
        "p99": pct(v, 99),
        "max": v[-1],
        "max_over_median": v[-1] / pct(v, 50) if pct(v, 50) > 0 else float("inf"),
    }


def topk_share(vals, ks=(0.01, 0.05, 0.10, 0.20)):
    desc = sorted(vals, reverse=True)
    tot = sum(desc)
    out = {}
    for k in ks:
        m = max(1, int(round(len(desc) * k)))
        out[f"top{int(k * 100)}pct"] = {
            "count": m,
            "share": sum(desc[:m]) / tot,
        }
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
    """一元线性回归，返回 (slope, intercept, R²)。"""
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


def ccdf(vals):
    v = sorted(vals)
    n = len(v)
    return v, [1.0 - (i + 1) / n for i in range(n)]


def tail_fit(vals, tail_pct=90):
    """对 x >= P{t} 的尾部做 log-log 与 semi-log 拟合，用于形状判定。"""
    v = sorted(vals)
    thr = pct(v, tail_pct)
    tail = [x for x in v if x >= thr and x > 0]
    n = len(v)
    xs, ys = [], []
    for i, x in enumerate(v):
        if x >= thr and x > 0:
            ccdf = 1.0 - (i + 1) / n
            if ccdf <= 0:      # 最大那个点的 CCDF 恰为 0，log10 会炸，必须跳过
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
        "loglog_slope": sl_ll,
        "loglog_alpha": -sl_ll,
        "loglog_r2": r2_ll,
        "semilog_slope": sl_sl,
        "semilog_r2": r2_sl,
    }


def detect_modality(vals, bins=24):
    """在 log10 尺度上做直方图，检测单峰/双峰。"""
    pos = [v for v in vals if v > 0]
    if len(pos) < 20:
        return {"shape": "样本不足"}
    logs = [math.log10(v) for v in pos]
    lo, hi = min(logs), max(logs)
    if hi <= lo:
        return {"shape": "退化（全部相同）"}
    step = (hi - lo) / bins
    hist = [0] * bins
    for x in logs:
        hist[min(bins - 1, int((x - lo) / step))] += 1
    # 找局部极大
    peaks = [i for i in range(bins)
             if hist[i] > 0 and (i == 0 or hist[i] >= hist[i - 1])
             and (i == bins - 1 or hist[i] >= hist[i + 1])]
    peaks = [i for i in peaks if hist[i] >= 0.15 * max(hist)]
    if len(peaks) < 2:
        return {"shape": "单峰", "peaks": peaks, "hist": hist,
                "log10_range": [lo, hi]}
    # 取最高的两个峰，检查中间是否有谷
    peaks.sort(key=lambda i: -hist[i])
    a, b = sorted(peaks[:2])
    valley = min(hist[a:b + 1]) if b > a else 0
    bimodal = valley <= 0.6 * min(hist[a], hist[b])
    return {
        "shape": "双峰" if bimodal else "单峰（次峰不明显）",
        "peaks": peaks[:3],
        "valley": valley,
        "hist": hist,
        "log10_range": [lo, hi],
    }


def coverage(intervals, step_total_us):
    """扫描线：任一时刻处于非活跃状态的张量总字节数。"""
    ev = []
    for r in intervals:
        if r["t_end_us"] <= r["t_start_us"]:
            continue
        ev.append((r["t_start_us"], +r["size_bytes"]))
        ev.append((r["t_end_us"], -r["size_bytes"]))
    if not ev:
        return {}
    ev.sort()
    cur = 0
    peak = 0
    area = 0.0
    prev = ev[0][0]
    for t, d in ev:
        area += cur * (t - prev)
        cur += d
        if cur > peak:
            peak = cur
        prev = t
    return {
        "peak_idle_bytes": peak,
        "mean_idle_bytes": area / step_total_us if step_total_us > 0 else float("nan"),
        "idle_byte_us": area,
        "mean_idle_over_peak_pct": 100.0 * (area / step_total_us) / peak if peak > 0 else float("nan"),
    }


# ---------------------------------------------------------------- 绘图
def make_plots(iv_clean, kk, cov, axis, outdir, shape_info, tail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图1：size vs inactive time（论文同款，x/y 双对数）
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for role, color, marker in (("activation", "navy", "o"),
                                ("grad", "darkorange", "s"),
                                ("weight", "green", "x"),
                                ("d_weight", "purple", "^"),
                                ("unknown", "grey", ".")):
        pts = [(r["dur_us"], r["size_bytes"]) for r in iv_clean if r["role"] == role]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       s=14, c=color, marker=marker, label=role, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inactive Time ($\\mu$s)")
    ax.set_ylabel("Size (byte)")
    ax.set_title("ResNet50 bs16: tensor inactive period vs size")
    ax.grid(True, which="both", ls="-", lw=0.4, color="grey", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig1_size_vs_time.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 图2：CCDF（log-log）+ 尾部拟合线
    fig, ax = plt.subplots(figsize=(8, 6))
    v = sorted(r["dur_us"] for r in iv_clean if r["dur_us"] > 0)
    xs, ys = ccdf(v)
    ax.plot(xs, ys, color="#1f4e79", lw=2, label="empirical CCDF")
    if tail.get("loglog_slope") is not None and not math.isnan(tail.get("loglog_slope", float("nan"))):
        a = tail["loglog_alpha"]
        x0 = tail["tail_threshold_us"]
        y0 = sum(ys[max(0, len(ys) - tail["tail_n"]):]) / max(1, tail["tail_n"])
        xx = [x0, max(xs)]
        yy = [y0 * (x / x0) ** (-a) for x in xx]
        ax.plot(xx, yy, "r--", lw=1.5,
                label=f"tail fit: $\\alpha$={a:.2f}, $R^2$={tail['loglog_r2']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inactive time ($\\mu$s)")
    ax.set_ylabel("P(X > x)")
    ax.set_title(f"CCDF of inactive windows (shape: {shape_info.get('shape','?')})")
    ax.grid(True, which="both", ls="-", lw=0.4, color="grey", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig2_ccdf.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 图3：按 size 分桶的箱线图
    edges = [(0, 1 << 20, "<1MB"), (1 << 20, 10 << 20, "1-10MB"),
             (10 << 20, 50 << 20, "10-50MB"), (50 << 20, 1 << 62, ">50MB")]
    data, labels = [], []
    for lo, hi, lab in edges:
        grp = [r["dur_us"] for r in iv_clean if lo <= r["size_bytes"] < hi]
        if grp:
            data.append(grp)
            labels.append(f"{lab}\n(n={len(grp)})")
    if data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_yscale("log")
        ax.set_ylabel("Inactive time ($\\mu$s)")
        ax.set_title("Inactive time by tensor size bucket")
        ax.grid(True, axis="y", ls="-", lw=0.4, color="grey", alpha=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"fig3_size_bucket.{ext}"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 图4：空闲字节随时间（扫描线）
    if cov:
        ev = []
        for r in iv_clean:
            if r["t_end_us"] <= r["t_start_us"]:
                continue
            ev.append((r["t_start_us"], +r["size_bytes"]))
            ev.append((r["t_end_us"], -r["size_bytes"]))
        ev.sort()
        ts, bs = [0.0], [0]
        cur = 0
        for t, d in ev:
            cur += d
            ts.append(t)
            bs.append(cur)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(ts, bs, color="#2e75b6", alpha=0.55, step="post")
        ax.set_xlabel("Time within one iteration ($\\mu$s)")
        ax.set_ylabel("Idle bytes")
        ax.set_title(f"Resident idle bytes (peak={cov['peak_idle_bytes']/2**20:.1f} MB, "
                     f"mean={cov['mean_idle_bytes']/2**20:.1f} MB)")
        fwd_end = axis.get("fwd_end_kernel")
        if fwd_end is not None and fwd_end < len(axis["prefix_us"]):
            ax.axvline(axis["prefix_us"][fwd_end], color="red", ls="--", lw=1.2,
                       label="forward/backward boundary")
            ax.legend(fontsize=9)
        ax.grid(True, ls="-", lw=0.4, color="grey", alpha=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"fig4_idle_bytes.{ext}"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 图5：逐 kernel 耗时 CDF（Q1）
    durs = sorted(k["dur_us"] for k in kk if k["dur_us"] > 0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(durs, [(i + 1) / len(durs) for i in range(len(durs))],
            color="#e05a2b", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Kernel execution time ($\\mu$s)")
    ax.set_ylabel("Cumulative fraction of kernels")
    ax.set_title("CDF of per-kernel execution time (G10 scheduling units)")
    ax.grid(True, which="both", ls="-", lw=0.4, color="grey", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig5_kernel_cdf.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="./task1_out")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tail-pct", type=float, default=90.0)
    args = ap.parse_args()

    iv, kk, axis = load(args.indir)

    n_all = len(iv)
    n_global = sum(1 for r in iv if r["is_global"])
    n_looped = sum(1 for r in iv if r["is_looped"])
    iv_clean = [r for r in iv if not r["is_global"] and not r["is_looped"]]

    durs = [r["dur_us"] for r in iv_clean]
    d_all = describe(durs)
    tk = topk_share(durs)
    g = gini(durs)
    tail = tail_fit(durs, args.tail_pct)
    shape = detect_modality(durs)
    cov = coverage(iv_clean, axis["step_total_us"])

    # 交叉验证：G10 的 est/1000 与本脚本的物理 dur 对比。
    # 注意 G10 的实现是 est = table[end] - table[start]（table 为 ns 前缀和），
    # 等于 sum(kernel[start..end-1])，**不含末端 kernel[end]**；
    # 本脚本的 dur = sum(kernel[start..end])（物理空闲期，含两端）。
    # 因此跨度为 1 的短窗口会被 G10 少算一个 kernel，必须按跨度分层看。
    pairs = [(r["est_raw"] / 1000.0, r["dur_us"], r["k_end"] - r["k_start"])
             for r in iv_clean if r["dur_us"] > 0]
    ratio = None
    if pairs:
        buckets = {"span=1": [], "span=2-10": [], "span>10": []}
        for e, d, span in pairs:
            if d <= 0 or e <= 0:
                continue
            key = "span=1" if span <= 1 else ("span=2-10" if span <= 10 else "span>10")
            buckets[key].append(e / d)
        ratio = {"n": len(pairs),
                 "median": pct(sorted(e / d for e, d, _ in pairs if d > 0), 50),
                 "by_span": {k: {"n": len(v),
                                 "median": pct(sorted(v), 50) if v else None}
                             for k, v in buckets.items()}}

    # 分层（跨相位 vs 相内）
    cross = [r["dur_us"] for r in iv_clean if r["cross_phase"] == 1]
    inner = [r["dur_us"] for r in iv_clean if r["cross_phase"] == 0]
    layer_stats = {
        "cross_phase": describe(cross) if cross else None,
        "inner_phase": describe(inner) if inner else None,
    }

    # 按角色
    by_role = {}
    for role in ("activation", "grad", "weight", "d_weight", "unknown"):
        grp = [r["dur_us"] for r in iv_clean if r["role"] == role]
        if grp:
            by_role[role] = describe(grp)

    kernel_desc = describe([k["dur_us"] for k in kk if k["dur_us"] > 0])
    top_kernels = sorted(kk, key=lambda k: -k["dur_us"])[:15]

    stats = {
        "source": {
            "n_intervals_all": n_all,
            "n_global_records": n_global,
            "n_looped_records": n_looped,
            "n_clean": len(iv_clean),
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
        "coverage": cov,
        "est_vs_measured_ratio": ratio,
        "by_phase": layer_stats,
        "by_role": by_role,
        "kernel_duration_us": kernel_desc,
        "top15_kernels": [{k: t[k] for k in ("kid", "name", "layer_id", "layer_name", "dur_us")}
                          for t in top_kernels],
    }

    with open(os.path.join(args.indir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Markdown 表格
    def mb(x):
        return f"{x/2**20:.2f}"

    md = []
    md.append("### 表 1 非活跃窗口描述统计（清洗后）\n")
    md.append("| N | 总时长(µs) | 均值 | 中位数 | P90 | P95 | P99 | 最大 | max/median |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    md.append(f"| {d_all['n']} | {d_all['sum']:.0f} | {d_all['mean']:.1f} | "
              f"{d_all['median']:.1f} | {d_all['p90']:.1f} | {d_all['p95']:.1f} | "
              f"{d_all['p99']:.1f} | {d_all['max']:.1f} | {d_all['max_over_median']:.1f}× |")

    md.append("\n### 表 2 长尾三指标\n")
    md.append("| 指标 | 值 | 判读 |")
    md.append("|---|---|---|")
    for k, v in tk.items():
        md.append(f"| {k} 贡献率 | {100*v['share']:.1f}% （{v['count']} 个窗口） | |")
    md.append(f"| Gini | {g:.3f} | 0=均匀，1=极度集中 |")
    if tail.get("loglog_alpha") is not None:
        md.append(f"| CCDF 尾指数 α | {tail['loglog_alpha']:.2f} （R²={tail['loglog_r2']:.3f}） | 越小尾越重 |")
        md.append(f"| semi-log 拟合 R² | {tail['semilog_r2']:.3f} | 高于 log-log 则更接近指数尾 |")
    md.append(f"| 形状判定 | **{shape.get('shape','?')}** | 由 log10 直方图峰谷判定 |")

    md.append("\n### 表 3 可卸载潜力（扫描线）\n")
    md.append("| 量 | 值 |")
    md.append("|---|---|")
    md.append(f"| 峰值同时空闲 | {mb(cov['peak_idle_bytes'])} MB |")
    md.append(f"| 平均同时空闲 | {mb(cov['mean_idle_bytes'])} MB |")
    md.append(f"| 平均/峰值 | {cov['mean_idle_over_peak_pct']:.1f}% |")

    md.append("\n### 表 4 分层（跨前后向 vs 相内）\n")
    md.append("| 类别 | N | 中位数(µs) | P90(µs) | 最大(µs) |")
    md.append("|---|---|---|---|---|")
    for lab, key in (("跨前后向", "cross_phase"), ("相位内", "inner_phase")):
        s = layer_stats[key]
        if s:
            md.append(f"| {lab} | {s['n']} | {s['median']:.1f} | {s['p90']:.1f} | {s['max']:.1f} |")

    if ratio:
        md.append("\n### 表 5 G10 估算 vs 物理窗口时长（按窗口跨度分层）\n")
        md.append("G10 的 `time_estimated = table[end] - table[start]`（table 为 ns 前缀和），")
        md.append("等于 `sum(kernel[start..end-1])`，**不含末端 kernel[end]**；")
        md.append("本表的 `dur_us` 为物理空闲期 `sum(kernel[start..end])`。\n")
        md.append("| 窗口跨度 (end-start) | N | median(est/1000 ÷ dur) | 判读 |")
        md.append("|---|---|---|---|")
        for k, v in ratio["by_span"].items():
            if v["n"] == 0:
                continue
            verdict = "两者口径一致" if v["median"] and v["median"] > 0.9 else "G10 少算末端 kernel，显著低估"
            md.append(f"| {k} | {v['n']} | {v['median']:.4f} | {verdict} |")
        md.append(f"| **全部** | {ratio['n']} | {ratio['median']:.4f} | 长窗口主导，整体接近 1 |")

    md.append("\n### 表 6 Top-15 慢 kernel\n")
    md.append("| kid | 名称 | 层 | 耗时(µs) | 占比 |")
    md.append("|---|---|---|---|---|")
    tot_k = kernel_desc["sum"]
    for t in top_kernels:
        md.append(f"| {t['kid']} | {t['name']} | {t['layer_id']} {t['layer_name']} | "
                  f"{t['dur_us']:.1f} | {100*t['dur_us']/tot_k:.2f}% |")

    txt = "\n".join(md)
    with open(os.path.join(args.indir, "tables.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    if not args.no_plot:
        try:
            make_plots(iv_clean, kk, cov, axis, args.indir, shape, tail)
            print("[OK] 图已输出 fig1..fig5 (pdf/png)")
        except ImportError as e:
            print(f"[跳过绘图] 缺少绘图库：{e}")
            print("  数字与表格已写入 stats.json / tables.md，可直接用于报告。")
            print("  装库后重跑：sudo apt install -y python3-matplotlib")
            print("             或 python3 -m pip install --user matplotlib")
        except Exception as e:  # noqa: BLE001
            print(f"[跳过绘图] 绘图失败：{e}")
            print("  数字与表格已写入 stats.json / tables.md，不受影响。")

    print(txt)
    print(f"\n[OK] stats.json / tables.md 已写入 {args.indir}")


if __name__ == "__main__":
    main()
