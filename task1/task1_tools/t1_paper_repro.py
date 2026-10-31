#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_paper_repro.py —— 用论文原版口径复现 ResNet50 的张量周期图

为什么需要这个脚本
------------------
G10 自带的 figure_drawing/plot_tensor_period_distribution.py 与 plot_tensor_time_cdf.py
把 `sd_time` 直接画在 x 轴上并标注 "Inactive Time ($\\mu$s)"。但实测表明
`sd_time` 的数值是 **ns** 量级：

    ResNet50_TensorPeriodLog.py:  sd_time max = 1.6267e8
    同一模型的 cudnn_kernel_times.txt 求和 = 162671（µs）
    → 1.6267e8 / 1000 = 162670 µs ≈ 整步 162671 µs

也就是说 max(sd_time) 正好等于"整步"，说明它的单位是 ns。若照论文脚本原样画，
x 轴会显示 ~1e8 µs（≈100 秒），与 trace 实测的 152940 µs 差 1000 倍。

本脚本做三件事：
    1) 用 ast 安全解析 TensorPeriodLog.py 的 sd_size / sd_time（不 exec）
    2) 默认把 sd_time ÷1000 换成 µs，与 trace 的 µs 口径对齐（可关）
    3) 按论文的两种图样式输出：size–time 散点、按百分位排布的时长 CDF

设计上的两点兜底
----------------
* matplotlib 缺失时不会裸奔 Traceback：先落盘 JSON 摘要，再打印安装指引。
* `--no-plot` 可跳过绘图，先拿数字（无 matplotlib 也能完成统计部分）。

用法
----
    python3 t1_paper_repro.py --period-log ../ResNet50_TensorPeriodLog.py
    python3 t1_paper_repro.py --period-log ... --unit-scale 1.0     # 照论文原样（不换算）
    python3 t1_paper_repro.py --period-log ... --no-plot            # 只要数字
"""

import argparse
import ast
import json
import os


def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    ns = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    if "sd_time" not in ns or "sd_size" not in ns:
        raise SystemExit(f"[ERROR] {path} 里没找到 sd_time / sd_size")
    return ns["sd_time"], ns["sd_size"]


def pct(v, q):
    if not v:
        return float("nan")
    idx = (len(v) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(v) - 1)
    f = idx - lo
    return v[lo] * (1 - f) + v[hi] * f


def plot_all(ts, sizes, outdir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图 A：size–inactive time 双对数散点（论文 plot_tensor_period_distribution.py 样式）
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(ts, sizes, color="navy", marker="o", s=10, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inactive Time ($\\mu$s)")
    ax.set_ylabel("Size (byte)")
    ax.set_title(f"ResNet50 bs16: tensor inactive period vs size (n={len(ts)})")
    ax.grid(True, which="both", ls="-", lw=0.4, color="grey", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{tag}_figA_size_vs_time.{ext}"),
                    dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 图 B：论文 plot_tensor_time_cdf.py 的口径——横轴是"窗口百分位"，纵轴 log 时长
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [(i) / (len(ts) - 1) for i in range(len(ts))]
    ax.plot(xs, ts, color="#2e7d32", lw=3, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("% of Tensor Inactive Periods")
    ax.set_ylabel("Inactive Time ($\\mu$s)")
    ax.set_title("CDF of tensor inactive periods")
    ax.grid(which="major", axis="both", ls="-", lw=0.4, color="grey", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{tag}_figB_time_cdf.{ext}"),
                    dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period-log", default="../ResNet50_TensorPeriodLog.py")
    ap.add_argument("--outdir", default="./task1_out")
    ap.add_argument("--unit-scale", type=float, default=1e-3,
                    help="sd_time 乘以此系数后作为 µs；默认 1e-3（ns→µs），"
                         "传 1.0 则为论文脚本原样")
    ap.add_argument("--tag", default="paper", help="输出文件前缀")
    ap.add_argument("--no-plot", action="store_true", help="只算数字，不绘图")
    args = ap.parse_args()

    times, sizes = load_log(args.period_log)
    if len(times) != len(sizes):
        raise SystemExit("[ERROR] sd_time 与 sd_size 长度不一致")
    os.makedirs(args.outdir, exist_ok=True)

    t_us = [x * args.unit_scale for x in times]
    ts = sorted(t_us)

    summary = {
        "n_windows": len(ts),
        "unit_scale_applied": args.unit_scale,
        "x_axis_unit": "us" if abs(args.unit_scale - 1e-3) < 1e-12 else "raw(sd_time)",
        "min": ts[0], "median": pct(ts, 50), "p90": pct(ts, 90),
        "p99": pct(ts, 99), "max": ts[-1],
        "size_min": min(sizes), "size_max": max(sizes),
        "note": "TensorPeriodLog 含全局权重窗口，未做 is_global / is_looped 过滤；"
                "t1_analyze 的清洗口径以 interval.config 为准",
    }

    # 先落盘，保证绘图失败也能拿到数字
    json_path = os.path.join(args.outdir, f"{args.tag}_repro.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[统计完成] 窗口数 {summary['n_windows']}，摘要已写入 {json_path}")
    print(f"  sd_time 换算  : ×{args.unit_scale}（x 轴单位 {summary['x_axis_unit']}）")
    print(f"  最小值/中位数 : {summary['min']:.2f} / {summary['median']:.2f}")
    print(f"  P90 / P99     : {summary['p90']:.2f} / {summary['p99']:.2f}")
    print(f"  最大值        : {summary['max']:.2f}")

    if args.no_plot:
        print("\n[跳过绘图] 已指定 --no-plot。")
        return

    try:
        plot_all(ts, sizes, args.outdir, args.tag)
    except ImportError as e:
        print(f"\n[跳过绘图] 缺少绘图库：{e}")
        print("  数字部分已完成，上面的摘要可直接用。装好 matplotlib 后重跑本命令即可出图：")
        print("    sudo apt install -y python3-matplotlib        # WSL2/Ubuntu 推荐")
        print("    python3 -m pip install --user matplotlib      # 或用户级安装")
        return
    except Exception as e:  # noqa: BLE001
        print(f"\n[跳过绘图] 绘图失败：{e}")
        return

    print(f"\n[论文同源图] 已输出 {args.tag}_figA / {args.tag}_figB（pdf+png）于 {args.outdir}")
    print("\n说明：本图与论文 figure_drawing 的样式一致，但含全局权重窗口，")
    print("      故中位数明显高于 t1_analyze 的清洗口径；报告里两者要分别标注。")


if __name__ == "__main__":
    main()
