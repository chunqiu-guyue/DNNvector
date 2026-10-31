#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_interval_config.py
=======================
G10 的 Tensor Vitality Analyzer 在「分析 / 模拟」阶段（阶段 B，`is_simulation=1`）
会写出 ``interval.config``，路径为 ``<output_folder>/statistics/interval.config``
（例如 ``G10/results/ResNet50/statistics/interval.config``）。

这个文件里，对**每一个张量**都记录了：
  * 生灭信息（Liveness: Birth / Death，即张量在哪些 kernel 序号之间存活）；
  * 若干个 **Hidding_Interval**（张量“活着但闲置”的非活跃窗口），每个窗口带一行
    ``Estimated Time:<t>``（窗口时长，但注意：经核对源码，该值并非物理毫秒，
    而是被 G10 按 `execution_cycles/(GPU_frequency*1000)` 放大过的中间量），
    并天然属于某个 ``Size in byte``（张量字节数）的张量。

另外，阶段 A（profiling）会产出 ``cudnn_kernel_times.txt``，每行是
``<kernel序号> <实测耗时ms>``，这是**逐算子真实实测时间**，作为动态时间真值。

本脚本支持「单数据源」与「双数据源」两种用法：

1) 只给 --input（== 原版本，任务一）：
   - 重建与 G10 自带 ``*_TensorPeriodLog.py`` 等价的 ``sd_time`` / ``sd_size``；
   - 打印非活跃窗口分布统计（个数、总/均值/中位数/分位数、长尾占比）；
   - 导出每个窗口的 CSV，并（可选）画 size–time 散点 + 窗口时长 CDF。

2) 再给 --kernel-times（== 增强版，兼顾任务二交叉验证）：
   - 解析逐 kernel 实测时间，输出算子时序分布（总/均值/中位数/分位数 + Top-N 最慢 kernel）；
   - 对每个 Hidding_Interval，把 [start, end] 之间的真实 kernel 时间累加，得到
     **物理窗口时长 physical_ms**（这才是可读的“这个闲置窗口实际占多少 ms”）；
   - 用 physical_ms 与 interval.config 的 Estimated Time 做**交叉验证**，给出两者
     的中位数比例，量化 G10 把时间“放大”到了多少倍（这正是任务二要求的
     “与实际训练测量交叉验证”的切入点）；
   - 在 PDF 里追加：算子耗时分布图、以及 Estimated-Time-vs-physical-ms 的对照散点。

依赖：解析/统计/CSV 仅用 Python 标准库；matplotlib 仅用于出图。

用法
----
    # 只解析 interval.config（任务一）
    python3 parse_interval_config.py --input interval.config

    # 双数据源：交叉验证（任务一 + 任务二铺垫）
    python3 parse_interval_config.py \
        --input     interval.config \
        --kernel-times cudnn_kernel_times.txt

    # 在 WSL2 上指向自己的文件
    python3 parse_interval_config.py \
        --input ~/26DNNvectorG10/G10/results/ResNet50/statistics/interval.config \
        --kernel-times ~/26DNNvectorG10/G10/results/ResNet50/cudnn_kernel_times.txt
"""

import argparse
import csv
import os
import re
import sys


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------

# tensor<id> Is weight (global)?: 0, Size in byte: 12345, Range:...
RE_TENSOR = re.compile(r"tensor(\d+).*?Size in byte:\s*(\d+)")
# Liveness: Birth: 5, Death: 12.   （Death 可能为 -1 表示到结尾仍存活）
RE_LIVENESS = re.compile(r"Liveness:\s*Birth:\s*(-?\d+),\s*Death:\s*(-?\d+)")
# interval 0: 6--------11
RE_INTERVAL = re.compile(r"interval\s+(\d+):\s*(\d+)\s*--------\s*(\d+)")
# Estimated Time: 41.562111
RE_ETIME = re.compile(r"Estimated Time:\s*([\d.eE+-]+)")
# cudnn_kernel_times.txt:  "000 2.251232 ms"
RE_KERNEL = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)")


def parse_interval_config(path, kernel_times=None):
    """返回 (intervals, liveness, meta)。

    intervals : list[dict]  每个非活跃窗口一条记录
    liveness  : dict[int->(birth,death,is_global)]
    meta      : dict        汇总信息
    """
    if not os.path.isfile(path):
        sys.exit(f"[错误] 找不到 interval.config：{path}\n"
                 f"        请用 --input 指定正确路径（见脚本头部用法说明）。")

    intervals = []
    liveness = {}
    n_tensors = 0
    n_global = 0

    cur_tid = None
    cur_size = None
    pending = None  # 正在收集 (start, end) 的窗口，等待 Estimated Time 收尾

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = RE_TENSOR.search(line)
            if m:
                cur_tid = int(m.group(1))
                cur_size = int(m.group(2))
                n_tensors += 1
                continue

            m = RE_LIVENESS.search(line)
            if m:
                birth = int(m.group(1))
                death = int(m.group(2))
                liveness[cur_tid] = (birth, death, False)
                continue

            if line.strip() == "Global!":
                # 全局权重（如卷积核）没有 Birth/Death 行，单独标记
                liveness[cur_tid] = (-1, -1, True)
                n_global += 1
                continue

            m = RE_INTERVAL.search(line)
            if m:
                start = int(m.group(2))
                end = int(m.group(3))
                pending = (start, end)
                continue

            m = RE_ETIME.search(line)
            if m and pending is not None:
                est = float(m.group(1))
                is_global = liveness.get(cur_tid, (None, None, False))[2]
                # 把窗口 [start, end] 之间的真实 kernel 时间累加，得到物理时长
                phys = window_physical_ms(kernel_times, pending[0], pending[1])
                intervals.append({
                    "tensor_id": cur_tid,
                    "is_global": is_global,
                    "size_bytes": cur_size,
                    "est_time": est,          # interval.config 里的（被放大的）值
                    "physical_ms": phys,      # 由 cudnn_kernel_times 累加得到的物理 ms
                    "kernel_start": pending[0],
                    "kernel_end": pending[1],
                    "is_looped": pending[0] > pending[1],
                })
                pending = None
                continue

    meta = {
        "n_tensors": n_tensors,
        "n_global": n_global,
        "n_intervals": len(intervals),
    }
    return intervals, liveness, meta


def parse_kernel_times(path):
    """解析 cudnn_kernel_times.txt，返回 {kernel_idx: time_ms}。

    文件格式：每行 ``<序号> <耗时> ms``，序号允许前导零；
    以 ``#`` 开头或解析不到序号的行会被忽略。
    """
    if path is None:
        return None
    if not os.path.isfile(path):
        sys.exit(f"[错误] 找不到 kernel time 文件：{path}\n"
                 f"        请检查 --kernel-times 路径。")

    times = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            m = RE_KERNEL.search(line)
            if m:
                idx = int(m.group(1))
                t = float(m.group(2))
                times[idx] = t
    return times


def window_physical_ms(kernel_times, start, end):
    """累加 [start, end] 区间内的 kernel 实测耗时（ms）。

    - 普通窗口 start <= end：直接求和；
    - 循环窗口 start > end（跨迭代边界，is_looped）：求和 [start..最大序号] + [0..end]；
    - 没有传入 kernel_times 时返回 None；缺号 kernel 自动跳过。
    """
    if not kernel_times:
        return None
    if not kernel_times:
        return None
    idxs = sorted(kernel_times.keys())
    lo, hi = idxs[0], idxs[-1]
    if start <= end:
        rng = range(start, end + 1)
    else:
        rng = list(range(start, hi + 1)) + list(range(lo, end + 1))
    total = 0.0
    for i in rng:
        if i in kernel_times:
            total += kernel_times[i]
    return total


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------

def percentile(sorted_vals, q):
    """线性插值百分位（q 取 0~100）。sorted_vals 必须已升序。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_intervals(intervals):
    """非活跃窗口（Hidding_Interval）分布统计。"""
    if not intervals:
        print("[警告] 未解析到任何 Hidding_Interval，请检查输入文件。")
        return

    times = sorted(iv["est_time"] for iv in intervals)
    sizes = sorted(iv["size_bytes"] for iv in intervals)
    total_time = sum(times)
    total_size = sum(sizes)

    print("=" * 60)
    print("非活跃窗口（Hidding_Interval）分布统计")
    print("=" * 60)
    print(f"窗口总数                : {len(times)}")
    print(f"Estimated Time 总 (raw) : {total_time:.4f}")
    print(f"单窗口 Estimated 均值   : {total_time / len(times):.4f}")
    print(f"单窗口 Estimated 中位数 : {percentile(times, 50):.4f}")
    for q in (90, 95, 99):
        print(f"单窗口 Estimated P{q}  : {percentile(times, q):.4f}")
    print(f"单窗口 Estimated 最大   : {times[-1]:.4f}")
    print(f"单窗口 Estimated 最小   : {times[0]:.4f}")

    print("-" * 60)
    print(f"张量字节数  均值 (B)    : {total_size / len(sizes):.1f}")
    print(f"张量字节数  中位数 (B)  : {percentile(sizes, 50):.1f}")
    print(f"张量字节数  最大 (B)    : {sizes[-1]:.1f}")

    # 长尾 / 重要性：按窗口 Estimated 降序，累计到总时长的 80% 需要前多少窗口
    desc = sorted(times, reverse=True)
    cum = 0.0
    top_n = 0
    for t in desc:
        cum += t
        top_n += 1
        if cum >= 0.8 * total_time:
            break
    print("-" * 60)
    print(f"长尾特征：前 {top_n} 个（占窗口数 "
          f"{100.0 * top_n / len(times):.1f}%）窗口")
    print(f"          贡献了 80% 的非活跃总时长(Estimated)。")
    print("=" * 60)


def summarize_kernel_times(kernel_times, top_n=15):
    """逐 kernel 实测时间的分布统计（任务一『采样算子时序』的直接产物）。"""
    if not kernel_times:
        return
    vals = sorted(kernel_times.values())
    total = sum(vals)
    print("=" * 60)
    print("逐 kernel 实测时间（cudnn_kernel_times.txt）分布统计")
    print("=" * 60)
    print(f"kernel 总数             : {len(vals)}")
    print(f"总耗时 (ms)             : {total:.4f}")
    print(f"单 kernel 均值 (ms)     : {total / len(vals):.4f}")
    print(f"单 kernel 中位数 (ms)   : {percentile(vals, 50):.4f}")
    for q in (90, 95, 99):
        print(f"单 kernel P{q} (ms)    : {percentile(vals, q):.4f}")
    print(f"单 kernel 最大 (ms)     : {vals[-1]:.4f}")
    print(f"单 kernel 最小 (ms)     : {vals[0]:.4f}")
    print("-" * 60)
    print(f"Top-{top_n} 最慢 kernel（序号 : 耗时ms : 占总耗时%）:")
    desc = sorted(kernel_times.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    for idx, t in desc:
        print(f"    {idx:>4} : {t:>14.4f} : {100.0 * t / total:5.2f}%")
    print("=" * 60)


def cross_validate(intervals):
    """用物理窗口时长(physical_ms) 与 interval.config 的 Estimated Time 交叉验证。

    返回 median_ratio = median(Estimated / physical_ms)，量化 G10 把时间放大倍数。
    """
    pairs = [(iv["est_time"], iv["physical_ms"])
             for iv in intervals
             if iv["physical_ms"] is not None and iv["physical_ms"] > 0
             and iv["est_time"] > 0]
    if not pairs:
        print("[提示] 无可交叉验证的窗口（需同时提供 interval.config 与 "
              "kernel time，且窗口非循环）。")
        return None

    ratios = sorted(e / p for e, p in pairs)
    median_ratio = percentile(ratios, 50)
    print("=" * 60)
    print("交叉验证：interval.config(Estimated) vs 实测物理窗口时长")
    print("=" * 60)
    print(f"参与比对窗口数          : {len(pairs)}")
    print(f"median(Estimated/phys)  : {median_ratio:.6f}")
    if median_ratio >= 1:
        print(f"  -> Estimated 约为物理 ms 的 {median_ratio:.1f} 倍"
              f"（interval.config 里的时间被放大过，非真实毫秒）。")
    else:
        print(f"  -> Estimated 约为物理 ms 的 {median_ratio:.6f} 倍"
              f"（即物理 ms ≈ Estimated / {median_ratio:.6f} = "
              f"Estimated × {1.0/median_ratio:.1f}）。")
    print(f"  （这个固定线性比例由 G10 内部 `execution_cycles/(GPU频率*1000)` "
          f"决定；脚本只如实汇报，不做假设。要得到可读的物理 ms，")
    print(f"   用本脚本算出的 physical_ms 列即可，无需手动换算。）")
    print("-" * 60)
    print("还原示例（取 median 比例估算每个窗口的物理 ms）：")
    shown = 0
    for iv in intervals:
        if iv["physical_ms"] is not None and iv["physical_ms"] > 0 and shown < 5:
            print(f"    tensor{iv['tensor_id']} 区间[{iv['kernel_start']}"
                  f"..{iv['kernel_end']}] : 实测={iv['physical_ms']:.4f}ms, "
                  f"Estimated={iv['est_time']:.4f}")
            shown += 1
    print("=" * 60)
    return median_ratio


# --------------------------------------------------------------------------
# 输出：CSV + 可选图片
# --------------------------------------------------------------------------

def write_csv(intervals, kernel_times, outdir, src_name):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, src_name + "_intervals.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor_id", "is_global_weight", "size_bytes",
                    "est_time_raw", "physical_ms",
                    "kernel_start", "kernel_end", "is_looped"])
        for iv in intervals:
            phys = "" if iv["physical_ms"] is None else f"{iv['physical_ms']:.6f}"
            w.writerow([iv["tensor_id"], int(iv["is_global"]),
                        iv["size_bytes"], iv["est_time"],
                        phys,
                        iv["kernel_start"], iv["kernel_end"],
                        int(iv["is_looped"])])
    print(f"[已写出 CSV] {csv_path}")
    return csv_path


def try_plot(intervals, kernel_times, outdir, src_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:
        print(f"[跳过绘图] 未安装 matplotlib（{e}）。"
              f"可用 `pip3 install matplotlib` 后重跑（去掉 --no-plot）。")
        return None

    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, src_name + "_analysis.pdf")

    est_times = [iv["est_time"] for iv in intervals]
    sizes = [iv["size_bytes"] for iv in intervals]
    phys = [iv["physical_ms"] for iv in intervals
            if iv["physical_ms"] is not None and iv["physical_ms"] > 0]
    phys_pairs = [(iv["est_time"], iv["physical_ms"]) for iv in intervals
                  if iv["physical_ms"] is not None and iv["physical_ms"] > 0]

    with PdfPages(pdf_path) as pdf:
        # 图1：size vs Estimated-time 双对数散点（论文 tensor-period 分布）
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(sizes, est_times, color="navy", marker="o", s=8)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Size (byte)")
        ax1.set_ylabel("Estimated Time (raw, scaled units)")
        ax1.set_title("Tensor inactive windows: size vs time (interval.config)")
        ax1.grid(True, which="both", linestyle="-", linewidth=0.5, color="grey")
        fig1.tight_layout()
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # 图2：窗口 Estimated-time 的经验 CDF
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        s = sorted(est_times)
        y = [i / (len(s) - 1) for i in range(len(s))] if len(s) > 1 else [0.0]
        ax2.plot(s, y, color="#23a8eb", linewidth=2)
        ax2.set_xscale("log")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Estimated Time (raw, scaled units)")
        ax2.set_ylabel("Cumulative fraction of windows")
        ax2.set_title("CDF of inactive-window time")
        ax2.grid(True, which="both", linestyle="-", linewidth=0.5, color="grey")
        fig2.tight_layout()
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # 图3（仅双数据源）：逐 kernel 实测耗时 CDF
        if kernel_times:
            kvals = sorted(kernel_times.values())
            fig3 = plt.figure(figsize=(8, 6))
            ax3 = fig3.add_subplot(111)
            y3 = [i / (len(kvals) - 1) for i in range(len(kvals))] if len(kvals) > 1 else [0.0]
            ax3.plot(kvals, y3, color="#e05a2b", linewidth=2)
            ax3.set_xscale("log")
            ax3.set_ylim(0, 1)
            ax3.set_xlabel("Kernel execution time (ms)")
            ax3.set_ylabel("Cumulative fraction of kernels")
            ax3.set_title("CDF of per-kernel execution time")
            ax3.grid(True, which="both", linestyle="-", linewidth=0.5, color="grey")
            fig3.tight_layout()
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

        # 图4（仅双数据源）：Estimated vs 物理窗口时长 对照散点（交叉验证）
        if phys_pairs:
            fig4 = plt.figure(figsize=(8, 8))
            ax4 = fig4.add_subplot(111)
            ex = [e for e, p in phys_pairs]
            px = [p for e, p in phys_pairs]
            ax4.scatter(px, ex, color="forestgreen", marker="o", s=8)
            ax4.set_xscale("log")
            ax4.set_yscale("log")
            # 参考线 y = x * median_ratio
            import statistics
            ratio = statistics.median([e / p for e, p in phys_pairs])
            lim = max(max(ex), max(px)) * 1.2
            ax4.plot([0, lim], [0, lim * ratio], color="red",
                     linestyle="--", linewidth=1.2,
                     label=f"y = {ratio:.1f} * x")
            ax4.set_xlabel("Physical window time (ms, from cudnn_kernel_times)")
            ax4.set_ylabel("Estimated Time (interval.config, raw units)")
            ax4.set_title("Cross-validation: Estimated vs physical window time")
            ax4.legend()
            ax4.grid(True, which="both", linestyle="-", linewidth=0.5, color="grey")
            fig4.tight_layout()
            pdf.savefig(fig4, bbox_inches="tight")
            plt.close(fig4)

    print(f"[已写出图片] {pdf_path}")
    return pdf_path


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(
        here, "G10", "results", "ResNet50", "statistics", "interval.config")
    default_kernel = os.path.join(
        here, "G10", "results", "ResNet50", "cudnn_kernel_times.txt")

    p = argparse.ArgumentParser(
        description="解析 G10 的 interval.config（+ 可选 kernel time 交叉验证），"
                    "统计张量非活跃窗口分布。")
    p.add_argument("--input", default=default_input,
                   help="interval.config 的路径")
    p.add_argument("--kernel-times", default=None,
                   help="cudnn_kernel_times.txt 路径（提供后做交叉验证 + "
                        "算子时序统计）。默认尝试同目录下的同名文件。")
    p.add_argument("--outdir", default=None,
                   help="输出目录（默认与 --input 同目录）")
    p.add_argument("--no-plot", action="store_true",
                   help="不生成图片")
    args = p.parse_args()

    # 若未显式给 kernel-times，尝试与 input 同目录的默认名
    kernel_path = args.kernel_times
    if kernel_path is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                             "cudnn_kernel_times.txt")
        if os.path.isfile(guess):
            kernel_path = guess

    kernel_times = parse_kernel_times(kernel_path)

    intervals, liveness, meta = parse_interval_config(args.input, kernel_times)

    print(f"[解析完成] 张量总数={meta['n_tensors']} "
          f"(其中全局权重={meta['n_global']})，"
          f"非活跃窗口数={meta['n_intervals']}")
    if kernel_times is not None:
        print(f"[已载入 kernel 实测时间] kernel 数={len(kernel_times)}"
              + (f"（来自 {kernel_path}）" if kernel_path else ""))
        summarize_kernel_times(kernel_times)
    else:
        print("[提示] 未提供 kernel time 文件，跳过算子时序统计与交叉验证。")

    summarize_intervals(intervals)

    if kernel_times is not None:
        cross_validate(intervals)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    src_name = os.path.splitext(os.path.basename(args.input))[0]
    write_csv(intervals, kernel_times, outdir, src_name)

    if not args.no_plot:
        try_plot(intervals, kernel_times, outdir, src_name)

    print("\n说明：")
    print("  * 单 interval.config 模式即任务一所需的『非活跃窗口分布特征』；")
    print("  * 带 --kernel-times 的模式额外给出算子时序分布，并用实测 kernel 时间")
    print("    累加出每个窗口的物理时长，与 interval.config 的 Estimated Time 对比，")
    print("    即任务二『与实际训练测量交叉验证』的量化切入点；")
    print("  * 跨迭代稳定性：把 config 的 num_iteration 调大（如 10）再跑阶段 B，")
    print("    对每次产出的 interval.config 各跑一遍本脚本对比即可。")


if __name__ == "__main__":
    main()
