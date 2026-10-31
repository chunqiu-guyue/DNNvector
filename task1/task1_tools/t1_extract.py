#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_extract.py —— 任务一：把 G10 的四个 config 解析成一张可计算的主表

读 4 个文件
-----------
    <g10-dir>/cudnn_kernel_times.txt          449 行，逐调度单元实测耗时
    <g10-dir>/statistics/interval.config      697 个非活跃窗口（Hidding_Interval）
    <g10-dir>/statistics/kernels.config       449 个 kernel 的 层归属 + 输入输出张量
    <g10-dir>/statistics/layers.config        每层的 Input/Output/d_Input/d_Output/Weight/d_Weight
    <g10-dir>/statistics/tensors.config       张量清单（size / 是否全局）

输出
----
    task1_out/kernels.csv    kid,name,layer_id,layer_name,dur_us,shape,n_in,n_out
    task1_out/tensors.csv    tid,size_bytes,is_global,role,birth,death
    task1_out/intervals.csv  tid,size_bytes,is_global,is_looped,k_start,k_end,
                             est_raw,dur_us,t_start_us,t_end_us,role,cross_phase
    task1_out/axis.json      {"prefix_us": [...], "fwd_end_kernel": N, "n_kernel": 449}
    task1_out/extract_report.txt

时间单位（本脚本的核心口径，报告里必须照抄）
--------------------------------------------
G10 把 cudnn_kernel_times.txt 的数值按 ms 读入，但实测数值本身是 µs 量级：
449 行求和 ≈ 162671，而 PyTorch trace 的单步墙钟 ≈ 152940 µs。
因此本脚本：
    1) 先判定 cudnn_kernel_times.txt 的数值单位（auto：sum>10000 视为 µs）
    2) 统一换算成 µs
    3) interval.config 的 Estimated Time 是 ns（= µs × 1000），÷1000 后与实测同量级
    4) 论文 figure_drawing/plot_tensor_period_distribution.py 的 x 轴就标 "Inactive Time (µs)"，
       与本脚本口径一致

用法
----
    python3 t1_extract.py --g10-dir ../ResNet50 --outdir ./task1_out
"""

import argparse
import csv
import json
import os
import re

# ---------------------------------------------------------------- 正则
RE_KT = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)")
RE_KSIZE = re.compile(r"^\s*(\d+)\s+(\d+)\s+B")
RE_KERNEL_HEAD = re.compile(r"^Kernel ID:\s*(\d+),\s*Name:\s*(\S+)")
RE_KERNEL_LAYER = re.compile(r"^Parent Layer ID:\s*(\d+);\s*Name:\s*(\S+)")
RE_KERNEL_TIME = re.compile(r"^Execution Time:\s*([\d.eE+-]+)")
RE_SHAPE = re.compile(r"^\((\d+(?:,\s*\d+)*)\)")
RE_TENSOR = re.compile(r"^tensor(\d+).*?Size in byte:\s*(\d+)")

RE_TENSOR_ANY = re.compile(r"tensor(\d+).*?Is weight \(global\)\?:\s*([01]).*?Size in byte:\s*(\d+)")
RE_LIVENESS = re.compile(r"Liveness:\s*Birth:\s*(-?\d+),\s*Death:\s*(-?\d+)")
RE_INTERVAL = re.compile(r"interval\s+(\d+):\s*(\d+)\s*--------\s*(\d+)")
RE_EST = re.compile(r"Estimated Time:\s*([\d.eE+-]+)")

RE_LAYER_HEAD = re.compile(r"^Layer ID:\s*(\d+);\s*Name:\s*(\S+)")
RE_TENSOR_LINE = re.compile(r"^(Input|Output|d_Input|d_Output|Weight|d_Weight) Tensor:\s*tensor(\d+)")

# 张量角色的优先级：同一张量若被多个层引用，取更"专有"的角色
ROLE_PRIORITY = ["d_Input", "d_Output", "d_Weight", "Weight", "Output", "Input"]
ROLE_NORMALIZE = {
    "Input": "activation", "Output": "activation",
    "d_Input": "grad", "d_Output": "grad",
    "Weight": "weight", "d_Weight": "d_weight",
}


# ---------------------------------------------------------------- 工具
def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def parse_kernel_times(path):
    out = {}
    for line in read_text(path):
        m = RE_KT.match(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def detect_unit(values, ref_step_us=None):
    """判定 cudnn_kernel_times.txt 数值的单位，返回 (unit, scale_to_us, note)。"""
    total = sum(values.values())
    if ref_step_us:
        ratio = total / ref_step_us
        if 0.5 <= ratio <= 2.0:
            return "us", 1.0, f"sum={total:.1f} 与参考步长 {ref_step_us:.0f}µs 同量级"
        if 500 <= ratio <= 2000:
            return "ms", 1000.0, f"sum={total:.1f} ≈ 参考步长×1000，判定为 ms"
    if total > 10000:
        return "us", 1.0, f"sum={total:.1f} 落在 1e4 以上，判定为 µs"
    return "ms", 1000.0, f"sum={total:.1f} 偏小，判定为 ms"


def parse_intervals(path):
    """返回 list[dict]：每个 Hidding_Interval 一条，含 est_raw（原始单位）。"""
    intervals = []
    cur_tid = cur_size = cur_global = None
    pending = None
    for line in read_text(path):
        m = RE_TENSOR_ANY.search(line)
        if m:
            cur_tid, cur_global, cur_size = int(m.group(1)), int(m.group(2)), int(m.group(3))
            continue
        m = RE_INTERVAL.search(line)
        if m:
            pending = (int(m.group(2)), int(m.group(3)))
            continue
        m = RE_EST.search(line)
        if m and pending is not None:
            intervals.append({
                "tid": cur_tid,
                "size_bytes": cur_size,
                "is_global": cur_global,
                "k_start": pending[0],
                "k_end": pending[1],
                "is_looped": 1 if pending[0] > pending[1] else 0,
                "est_raw": float(m.group(1)),
            })
            pending = None
    return intervals


def parse_liveness(path):
    """返回 {tid: (birth, death, is_global)}；全局权重没有 Liveness 行，记 -1/-1。"""
    live = {}
    cur_tid = None
    for line in read_text(path):
        m = RE_TENSOR_ANY.search(line)
        if m:
            cur_tid = int(m.group(1))
            if int(m.group(2)) == 1:
                live.setdefault(cur_tid, (-1, -1, True))
            continue
        m = RE_LIVENESS.search(line)
        if m:
            live[cur_tid] = (int(m.group(1)), int(m.group(2)), False)
    return live


def parse_kernels(path):
    """返回 (list[dict], fwd_end_kernel)。"""
    kernels = []
    cur = None
    section = None
    for line in read_text(path):
        m = RE_KERNEL_HEAD.match(line)
        if m:
            if cur:
                kernels.append(cur)
            cur = {"kid": int(m.group(1)), "name": m.group(2),
                   "layer_id": None, "layer_name": None, "time_raw": None,
                   "shape": "", "inputs": [], "outputs": []}
            section = None
            continue
        if cur is None:
            continue
        m = RE_KERNEL_LAYER.match(line)
        if m:
            cur["layer_id"] = int(m.group(1))
            cur["layer_name"] = m.group(2)
            continue
        m = RE_KERNEL_TIME.match(line)
        if m:
            cur["time_raw"] = float(m.group(1))
            continue
        m = RE_SHAPE.match(line)
        if m:
            cur["shape"] = m.group(1).replace(" ", "")
            continue
        if line.startswith("Input Tensors"):
            section = "in"
            continue
        if line.startswith("Output Tensors"):
            section = "out"
            continue
        m = RE_TENSOR.match(line)
        if m and section:
            key = "inputs" if section == "in" else "outputs"
            cur[key].append(int(m.group(1)))
    if cur:
        kernels.append(cur)

    fwd_end = None
    for k in kernels:
        if "Backward" in k["name"]:
            fwd_end = k["kid"]
            break
    return kernels, fwd_end


def parse_layers(path):
    """返回 {tid: role}，role ∈ {activation, grad, weight, d_weight}。"""
    role_hit = {}
    cur_lines = []

    def flush(block):
        for line in block:
            m = RE_TENSOR_LINE.match(line)
            if not m:
                continue
            kind, tid = m.group(1), int(m.group(2))
            prev = role_hit.get(tid)
            if prev is None or ROLE_PRIORITY.index(kind) < ROLE_PRIORITY.index(prev):
                role_hit[tid] = kind

    for line in read_text(path):
        if RE_LAYER_HEAD.match(line):
            flush(cur_lines)
            cur_lines = []
        cur_lines.append(line)
    flush(cur_lines)

    return {tid: ROLE_NORMALIZE[k] for tid, k in role_hit.items()}


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g10-dir", default="../ResNet50",
                    help="含 cudnn_kernel_times.txt 与 statistics/ 的目录")
    ap.add_argument("--outdir", default="./task1_out")
    ap.add_argument("--time-unit", default="auto", choices=["auto", "us", "ms"])
    ap.add_argument("--ref-step-us", type=float, default=0.0,
                    help="可选：PyTorch trace 的单步墙钟（µs），用于单位 auto 判定")
    args = ap.parse_args()

    stat = os.path.join(args.g10_dir, "statistics")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) kernel 时间
    kt_raw = parse_kernel_times(os.path.join(args.g10_dir, "cudnn_kernel_times.txt"))
    if not kt_raw:
        raise SystemExit(f"[ERROR] 没读到 kernel 时间，检查 {args.g10_dir}/cudnn_kernel_times.txt")
    if args.time_unit == "auto":
        unit, scale, note = detect_unit(kt_raw, args.ref_step_us or None)
    elif args.time_unit == "us":
        unit, scale, note = "us", 1.0, "用户指定"
    else:
        unit, scale, note = "ms", 1000.0, "用户指定"
    kt_us = {k: v * scale for k, v in kt_raw.items()}

    n_kernel = max(kt_us) + 1
    prefix = [0.0] * (n_kernel + 1)
    for i in range(n_kernel):
        prefix[i + 1] = prefix[i] + kt_us.get(i, 0.0)

    # 2) kernels / layers / interval
    kernels, fwd_end = parse_kernels(os.path.join(stat, "kernels.config"))
    roles = parse_layers(os.path.join(stat, "layers.config"))
    live = parse_liveness(os.path.join(stat, "interval.config"))
    intervals = parse_intervals(os.path.join(stat, "interval.config"))

    # 3) 写 kernels.csv
    with open(os.path.join(args.outdir, "kernels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kid", "name", "layer_id", "layer_name", "dur_us", "shape",
                    "n_in", "n_out"])
        for k in kernels:
            w.writerow([k["kid"], k["name"], k["layer_id"], k["layer_name"],
                        f"{kt_us.get(k['kid'], 0.0):.6f}", k["shape"],
                        len(k["inputs"]), len(k["outputs"])])

    # 4) 写 tensors.csv
    with open(os.path.join(args.outdir, "tensors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tid", "size_bytes", "is_global", "role", "birth", "death"])
        for tid, (birth, death, is_g) in sorted(live.items()):
            size = 0
            for iv in intervals:
                if iv["tid"] == tid:
                    size = iv["size_bytes"]
                    break
            w.writerow([tid, size, int(is_g), roles.get(tid, "unknown"), birth, death])

    # 5) 写 intervals.csv（核心主表）
    n_cross = 0
    n_cross_clean = 0
    with open(os.path.join(args.outdir, "intervals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tid", "size_bytes", "is_global", "is_looped", "k_start", "k_end",
                    "est_raw", "dur_us", "t_start_us", "t_end_us", "role", "cross_phase"])
        for iv in intervals:
            s, e = iv["k_start"], iv["k_end"]
            if iv["is_looped"]:
                dur = sum(kt_us.get(i, 0.0) for i in range(s, n_kernel)) + \
                      sum(kt_us.get(i, 0.0) for i in range(0, e + 1))
                t0, t1 = prefix[s], prefix[n_kernel] + prefix[e + 1]
            else:
                dur = prefix[min(e + 1, n_kernel)] - prefix[s]
                t0, t1 = prefix[s], prefix[min(e + 1, n_kernel)]
            cross = 0
            if fwd_end is not None and not iv["is_looped"]:
                cross = 1 if (s < fwd_end <= e) else 0
                n_cross += cross
                if not iv["is_global"]:
                    n_cross_clean += cross
            w.writerow([iv["tid"], iv["size_bytes"], iv["is_global"], iv["is_looped"],
                        s, e, f"{iv['est_raw']:.6f}", f"{dur:.6f}",
                        f"{t0:.6f}", f"{t1:.6f}",
                        roles.get(iv["tid"], "unknown"), cross])

    # 6) 轴
    with open(os.path.join(args.outdir, "axis.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_kernel": n_kernel,
            "fwd_end_kernel": fwd_end,
            "prefix_us": prefix,
            "step_total_us": prefix[-1],
            "kernel_time_unit_detected": unit,
            "scale_to_us": scale,
            "detect_note": note,
        }, f, indent=2)

    # 7) 报告
    lines = [
        "=== t1_extract 口径报告 ===",
        f"kernel 文件数值单位判定 : {unit}（{note}）",
        f"换算到 µs 的系数        : {scale}",
        f"kernel 数               : {len(kt_raw)}（G10 调度单元）",
        f"单步总时长(µs)          : {prefix[-1]:.1f}",
        f"前向/反向分界 kernel    : {fwd_end}",
        f"张量数(含全局)          : {len(live)}",
        f"  其中全局权重          : {sum(1 for v in live.values() if v[2])}",
        f"非活跃窗口记录          : {len(intervals)}",
        f"  其中 is_looped=1      : {sum(iv['is_looped'] for iv in intervals)}",
        f"  其中 is_global=1      : {sum(iv['is_global'] for iv in intervals)}",
        f"  跨前后向(全部记录)    : {n_cross}",
        f"  跨前后向(清洗后)      : {n_cross_clean}  ← 分析脚本表 4 用的是这个",
        "",
        "下一步：python3 t1_analyze.py --indir ./task1_out",
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "extract_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
