#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_peak_mem.py —— 可选：显存峰值对账（不需要新测量，用现成文件）

作用
----
把 G10 模拟出的逐 kernel 显存占用（statistics/real_mem.config）与
PyTorch 实测的分配快照（resnet50_bs16_mem_snapshot.pickle）放在一起比，
回答一个报告里很自然会被问到的问题：

    "G10 的静态分析算出来的显存曲线，与真机分配器看到的峰值一致吗？"

这不是任务一的必需项，但它能把"非活跃窗口"的可卸载量与真实显存占用挂上钩，
让"可卸载比例"这个数字有分母。

用法
----
    python3 t1_peak_mem.py \
        --real-mem ../ResNet50/statistics/real_mem.config \
        --snapshot ../resnet50_bs16_mem_snapshot.pickle
"""

import argparse
import os
import pickle


def mb(x):
    return x / 2 ** 20


def read_real_mem(path):
    vals = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" not in line:
                continue
            _, rhs = line.split(":", 1)
            rhs = rhs.strip()
            try:
                vals.append(int(rhs))
            except ValueError:
                continue
    return vals


def peak_from_snapshot(path):
    """返回 (peak_bytes, n_alloc, n_free, note)。解析失败时返回 None。"""
    try:
        with open(path, "rb") as f:
            snap = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        return None, 0, 0, f"无法读取 pickle：{e}"

    traces = snap.get("device_traces")
    if not traces:
        return None, 0, 0, "pickle 里没有 device_traces 字段（torch 版本差异？）"

    # 取分配事件最多的那张卡
    dev = max(traces, key=lambda d: len(d) if d else 0)
    ev = []
    for e in dev or []:
        act = e.get("action")
        size = e.get("size", 0)
        t = e.get("time_us", e.get("timestamp", 0))
        if act == "alloc":
            ev.append((t, +size))
        elif act in ("free_requested", "free_completed", "free"):
            ev.append((t, -size))
    if not ev:
        return None, 0, 0, "device_traces 里没有 alloc/free 事件"

    ev.sort(key=lambda x: x[0])
    cur = peak = 0
    n_alloc = sum(1 for x in ev if x[1] > 0)
    n_free = len(ev) - n_alloc
    for _, d in ev:
        cur += d
        if cur > peak:
            peak = cur
    return peak, n_alloc, n_free, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-mem", default="../ResNet50/statistics/real_mem.config")
    ap.add_argument("--snapshot", default="../resnet50_bs16_mem_snapshot.pickle")
    args = ap.parse_args()

    lines = []
    if os.path.isfile(args.real_mem):
        vals = read_real_mem(args.real_mem)
        if vals:
            lines.append(f"G10 real_mem.config : {len(vals)} 个 kernel，"
                         f"峰值 {mb(max(vals)):.2f} MB，末值 {mb(vals[-1]):.2f} MB")
        else:
            lines.append("G10 real_mem.config : 解析到 0 条记录")
    else:
        lines.append(f"G10 real_mem.config : 文件不存在（{args.real_mem}）")

    peak, na, nf, note = peak_from_snapshot(args.snapshot)
    if peak is None:
        lines.append(f"实测 mem_snapshot  : 跳过（{note}）")
    else:
        lines.append(f"实测 mem_snapshot  : 峰值 {mb(peak):.2f} MB，"
                     f"alloc={na}，free={nf}（{note}）")

    if peak and os.path.isfile(args.real_mem):
        vals = read_real_mem(args.real_mem)
        if vals and max(vals) > 0:
            lines.append(f"两者比值（G10/实测）: {max(vals)/peak:.3f}")

    lines.append("")
    lines.append("说明：G10 的 real_mem 是静态分析估算的逐 kernel 显存占用，")
    lines.append("      实测快照是 PyTorch caching allocator 的 alloc/free 记录，")
    lines.append("      两者口径不同（前者含 UVM 常驻、后者含缓存块碎片），")
    lines.append("      比值落在 0.8–1.2 即认为自洽；偏离过大要在报告局限里说明。")

    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(os.path.dirname(os.path.abspath(args.real_mem)), "peak_mem_check.txt")
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(f"\n[OK] {out}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
