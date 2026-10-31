#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig5_dispatch.py —— 分组卷积的两种派发形态

(a) 原生单核：cuDNN 为该组宽特化的 conv2d_grouped_direct，一次发射算完整组；
(b) 逐组拆分：拆成 32 次发射，且每次前后各夹一次布局变换。

横轴按各层自身执行时长归一化，只比较形态、不比较性能
（两层通道数与空间尺寸不同，绝对耗时不可直接对比）。
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import CROSS, GREY, INTRA, grid, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "..", "traces",
                     "resnext50_32x4d_bs16_kernel_trace.json")
OUT = os.path.join(HERE, "out")


def brief(name):
    n = re.sub(r"<.*", "", name)
    n = re.sub(r"\(.*", "", n).strip()
    if "nchwToNhwc" in n:
        return "in"
    if "nhwcToNchw" in n:
        return "out"
    if "cutlass" in n.lower() or "sgemm" in n.lower():
        return "compute"
    if "grouped_direct" in n:
        return "native"
    return "other"


def load():
    d = json.load(open(TRACE))
    ev = d["traceEvents"]
    by = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") != "kernel":
            continue
        x = (e.get("args") or {}).get("External id")
        if x is not None:
            by[x].append(e)
    ops = [e for e in ev if e.get("cat") == "cpu_op"
           and e.get("name") == "aten::cudnn_convolution"]

    def pick(ci, n):
        for o in ops:
            dims = o["args"].get("Input Dims")
            c = dims[0][1] if dims and dims[0] else None
            if c == ci and len(by.get(o["args"]["External id"], [])) == n:
                return sorted(by[o["args"]["External id"]], key=lambda e: e["ts"])
        return None

    return pick(128, 1), pick(1024, 128)


def spans(kk):
    t0 = kk[0]["ts"]
    t1 = max(e["ts"] + e.get("dur", 0) for e in kk)
    span = t1 - t0
    return [(brief(e["name"]), (e["ts"] - t0) / span,
             max(e.get("dur", 0) / span, 0.0045)) for e in kk], t1 - t0


def main():
    setup()
    A, B = load()
    sa, dur_a = spans(A)
    sb, dur_b = spans(B)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharex=True,
                             gridspec_kw={"width_ratios": [1, 3.4]},
                             constrained_layout=True)

    # (a) 原生单核
    ax = axes[0]
    for _, x, w in sa:
        ax.add_patch(Rectangle((x, 0.25), w, 0.5, facecolor=CROSS,
                               edgecolor="white", linewidth=0.6, zorder=3))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["native\nkernel"])
    ax.set_title(f"(a) one kernel  ·  {len(A)} launch", fontsize=9)
    ax.text(0.5, 0.86, f"{dur_a:.0f} µs", ha="center", va="center",
            fontsize=8.5, color=CROSS, transform=ax.transAxes)

    # (b) 逐组拆分：三类横条按类别分行
    ax = axes[1]
    rows = [("in", 2, "layout in"), ("compute", 1, "compute"),
            ("out", 0, "layout out")]
    for key, y, _ in rows:
        for kind, x, w in sb:
            if kind != key:
                continue
            color = CROSS if key == "compute" else "#2f5597"
            ax.add_patch(Rectangle((x, y * 0.32 + 0.05), w, 0.22,
                                   facecolor=color, edgecolor="none", zorder=3))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.16, 0.48, 0.80])
    ax.set_yticklabels(["layout out", "compute", "layout in"])
    ax.set_title(f"(b) 32-way split  ·  {len(B)} launches  ·  {dur_b:.0f} µs",
                 fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Time within the layer (normalised)")

    for ax in axes:
        grid(ax, "x")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=CROSS, label="convolution compute"),
               Patch(facecolor="#2f5597", label="layout transform (NCHW ↔ NHWC)")]
    fig.legend(handles=handles, ncol=2, frameon=False, fontsize=8.5,
               loc="outside lower center", handlelength=1.4,
               columnspacing=1.8)

    save(fig, OUT, "fig5_dispatch")
    n_comp = sum(1 for k, _, _ in sb if k == "compute")
    print(f"  A: {len(A)} kernel / {dur_a:.0f} µs；"
          f"B: {len(B)} kernel（其中计算 {n_comp}）/ {dur_b:.0f} µs")


if __name__ == "__main__":
    main()
