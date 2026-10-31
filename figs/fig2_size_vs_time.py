#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig2_size_vs_time.py —— 张量尺寸与非活跃时长的联合分布（仿 G10 论文 Figure 4）

横轴为时长、纵轴为尺寸，与论文 Figure 4 一致。
颜色直接编码"是否跨越前向—反向分界"，而不是按张量角色：
要讲的结论是双峰，颜色就该服务于双峰。
"""
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import data,  ACCENT, CROSS, GREY, INTRA, grid, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = data("task1_out", "intervals.csv")
OUT = os.path.join(HERE, "out")


def load():
    rows = []
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_looped"] != "0":
                continue
            if r["is_global"] == "1":
                kind = "weight"
            elif r["cross_phase"] == "1":
                kind = "cross"
            else:
                kind = "intra"
            rows.append({"t": float(r["dur_us"]),
                         "s": int(r["size_bytes"]), "kind": kind})
    return rows


def main():
    setup()
    rows = load()
    intra = [r for r in rows if r["kind"] == "intra"]
    cross = [r for r in rows if r["kind"] == "cross"]
    weight = [r for r in rows if r["kind"] == "weight"]
    longw = cross + weight
    med_i = st.median([r["t"] for r in intra])
    med_c = st.median([r["t"] for r in longw])

    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)

    ax.scatter([r["t"] for r in intra], [r["s"] for r in intra],
               s=13, c=INTRA, alpha=0.70, linewidths=0, zorder=3,
               label=f"intra-phase activations (n={len(intra)})")
    ax.scatter([r["t"] for r in cross], [r["s"] for r in cross],
               s=15, c=CROSS, alpha=0.75, linewidths=0, zorder=3,
               label=f"cross-phase activations (n={len(cross)})")
    ax.scatter([r["t"] for r in weight], [r["s"] for r in weight],
               s=17, c=ACCENT, alpha=0.85, linewidths=0, zorder=4,
               label=f"global weights (n={len(weight)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inactive time ($\\mu$s)")
    ax.set_ylabel("Tensor size (byte)")
    ax.set_xlim(3e-3, 6e5)
    ax.set_ylim(6e3, 2.0e8)

    # 两组的中位时长：竖虚线（时长在横轴），两端各标数值
    for med, color in ((med_i, INTRA), (med_c, CROSS)):
        ax.vlines(med, 6e3, 9.0e7, colors=color, lw=1.0,
                  linestyles=(0, (4, 3)), alpha=0.9, zorder=2)
    ax.text(med_i, 1.0e8, "median 90.9 µs", ha="center", va="bottom",
            fontsize=8.5, color=INTRA)
    ax.text(med_c, 1.0e8, "median 50.4 ms", ha="center", va="bottom",
            fontsize=8.5, color=CROSS)

    grid(ax, "both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, frameon=False,
               fontsize=8.5, loc="outside lower center",
               handletextpad=0.4, columnspacing=1.6)

    save(fig, OUT, "fig2_size_vs_time")
    print(f"  相内 {len(intra)} / 跨相位 {len(cross)} / 权重 {len(weight)}；中位 {med_c:.0f} vs {med_i:.1f} µs")


if __name__ == "__main__":
    main()
