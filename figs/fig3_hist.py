#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig3_hist.py —— 非活跃时长的分布直方图（双峰的直接证据）

与正文"24 桶直方图"对应：横轴取 log10(时长)，纵轴为该桶内的窗口数，
按窗口类别堆叠，使"左峰是相位内间隙、右峰是跨前后向驻留"一眼可见。
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import data,  ACCENT, CROSS, GREY, INTRA, grid, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
            rows.append((float(r["dur_us"]), kind))
    return rows


def main():
    setup()
    rows = load()
    vals = np.array([v for v, _ in rows])
    logs = np.log10(np.clip(vals, 1e-3, None))

    edges = np.linspace(-3.0, 5.2, 25)          # 24 个桶
    centers = (edges[:-1] + edges[1:]) / 2

    series = [("intra", INTRA, "intra-phase activations"),
              ("cross", CROSS, "cross-phase activations"),
              ("weight", ACCENT, "global weights")]
    counts = {}
    for key, _, _ in series:
        sub = np.log10(np.clip([v for v, k in rows if k == key], 1e-3, None))
        counts[key], _ = np.histogram(sub, bins=edges)

    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    width = (edges[1] - edges[0]) * 0.92
    bottom = np.zeros(len(centers))
    for key, color, label in series:
        c = counts[key]
        ax.bar(centers, c, width=width, bottom=bottom, color=color,
               alpha=0.88, linewidth=0, label=f"{label} (n={c.sum()})", zorder=3)
        bottom += c

    ax.set_xlabel("Inactive time ($\\mu$s)")
    ax.set_ylabel("Number of windows")
    ax.set_xlim(-3.2, 5.4)
    ax.set_ylim(0, bottom.max() * 1.22)

    ticks = list(range(-3, 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks])

    # 标出两峰与中间低谷
    i_left = int(np.argmax(counts["intra"]))
    i_right = int(np.argmax(counts["cross"] + counts["weight"]))
    for idx, txt, dy in ((i_left, "phase-inner peak", 1.06),
                         (i_right, "cross-phase peak", 1.06)):
        ax.annotate(txt, xy=(centers[idx], bottom[idx]), xytext=(centers[idx], bottom[idx] + bottom.max() * 0.04),
                    ha="center", va="bottom", fontsize=8, color="#444444",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY, shrinkA=2, shrinkB=2))

    i0 = int(np.searchsorted(edges, 3.0))
    i1 = int(np.searchsorted(edges, 4.0))
    n_gap = int(((vals >= 1e3) & (vals < 1e4)).sum())   # 精确计数，不用桶近似
    ax.annotate(f"{n_gap} windows\nin 1–10 ms",
                xy=(centers[(i0 + i1) // 2 - 1], 4),
                xytext=(centers[(i0 + i1) // 2 - 1] - 0.15, bottom.max() * 0.34),
                fontsize=8, color=GREY, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY,
                                shrinkA=2, shrinkB=2))

    grid(ax, "y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left",
              handlelength=1.0, handletextpad=0.5, borderaxespad=0.5)

    save(fig, OUT, "fig3_hist")
    print(f"  n={len(rows)}；左峰桶={counts['intra'][i_left]}，"
          f"右峰桶={int(counts['cross'][i_right]+counts['weight'][i_right])}；"
          f"1–10ms 区间 {n_gap} 个（{100*n_gap/len(rows):.1f}%）")


if __name__ == "__main__":
    main()
