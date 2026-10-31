#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig1_lifecycle.py —— 张量生命周期甘特图（仿 G10 论文 Figure 6）

取 ResNet50 第一个 bottleneck 块（layer 4–13，含 downsample 支路），
横轴只列出该块的前向算子与反向算子，中间以断轴表示其余算子。

每个张量的生灭时刻由 kernels.config 自动判定：
  born  = 首次作为某 kernel 输入或输出的 kernel 序号
  death = 最后一次作为某 kernel 输入或输出的 kernel 序号
跨相位/相内也由 death 是否落进反向段自动判定，不手工指定。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (ACCENT, BWD_BG, CROSS, FWD_BG, GREY, INTRA,  # noqa: E402
                       data, grid, save, setup)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KCFG = data("ResNet50",
                    "statistics", "kernels.config")
OUT = os.path.join(HERE, "out")

# 该块的张量（编号经 kernels.config 逐个核对）
ROWS = [
    ("t23", "W1"), ("t20", "A0"), ("t26", "A1"), ("t44", "A2"),
    ("t62", "A3"), ("t81", "A4"), ("t95", "A5"),
]
FWD_K = list(range(5, 15))                       # Conv1 → Add
BWD_K = list(range(416, 441))                    # 该块反向全过程
GAP = 1

# 前向全部标注；反向只标有意义的算子，其余留空以免拥挤
LABEL_FWD = {5: "Conv1", 6: "BN1", 7: "Conv2", 8: "BN2", 9: "Conv3",
             10: "BN3", 11: "ReLU", 12: "ConvS", 13: "BNS", 14: "Add"}
LABEL_BWD = {416: "Conv1-I", 419: "Add", 420: "BN3", 422: "Conv3-I",
             425: "ReLU", 428: "Conv2-I", 431: "BN2", 433: "Conv2-W",
             436: "BN1", 438: "Conv1-W", 440: "Apply"}


def parse_kernels(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {}
    for b in txt.split("_" * 20):
        m = re.search(r"Kernel ID: (\d+), Name: (\S+)", b)
        if not m:
            continue
        kid = int(m.group(1))
        tin, tout = [], []
        if "Input Tensors:" in b:
            seg = b.split("Input Tensors:")[1].split("Output Tensors:")[0]
            tin = [int(x) for x in re.findall(r"\ntensor(\d+) Is weight", seg)]
        if "Output Tensors:" in b:
            seg = b.split("Output Tensors:")[1]
            tout = [int(x) for x in re.findall(r"\ntensor(\d+) Is weight", seg)]
        out[kid] = {"name": m.group(2), "in": tin, "out": tout}
    return out


def lifespan(K, tid):
    use = []
    for kid in sorted(K):
        if tid in K[kid]["out"] or tid in K[kid]["in"]:
            use.append(kid)
    if not use:
        return None
    return min(use), max(use), sorted(set(use))


def main():
    setup()
    K = parse_kernels(KCFG)
    xmap = {}
    for i, k in enumerate(FWD_K):
        xmap[k] = i
    for i, k in enumerate(BWD_K):
        xmap[k] = len(FWD_K) + GAP + i
    total_w = len(FWD_K) + GAP + len(BWD_K)
    lo_b, hi_x = 0, total_w - 1
    n = len(ROWS)

    fig, ax = plt.subplots(figsize=(6.6, 2.95), constrained_layout=True)

    ax.axvspan(-0.5, len(FWD_K) - 0.5, color=FWD_BG, zorder=0)
    ax.axvspan(len(FWD_K) + GAP - 0.5, total_w - 0.5, color=BWD_BG, zorder=0)

    stats = {"cross": 0, "intra": 0}

    for r, (tkey, label) in enumerate(ROWS):
        y = n - 1 - r
        tid = int(tkey[1:])
        born, death, used = lifespan(K, tid)
        is_global = (label == "W1")
        if is_global:
            kind = "weight"
        else:
            kind = "cross" if death >= min(BWD_K) else "intra"
            stats[kind] += 1
        color = {"weight": ACCENT, "cross": CROSS, "intra": INTRA}[kind]

        def to_x(k):
            if k in xmap:
                return xmap[k]
            if k < min(FWD_K):
                return -0.5
            if k > max(BWD_K):
                return total_w - 0.5
            return len(FWD_K) + GAP / 2.0 - 0.5       # 落在被省略的中间段
        x0, x1 = to_x(born), to_x(death)

        ax.add_patch(Rectangle((x0, y - 0.30), max(x1 - x0, 0.05), 0.60,
                               facecolor=color, alpha=0.20,
                               edgecolor="none", zorder=2))
        if True:
            for k in used:
                if k not in xmap:
                    continue          # 落在省略段，不画点，避免与断轴混淆
                xk = xmap[k]
                ax.add_patch(Rectangle((xk - 0.32, y - 0.30), 0.64, 0.60,
                                       facecolor=color, edgecolor="white",
                                       linewidth=0.5, zorder=3))

    # 断轴：仅在轴线处画一对斜线；数据区的色条保持连续（张量在省略段仍存活）
    xb = len(FWD_K) + GAP / 2.0 - 0.5
    tr = ax.get_xaxis_transform()
    for off in (-0.20, 0.20):
        ax.plot([xb + off - 0.09, xb + off + 0.09], [-0.030, 0.030],
                color=GREY, lw=1.1, zorder=7, clip_on=False,
                transform=tr, solid_capstyle="butt")

    # x 轴：只给有意义的算子加标签
    ticks, labels = [], []
    for k in FWD_K:
        ticks.append(xmap[k])
        labels.append(LABEL_FWD.get(k, ""))
    for k in BWD_K:
        if k in LABEL_BWD:
            ticks.append(xmap[k])
            labels.append(LABEL_BWD[k])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="x", length=2.5)

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([r[1] for r in ROWS])
    ax.set_ylim(-0.7, n - 0.5)
    ax.set_xlim(-0.5, total_w - 0.5)
    ax.set_ylabel("Tensor")
    ax.set_xlabel("Operators in execution order")

    ax.text((len(FWD_K) - 1) / 2, n - 0.42, "Forward",
            ha="center", va="bottom", fontsize=9, style="italic")
    ax.text(len(FWD_K) + GAP + (len(BWD_K) - 1) / 2, n - 0.42, "Backward",
            ha="center", va="bottom", fontsize=9, style="italic")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Patch(facecolor=CROSS, alpha=0.8, label=f"cross-phase ({stats['cross']})"),
        Patch(facecolor=INTRA, alpha=0.8, label=f"intra-phase ({stats['intra']})"),
        Patch(facecolor=ACCENT, alpha=0.8, label="global weight (W1)"),
    ]
    fig.legend(handles=handles, ncol=3, frameon=False, fontsize=9,
               loc="outside upper center", handlelength=1.2,
               columnspacing=1.8, handletextpad=0.5)

    save(fig, OUT, "fig1_lifecycle")
    print(f"  跨相位 {stats['cross']} 个、相内 {stats['intra']} 个")


if __name__ == "__main__":
    main()
