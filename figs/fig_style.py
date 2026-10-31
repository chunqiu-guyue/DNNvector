#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_style.py —— 全报告统一图样式

参数取自 G10 artifact 的 src/resources/figure_drawing/fig_common.py：
serif 字体、Type-42 字体嵌入、细灰网格、无图内标题（信息交给 LaTeX 图注）。

用法：
    from fig_style import setup, save, CROSS, INTRA
    setup()
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def data(*parts):
    """数据根目录：本地为 forresnet50/，仓库中为 task1/。"""
    for base in ("../forresnet50", "../task1"):
        p = os.path.join(HERE, base, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, "../task1", *parts)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── 语义配色（不是默认色板）─────────────────────────────
CROSS = "#c0504d"      # 跨前向—反向的窗口（长时间驻留）
INTRA = "#4f81bd"      # 相位内窗口（短间隙）
FWD_BG = "#f4f4f4"     # 前向区域底色
BWD_BG = "#e9eff7"     # 反向区域底色
GREY = "#7f7f7f"
ACCENT = "#d9a441"     # 标注、峰值点
PALETTE4 = ["brown", "royalblue", "peru", "forestgreen"]  # G10 fig_common 原配色
PALETTE_LIGHT = ["#ff796c", "#fac205", "#95d0fc", "#96f97b"]

# ── 版面尺寸（英寸）──────────────────────────────────
W_SINGLE = (3.4, 2.6)
W_FULL = (6.8, 3.0)
W_TALL = (6.8, 3.6)


def setup():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "hatch.linewidth": 0.7,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def grid(ax, axis="y"):
    """统一网格：仅主刻度、细灰线、置于数据下方。"""
    ax.grid(True, which="major", axis=axis, linestyle="-",
            linewidth=0.5, color=GREY, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  [OK] {name}.pdf / .png")
