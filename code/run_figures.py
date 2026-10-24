# -*- coding: utf-8 -*-
"""本文件由原分析脚本改写，仅调整路径；计算逻辑未改。
运行前可用环境变量指定数据与输出位置：
    PAPER_DATA=/path/to/raw_data  PAPER_OUT=/path/to/results
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("PAPER_DATA", os.path.join(ROOT, "data"))
OUT  = os.environ.get("PAPER_OUT",  os.path.join(ROOT, "results"))
AUX  = os.path.join(HERE, "..", "aux")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
if not os.path.isdir(DATA):
    raise SystemExit("未找到数据目录：%s\n请用 PAPER_DATA 指定（见 README 第 3 节）" % DATA)
os.chdir(DATA)                      # 引擎按相对路径读取原始数据
# -*- coding: utf-8 -*-
"""图件重绘（v2）：参考原排版稿的视觉语言——柔和填充 + 深色描边 + 低饱和配色。

设计要点：
  1. 配色选用"转灰度后亮度差明显"的低饱和色，彩色好看、黑白也清晰；
  2. 方块用浅色填充 + 深色描边 + 圆角，恢复原稿的层次感；
  3. 图1 机制图按修订后逻辑重绘，并在图下加注（与原稿一致）；
  4. 图表统一细网格、外框、留白，字号适配期刊排版。
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

import generate_paper_tables_figures as G
OUT_BASE = os.environ.get("PAPER_OUT", os.path.join(ROOT, "results"))
G.RAW_CACHE = os.path.join(OUT_BASE, "raw_cache.pkl")
G.OUT_DIR   = os.path.join(OUT_BASE, "_scratch")

for f in font_manager.fontManager.ttflist:
    if f.name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Source Han Sans SC", "AR PL UMing CN"):
        plt.rcParams["font.sans-serif"] = [f.name]; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11.5, "axes.titlesize": 13, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.facecolor": "white",
})

# ---------- 黑白灰阶配色（印刷安全：线型+标记+纹理+灰阶四重区分） ----------
BLUE   = "#1a1a1a"   # 主色 → 近黑
RED    = "#4d4d4d"   # 强调 → 深灰
GREEN  = "#1a1a1a"   # 主渠道 → 近黑（加粗）
ORANGE = "#4d4d4d"   # 次色 → 深灰
PURPLE = "#6e6e6e"   # 弱化 → 中灰
TEAL   = "#4d4d4d"
GREY   = "#9a9a9a"   # 边框 → 浅灰
DARK   = "#1a1a1a"
LIGHT  = {"blue": "#e6e6e6", "red": "#f0f0f0", "green": "#e6e6e6",
          "orange": "#f0f0f0", "purple": "#f5f5f5", "grey": "#f7f7f7"}

OUT = os.path.join(OUT_BASE, "figures")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  saved:", p, os.path.getsize(p), "bytes")

def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#cccccc", linewidth=0.7, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ============================================================
# 图1 传导与溢出机制图
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 7.9); ax.axis("off")

    def box(x, y, w, h, main, sub, fc, ec, lw=1.9, dashed=False, ms=12.5, ss=10):
        ax.add_patch(FancyBboxPatch((x + 0.045, y - 0.055), w, h,
                     boxstyle="round,pad=0.11,rounding_size=0.16",
                     linewidth=0, facecolor="#bfbfbf", alpha=0.55, zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.11,rounding_size=0.16",
                     linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
                     linestyle=(0, (5, 3)) if dashed else "solid"))
        ax.text(x + w/2, y + h*0.68, main, ha="center", va="center",
                fontsize=ms, fontweight="bold", color=DARK, zorder=3)
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center",
                fontsize=ss, color="#3a3a3a", zorder=3, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color, lw=2.0, dashed=False, rad=0.0, ms=15):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=ms, linewidth=lw, color=color, zorder=4,
                     linestyle=(0, (5, 3)) if dashed else "solid",
                     connectionstyle=f"arc3,rad={rad}"))

    box(0.30, 3.15, 2.45, 1.55, "数智技术赋能 (Digit)",
        "年报关键词词频\n+ 数字化无形资产占比", LIGHT["blue"], BLUE, lw=2.0, ms=12)
    box(3.70, 5.85, 3.25, 1.30, "绿色技术创新", "a=0.0549**，b=0.1821***\n中介占比 50.05%",
        LIGHT["green"], GREEN, lw=2.6, ms=12.5, ss=10)
    box(3.70, 3.58, 3.25, 1.30, "管理效率（内控质量）", "a=0.0020，b=0.0015\n（Sobel 不显著）",
        LIGHT["grey"], GREY, lw=1.4, dashed=True, ms=11, ss=10)
    box(3.70, 1.05, 3.25, 1.30, "资源配置优化（绿色并购）", "a=0.0294*，b=0.0121*\n中介占比 1.40%",
        LIGHT["grey"], GREY, lw=1.4, dashed=True, ms=11, ss=10)
    box(8.30, 3.15, 3.30, 1.55, "企业环境绩效 (EP)",
        "绿色创新产出 + 碳信息披露\n+ 环境治理投入", LIGHT["orange"], ORANGE, lw=2.0, ms=12)

    # 主渠道（粗实线）
    arrow(2.78, 4.72, 3.70, 6.25, GREEN, lw=2.6)
    arrow(6.95, 6.35, 8.30, 4.64, GREEN, lw=2.6)
    ax.text(2.90, 5.62, "促进\n(H1)", fontsize=10.5, color=GREEN, fontweight="bold",
            ha="center", rotation=40, linespacing=1.4)
    ax.text(7.52, 5.70, "改善\n(H2)", fontsize=10.5, color=GREEN, fontweight="bold",
            ha="center", rotation=-40, linespacing=1.4)
    # 未获支持路径（虚线）
    arrow(2.78, 4.45, 3.70, 4.40, GREY, lw=1.4, dashed=True, rad=-0.08)
    arrow(2.78, 3.28, 3.70, 2.30, GREY, lw=1.4, dashed=True, rad=0.10)
    arrow(6.95, 4.45, 8.30, 4.35, GREY, lw=1.4, dashed=True, rad=-0.08)
    arrow(6.95, 2.30, 8.30, 3.30, GREY, lw=1.4, dashed=True, rad=-0.10)
    # 直接效应：走"管理效率"与"资源配置优化"之间的空档（y: 2.26~3.28）+ 右侧空档
    ax.add_patch(FancyArrowPatch((2.78, 3.42), (8.30, 3.42), arrowstyle="-|>",
                 mutation_scale=15, linewidth=1.9, color=RED, zorder=5,
                 connectionstyle="arc3,rad=-0.055"))
    ax.text(2.62, 2.86, "直接效应 c′=0.0100（不显著）", fontsize=10.2, color=RED,
            ha="left", style="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.95))
    # 空间溢出
    ax.add_patch(FancyBboxPatch((8.65+0.045, 0.55-0.055), 2.85, 1.15,
                 boxstyle="round,pad=0.11,rounding_size=0.16",
                 linewidth=0, facecolor="#bfbfbf", alpha=0.5, zorder=1))
    ax.add_patch(FancyBboxPatch((8.65, 0.55), 2.85, 1.15,
                 boxstyle="round,pad=0.11,rounding_size=0.16",
                 linewidth=1.5, edgecolor=PURPLE, facecolor=LIGHT["purple"],
                 zorder=2, linestyle=(0, (2, 2))))
    ax.text(10.08, 1.32, "跨企业空间溢出", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK, zorder=3)
    ax.text(10.08, 0.88, "间接效应不显著\n（H5 未获支持）", ha="center", va="center",
            fontsize=9.8, color="#3a3a3a", zorder=3, linespacing=1.5)
    arrow(9.95, 3.15, 10.08, 1.75, PURPLE, lw=1.4, dashed=True)
    # 图下注释（与原稿风格一致）
    ax.text(0.30, -0.30, "注：***、**、* 分别表示 1%、5%、10% 水平显著；a 为 Digit→中介变量的系数，b 为中介变量→EP 的系数；"
            "样本期 2007—2024 年，企业—年份双向固定效应，企业层面聚类稳健标准误。",
            fontsize=9.2, color="#3a3a3a", ha="left", va="bottom")
    ax.set_ylim(-0.62, 7.55)
    save(fig, "图1_传导与溢出机制.png")

# ============================================================
# 图2—图8
# ============================================================
def data_figs():
    raw = G.build_raw(); df = G.build_variables(raw); df["Scode"] = df["Scode"].astype(str)
    years = sorted(df["Year"].unique()); g = df.groupby("Year")

    # 图2
    fig, ax = plt.subplots(figsize=(9.0, 4.15))
    ax.plot(years, g["Digit"].mean(), marker="o", ms=5.5, color=BLUE, lw=2.3,
            label="数智技术赋能指数均值", zorder=3, markerfacecolor="white", markeredgewidth=1.6)
    ax.fill_between(years, g["Digit"].mean(), min(g["Digit"].mean())*1.02,
                    color=BLUE, alpha=0.07, zorder=0)
    ax.set_xlabel("年份"); ax.set_ylabel("数智技术赋能指数均值", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.plot(years, g["IntanRatio"].mean()*100, marker="s", ms=5, color=RED, lw=2.0,
             linestyle="--", label="数字化无形资产占比（%）")
    ax2.set_ylabel("数字化无形资产占比（%）", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    grid(ax); ax.grid(False); ax2.grid(False)
    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", frameon=True, framealpha=0.94,
              edgecolor="#cccccc")
    ax.set_title("企业数智化投入的年度演变（2007—2024年）", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    save(fig, "图2_企业数智化投入年度演变.png")

    # 图3
    fig, ax = plt.subplots(figsize=(9.0, 4.05))
    y3 = g["Carbon"].mean()
    ax.fill_between(years, y3, color=ORANGE, alpha=0.18, zorder=0)
    ax.plot(years, y3, marker="o", ms=5, color=ORANGE, lw=2.3, zorder=3,
            markerfacecolor="white", markeredgewidth=1.6)
    ax.axvspan(2019.5, 2024.5, color=ORANGE, alpha=0.07, zorder=0)
    ax.set_xlabel("年份"); ax.set_ylabel("词频企业均值（次）")
    grid(ax); ax.set_title("上市公司年报碳信息披露词频（2007—2024年）", pad=10)
    ax.annotate("2020年“双碳”目标提出后\n披露词频快速上升",
                xy=(2021.2, 8.2), xytext=(2011.5, 10.2), fontsize=10, color="#3a3a3a",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3,
                                connectionstyle="arc3,rad=-0.2"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    save(fig, "图3_年报碳信息披露词频.png")

    # 图4
    env = G.rd("全国环境污染治理投资情况/全国环境污染治理投资情况.xlsx")
    env["Y"] = pd.to_numeric(env["年份"], errors="coerce")
    env = env[(env["Y"] >= 2007) & (env["Y"] <= 2024)].sort_values("Y")
    v = env["环境污染治理投资总额（亿元）"]
    fig, ax = plt.subplots(figsize=(9.0, 4.05))
    ax.fill_between(env["Y"], v, v.min()*0.96, color=GREEN, alpha=0.16, zorder=0)
    ax.plot(env["Y"], v, marker="o", ms=5, color=GREEN, lw=2.3, zorder=3,
            markerfacecolor="white", markeredgewidth=1.6)
    ax.set_xlabel("年份"); ax.set_ylabel("投资总额（亿元）")
    grid(ax); ax.set_title("全国环境污染治理投资（2007—2024年）", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    save(fig, "图4_全国环境污染治理投资.png")

    # 图5
    fig, ax = plt.subplots(figsize=(9.0, 4.15))
    ax.bar(years, g["GreenApp"].sum(), color=GREEN, alpha=0.30, edgecolor=GREEN,
           linewidth=1.1, width=0.64, label="绿色专利申请量（件）", zorder=1)
    ax.set_xlabel("年份"); ax.set_ylabel("绿色专利申请量（件）")
    grid(ax); ax.grid(False); ax.grid(True, axis="y", color="#cccccc", linewidth=0.7)
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(years, g["Digit"].mean(), marker="o", ms=5.5, color=BLUE, lw=2.4,
             label="数智技术赋能指数均值", zorder=3, markerfacecolor="white", markeredgewidth=1.6)
    ax2.set_ylabel("数智技术赋能指数均值", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE); ax2.spines["top"].set_visible(False)
    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", frameon=True, framealpha=0.94,
              edgecolor="#cccccc")
    ax.set_title("数智技术赋能水平与绿色专利产出趋势（2007—2024年）", pad=10)
    ax.annotate("2024年受公开时滞影响\n数据不完整",
                xy=(2024, 4700), xytext=(2014.6, 4300), fontsize=10, color="#1a1a1a",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3,
                                connectionstyle="arc3,rad=-0.22"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    save(fig, "图5_数智赋能与绿色专利趋势.png")

    # 图6
    gp = G.rd("各省市绿色专利申请情况/各省市绿色专利申请情况.xlsx")
    gp["Prov"] = gp["省份名称"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
    east = ["北京","天津","河北","辽宁","上海","江苏","浙江","福建","山东","广东","海南"]
    central = ["山西","吉林","黑龙江","安徽","江西","河南","湖北","湖南"]
    west = ["内蒙古","广西","重庆","四川","贵州","云南","西藏","陕西","甘肃","青海","宁夏","新疆"]
    gp["Region"] = np.select([gp["Prov"].isin(east), gp["Prov"].isin(central), gp["Prov"].isin(west)],
                             ["东部","中部","西部"], "其他")
    gp = gp[(gp["会计年度"]>=2007)&(gp["会计年度"]<=2024)]
    gp["Total"] = gp["当年申请的绿色发明数量"].fillna(0)+gp["当年申请的绿色实用新型数量"].fillna(0)
    rg = gp.groupby(["Region","会计年度"])["Total"].sum().unstack(0)
    fig, ax = plt.subplots(figsize=(9.0, 4.05))
    for r,c,ls,mk in [("东部","#1a1a1a","-","o"),("中部","#5a5a5a","--","s"),("西部","#8c8c8c","-.","^")]:
        if r in rg:
            ax.plot(rg.index, rg[r], color=c, linestyle=ls, marker=mk, ms=4.8, lw=2.1,
                    label=r+"地区", markerfacecolor="white", markeredgewidth=1.4)
    ax.set_title("东中西部绿色专利申请趋势（2007—2024年）", pad=10)
    ax.set_xlabel("年份"); ax.set_ylabel("绿色专利申请量（件）")
    grid(ax); ax.legend(frameon=True, framealpha=0.94, edgecolor="#cccccc")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    save(fig, "图6_东中西部绿色专利趋势.png")

    # 图7
    top = gp.groupby("Prov")["Total"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9.0, 4.05))
    cols = ["#4d4d4d" if i < 3 else "#d9d9d9" for i in range(len(top))]
    bars = ax.bar(range(len(top)), top.values, color=cols, edgecolor="#1a1a1a",
                  linewidth=1.0, width=0.68)
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top.index, rotation=30, ha="right")
    ax.set_title("省域绿色专利申请累计TOP10（2007—2024年）", pad=10)
    ax.set_xlabel("省份"); ax.set_ylabel("绿色专利申请量（件）")
    grid(ax)
    for b,v_ in zip(bars, top.values):
        ax.text(b.get_x()+b.get_width()/2, v_*1.01, f"{int(v_)}", ha="center",
                va="bottom", fontsize=9.2, color="#1a1a1a")
    save(fig, "图7_省域绿色专利TOP10.png")

    # 图8（实心/斜纹区分，兼顾彩色与黑白）
    groups = [("重污染行业", df[df["Heavy"]]), ("非重污染行业", df[~df["Heavy"]]),
              ("东部地区", df[df["Region"]=="东部"]), ("中部地区", df[df["Region"]=="中部"]),
              ("西部地区", df[df["Region"]=="西部"])]
    names = [x[0] for x in groups]
    dm = [x[1]["Digit"].mean() for x in groups]; em = [x[1]["EP"].mean() for x in groups]
    x = np.arange(len(names)); w = 0.37
    fig, ax = plt.subplots(figsize=(9.2, 4.25))
    b1 = ax.bar(x-w/2, dm, w, color=BLUE, edgecolor="#1a1a1a", linewidth=1.1,
                label="数智技术赋能均值")
    b2 = ax.bar(x+w/2, em, w, color="white", edgecolor=RED, linewidth=1.3,
                hatch="////", label="环境绩效均值")
    ax.axhline(0, color="#3a3a3a", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("均值")
    grid(ax)
    ax.set_title("行业与区域分组的核心指标均值对比（2007—2024年实证样本）", pad=10)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#cccccc", loc="upper right")
    for bars in (b1, b2):
        for b in bars:
            v_ = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, v_+(0.013 if v_>=0 else -0.013),
                    f"{v_:.3f}", ha="center", va="bottom" if v_>=0 else "top",
                    fontsize=8.8, color="#3a3a3a")
    ax.set_ylim(min(dm+em)-0.09, max(dm+em)+0.10)
    save(fig, "图8_行业区域分组对比.png")

print("=== 图件重绘 v2（低饱和配色 + 柔和填充）===")
fig1(); data_figs()
print("完成")
