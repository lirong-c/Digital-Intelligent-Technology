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

import generate_paper_tables_figures as G
G.RAW_CACHE = os.path.join(OUT, "raw_cache.pkl")
G.OUT_DIR   = os.path.join(OUT, "_scratch")

raw = G.build_raw()
df = G.build_variables(raw)
df["Scode"] = df["Scode"].astype(str)
print("样本观测:", len(df), "| 企业数:", df["Scode"].nunique(), "| 省份:", df["Prov"].nunique())
print("年份范围:", df.Year.min(), "-", df.Year.max())
m1 = G.pooled_ols(df, "EP", ["Digit"])
m2 = G.pooled_ols(df, "EP", ["Digit"] + G.CONTROLS)
m3 = G.pooled_ols(df, "EP", ["Digit"] + G.ALL_CTRL)
m4 = G.within_est(df, "EP", ["Digit"] + G.ALL_CTRL, ["Scode", "Year"], iters=300)
for i, m in enumerate([m1, m2, m3, m4], 1):
    k = 1 if i < 4 else 0
    print("模型(%d) Digit beta=%.4f t=%.3f N=%d R2=%.4f" % (i, m["beta"][k], m["t"][k], m["N"], m["R2"]))
g = G.within_est(df, "GreenTech", ["Digit"] + G.ALL_CTRL, ["Scode", "Year"], iters=300)
print("绿色技术创新: beta=%.4f t=%.3f" % (g["beta"][0], g["t"][0]))
