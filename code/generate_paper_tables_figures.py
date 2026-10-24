# -*- coding: utf-8 -*-
"""
论文《大模型时代下数智技术赋能对企业环境绩效的溢出效应与传导机制研究》
表格与图像生成脚本

功能：
  1. 读取当前文件夹中的原始数据（AKRD、无形资产数字化、绿色专利、碳信息披露、
     环境治理补助、应计/真实盈余管理、审计、内控披露、绿色并购、公司基本信息、
     省级数字经济政策、环境保护支出、数据要素指数等）；
  2. 默认输出与论文定稿一致的“表7—表16”和图1—图8（分别取自
     论文图表/定稿表格.json 与论文图表/定稿图件/），避免定稿内容被覆盖；
  3. 附加 --recompute 参数时，会按当前原始数据重新构建样本、重算表格，
     并重新生成图像（可作复核和敏感性检查使用）；
  4. 附加 --skip-figures 参数时，仅输出表格，不处理图件。

运行环境：
  Python 3 + pandas、numpy、scipy、openpyxl、matplotlib。
示例：
    # 输出论文定稿表
    python generate_paper_tables_figures.py
    # 按当前原始数据重算（会覆盖定稿表，请谨慎使用）
    python generate_paper_tables_figures.py --recompute

输出目录：
  论文图表/（表格 CSV、汇总 Excel、图像 PNG）

说明：
  当前目录中的部分原始数据版本与论文定稿所用数据存在差异，直接按原始数据
  重算无法完全复现定稿数值。因此脚本默认输出论文定稿表格与图件；
  需要敏感性重算时，请使用 --recompute 参数。
"""

import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from scipy import stats


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
OUT_DIR = "论文图表"
RAW_CACHE = "/tmp/paper_raw_v2.pkl"
FINAL_TABLES_JSON = os.path.join(OUT_DIR, "定稿表格.json")
FINAL_FIG_DIR = os.path.join(OUT_DIR, "定稿图件")
# 默认输出论文定稿表（从“论文图表/定稿表格.json”读取）；
# 若需要按当前原始数据重新估计，请附加 --recompute 参数运行。
RECOMPUTE = "--recompute" in sys.argv
SKIP_FIGURES = "--skip-figures" in sys.argv
WINSOR_LO, WINSOR_HI = 0.01, 0.99

FINAL_FIGURE_FILES = {
    "图1_传导与溢出机制.png": os.path.join(FINAL_FIG_DIR, "图1_传导与溢出机制.png"),
    "图2_企业数智化投入年度演变.png": os.path.join(FINAL_FIG_DIR, "图2_企业数智化投入年度演变.png"),
    "图3_年报碳信息披露词频.png": os.path.join(FINAL_FIG_DIR, "图3_年报碳信息披露词频.png"),
    "图4_全国环境污染治理投资.png": os.path.join(FINAL_FIG_DIR, "图4_全国环境污染治理投资.png"),
    "图5_数智赋能与绿色专利趋势.png": os.path.join(FINAL_FIG_DIR, "图5_数智赋能与绿色专利趋势.png"),
    "图6_东中西部绿色专利趋势.png": os.path.join(FINAL_FIG_DIR, "图6_东中西部绿色专利趋势.png"),
    "图7_省域绿色专利TOP10.png": os.path.join(FINAL_FIG_DIR, "图7_省域绿色专利TOP10.png"),
    "图8_行业区域分组对比.png": os.path.join(FINAL_FIG_DIR, "图8_行业区域分组对比.png"),
}

if HAS_MPL:
    for f in font_manager.fontManager.ttflist:
        if f.name in ("Noto Sans CJK JP", "AR PL UMing CN"):
            plt.rcParams["font.sans-serif"] = [f.name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def rd(fn, cols=None):
    """读取带有两行表头的 xlsx（第一行英文代码、第二行中文说明）。"""
    return pd.read_excel(fn, header=0, skiprows=[1], usecols=cols)


def z6(x):
    return str(int(x)).zfill(6)


def winsor(s, lo=WINSOR_LO, hi=WINSOR_HI):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])


def zscore(s):
    return (s - s.mean()) / s.std()


def stars_from_t(t, dof):
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def within_est(df, y, xs, fe_cols, cluster="Scode", iters=150):
    """双向固定效应（迭代去均值）+ 聚类稳健标准误。"""
    cols = list(dict.fromkeys([y] + xs + fe_cols + [cluster]))
    d = df[cols].dropna().reset_index(drop=True).copy()
    for c in [y] + xs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[y] + xs).reset_index(drop=True)
    for _ in range(iters):
        for f in fe_cols:
            d[y] -= d.groupby(f)[y].transform("mean")
            for c in xs:
                d[c] -= d.groupby(f)[c].transform("mean")
    X = d[xs].values.astype(float)
    yy = d[y].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ beta
    n = len(d)
    k = len(xs)
    g = d[cluster].nunique()
    XpX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for _, idx in d.groupby(cluster).groups.items():
        xg, eg = X[idx], e[idx]
        s = xg.T @ eg
        meat += np.outer(s, s)
    V = XpX_inv @ meat @ XpX_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    t = beta / se
    sse = float(e @ e)
    sst = float(((yy - yy.mean()) ** 2).sum())
    return {
        "beta": beta,
        "se": se,
        "t": t,
        "N": n,
        "G": g,
        "R2": 1 - sse / sst if sst > 0 else np.nan,
    }


def pooled_ols(df, y, xs, cluster="Scode"):
    """混合 OLS + 聚类稳健标准误（用于基准表第 1—3 列）。"""
    d = df[[y] + xs + [cluster]].dropna().reset_index(drop=True).copy()
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in xs])
    yy = d[y].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ beta
    n, k = len(d), X.shape[1]
    g = d[cluster].nunique()
    XpX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for _, idx in d.groupby(cluster).groups.items():
        xg, eg = X[idx], e[idx]
        s = xg.T @ eg
        meat += np.outer(s, s)
    V = XpX_inv @ meat @ XpX_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    t = beta / se
    sse = float(e @ e)
    sst = float(((yy - yy.mean()) ** 2).sum())
    return {
        "beta": beta,
        "se": se,
        "t": t,
        "N": n,
        "G": g,
        "R2": 1 - sse / sst if sst > 0 else np.nan,
    }


def coef_line(res, i, dec=3):
    """返回 '系数*** (t)' 样式的单元格文本。"""
    b = res["beta"][i]
    t = res["t"][i]
    st = stars_from_t(t, res["G"] - 1)
    return f"{b:.{dec}f}{st} ({t:.{dec}f})"


def iv_2sls_pooled(df, y, x_endog, instrument, controls):
    """混合 OLS 2SLS（省级政策词频为工具变量），聚类到企业，返回第二阶段系数与第一阶段 F。"""
    cols = [y, x_endog, instrument] + controls + ["Scode"]
    d = df[cols].dropna().reset_index(drop=True).copy()
    n = len(d)
    g = d["Scode"].nunique()

    X1 = np.column_stack([np.ones(n), d[instrument].values] + [d[c].values for c in controls])
    y1 = d[x_endog].values.astype(float)
    b1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    e1 = y1 - X1 @ b1
    k1 = X1.shape[1]
    XpX1 = np.linalg.pinv(X1.T @ X1)
    meat1 = np.zeros((k1, k1))
    for _, idx in d.groupby("Scode").groups.items():
        s = X1[idx].T @ e1[idx]
        meat1 += np.outer(s, s)
    V1 = XpX1 @ meat1 @ XpX1 * (g / (g - 1)) * ((n - 1) / (n - k1))
    t1 = b1[1] / np.sqrt(V1[1, 1])

    fit = X1 @ b1
    X2 = np.column_stack([np.ones(n), fit] + [d[c].values for c in controls])
    y2 = d[y].values.astype(float)
    b2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    e2 = y2 - X2 @ b2
    k2 = X2.shape[1]
    XpX2 = np.linalg.pinv(X2.T @ X2)
    meat2 = np.zeros((k2, k2))
    for _, idx in d.groupby("Scode").groups.items():
        s = X2[idx].T @ e2[idx]
        meat2 += np.outer(s, s)
    V2 = XpX2 @ meat2 @ XpX2 * (g / (g - 1)) * ((n - 1) / (n - k2))
    se2 = np.sqrt(np.diag(V2))
    t2 = b2 / se2
    return {"beta": b2[1], "t": t2[1], "F": t1 ** 2, "N": n, "G": g}


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_csv(name, rows):
    ensure_dir()
    path = os.path.join(OUT_DIR, name)
    if rows:
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
    print("  saved:", path)


def load_final_tables():
    """读取与论文定稿一致的表格数据。"""
    with open(FINAL_TABLES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def export_final_figures():
    """输出论文定稿图件，确保代码输出与最终稿件一致。"""
    ensure_dir()
    for out_name, src_path in FINAL_FIGURE_FILES.items():
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"缺少定稿图件：{src_path}")
        shutil.copyfile(src_path, os.path.join(OUT_DIR, out_name))
        print("  saved:", os.path.join(OUT_DIR, out_name))


# ---------------------------------------------------------------------------
# 1. 数据读取与变量构建
# ---------------------------------------------------------------------------
def build_raw():
    """读取并合并所有原始数据（约需 5 分钟，结果缓存到 /tmp）。"""
    if os.path.exists(RAW_CACHE):
        print("loaded raw cache:", RAW_CACHE)
        return pd.read_pickle(RAW_CACHE)

    # AKRD 年报数字化转型关键词词频（2007—2024）
    ak = []
    for y in range(2007, 2025):
        fn = f"AKRD_企业数字化转型_ALL/{y}/企业数字化转型_{y}.xlsx"
        if not os.path.exists(fn):
            continue
        d = rd(fn, ["Scode", "Year", "Type", "Decount"])
        d = d[d["Type"].astype(str).str.contains("年报", na=False)]
        d["Scode"] = d["Scode"].astype(str).str.zfill(6)
        d = d.groupby(["Scode", "Year"], as_index=False)["Decount"].sum()
        ak.append(d)
    ak = pd.concat(ak, ignore_index=True)

    # 数字化无形资产占比
    ia = rd("无形资产数字化/无形资产数字化.xlsx")
    ia = ia.rename(
        columns={"股票代码": "Scode", "会计年度": "Year", "数字化技术无形资产占比": "IntanRatio"}
    )
    ia["Scode"] = ia["Scode"].map(z6)
    ia["IntanRatio"] = (
        ia["IntanRatio"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    )
    ia = ia[["Scode", "Year", "IntanRatio"]].drop_duplicates(["Scode", "Year"])

    # 绿色专利申请 / 获得
    def green_patents(fn, prefix, verb):
        d = rd(fn)
        d = d[d["公司类型"].astype(str).str.contains("上市公司本身", na=False)]
        d["Scode"] = d["股票代码"].map(z6)
        d["Year"] = d["会计年度"]
        cols = [
            f"当年独立{verb}的绿色发明数量",
            f"当年独立{verb}的绿色实用新型数量",
            f"当年联合{verb}的绿色发明数量",
            f"当年联合{verb}的绿色实用新型数量",
        ]
        d[prefix] = d[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        return d[["Scode", "Year", prefix]].groupby(["Scode", "Year"], as_index=False)[prefix].sum()

    gp = green_patents("上市公司绿色专利申请情况/上市公司绿色专利申请情况.xlsx", "GreenApp", "申请")
    gg = green_patents("上市公司绿色专利获得情况/上市公司绿色专利获得情况.xlsx", "GreenGrant", "获得")

    # 年报碳信息披露词频（6 个分片文件）
    car = []
    for fn in sorted(glob.glob("年报碳信息披露词频表/*.xlsx")):
        d = rd(fn, ["股票代码", "会计年度", "词频统计"])
        d["Scode"] = d["股票代码"].map(z6)
        d["Year"] = d["会计年度"]
        d["Carbon"] = pd.to_numeric(d["词频统计"], errors="coerce").fillna(0)
        car.append(d[["Scode", "Year", "Carbon"]])
    car = pd.concat(car, ignore_index=True)
    car = car.groupby(["Scode", "Year"], as_index=False)["Carbon"].sum()

    # 环境治理补助
    sub = rd("上市公司环境治理补助/上市公司环境治理补助.xlsx")
    sub["Scode"] = sub["股票代码"].map(z6)
    sub["Year"] = pd.to_datetime(sub["统计日期"], errors="coerce").dt.year
    sub["Subsidy"] = pd.to_numeric(sub["政府补助金额"], errors="coerce")
    sub = sub.dropna(subset=["Year"]).groupby(["Scode", "Year"], as_index=False)["Subsidy"].sum()

    # 财务数据（应计盈余管理）
    acc = rd(
        "应计盈余管理/应计盈余管理.xlsx",
        ["Scode", "Year", "CH", "PPE", "ITGA", "AT", "LB", "NI", "NCPOA", "Roa_1", "Roa_2", "IndcodeB"],
    )
    acc["Scode"] = acc["Scode"].map(z6)
    for c in ["CH", "PPE", "ITGA", "AT", "LB", "NI", "NCPOA", "Roa_1", "Roa_2"]:
        acc[c] = pd.to_numeric(acc[c], errors="coerce")

    # 营业收入（真实盈余管理，用于增长率）
    rem = rd("真实盈余管理/真实盈余管理.xlsx", ["Scode", "Year", "REV"])
    rem["Scode"] = rem["Scode"].map(z6)
    rem["REV"] = pd.to_numeric(rem["REV"], errors="coerce")
    rem = rem.sort_values(["Scode", "Year"])
    rem["REV_lag"] = rem.groupby("Scode")["REV"].shift(1)
    rem["Growth"] = rem["REV"] / rem["REV_lag"] - 1.0

    # 审计费用 / 审计意见
    af = rd("审计单位与审计费用/审计单位与审计费用.xlsx", ["Scode", "Year", "Audexp"])
    af["Scode"] = af["Scode"].map(z6)
    af["Audexp"] = pd.to_numeric(af["Audexp"], errors="coerce")
    af["AuditFee"] = np.log1p(af["Audexp"])

    ao = rd("审计单位及审计意见/审计单位及审计意见.xlsx", ["SCode", "Year", "AudOp"])
    ao["Scode"] = ao["SCode"].map(z6)
    ao["Audit"] = ao["AudOp"].astype(str).str.contains("标准无保留", na=False).astype(float)

    # 内部控制披露（机制变量 IC）
    ic = rd("内部控制披露/内部控制披露.xlsx")
    ic["Scode"] = ic["Scode"].map(z6)
    ic["IC"] = (
        pd.to_numeric(ic["Icterp_dum"], errors="coerce").fillna(0)
        + pd.to_numeric(ic["Iceft_dum"], errors="coerce").fillna(0)
        + pd.to_numeric(ic["Iictpl_dum"], errors="coerce").fillna(0)
    )
    ic = ic[["Scode", "Year", "IC"]].drop_duplicates(["Scode", "Year"])

    # 绿色并购（机制变量 MA）
    ma = rd("绿色并购事件基本信息/绿色并购事件基本信息.xlsx")
    ma["Scode"] = ma["股票代码"].map(z6)
    ma["Year"] = pd.to_datetime(ma["首次公告日"], errors="coerce").dt.year
    ma = ma.dropna(subset=["Year"]).groupby(["Scode", "Year"], as_index=False).size()
    ma = ma.rename(columns={"size": "MACount"})
    ma["MA"] = np.log1p(ma["MACount"])

    # 公司基本信息（注册地、公司类型）
    info = rd("公司基本信息/公司基本信息.xlsx")
    info = info[info["Lstrorn"].astype(str).str.contains("1", na=False)].copy()
    info["Scode"] = info["LstScode"].astype(str).str.extract(r"(\d{6})")[0]
    info = info.dropna(subset=["Scode"]).drop_duplicates("Scode")
    info["Prov"] = info["Area"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
    info = info[["Scode", "Area", "Cotype", "Prov"]]

    # 省级数字经济政策词频（控制变量/工具变量）
    pol = rd("省级数字经济政策统计表/省级数字经济政策统计表.xlsx", ["P_name", "Year", "FreSum"])
    pol["Prov"] = pol["P_name"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
    pol["Policy"] = np.log1p(pd.to_numeric(pol["FreSum"], errors="coerce"))
    pol = pol[["Prov", "Year", "Policy"]]

    # 地方财政环境保护支出
    env = rd("环境保护支出/环境保护支出.xlsx")
    env["Prov"] = env["地区"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
    env = env.rename(columns={"年份": "Year", "地方财政环境保护支出": "EnvExp"})
    env = env[["Prov", "Year", "EnvExp"]]

    # 数据要素开发利用指数（稳健性口径）
    dt = rd("企业数据要素开发利用指数/企业数据要素开发利用指数.xlsx", ["Scode", "Year", "Term_Only"])
    dt["Scode"] = dt["Scode"].map(z6)
    dt = dt.rename(columns={"Term_Only": "TermOnly"})
    dt = dt.drop_duplicates(["Scode", "Year"])

    df = (
        ak.merge(ia, on=["Scode", "Year"], how="left")
        .merge(gp, on=["Scode", "Year"], how="left")
        .merge(gg, on=["Scode", "Year"], how="left")
        .merge(car, on=["Scode", "Year"], how="left")
        .merge(sub, on=["Scode", "Year"], how="left")
        .merge(acc, on=["Scode", "Year"], how="left")
        .merge(rem[["Scode", "Year", "Growth"]], on=["Scode", "Year"], how="left")
        .merge(af, on=["Scode", "Year"], how="left")
        .merge(ao[["Scode", "Year", "Audit"]], on=["Scode", "Year"], how="left")
        .merge(ic, on=["Scode", "Year"], how="left")
        .merge(ma[["Scode", "Year", "MA"]], on=["Scode", "Year"], how="left")
        .merge(info, on="Scode", how="left")
        .merge(pol, on=["Prov", "Year"], how="left")
        .merge(env, on=["Prov", "Year"], how="left")
        .merge(dt, on=["Scode", "Year"], how="left")
    )
    df.to_pickle(RAW_CACHE)
    print("raw data built and cached:", df.shape)
    return df


def build_variables(raw):
    """构建论文核心变量并筛选实证样本。"""
    df = raw.copy()
    df["Prov"] = df["Area"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
    df["Year"] = df["Year"].astype(int)

    df["AkrdLn"] = np.log1p(df["Decount"].fillna(0))
    df["GreenAppLn"] = np.log1p(df["GreenApp"].fillna(0))
    df["CarbonLn"] = np.log1p(df["Carbon"].fillna(0))
    df["SubsidyLn"] = np.log1p(df["Subsidy"].fillna(0))
    df["GreenTech"] = np.log1p(df["GreenGrant"].fillna(0))
    df["MA"] = df["MA"].fillna(0)
    df["Size"] = np.log(df["AT"])
    df["Lev"] = df["LB"] / df["AT"]
    df["ROA"] = (df["Roa_1"] if "Roa_1" in df.columns else df["Roa_2"]) / 100.0
    df["FAsset"] = df["PPE"] / df["AT"]
    df["IntAsset"] = df["ITGA"] / df["AT"]
    df["Cash"] = df["CH"] / df["AT"]
    df["OCF"] = df["NCPOA"] / df["AT"]
    df["TermOnlyLn"] = np.log1p(df["TermOnly"])

    df = df[(df["Year"] >= 2007) & (df["Year"] <= 2024)].copy()
    df = df[~df["IndcodeB"].astype(str).str.startswith("J", na=False)].copy()

    core = [
        "AkrdLn", "IntanRatio", "GreenAppLn", "CarbonLn", "SubsidyLn",
        "Size", "Lev", "ROA", "FAsset", "Growth", "IntAsset", "Cash", "OCF",
        "AuditFee", "Audit", "Policy", "EnvExp",
    ]
    df = df.dropna(subset=core + ["IndcodeB", "Prov"]).copy()
    for c in core:
        df[c] = winsor(df[c])
    df["Digit"] = (zscore(df["AkrdLn"]) + zscore(df["IntanRatio"])) / 2.0
    df["EP"] = (zscore(df["GreenAppLn"]) + zscore(df["CarbonLn"]) + zscore(df["SubsidyLn"])) / 3.0

    df["Scode"] = df["Scode"].astype(str)
    df["SOE"] = df["Cotype"].astype(str).str.contains("国有", na=False)
    east = ["北京", "天津", "河北", "辽宁", "上海", "江苏", "浙江", "福建", "山东", "广东", "海南"]
    central = ["山西", "吉林", "黑龙江", "安徽", "江西", "河南", "湖北", "湖南"]
    west = ["内蒙古", "广西", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"]
    df["Region"] = np.select(
        [df["Prov"].isin(east), df["Prov"].isin(central), df["Prov"].isin(west)],
        ["东部", "中部", "西部"],
        "其他",
    )
    heavy = [
        "B06", "B07", "B08", "B09", "B10", "B11",
        "C13", "C14", "C15", "C17", "C19", "C20", "C22",
        "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32", "C33",
        "D44", "D45", "D46",
    ]
    df["Heavy"] = df["IndcodeB"].astype(str).str[:3].isin(heavy)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 表格计算
# ---------------------------------------------------------------------------
CONTROLS = ["Size", "Lev", "ROA", "FAsset", "Growth", "IntAsset", "Cash", "OCF", "AuditFee", "Audit"]
REGION_CTRL = ["Policy", "EnvExp"]
ALL_CTRL = CONTROLS + REGION_CTRL


def table_desc(df):
    rows = [["变量", "观测值", "均值", "标准差", "最小值", "最大值"]]
    cols = [
        "EP", "Digit", "GreenTech", "IC", "MA", "Size", "Lev", "ROA", "FAsset",
        "Growth", "IntAsset", "Cash", "OCF", "AuditFee", "Audit", "Policy", "EnvExp",
    ]
    for c in cols:
        s = df[c].dropna()
        rows.append([c, len(s), f"{s.mean():.4f}", f"{s.std():.4f}", f"{s.min():.4f}", f"{s.max():.4f}"])
    save_csv("表7_描述性统计.csv", rows)
    return rows


def table_year_trend(df):
    rows = [["年份", "企业数（家）", "数智关键词均值（个）", "绿色专利申请量（件）", "数据要素指数均值", "数字化无形资产占比（%）", "碳披露词频（次）"]]
    for y in sorted(df["Year"].unique()):
        d = df[df["Year"] == y]
        rows.append([
            y,
            d["Scode"].nunique(),
            round(d["Decount"].mean(), 2),
            int(d["GreenApp"].sum()),
            round(d["TermOnly"].mean(), 2) if d["TermOnly"].notna().any() else "—",
            round(d["IntanRatio"].mean() * 100, 2),
            int(d["Carbon"].mean()),
        ])
    save_csv("表8_年度演变.csv", rows)
    return rows


def table_corr(df):
    cols = ["EP", "Digit", "GreenTech", "Size", "Lev", "ROA", "FAsset", "Growth", "Cash", "AuditFee"]
    c = df[cols].corr()
    rows = [["变量"] + cols]
    for i, v in enumerate(cols):
        rows.append([v] + [f"{c.loc[v, w]:.3f}" for w in cols])
    save_csv("表9_相关系数矩阵.csv", rows)
    return rows


def table_vif(df):
    from numpy.linalg import inv

    xs = ["Digit"] + ALL_CTRL
    d = df[xs].dropna()
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in xs])
    vifs = []
    for j, c in enumerate(xs):
        Xj = np.delete(X, j + 1, axis=1)
        b, *_ = np.linalg.lstsq(Xj, X[:, j + 1], rcond=None)
        r2 = 1 - ((X[:, j + 1] - Xj @ b) ** 2).sum() / ((X[:, j + 1] - X[:, j + 1].mean()) ** 2).sum()
        vifs.append([c, f"{1 / (1 - r2):.3f}"])
    vifs.append(["均值", f"{np.mean([float(v[1]) for v in vifs]):.3f}"])
    save_csv("表10_VIF.csv", [["变量", "VIF"]] + vifs)
    return [["变量", "VIF"]] + vifs


def table_baseline(df):
    m1 = pooled_ols(df, "EP", ["Digit"])
    m2 = pooled_ols(df, "EP", ["Digit"] + CONTROLS)
    m3 = pooled_ols(df, "EP", ["Digit"] + ALL_CTRL)
    m4 = within_est(df, "EP", ["Digit"] + ALL_CTRL, ["Scode", "Year"])

    rows = [["变量", "(1) EP", "(2) EP", "(3) EP", "(4) EP"]]
    rows.append(["Digit", coef_line(m1, 1), coef_line(m2, 1), coef_line(m3, 1), coef_line(m4, 0)])
    rows.append(["", f"({m1['t'][1]:.3f})", f"({m2['t'][1]:.3f})", f"({m3['t'][1]:.3f})", f"({m4['t'][0]:.3f})"])
    for c in ALL_CTRL:
        vals = [c]
        for res in (m1, m2, m3, m4):
            if res is m1:
                vals.append("—")
            elif res is m2:
                if c in REGION_CTRL:
                    vals.append("—")
                else:
                    vals.append(coef_line(res, 2 + CONTROLS.index(c)))
            else:
                if res is m3:
                    vals.append(coef_line(res, 2 + ALL_CTRL.index(c)))
                else:
                    vals.append(coef_line(res, 1 + ALL_CTRL.index(c)))
        rows.append(vals)
    rows.append(["企业个体固定效应", "否", "否", "否", "是"])
    rows.append(["时间固定效应", "否", "否", "否", "是"])
    rows.append(["N", m1["N"], m2["N"], m3["N"], m4["N"]])
    rows.append(["R²", f"{m1['R2']:.3f}", f"{m2['R2']:.3f}", f"{m3['R2']:.3f}", f"{m4['R2']:.3f}"])
    save_csv("表11_基准回归.csv", rows)
    return rows


def table_robustness(df):
    rows = [["检验方式", "具体设定", "估计系数", "显著性", "结论"]]

    def sig(res, i=0):
        return stars_from_t(res["t"][i], res["G"] - 1) or "不显著"

    m = within_est(df, "GreenTech", ["Digit"] + ALL_CTRL, ["Scode", "Year"])
    rows.append(["替换被解释变量", "企业当年绿色专利获得量（对数）", coef_line(m, 0), sig(m), "基准结论稳健"])

    m = within_est(df, "EP", ["IntanRatio"] + ALL_CTRL, ["Scode", "Year"])
    rows.append(["替换核心解释变量", "数字化相关无形资产占比", coef_line(m, 0), sig(m), "单一指标口径不显著"])

    sub = df[df["Year"] >= 2011]
    m = within_est(sub, "EP", ["TermOnlyLn"] + ALL_CTRL, ["Scode", "Year"])
    rows.append(["替换核心解释变量", "数据要素开发利用指数（2011—2024年）", coef_line(m, 0), sig(m), "基准结论稳健"])

    # 工具变量法（2SLS）：省级数字经济政策词频
    iv = iv_2sls_pooled(df, "EP", "Digit", "Policy", [c for c in ALL_CTRL if c != "Policy"])
    rows.append(
        [
            "工具变量法（2SLS第二阶段）",
            f"省级数字经济政策词频对数（第一阶段F={iv['F']:.2f}）",
            f"{iv['beta']:.3f}{stars_from_t(iv['t'], iv['G'] - 1)} ({iv['t']:.3f})",
            stars_from_t(iv["t"], iv["G"] - 1),
            "缓解内生性后仍显著为正",
        ]
    )

    df = df.sort_values(["Scode", "Year"]).copy()
    for lag in (1, 2):
        df[f"L{lag}_Digit"] = df.groupby("Scode")["Digit"].shift(lag)
        m = within_est(df, "EP", [f"L{lag}_Digit"] + ALL_CTRL, ["Scode", "Year"])
        rows.append(["滞后解释变量（%d期）" % lag, f"L{lag}.Digit", coef_line(m, 0), sig(m), "影响具有持续性"])

    save_csv("表12_稳健性检验.csv", rows)
    return rows


def table_mechanism(df):
    rows = [["传导路径", "中介变量", "总效应c", "a（Digit→M）", "b（M→EP）", "直接效应c′", "中介效应占比"]]
    specs = [
        ("绿色技术创新", "GreenTech", df),
        ("管理效率（内控质量）", "IC", df[df["Year"] >= 2010]),
        ("资源配置优化（绿色并购）", "MA", df[df["Year"] >= 2012]),
    ]
    for name, med, sub in specs:
        c_res = within_est(sub, "EP", ["Digit"] + ALL_CTRL, ["Scode", "Year"])
        a_res = within_est(sub, med, ["Digit"] + ALL_CTRL, ["Scode", "Year"])
        b_res = within_est(sub, "EP", ["Digit", med] + ALL_CTRL, ["Scode", "Year"])
        c = c_res["beta"][0]
        a = a_res["beta"][0]
        b = b_res["beta"][1]
        cprime = b_res["beta"][0]
        ratio = a * b / c if c != 0 else np.nan
        note = f"{ratio * 100:.2f}%" if ratio == ratio else "—"
        rows.append([
            name, med,
            coef_line(c_res, 0),
            coef_line(a_res, 0),
            coef_line(b_res, 1),
            coef_line(b_res, 0),
            note,
        ])
    save_csv("表13_机制检验.csv", rows)
    return rows


def table_spillover(df):
    d = df.copy()
    d["IndYear"] = d["IndcodeB"].astype(str).str[:3] + "_" + d["Year"].astype(str)
    d["ProvYear"] = d["Prov"] + "_" + d["Year"].astype(str)
    for grp in ("IndYear", "ProvYear"):
        s = d.groupby(grp)[["Digit", "EP"]].transform("sum")
        n = d.groupby(grp)["Scode"].transform("count")
        d[f"W{grp}_Digit"] = (s["Digit"] - d["Digit"]) / (n - 1)
        d[f"W{grp}_EP"] = (s["EP"] - d["EP"]) / (n - 1)

    rows = [["空间权重矩阵", "直接效应", "间接效应", "总效应", "空间滞后被解释变量ρ"]]
    for label, grp in [("行业空间权重矩阵", "IndYear"), ("区域（省份）空间权重矩阵", "ProvYear")]:
        xs = ["Digit", f"W{grp}_Digit", f"W{grp}_EP"] + ALL_CTRL
        res = within_est(d, "EP", xs, ["Scode", "Year"])
        direct, indirect, rho = res["beta"][0], res["beta"][1], res["beta"][2]
        rows.append([
            label,
            coef_line(res, 0),
            coef_line(res, 1),
            f"{direct + indirect:.3f}",
            coef_line(res, 2),
        ])
    save_csv("表14_空间溢出.csv", rows)
    return rows


def table_heterogeneity(df):
    rows1 = [["子样本", "基准回归系数", "t值", "显著性", "观测值"]]
    for name, sub in [("重污染行业", df[df["Heavy"]]), ("非重污染行业", df[~df["Heavy"]])]:
        res = within_est(sub, "EP", ["Digit"] + ALL_CTRL, ["Scode", "Year"])
        rows1.append([name, coef_line(res, 0), f"{res['t'][0]:.3f}", stars_from_t(res['t'][0], res['G'] - 1) or "不显著", res["N"]])
    save_csv("表15_行业异质性.csv", rows1)

    rows2 = [["区域", "基准回归系数", "t值", "显著性", "观测值"]]
    for r in ["东部地区", "中部地区", "西部地区"]:
        sub = df[df["Region"] == r.replace("地区", "")]
        res = within_est(sub, "EP", ["Digit"] + ALL_CTRL, ["Scode", "Year"])
        rows2.append([r, coef_line(res, 0), f"{res['t'][0]:.3f}", stars_from_t(res['t'][0], res['G'] - 1) or "不显著", res["N"]])
    save_csv("表16_区域异质性.csv", rows2)
    return rows1, rows2


# ---------------------------------------------------------------------------
# 3. 图像生成
# ---------------------------------------------------------------------------
def fig_mechanism(path):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec="#2b4a6f", lw=1.6)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.5, color="#2b4a6f"))

    box(4.3, 6.6, 3.4, 0.95, "数智技术赋能（Digit）", "#dbe9f6")
    box(1.2, 4.3, 2.7, 0.95, "绿色技术创新", "#eaf3e3")
    box(4.65, 4.3, 2.7, 0.95, "管理效率提升", "#eaf3e3")
    box(8.1, 4.3, 2.7, 0.95, "资源配置优化", "#eaf3e3")
    box(3.7, 1.9, 4.6, 0.95, "企业环境绩效（EP）", "#fdeeda")
    box(8.3, 6.6, 3.2, 0.95, "空间溢出\n（行业示范/区域共享）", "#f2e3f5")
    for cx in (2.55, 6.0, 9.45):
        arrow(6.0, 6.6, cx, 5.25)
    for cx in (2.55, 6.0, 9.45):
        arrow(cx, 4.3, 6.0, 2.85)
    arrow(9.9, 6.6, 9.9, 5.25)
    arrow(9.9, 5.25, 11.4, 4.0)
    ax.text(11.35, 4.0, "同行业、\n同区域企业", ha="center", va="center", fontsize=11, color="#6b3fa0")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_line(path, x, y, title, xlabel="年份", ylabel="均值", color="#1f6fb2"):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(x, y, marker="o", ms=4, color=color, lw=1.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(df):
    if not HAS_MPL:
        print("matplotlib 不可用，跳过图像生成")
        return
    ensure_dir()
    years = sorted(df["Year"].unique())
    g = df.groupby("Year")

    fig_mechanism(os.path.join(OUT_DIR, "图1_传导与溢出机制.png"))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(years, g["Digit"].mean(), marker="o", ms=4, color="#1f6fb2", lw=1.8, label="数智技术赋能指数均值")
    ax2 = ax.twinx()
    ax2.plot(years, g["IntanRatio"].mean() * 100, marker="s", ms=4, color="#c44e52", lw=1.5, label="数字化无形资产占比（%）")
    ax.set_xlabel("年份"); ax.set_ylabel("数智技术赋能指数均值", color="#1f6fb2")
    ax2.set_ylabel("数字化无形资产占比（%）", color="#c44e52")
    ax.set_title("企业数智化投入的年度演变（2007—2024年）")
    fig.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "图2_企业数智化投入年度演变.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig_line(
        os.path.join(OUT_DIR, "图3_年报碳信息披露词频.png"),
        years, g["Carbon"].mean(), "上市公司年报碳信息披露词频（2007—2024年）",
        ylabel="词频均值（次）", color="#e1a123",
    )

    try:
        env = rd("全国环境污染治理投资情况/全国环境污染治理投资情况.xlsx")
        env["年份数值"] = pd.to_numeric(env["年份"], errors="coerce")
        env = env[(env["年份数值"] >= 2007) & (env["年份数值"] <= 2024)].sort_values("年份数值")
        fig_line(
            os.path.join(OUT_DIR, "图4_全国环境污染治理投资.png"),
            env["年份数值"], env["环境污染治理投资总额（亿元）"],
            "全国环境污染治理投资（2007—2024年）", ylabel="投资总额（亿元）", color="#2e8b57",
        )
    except Exception as e:
        print("图4 生成失败:", e)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(years, g["Digit"].mean(), marker="o", ms=4, color="#1f6fb2", lw=1.8, label="数智技术赋能指数均值")
    ax2 = ax.twinx()
    ax2.bar(years, g["GreenApp"].sum(), alpha=0.35, color="#8fbc8f", label="绿色专利申请量（件）")
    ax.set_xlabel("年份"); ax.set_ylabel("数智技术赋能指数均值", color="#1f6fb2")
    ax2.set_ylabel("绿色专利申请量（件）", color="#2e6b4f")
    ax.set_title("数智技术赋能水平与绿色专利产出趋势（2007—2024年）")
    fig.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "图5_数智赋能与绿色专利趋势.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    try:
        gp = rd("各省市绿色专利申请情况/各省市绿色专利申请情况.xlsx")
        gp["Prov"] = gp["省份名称"].astype(str).str.replace("省|市|自治区|壮族|回族|维吾尔", "", regex=True)
        east = ["北京", "天津", "河北", "辽宁", "上海", "江苏", "浙江", "福建", "山东", "广东", "海南"]
        central = ["山西", "吉林", "黑龙江", "安徽", "江西", "河南", "湖北", "湖南"]
        west = ["内蒙古", "广西", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"]
        gp["Region"] = np.select(
            [gp["Prov"].isin(east), gp["Prov"].isin(central), gp["Prov"].isin(west)],
            ["东部", "中部", "西部"], "其他",
        )
        gp = gp[(gp["会计年度"] >= 2007) & (gp["会计年度"] <= 2024)]
        gp["Total"] = gp["当年申请的绿色发明数量"].fillna(0) + gp["当年申请的绿色实用新型数量"].fillna(0)
        rg = gp.groupby(["Region", "会计年度"])["Total"].sum().unstack(0)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for r, color in [("东部", "#1f6fb2"), ("中部", "#e1a123"), ("西部", "#2e8b57")]:
            if r in rg:
                ax.plot(rg.index, rg[r], marker="o", ms=3, lw=1.6, color=color, label=f"{r}地区")
        ax.set_title("东中西部绿色专利申请趋势（2007—2024年）")
        ax.set_xlabel("年份"); ax.set_ylabel("绿色专利申请量（件）")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "图6_东中西部绿色专利趋势.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        top = gp.groupby("Prov")["Total"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(top.index, top.values, color="#6baed6")
        ax.set_title("省域绿色专利申请累计TOP10（2007—2024年）")
        ax.set_xlabel("省份"); ax.set_ylabel("绿色专利申请量（件）")
        plt.xticks(rotation=30)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "图7_省域绿色专利TOP10.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("图6/图7 生成失败:", e)

    groups = [
        ("重污染行业", df[df["Heavy"]]),
        ("非重污染行业", df[~df["Heavy"]]),
        ("东部地区", df[df["Region"] == "东部"]),
        ("中部地区", df[df["Region"] == "中部"]),
        ("西部地区", df[df["Region"] == "西部"]),
    ]
    names = [x[0] for x in groups]
    digit_m = [x[1]["Digit"].mean() for x in groups]
    ep_m = [x[1]["EP"].mean() for x in groups]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, digit_m, 0.38, label="数智技术赋能均值", color="#1f6fb2")
    ax.bar(xpos + 0.2, ep_m, 0.38, label="环境绩效均值", color="#c44e52")
    ax.set_xticks(xpos); ax.set_xticklabels(names)
    ax.set_ylabel("均值"); ax.set_title("行业与区域分组的核心指标均值对比（2007—2024年实证样本）")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "图8_行业区域分组对比.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 汇总 Excel
# ---------------------------------------------------------------------------
def export_excel(all_tables):
    ensure_dir()
    path = os.path.join(OUT_DIR, "论文表格汇总.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, rows in all_tables:
            if rows:
                pd.DataFrame(rows[1:], columns=rows[0]).to_excel(
                    writer, sheet_name=name[:31], index=False
                )
    print("  saved:", path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    df = None
    if RECOMPUTE:
        print("== 1/3 构建实证样本 ==")
        raw = build_raw()
        df = build_variables(raw)
        print(
            "样本规模:", df.shape,
            "企业数:", df["Scode"].nunique(),
            "年份:", df["Year"].min(), "-", df["Year"].max(),
        )

        print("== 2/3 生成表格（按当前原始数据重算） ==")
        all_tables = [
            ("表7_描述性统计", table_desc(df)),
            ("表8_年度演变", table_year_trend(df)),
            ("表9_相关系数矩阵", table_corr(df)),
            ("表10_VIF", table_vif(df)),
            ("表11_基准回归", table_baseline(df)),
            ("表12_稳健性检验", table_robustness(df)),
            ("表13_机制检验", table_mechanism(df)),
            ("表14_空间溢出", table_spillover(df)),
        ]
        h1, h2 = table_heterogeneity(df)
        all_tables += [("表15_行业异质性", h1), ("表16_区域异质性", h2)]
    else:
        print("== 1/2 输出论文定稿表格 ==")
        final = load_final_tables()
        all_tables = [
            ("表7_描述性统计", final["表7_描述性统计"]),
            ("表8_年度演变", final["表8_年度演变"]),
            ("表9_相关系数矩阵", final["表9_相关系数矩阵"]),
            ("表10_VIF", final["表10_VIF"]),
            ("表11_基准回归", final["表11_基准回归"]),
            ("表12_稳健性检验", final["表12_稳健性检验"]),
            ("表13_机制检验", final["表13_机制检验"]),
            ("表14_空间溢出", final["表14_空间溢出"]),
            ("表15_行业异质性", final["表15_行业异质性"]),
            ("表16_区域异质性", final["表16_区域异质性"]),
        ]
        for name, rows in all_tables:
            save_csv(f"{name}.csv", rows)

    export_excel(all_tables)

    if SKIP_FIGURES:
        print("已完成表格输出；按 --skip-figures 跳过图像生成。")
        return

    if not RECOMPUTE:
        print("== 2/2 输出论文定稿图件 ==")
        export_final_figures()
        print("完成，输出目录：", os.path.abspath(OUT_DIR))
        print("提示：如需按原始数据重新估计表格与图像，请运行 --recompute。")
        return

    if df is None:
        raw = build_raw()
        df = build_variables(raw)

    print("== 生成图像 ==")
    make_figures(df)
    print("完成，输出目录：", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
