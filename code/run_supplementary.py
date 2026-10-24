# -*- coding: utf-8 -*-
"""本文件由原分析脚本改写，仅调整路径；计算逻辑未改。
注意：原脚本用 OUT 作为“结果字典”变量，故此处输出目录命名为 OUTDIR，避免冲突。
运行前可用环境变量指定数据与输出位置：
    PAPER_DATA=/path/to/raw_data  PAPER_OUT=/path/to/results
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA   = os.environ.get("PAPER_DATA", os.path.join(ROOT, "data"))
OUTDIR = os.environ.get("PAPER_OUT",  os.path.join(ROOT, "results"))
AUX    = os.path.join(HERE, "..", "aux")
os.makedirs(OUTDIR, exist_ok=True)
sys.path.insert(0, HERE)
if not os.path.isdir(DATA):
    raise SystemExit("未找到数据目录：%s\n请用 PAPER_DATA 指定（见 README 第 3 节）" % DATA)
os.chdir(DATA)                      # 引擎按相对路径读取原始数据
# -*- coding: utf-8 -*-
"""回应评议意见的补充计量分析（向量化版，快速）。

① EP 测量效度：熵权法赋权、三分维度平行检验、碳披露结构断点、剔除2020+
② 中介变量替换：总资产周转率（管理效率）、绿色专利强度（资源配置）
③ 结果复核：随机子样本重复、Bootstrap 中介检验（向量化）
④ AI 专属强度指标（回应概念—测量鸿沟）
"""
import numpy as np, pandas as pd, sys, json, time
_conv = lambda o: o.item() if isinstance(o, np.generic) else str(o)
def jdump(o, f): json.dump(o, open(f,"w",encoding="utf-8"), ensure_ascii=False, indent=1, default=_conv)
import generate_paper_tables_figures as G
G.RAW_CACHE = os.path.join(OUTDIR, "raw_cache.pkl")
G.OUT_DIR   = os.path.join(OUTDIR, "_scratch")

t0 = time.time()
raw = G.build_raw(); df = G.build_variables(raw); df["Scode"] = df["Scode"].astype(str)
print("基础样本 N=%d 企业=%d  (%.0fs)" % (len(df), df["Scode"].nunique(), time.time()-t0))
OUT = {}

def fe(d, y, xs, fes=("Scode", "Year"), cluster="Scode", iters=300):
    return G.within_est(d, y, xs, list(fes), cluster=cluster, iters=iters)

# ---------- 预计算去均值数据（供 bootstrap 复用） ----------
def demean_matrix(d, ys, xs, fes=("Scode", "Year"), iters=200):
    cols = list(dict.fromkeys(ys + xs + list(fes) + ["Scode"]))
    dd = d[cols].dropna().reset_index(drop=True).copy()
    for c in ys + xs:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd = dd.dropna(subset=ys + xs).reset_index(drop=True)
    for _ in range(iters):
        for f in fes:
            for c in ys + xs:
                dd[c] = dd[c] - dd.groupby(f)[c].transform("mean")
    return dd

def ols_cluster(dd, y, xs, cluster="Scode"):
    X = dd[xs].values.astype(float); yy = dd[y].values.astype(float)
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ b
    n, k = len(dd), len(xs)
    XpX = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for _, idx in dd.groupby(cluster).groups.items():
        s = X[idx].values.T @ e[idx]
        meat += np.outer(s, s)
    g = dd[cluster].nunique()
    V = XpX @ meat @ XpX * (g/(g-1)) * ((n-1)/(n-k))
    return b, b/np.sqrt(np.diag(V))

# ============================================================
SKIP1 = "--skip1" in sys.argv
print("\n" + "="*74); print("① EP 测量效度"); print("="*74)
# 1-a 熵权法（各维度同向，权重为正，可解释）
Z = np.column_stack([G.zscore(df["GreenAppLn"]), G.zscore(df["CarbonLn"]), G.zscore(df["SubsidyLn"])])
Zpos = Z - Z.min(0) + 1e-6
P = Zpos / Zpos.sum(0, keepdims=True)
E = -(P * np.log(P)).sum(0) / np.log(len(P))
d_ = 1 - E; w_ent = d_ / d_.sum()
print("熵权法权重: 绿色创新=%.4f 碳披露=%.4f 环保投入=%.4f" % tuple(w_ent))
df["EP_ent"] = G.zscore(Z @ w_ent)
m = fe(df, "EP_ent", ["Digit"] + G.ALL_CTRL)
print("熵权-EP 基准: Digit=%.4f (t=%.3f)" % (m['beta'][0], m['t'][0]))
OUT["熵权法"] = [["维度", "等权", "熵权法权重"],
              ["绿色创新产出", "0.3333", f"{w_ent[0]:.4f}"],
              ["碳信息披露", "0.3333", f"{w_ent[1]:.4f}"],
              ["环保治理投入", "0.3333", f"{w_ent[2]:.4f}"],
              ["Digit系数(t)", "0.0200 (1.335)", f"{m['beta'][0]:.4f} ({m['t'][0]:.3f})"]]

# 1-b 三分维度平行检验
rows = [["被解释变量", "Digit系数", "t值", "显著性", "N"]]
print("\n三分维度平行检验:")
for nm, v in [("绿色创新产出", "GreenAppLn"), ("碳信息披露", "CarbonLn"), ("环保治理投入", "SubsidyLn")]:
    r = fe(df, v, ["Digit"] + G.ALL_CTRL)
    st = G.stars_from_t(r['t'][0], r['G']-1) or "不显著"
    print(f"  {nm:8s}: {r['beta'][0]:8.4f} (t={r['t'][0]:6.3f}) {st}  N={r['N']}")
    rows.append([nm, f"{r['beta'][0]:.4f}", f"{r['t'][0]:.3f}", st, r["N"]])
OUT["三分维度平行检验"] = rows

# 1-c 碳披露结构断点
df["Post2020"] = (df["Year"] >= 2020).astype(float)
df["Dig_post"] = df["Digit"] * df["Post2020"]
r = fe(df, "EP", ["Digit", "Post2020", "Dig_post"] + G.ALL_CTRL)
print("\n控制碳披露口径断点(Post2020):")
print("  Digit(2020前)=%.4f (t=%.3f)  Post2020=%.4f (t=%.3f)  Digit×Post=%.4f (t=%.3f)"
      % (r['beta'][0], r['t'][0], r['beta'][1], r['t'][1], r['beta'][2], r['t'][2]))
OUT["结构断点"] = [["变量", "系数", "t值", "显著性"],
                ["Digit（2020年前）", f"{r['beta'][0]:.4f}", f"{r['t'][0]:.3f}", G.stars_from_t(r['t'][0], r['G']-1) or "不显著"],
                ["Post2020", f"{r['beta'][1]:.4f}", f"{r['t'][1]:.3f}", G.stars_from_t(r['t'][1], r['G']-1) or "不显著"],
                ["Digit×Post2020", f"{r['beta'][2]:.4f}", f"{r['t'][2]:.3f}", G.stars_from_t(r['t'][2], r['G']-1) or "不显著"]]
jdump(OUT, os.path.join(OUTDIR, "supplementary.json"))

# ============================================================
print("\n" + "="*74); print("② 中介变量代理替换"); print("="*74)
revcol = next((c for c in ("REV","REV_r","REV_y","REV_x","REV_src") if c in df.columns), None)
if revcol is None:
    rem = G.rd("真实盈余管理/真实盈余管理.xlsx", ["Scode","Year","REV"])
    rem["Scode"] = rem["Scode"].map(G.z6); rem["REV"] = pd.to_numeric(rem["REV"], errors="coerce")
    df = df.merge(rem.rename(columns={"REV":"REV_src"})[["Scode","Year","REV_src"]], on=["Scode","Year"], how="left")
    revcol = "REV_src"
df["TATurn"] = pd.to_numeric(df[revcol], errors="coerce") / pd.to_numeric(df["AT"], errors="coerce")
df["TATurn"] = G.winsor(df["TATurn"].replace([np.inf,-np.inf], np.nan))
s = df["TATurn"].dropna()
print("TATurn: N=%d mean=%.4f sd=%.4f CV=%.1f%%" % (len(s), s.mean(), s.std(), s.std()/s.mean()*100))
print("IC    : N=%d mean=%.4f sd=%.4f CV=%.1f%%" % (df['IC'].notna().sum(), df['IC'].mean(), df['IC'].std(), df['IC'].std()/df['IC'].mean()*100))
# 资源配置优化代理：剔除绿色创新维度后的环境绩效（剥离机械关联，反映资源在非创新维度的配置效果）
df["EP_noG"] = G.zscore(Z[:, 1] + Z[:, 2])
OUT["代理变量对比"] = [["代理变量", "N", "均值", "标准差", "变异系数"],
                  ["IC（内控得分，原）", df['IC'].notna().sum(), f"{df['IC'].mean():.4f}", f"{df['IC'].std():.4f}", f"{df['IC'].std()/df['IC'].mean()*100:.1f}%"],
                  ["TATurn（总资产周转率，新）", len(s), f"{s.mean():.4f}", f"{s.std():.4f}", f"{s.std()/s.mean()*100:.1f}%"],
                  ["MA（绿色并购，原）", df['MA'].notna().sum(), f"{df['MA'].mean():.4f}", f"{df['MA'].std():.4f}", f"{df['MA'].std()/df['MA'].mean()*100:.1f}%"],
                  ["EP_noG（非创新维度，新）", df['EP_noG'].notna().sum(), f"{df['EP_noG'].mean():.4f}", f"{df['EP_noG'].std():.4f}", "—"]]
rows = [["传导路径", "中介变量", "总效应c", "a(Digit→M)", "b(M→EP)", "中介效应a×b", "占比", "Sobel Z", "N"]]
for nm, med in [("管理效率（总资产周转率）","TATurn"), ("资源配置（非创新维度绩效）","EP_noG")]:
    sub = df[df[med].notna()]
    c = fe(sub,"EP",["Digit"]+G.ALL_CTRL); a = fe(sub,med,["Digit"]+G.ALL_CTRL); b = fe(sub,"EP",["Digit",med]+G.ALL_CTRL)
    ab = a['beta'][0]*b['beta'][1]
    se = np.sqrt(a['beta'][0]**2*b['se'][1]**2 + b['beta'][1]**2*a['se'][0]**2)
    z = ab/se if se > 0 else np.nan
    ratio = ab/c['beta'][0]*100 if c['beta'][0] != 0 else np.nan
    print("\n  %s" % nm)
    print("    c=%.4f(t=%.2f) a=%.4f(t=%.2f) b=%.4f(t=%.2f) c'=%.4f(t=%.2f)"
          % (c['beta'][0],c['t'][0],a['beta'][0],a['t'][0],b['beta'][1],b['t'][1],b['beta'][0],b['t'][0]))
    print("    a×b=%.4f 占比=%.2f%% Sobel Z=%.3f N=%d" % (ab, ratio, z, b['N']))
    rows.append([nm, med, f"{c['beta'][0]:.4f}({c['t'][0]:.2f})", f"{a['beta'][0]:.4f}({a['t'][0]:.2f})",
                 f"{b['beta'][1]:.4f}({b['t'][1]:.2f})", f"{ab:.4f}", f"{ratio:.2f}%", f"{z:.3f}", b['N']])
OUT["中介变量替换"] = rows
jdump(OUT, os.path.join(OUTDIR, "supplementary.json"))

# ============================================================
print("\n" + "="*74); print("③ 结果复核与一致性"); print("="*74)
rng = np.random.default_rng(20240916)
codes = df["Scode"].unique()
res = []
for rep in range(10):
    pick = set(rng.choice(codes, size=int(len(codes)*0.8), replace=False))
    sub = df[df["Scode"].isin(pick)]
    r = fe(sub,"EP",["Digit"]+G.ALL_CTRL, iters=200)
    res.append((r['beta'][0], r['t'][0], r['N']))
    print("  随机子样本%2d: Digit=%7.4f t=%6.3f N=%d" % (rep+1, r['beta'][0], r['t'][0], r['N']))
arr = np.array([[a,b] for a,b,_ in res])
print("\n  10次重复: 系数均值=%.4f 标准差=%.4f 区间=[%.4f, %.4f]" % (arr[:,0].mean(), arr[:,0].std(), arr[:,0].min(), arr[:,0].max()))
print("  t值均值=%.3f  显著次数=%d/10" % (arr[:,1].mean(), int((np.abs(arr[:,1])>1.96).sum())))
OUT["随机子样本复核"] = [["重复","Digit系数","t值","N"]] + \
    [[f"{i+1}", f"{a:.4f}", f"{b:.3f}", n] for i,(a,b,n) in enumerate(res)] + \
    [["均值", f"{arr[:,0].mean():.4f}", f"{arr[:,1].mean():.3f}", ""],
     ["标准差", f"{arr[:,0].std():.4f}", f"{arr[:,1].std():.3f}", ""],
     ["系数区间", f"[{arr[:,0].min():.4f}, {arr[:,0].max():.4f}]", "", ""]]

# Bootstrap 中介（向量化：预去均值一次，重复抽样企业簇）
print("\n  Bootstrap 中介检验（绿色技术创新路径，1000次，向量化）")
dd = demean_matrix(df, ["EP"], ["Digit","GreenTech"]+G.ALL_CTRL, iters=200)
X = dd[["Digit"]+G.ALL_CTRL].values.astype(float)          # a 方程
Xb = dd[["Digit","GreenTech"]+G.ALL_CTRL].values.astype(float)  # b 方程
yM = dd["GreenTech"].values.astype(float); yE = dd["EP"].values.astype(float)
cl = dd["Scode"].values
groups = dd.groupby("Scode").indices
keys = list(groups.keys()); kidx = [groups[k] for k in keys]
boot = []
for _ in range(1000):
    sel = rng.integers(0, len(keys), len(keys))
    idx = np.concatenate([kidx[i] for i in sel])
    Xs, Xbs, yMs, yEs = X[idx], Xb[idx], yM[idx], yE[idx]
    ba, *_ = np.linalg.lstsq(Xs, yMs, rcond=None)
    bb, *_ = np.linalg.lstsq(Xbs, yEs, rcond=None)
    boot.append(ba[0]*bb[1])
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
print("    a×b 均值=%.4f  95%%CI=[%.4f, %.4f]  区间不含0: %s" % (boot.mean(), lo, hi, lo*hi>0))
OUT["Bootstrap中介"] = [["项目","值"],["a×b均值",f"{boot.mean():.4f}"],
                     ["95%CI下限",f"{lo:.4f}"],["95%CI上限",f"{hi:.4f}"],
                     ["结论","不含0，中介效应显著" if lo*hi>0 else "含0"]]
jdump(OUT, os.path.join(OUTDIR, "supplementary.json"))

# ============================================================
print("\n" + "="*74); print("④ AI 专属技术强度指标"); print("="*74)
ai = pd.read_pickle(os.path.join(AUX, "ai_count.pkl"))
ai = ai[(ai.Year >= 2007) & (ai.Year <= 2024)]
ai["AILn"] = np.log1p(ai["AICount"])
d2 = df.merge(ai[["Scode","Year","AILn"]], on=["Scode","Year"], how="left")
d2["AILn"] = d2["AILn"].fillna(0)
print("合并后 AI 指标覆盖: %d/%d (%.1f%%)" % (d2['AILn'].gt(0).sum(), len(d2), d2['AILn'].gt(0).mean()*100))
d2["AILn"] = G.winsor(d2["AILn"])
r = fe(d2, "EP", ["AILn"]+G.ALL_CTRL)
r2 = fe(d2, "GreenTech", ["AILn"]+G.ALL_CTRL)
print("AI强度 → EP      : %.4f (t=%.3f) %s" % (r['beta'][0], r['t'][0], G.stars_from_t(r['t'][0], r['G']-1) or "不显著"))
print("AI强度 → GreenTech: %.4f (t=%.3f) %s" % (r2['beta'][0], r2['t'][0], G.stars_from_t(r2['t'][0], r2['G']-1) or "不显著"))
OUT["AI专属指标"] = [["被解释变量","AI强度系数","t值","显著性","N"],
                  ["EP", f"{r['beta'][0]:.4f}", f"{r['t'][0]:.3f}", G.stars_from_t(r['t'][0], r['G']-1) or "不显著", r['N']],
                  ["GreenTech", f"{r2['beta'][0]:.4f}", f"{r2['t'][0]:.3f}", G.stars_from_t(r2['t'][0], r2['G']-1) or "不显著", r2['N']]]
jdump(OUT, os.path.join(OUTDIR, "supplementary.json"))
print("\n完成，用时 %.0f 秒" % (time.time()-t0))
