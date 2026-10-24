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
"""最终版：生成论文所需的全部修正结果（含机制分解、Sobel、描述统计、年度表）。"""
import numpy as np, pandas as pd, sys, json, os
import generate_paper_tables_figures as G
G.RAW_CACHE = os.path.join(OUT, "raw_cache.pkl")
G.OUT_DIR   = os.path.join(OUT, "_scratch")
from scipy import stats

raw = G.build_raw(); df = G.build_variables(raw); df["Scode"]=df["Scode"].astype(str)
N, FIRMS = len(df), df["Scode"].nunique()
R = {}

def fe(d,y,xs,fes=("Scode","Year"),cluster="Scode"):
    return G.within_est(d,y,xs,list(fes),cluster=cluster,iters=300)

# ---------- 表7 描述性统计 ----------
rows=[["变量","观测值","均值","标准差","最小值","最大值"]]
for c in ["EP","Digit","GreenTech","IC","MA","Size","Lev","ROA","FAsset","Growth","IntAsset","Cash","OCF","AuditFee","Audit","Policy","EnvExp"]:
    s=df[c].dropna(); rows.append([c,len(s),f"{s.mean():.4f}",f"{s.std():.4f}",f"{s.min():.4f}",f"{s.max():.4f}"])
R["表7"]=rows

# ---------- 表8 年度演变（统一为“企业均值”口径，与图3一致）----------
rows=[["年份","企业数（家）","数字化关键词词频企业均值（个）","绿色专利申请量（件）","数据要素指数企业均值","数字化无形资产占比（%）","碳信息披露词频企业均值（次）"]]
for y in sorted(df.Year.unique()):
    d=df[df.Year==y]
    rows.append([y,d.Scode.nunique(),f"{d.Decount.mean():.2f}",int(d.GreenApp.sum()),
                 f"{d.TermOnly.mean():.2f}" if d.TermOnly.notna().any() else "—",
                 f"{d.IntanRatio.mean()*100:.2f}",f"{d.Carbon.mean():.2f}"])
R["表8"]=rows

# ---------- 表9 相关系数 ----------
cols=["EP","Digit","GreenTech","Size","Lev","ROA","FAsset","Growth","Cash","AuditFee"]
c=df[cols].corr(); rows=[["变量"]+cols]+[[v]+[f"{c.loc[v,w]:.3f}" for w in cols] for v in cols]
R["表9"]=rows

# ---------- 表11 基准回归（含完整控制变量）----------
m=[G.pooled_ols(df,"EP",["Digit"]),
   G.pooled_ols(df,"EP",["Digit"]+G.CONTROLS),
   G.pooled_ols(df,"EP",["Digit"]+G.ALL_CTRL),
   fe(df,"EP",["Digit"]+G.ALL_CTRL)]
rows=[["变量","(1) EP","(2) EP","(3) EP","(4) EP"]]
rows.append(["Digit"]+[f"{mm['beta'][1 if i<3 else 0]:.3f}{G.stars_from_t(mm['t'][1 if i<3 else 0],mm['G']-1)}" for i,mm in enumerate(m)])
rows.append(["",*[f"({mm['t'][1 if i<3 else 0]:.3f})" for i,mm in enumerate(m)]])
for cc in G.ALL_CTRL:
    vals=[cc]
    for i,mm in enumerate(m):
        if i==0: vals.append("—")
        elif i==1: vals.append("—" if cc in G.REGION_CTRL else f"{mm['beta'][2+G.CONTROLS.index(cc)]:.3f}{G.stars_from_t(mm['t'][2+G.CONTROLS.index(cc)],mm['G']-1)}")
        elif i==2: vals.append(f"{mm['beta'][2+G.ALL_CTRL.index(cc)]:.3f}{G.stars_from_t(mm['t'][2+G.ALL_CTRL.index(cc)],mm['G']-1)}")
        else: vals.append(f"{mm['beta'][1+G.ALL_CTRL.index(cc)]:.3f}{G.stars_from_t(mm['t'][1+G.ALL_CTRL.index(cc)],mm['G']-1)}")
    rows.append(vals)
rows.append(["企业个体固定效应","否","否","否","是"])
rows.append(["时间固定效应","否","否","否","是"])
rows.append(["N",*[mm["N"] for mm in m]])
rows.append(["within R²",*[f"{mm['R2']:.3f}" for mm in m]])
R["表11"]=rows

# ---------- 表12 稳健性 ----------
rows=[["检验方式","具体设定","系数","t值","N","结论"]]
def add(tag,spec,r,concl):
    rows.append([tag,spec,f"{r['beta'][0]:.4f}",f"{r['t'][0]:.3f}",r["N"],concl])
add("替换被解释变量","绿色专利申请量对数",fe(df,"GreenAppLn",["Digit"]+G.ALL_CTRL),"1%水平显著为正")
add("替换被解释变量","绿色专利获得量对数",fe(df,"GreenTech",["Digit"]+G.ALL_CTRL),"5%水平显著为正")
d2=df.copy(); d2["EP_nc"]=(G.zscore(d2.GreenAppLn)+G.zscore(d2.SubsidyLn))/2
add("替换被解释变量","环境绩效指数（剔除碳披露维度）",fe(d2,"EP_nc",["Digit"]+G.ALL_CTRL),"10%水平显著为正")
add("替换核心解释变量","数字化无形资产占比",fe(df,"EP",["IntanRatio"]+G.ALL_CTRL),"不显著")
add("替换核心解释变量","数据要素开发利用指数（2013—2024年）",fe(df[df.Year>=2013],"EP",["TermOnlyLn"]+G.ALL_CTRL),"5%水平显著为正")
add("样本调整","剔除2024年（专利数据不完整）",fe(df[df.Year<=2023],"EP",["Digit"]+G.ALL_CTRL),"不显著")
dd=df.sort_values(["Scode","Year"]).copy()
for lag in (1,2):
    dd[f"L{lag}"]=dd.groupby("Scode")["Digit"].shift(lag)
    add(f"滞后解释变量（{lag}期）",f"L{lag}.Digit",fe(dd,"EP",[f"L{lag}"]+G.ALL_CTRL),G.stars_from_t(fe(dd,"EP",[f"L{lag}"]+G.ALL_CTRL)['t'][0],1200) or "不显著")
iv=G.iv_2sls_pooled(df,"EP","Digit","Policy",[c for c in G.ALL_CTRL if c!="Policy"])
rows.append(["工具变量法（2SLS第二阶段）",f"省级数字经济政策词频对数（第一阶段F={iv['F']:.2f}）",f"{iv['beta']:.4f}",f"{iv['t']:.3f}",iv["N"],"5%水平显著为正；标准误需按残差生成法重估"])
R["表12"]=rows

# ---------- 表13 机制（含Sobel）----------
rows=[["传导路径","中介变量","总效应c","a（Digit→M）","b（M→EP）","直接效应c′","中介效应a×b","占比a×b/c","Sobel Z","N"]]
for nm,med,sub in [("绿色技术创新","GreenTech",df),("管理效率（内控质量）","IC",df[df.Year>=2010]),
                   ("资源配置优化（绿色并购）","MA",df[df.Year>=2012])]:
    c=fe(sub,"EP",["Digit"]+G.ALL_CTRL); a=fe(sub,med,["Digit"]+G.ALL_CTRL); b=fe(sub,"EP",["Digit",med]+G.ALL_CTRL)
    ab=a["beta"][0]*b["beta"][1]
    se_ab=np.sqrt(a["beta"][0]**2*b["se"][1]**2+b["beta"][1]**2*a["se"][0]**2)
    z=ab/se_ab
    rows.append([nm,med,f"{c['beta'][0]:.4f}{G.stars_from_t(c['t'][0],c['G']-1)} ({c['t'][0]:.2f})",
                 f"{a['beta'][0]:.4f}{G.stars_from_t(a['t'][0],a['G']-1)} ({a['t'][0]:.2f})",
                 f"{b['beta'][1]:.4f}{G.stars_from_t(b['t'][1],b['G']-1)} ({b['t'][1]:.2f})",
                 f"{b['beta'][0]:.4f}{G.stars_from_t(b['t'][0],b['G']-1)} ({b['t'][0]:.2f})",
                 f"{ab:.4f}",f"{ab/c['beta'][0]*100:.2f}%",f"{z:.3f}",b["N"]])
R["表13"]=rows
print("Sobel(绿色技术创新) Z =", rows[1][-2])

# ---------- 表14 溢出 ----------
d=df.copy(); d["IndYear"]=d.IndcodeB.astype(str).str[:3]+"_"+d.Year.astype(str); d["ProvYear"]=d.Prov+"_"+d.Year.astype(str)
for g in ("IndYear","ProvYear"):
    s=d.groupby(g)[["Digit","EP"]].transform("sum"); n=d.groupby(g)["Scode"].transform("count")
    d[f"WD_{g}"]=(s.Digit-d.Digit)/(n-1); d[f"WE_{g}"]=(s.EP-d.EP)/(n-1)
rows=[["空间权重矩阵","直接效应","t值","间接效应","t值","总效应","ρ","t值","N"]]
for lab,g in [("行业空间权重矩阵","IndYear"),("区域（省份）空间权重矩阵","ProvYear")]:
    r=fe(d,"EP",["Digit",f"WD_{g}",f"WE_{g}"]+G.ALL_CTRL)
    rows.append([lab,f"{r['beta'][0]:.4f}",f"{r['t'][0]:.3f}",f"{r['beta'][1]:.4f}",f"{r['t'][1]:.3f}",
                 f"{r['beta'][0]+r['beta'][1]:.4f}",f"{r['beta'][2]:.4f}",f"{r['t'][2]:.3f}",r["N"]])
R["表14"]=rows

# ---------- 表15/16 异质性 ----------
rows=[["分组维度","子样本","系数","t值","N","组间差异检验(p值)"]]
for lab,f in [("重污染行业",df.Heavy),("非重污染行业",~df.Heavy)]:
    r=fe(df[f],"EP",["Digit"]+G.ALL_CTRL); rows.append(["行业",lab,f"{r['beta'][0]:.4f}",f"{r['t'][0]:.3f}",r["N"],""])
chow=df.copy(); chow["D_Heavy"]=chow.Heavy.astype(float); chow["Int"]=chow.Digit*chow.D_Heavy
r=fe(chow,"EP",["Digit","D_Heavy","Int"]+G.ALL_CTRL)
rows[-2][5]=f"{r['t'][2]:.3f} (t)"
for lab in ["东部","中部","西部"]:
    r=fe(df[df.Region==lab],"EP",["Digit"]+G.ALL_CTRL); rows.append(["区域",lab+"地区",f"{r['beta'][0]:.4f}",f"{r['t'][0]:.3f}",r["N"],""])
R["表15_表16"]=rows

# ---------- 元信息 ----------
R["元信息"]=[["项目","值"],["观测值",N],["企业数",FIRMS],["省级地区数",df.Prov.nunique()],
             ["样本期","2007—2024"],["2024年绿色专利申请量占2023年比重",f"{df[df.Year==2024].GreenApp.sum()/df[df.Year==2023].GreenApp.sum()*100:.1f}%"]]

def _c(o):
    import numpy as _np
    return o.item() if isinstance(o,_np.generic) else str(o)
json.dump(R, open(os.path.join(OUT, "tables.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1, default=_c)
print("\n==== 修正版全部结果 ====")
for k,v in R.items():
    print("\n----",k)
    for r in v: print(" | ".join(str(x) for x in r))
