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
"""构造 AI/智能化专属技术强度指标（回应评议：概念—测量鸿沟）。"""
import pandas as pd, numpy as np, glob, json, sys, re

# AI/新一代智能技术专属术语（从222词词表中筛选）
AI_TERMS = [
    '人工智能', '人工智能基础设施', '人工智能实验室', '人工智能平台', '人工智能系统', '人工智能设备',
    '机器学习', '神经网络', '深度学习', '学习算法',
    '大数据', '大数据信息系统', '大数据实验室', '大数据平台', '大数据技术系统', '大数据设备',
    '云计算', '区块链', '数字孪生', '物联网',
    '智能机器人', '机器人',
    '自然语言处理', '语音识别', '计算机视觉', '边缘计算', '智能制造',
    '虚拟现实', '增强现实', '混合现实', '智能穿戴', '智能终端', '智能数据分析', '智能规划',
    '数据挖掘', '分布式计算', '网格计算', '自动推理', 'OCR技术', '智能控制', '自动检测',
]
# 先读词表，确认术语确实存在
vocab = json.load(open(os.path.join(AUX, "vocab.json"), encoding='utf-8'))
AI_TERMS = [t for t in AI_TERMS if t in vocab]
print("匹配到的 AI 专属术语 %d 个：" % len(AI_TERMS))
print("  ", AI_TERMS)

frames = []
for i, f in enumerate(sorted(glob.glob('年报文本数字化/*.xlsx')), 1):
    d = pd.read_excel(f, header=0, skiprows=[1], usecols=['股票代码', '会计年度', '数字化转型词汇', '词频统计'])
    d = d[d['数字化转型词汇'].isin(AI_TERMS)]
    if len(d) == 0:
        continue
    d['Scode'] = d['股票代码'].map(lambda x: str(int(x)).zfill(6))
    d['Year'] = pd.to_numeric(d['会计年度'], errors='coerce')
    d['cnt'] = pd.to_numeric(d['词频统计'], errors='coerce').fillna(0)
    frames.append(d[['Scode', 'Year', 'cnt']])
    print(f"  [{i}] {len(d)} 行")
ai = pd.concat(frames, ignore_index=True)
ai = ai.dropna(subset=['Year'])
ai['Year'] = ai['Year'].astype(int)
ai = ai.groupby(['Scode', 'Year'], as_index=False)['cnt'].sum().rename(columns={'cnt': 'AICount'})
print("\nAI 词频面板:", ai.shape, " 企业数:", ai['Scode'].nunique(), " 年份:", ai['Year'].min(), "-", ai['Year'].max())
print(ai['AICount'].describe())
ai.to_pickle(os.path.join(OUT, "ai_count.pkl"))
print("已保存 .rev/ai_count.pkl")
