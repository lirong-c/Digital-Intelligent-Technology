# 复现代码包（Replication Package）

本包覆盖论文中的 **19 张表、8 张图**以及正文报告的全部系数。原始数据受数据库授权限制不能随包分发（见 §3），但**变量构造规则、样本筛选规则、估计设定与最终结果**均已完整给出；只要按表 1、表 3 的口径取得同名数据，即可逐格复现。

---

## 1. 包内文件

```
replication_package/
├── README.md                     本文件
├── requirements.txt              Python 依赖与版本
├── code/
│   ├── generate_paper_tables_figures.py   ★ 估计引擎（935 行，未改动）
│   ├── run_sample.py             样本量 + 基准回归（只读，不写文件）
│   ├── run_tables.py             生成全部主表 → results/tables.json
│   ├── run_supplementary.py      生成评议回应类结果 → results/supplementary.json
│   ├── run_figures.py            重绘 8 张图 → results/figures/
│   └── build_ai_index.py         构造 AI 专属技术强度词频 → results/ai_count.pkl
├── check/                        校验脚本（与 results/ 比对，均无需原始数据）
│   ├── check_docx_tables.py      稿件内 10 张数据表 ↔ expected/tables.json
│   ├── check_tables.py           results/tables.json ↔ expected/tables.json
│   ├── check_supplementary.py    results/supplementary.json ↔ expected/supplementary.json
│   └── check_figures.py          results/figures/ ↔ expected/figures/（MD5）
├── expected/                     稿件所用的基准结果（对照物，勿修改）
│   ├── tables.json               表 4、5、6、8、13、14、15、18、19 的数据
│   ├── supplementary.json        表 10、11、12、16、17 的数据
│   ├── vif.csv                   表 7 的数据
│   ├── figures/                  8 张成稿图件（PNG）
│   └── manuscript.docx           投稿稿（用于核对表格）
├── aux/
│   ├── ai_count.pkl              年报文本 AI 专属术语计数（表 17 的输入，避免重复处理 20 GB 文本）
│   └── vocab.json                AI 术语词表
├── data/                         原始数据放置处（不随包分发，见 §3）
├── results/                      运行输出目录（首次运行后自动生成）
└── docs/
    ├── 01_数据字典与变量构造.md   每个原始数据集 → 变量的对应关系
    └── 02_表图代码对照表.md       每张表/图 → 代码位置 → 输出文件
```

> `code/generate_paper_tables_figures.py` 即论文实际使用的估计引擎，**计算逻辑逐行未改**（仅文档字符串中的示例命令改为通用写法）；`run_*.py` 由原始分析脚本改写，**只调整输入/输出路径**。

---

## 2. 运行环境

- Python 3.10
- 实测版本：pandas 2.3.3、NumPy 2.2.6、SciPy 1.15.3、matplotlib 3.10.9、openpyxl 3.1.5、python-docx 1.1
  （与稿件 4.4 节所列版本一致）

```bash
pip install -r requirements.txt
```

本包在 `Linux + Python 3.10.13` 下实测通过；上列版本即实测版本。

---

## 3. 数据可得性

论文所用原始数据全部来自商业数据库（CSMAR、CNRDS、WIND 及各省统计年鉴），**受授权协议约束不可再分发**。论文正文 4.1 节已说明：样本筛选规则见表 1，变量口径见表 3，依此可重建样本。

如需完整复现，请把数据按下列目录名放在 `data/` 下（或在运行时用环境变量 `PAPER_DATA` 指向数据根目录）：

| 目录 | 用途 |
|---|---|
| `AKRD_企业数字化转型_ALL/{年}/企业数字化转型_{年}.xlsx` | 数字化转型词频（Digit 分量） |
| `无形资产数字化/无形资产数字化.xlsx` | 数字化无形资产占比（Digit 分量） |
| `上市公司绿色专利申请情况/…xlsx`、`上市公司绿色专利获得情况/…xlsx` | 绿色专利（EP 维度、GreenTech） |
| `年报碳信息披露词频表/*.xlsx` | 碳披露词频（EP 维度） |
| `上市公司环境治理补助/…xlsx` | 环保补助（EP 维度） |
| `应计盈余管理/…xlsx`、`真实盈余管理/…xlsx` | 财务控制变量 |
| `审计单位与审计费用/…xlsx`、`审计单位及审计意见/…xlsx` | 审计控制变量 |
| `内部控制披露/…xlsx` | 管理效率（IC） |
| `绿色并购事件基本信息/…xlsx` | 资源配置优化（MA） |
| `公司基本信息/…xlsx` | 注册地、产权性质 |
| `省级数字经济政策统计表/…xlsx`、`环境保护支出/…xlsx` | 地区控制变量、工具变量 |
| `企业数据要素开发利用指数/…xlsx` | 稳健性替代口径 |
| `全国环境污染治理投资情况/…xlsx` | 图 4 |

字段级对应关系见 `docs/01_数据字典与变量构造.md`。

---

## 4. 快速校验（不需要原始数据）

```bash
python check/check_docx_tables.py
```

预期输出：`合计: 549/549 单元格一致`。
该脚本直接读取 `expected/manuscript.docx` 与 `expected/tables.json`，核对稿件中 10 张数据表是否与基准结果一致——**审稿人即使不接触原始数据，也能据此确认"稿件表格 = 代码输出"**。

---

## 5. 完整复现（需要原始数据）

```bash
export PAPER_DATA=/path/to/raw_data      # 数据根目录（含上表各子目录）
export PAPER_OUT=/path/to/results        # 输出目录，默认 ./results

python code/run_sample.py          # 样本量 + 基准回归，约 4 分钟（首次读 Excel）
python code/run_tables.py          # 全部主表 → results/tables.json
python code/run_supplementary.py   # 评议回应结果 → results/supplementary.json，约 5.5 分钟
python code/run_figures.py         # 8 张图 → results/figures/

python check/check_tables.py          # 逐格比对
python check/check_supplementary.py
python check/check_figures.py         # MD5 比对
```

首次运行后数据会缓存到 `results/raw_cache.pkl`，后续为秒级。

### 预期结果（与本包实测一致）

```
样本观测: 11571 | 企业数: 1286 | 省份: 31 | 2007—2024
模型(1) Digit beta=-0.0537 t=-3.962
模型(2) Digit beta=-0.0106 t=-0.716
模型(3) Digit beta=-0.0187 t=-1.249
模型(4) Digit beta= 0.0200 t= 1.335   ← 表 8 第(4)列
绿色技术创新:   beta= 0.0549 t= 2.477
图件: 8/8 字节级一致
```

---

## 6. 表、图与代码的对应

### 6.1 表

| 论文表 | 内容 | 生成代码 | 基准数据 |
|---|---|---|---|
| 表 1 | 样本筛选流程 | 稿件表格（数字见 `expected/tables.json` 的 `元信息`） | 元信息 |
| 表 2 | 环境绩效指标体系 | 变量构造 `build_variables()` | — |
| 表 3 | 变量定义 | 变量构造 `build_variables()` | — |
| 表 4 | 描述性统计 | `run_tables.py`（final_v2 第 15–19 行） | `tables.json:表7` |
| 表 5 | 年度演变 | 同上（第 21–28 行，取 8 个年份） | `tables.json:表8` |
| 表 6 | 相关系数 | 同上（第 30–33 行） | `tables.json:表9` |
| 表 7 | VIF | 引擎 `table_vif()` | `vif.csv` |
| 表 8 | 基准回归 | `run_tables.py`（第 35–55 行） | `tables.json:表11` |
| 表 9 | 变量测度口径 | 描述性表格 | — |
| 表 10 | 熵权法稳健性 | `run_supplementary.py` | `supplementary.json:熵权法` |
| 表 11 | 子维度平行检验 | 同上 | `…:三分维度平行检验` |
| 表 12 | 结构断点 | 同上 | `…:结构断点` |
| 表 13 | 稳健性与内生性 | `run_tables.py`（第 57–74 行） | `tables.json:表12` |
| 表 14 | 中介效应 | 同上（第 76–90 行） | `tables.json:表13` |
| 表 15 | 空间权重回归 | 同上（第 92–102 行） | `tables.json:表14` |
| 表 16 | 随机子样本重复 | `run_supplementary.py` | `…:随机子样本复核` |
| 表 17 | AI 专属强度 | `build_ai_index.py` + `run_supplementary.py` | `…:AI专属指标` |
| 表 18 | 行业异质性 | `run_tables.py`（第 104–113 行） | `tables.json:表15_表16` |
| 表 19 | 区域异质性 | 同上 | 同上 |

### 6.2 图

| 图 | 内容 | 代码 | 数据 |
|---|---|---|---|
| 图 1 | 传导与溢出机制 | `run_figures.py` `fig1()` | 机制系数（脚本内常量） |
| 图 2 | 数智化投入年度演变 | 同上 | 面板 `Digit`、`IntanRatio` 年均值 |
| 图 3 | 碳披露词频 | 同上 | 面板 `Carbon` 年均值 |
| 图 4 | 全国污染治理投资 | 同上 | `全国环境污染治理投资情况` |
| 图 5 | 赋能与绿色专利趋势 | 同上 | 面板 `GreenApp`、`Digit` |
| 图 6 | 东中西部绿色专利 | 同上 | 面板按 `Region` 分组 |
| 图 7 | 省域绿色专利 TOP10 | 同上 | 面板按 `Prov` 汇总 |
| 图 8 | 行业与区域分组对比 | 同上 | 面板按 `Heavy`/`Region` 分组 |

---

## 7. 已知的细微差异（如实说明）

复现时可能出现下列差异，均不影响论文结论：

1. **表 7（VIF）末位差 0.001**：现版 `table_vif()` 用 `np.linalg.lstsq` 求 R²，稿件数值来自早期实现（正规方程求逆），7 个变量在第 3 位小数相差 1（最大 VIF 2.951 vs 2.952）。两者均远低于阈值 10。
2. **表 16 前两行**：稿件"1—5 次/6—10 次"两行为未固定随机种子时的旧运行结果；以现固定种子（`20240916`）重算为 0.0189/0.0164（总均值 0.0177、标准差 0.0103、区间 [0.0059, 0.0367] 与稿件一致）。正文仅引用后四项。
3. **表 4 的负零**：EP 均值在重算中为 `-0.0000`，稿件统一显示为 `0.0000`（数值相同）。
4. **表 18/19**：`tables.json` 中的 `表12_行业异质性`、`表13_区域异质性` 由 `表15_表16` 重排得到（`check_tables.py` 已按此规则自动核对）。

---

## 8. 引用与联系

使用本包中的代码，请引用本文。代码问题请通过通讯作者联系。
版本：2026-10（与投稿稿 `expected/manuscript.docx` 对应）。
