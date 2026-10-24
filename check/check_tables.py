# -*- coding: utf-8 -*-
"""校验：results/tables.json 与 expected/tables.json 是否逐格一致。
无需原始数据即可运行；若 results 不存在，则只检查 expected 的完整性。"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "expected", "tables.json")
RES = os.path.join(os.environ.get("PAPER_OUT", os.path.join(ROOT, "results")), "tables.json")

exp = json.load(open(EXP, encoding="utf-8"))
print("基准结果键:", list(exp.keys()))
print("基准结果文件:", os.path.relpath(EXP, ROOT))

if not os.path.exists(RES):
    print("\n[跳过对比] 未找到 %s" % os.path.relpath(RES, ROOT))
    print("如需完整复现，请先运行 code/run_tables.py（需要原始数据）。")
    sys.exit(0)

res = json.load(open(RES, encoding="utf-8"))
print("复现结果键:", list(res.keys()))

DERIVED = {                      # 这两个键由 表15_表16 重排得到（见 docs/02）
    "表12_行业异质性": ("表15_表16", slice(1, 3)),
    "表13_区域异质性": ("表15_表16", slice(3, 6)),
}
total = same = 0
problems = []
for k, rows in exp.items():
    if k in DERIVED:
        src_key, sl = DERIVED[k]
        base = res.get(src_key, [])
        if not base:
            problems.append((k, "复现结果缺少来源键 %s" % src_key)); continue
        got = [["子样本", "系数", "t值", "N", "组间差异检验"]] + [[r[1], r[2], r[3], str(r[4]), r[5] if len(r) > 5 and r[5] else "—"] for r in base[sl]]
        want = [[c for c in r] for r in rows]
        # 基准文件首行为表头
        want = [want[0]] + [w[:5] for w in want[1:]]
        for w, g in zip(want, got):
            for a, b in zip(w, g):
                total += 1
                same += (str(a) == str(b))
        continue
    if k not in res:
        problems.append((k, "复现结果缺少该键")); continue
    for i, (ra, rb) in enumerate(zip(rows, res[k])):
        if len(ra) != len(rb):
            problems.append((k, "第 %d 行列数不同" % i)); continue
        for a, b in zip(ra, rb):
            total += 1
            same += (str(a) == str(b))

print("\n逐格比对: %d / %d 一致 (%.1f%%)" % (same, total, 100.0 * same / total if total else 0))
if problems:
    print("结构性问题:")
    for p in problems: print("   ", p)
else:
    print("结构完整，无缺表。")
