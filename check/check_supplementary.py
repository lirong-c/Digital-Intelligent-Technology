# -*- coding: utf-8 -*-
"""校验：results/supplementary.json 与 expected/supplementary.json 是否一致。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "expected", "supplementary.json")
RES = os.path.join(os.environ.get("PAPER_OUT", os.path.join(ROOT, "results")), "supplementary.json")

exp = json.load(open(EXP, encoding="utf-8"))
print("基准结果键:", list(exp.keys()))
if not os.path.exists(RES):
    print("\n[跳过对比] 未找到 %s（需先运行 code/run_supplementary.py）" % os.path.relpath(RES, ROOT))
    sys.exit(0)
res = json.load(open(RES, encoding="utf-8"))
bad = 0
for k in exp:
    a = [[str(c) for c in r] for r in exp[k]]
    b = [[str(c) for c in r] for r in res.get(k, [])]
    ok = a == b
    print(("✓ " if ok else "✗ ") + k + ("  (%d 行)" % len(a)))
    if not ok:
        bad += 1
        for x, y in zip(a, b):
            if x != y: print("    期望:", x, "\n    复现:", y)
print("\n结论:", "全部一致" if bad == 0 else "%d 组不一致" % bad)
