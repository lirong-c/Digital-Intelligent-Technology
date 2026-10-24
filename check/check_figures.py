# -*- coding: utf-8 -*-
"""校验：results/figures/*.png 与 expected/figures/*.png 是否字节一致（无需原始数据）。"""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "expected", "figures")
RES = os.path.join(os.environ.get("PAPER_OUT", os.path.join(ROOT, "results")), "figures")

def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
names = sorted(os.listdir(EXP))
print("基准图件 %d 张：%s" % (len(names), EXP))
if not os.path.isdir(RES):
    print("\n[跳过对比] 未找到 %s（需先运行 code/run_figures.py）" % os.path.relpath(RES, ROOT))
    sys.exit(0)
ok = 0
for n in names:
    a, b = md5(os.path.join(EXP, n)), md5(os.path.join(RES, n)) if os.path.exists(os.path.join(RES, n)) else None
    same = a == b
    ok += same
    print(("✓ " if same else "✗ ") + n)
print("\n结果: %d/%d 字节一致" % (ok, len(names)))
