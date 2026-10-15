#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""包完整性门禁：检查所有 `from g05.X import ...` 的目标模块是否真实存在。

为什么需要它（真实踩过的坑）：
    本包是从上游仓库"抽取"出来的，抽取时可能漏掉整个目录；而且
    .gitignore 若写了【裸目录名】（如 `eval/`），会把同名源码目录一起忽略，
    导致文件从未被 push —— 容器里才报
        ModuleNotFoundError: No module named 'g05.utils.eval'
    （真实案例：finetune.py:451 因此崩溃，而 MODE=setup 全过、看不出问题，
      因为 setup 只下载权重和数据，不 import 模型代码。）

    静态检查只要几秒，却能避免"跑到 L1 才发现、白排一次 GPU 队列"。

用法：
    python check_package.py            # 默认检查脚本所在目录
    python check_package.py <repo根>

退出码：0 = 通过；1 = 有 import 无法解析（附排查方向）
"""
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                       else os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PKG = os.path.join(SRC, "g05")


def collect_modules():
    mods = set()
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, SRC).replace(os.sep, ".")
        if "__init__.py" in filenames:
            mods.add(rel)
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                mods.add(rel + "." + f[:-3])
    return mods


def main():
    if not os.path.isdir(PKG):
        print(f"[FAIL] 找不到 {PKG}", file=sys.stderr)
        return 1

    mods = collect_modules()
    print(f"  src 下可解析的 g05 模块数: {len(mods)}")

    pat = re.compile(r"^\s*(?:from\s+(g05[\w.]*)\s+import|import\s+(g05[\w.]*))", re.M)
    files = []
    for base in ("scripts", "tools", "src"):
        d = os.path.join(ROOT, base)
        if not os.path.isdir(d):
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for f in filenames:
                if f.endswith(".py"):
                    files.append(os.path.join(dirpath, f))

    missing = {}
    n_imports = 0
    for p in files:
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in pat.finditer(txt):
            mod = m.group(1) or m.group(2)
            n_imports += 1
            if mod not in mods:
                missing.setdefault(mod, set()).add(os.path.relpath(p, ROOT))

    print(f"  扫描 {len(files)} 个 .py 文件，共 {n_imports} 条 g05 import")
    if not missing:
        print("  [ OK ] 所有 g05.* import 都能解析")
        return 0

    print(f"\n[FAIL] {len(missing)} 个 import 指向不存在的模块：", file=sys.stderr)
    for mod, where in sorted(missing.items()):
        print(f"  {mod}", file=sys.stderr)
        for w in sorted(where)[:4]:
            print(f"      <- {w}", file=sys.stderr)
    print("\n  排查方向：", file=sys.stderr)
    print("    1) 抽取包时是否漏拷了整个目录（与上游 src/g05 的同名目录对比）", file=sys.stderr)
    print("    2) 文件是否被 .gitignore 吞了：", file=sys.stderr)
    print("       git check-ignore -v <该文件路径>", file=sys.stderr)
    print("       ⚠️ 裸目录名（如 `eval/`）会匹配任意层级，必须写成 `/eval/`", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
