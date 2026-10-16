#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""定位 torch 到底从哪个文件加载 NCCL（多份 libnccl.so.2 时谁赢）。

背景：升级 venv 的 nvidia-nccl-cu12 到 2.29.7 成功，但
torch.cuda.nccl.version() 仍报 (2,26,2) —— 说明磁盘上还有另一份 libnccl.so.2
被优先加载（通常是 LD_LIBRARY_PATH 里容器自带的那份）。

用法（在容器里）：
    .venv/bin/python tools/which_nccl.py
或
    python tools/which_nccl.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys


def mb(n: int) -> str:
    return f"{n / 1e6:.1f}MB"


def main() -> int:
    print("=" * 72)
    print("torch 实际加载的 NCCL")
    print("=" * 72)
    try:
        import torch
        print(f"  python        : {sys.executable}")
        print(f"  torch         : {torch.__version__}")
        print(f"  torch 目录     : {os.path.dirname(torch.__file__)}")
        try:
            print(f"  nccl.version(): {torch.cuda.nccl.version()}   <-- torch 报出来的")
        except Exception as e:  # noqa: BLE001
            # CPU-only 构建没有 _nccl_version；这不影响后面找 .so
            print(f"  nccl.version(): 不可用（{type(e).__name__}: {e}）")
            print("                   （CPU-only 构建或未链接 NCCL —— 继续找 .so）")
    except Exception as e:  # noqa: BLE001
        print(f"  import torch 失败: {type(e).__name__}: {e}")
        return 1

    print()
    print("=" * 72)
    print("磁盘上所有 libnccl.so.2（按加载优先级排列：LD_LIBRARY_PATH 最前）")
    print("=" * 72)
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"  LD_LIBRARY_PATH = {ldp or '(空)'}")
    print(f"  LD_PRELOAD      = {os.environ.get('LD_PRELOAD') or '(空)'}")

    roots = [p for p in ldp.split(os.pathsep) if p]
    roots += [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/opt/hpcx",
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/lib/python3/dist-packages",
    ]
    try:
        import site
        roots += list(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    # venv 的 site-packages（含 nvidia/nccl/lib）
    roots += glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages"))
    roots += glob.glob(os.path.join(sys.prefix, "lib64", "python*", "site-packages"))

    # ⚠️ 不要用 glob('**/libnccl.so*', recursive=True) —— 搜索范围无边界，
    #    在容器里可能扫整个 site-packages 甚至更外层，很慢。改为有界遍历：
    #    每个 root 最多向下 4 层，并把 nvidia/nccl/lib 这种已知位置优先看。
    SKIP = {".git", "__pycache__", "node_modules", ".cache"}

    def find_in(root: str, max_depth: int = 4) -> list[str]:
        hits: list[str] = []
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP]
            if dirpath.count("/") - base_depth >= max_depth:
                dirnames[:] = []
            for fn in filenames:
                if fn.startswith("libnccl.so"):
                    hits.append(os.path.join(dirpath, fn))
        return hits

    seen: dict[str, str] = {}
    order = 0
    for r in dict.fromkeys(roots):
        if not r or not os.path.isdir(r):
            continue
        try:
            hits = find_in(r)
        except Exception:  # noqa: BLE001
            continue
        for f in sorted(hits):
            if f in seen:
                continue
            order += 1
            try:
                st = os.stat(f)
                real = os.path.realpath(f)
                seen[f] = f"#{order} {mb(st.st_size):>9}  {f}"
                if real != f:
                    seen[f] += f"  -> {real}"
            except Exception as e:  # noqa: BLE001
                seen[f] = f"#{order} ?  {f}  ({e})"

    print(f"  搜索了 {len([r for r in dict.fromkeys(roots) if os.path.isdir(r)])} 个目录")
    if not seen:
        print("  ⚠️ 一个 libnccl.so* 都没找到（NCCL 可能是静态链接进 torch）")
    for v in seen.values():
        print("  " + v)

    print()
    print("=" * 72)
    print("用 ldd 看 torch 的 C 扩展实际链接到哪个 libnccl")
    print("=" * 72)
    try:
        import torch as _t
        libdir = os.path.join(os.path.dirname(_t.__file__), "lib")
        cands = glob.glob(os.path.join(libdir, "libtorch_cuda.so")) + \
            glob.glob(os.path.join(libdir, "libtorch_python.so"))
        if cands:
            for c in cands[:2]:
                print(f"  --- ldd {os.path.basename(c)} | grep -i nccl ---")
                r = subprocess.run(["ldd", c], capture_output=True, text=True)
                hits = [ln.strip() for ln in (r.stdout or "").splitlines()
                        if "nccl" in ln.lower()]
                print("    " + ("\n    ".join(hits) if hits else "(无 nccl 条目)"))
        else:
            print("  找不到 libtorch_cuda.so")
    except Exception as e:  # noqa: BLE001
        print(f"  ldd 检查失败: {e}")

    print()
    print("=" * 72)
    print("uv pip show（确认装到哪个环境）")
    print("=" * 72)
    for cmd in (["uv", "pip", "show", "--python", sys.executable, "nvidia-nccl-cu12"],
                ["uv", "pip", "list", "--python", sys.executable]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = (r.stdout or r.stderr or "").strip()
            if "list" in cmd:
                out = "\n".join(ln for ln in out.splitlines() if "nccl" in ln.lower())
            print(f"  $ {' '.join(cmd)}")
            print("    " + out.replace("\n", "\n    ") if out else "    (无输出)")
        except FileNotFoundError:
            print(f"  uv 不在 PATH —— 跳过 {' '.join(cmd)}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {' '.join(cmd)} 失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
