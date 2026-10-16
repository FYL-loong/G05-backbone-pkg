#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""把 nvidia-nccl-cu12 的 override 注入 pyproject.toml（幂等、可验证）。

为什么需要独立脚本（而不是塞在 setup_env.sh 的内联 Python 里）：
    内联版本曾把中文注释行吞进 TOML 值里，导致 `TOMLDecodeError: Invalid value`。
    在 shell heredoc + 字符串拼接里调试太容易出错，所以抽出来，本地能直接测。

用法：
    python tools/patch_nccl_override.py <pyproject.toml> <版本>
    python tools/patch_nccl_override.py pyproject.toml 2.29.7
    python tools/patch_nccl_override.py pyproject.toml --revert
"""
from __future__ import annotations

import re
import sys

# 我们自己注入的段落用固定标记，便于幂等替换/回退
MARK_BEGIN = "# >>> G05 NCCL override (由 tools/patch_nccl_override.py 注入) >>>"
MARK_END = "# <<< G05 NCCL override <<<"
SECTION = "[tool.uv.override-dependencies]"


def _strip_existing(src: str) -> str:
    """删掉我们之前注入的段落（带标记的），以及任何裸的 override-dependencies 段。"""
    # 1) 带标记的
    src = re.sub(
        re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"\n?",
        "",
        src,
        flags=re.S,
    )
    # 2) 裸段（行首是该 section 头，一直到下一个 [ 开头的行之前）
    src = re.sub(
        r"(?m)^\[tool\.uv\.override-dependencies\]\s*\n(?:[^\[].*\n?)*",
        "",
        src,
    )
    return src


def _block(ver: str) -> str:
    return (
        f"{MARK_BEGIN}\n"
        f"# 目的：GPU 是 sm_120(Blackwell)，而 uv.lock 原本锁 nvidia-nccl-cu12==2.26.2\n"
        f"# （2025-02），该版本不认识 sm_120，4 卡 NCCL 会报\n"
        f"#   Cuda failure 1 'invalid argument'\n"
        f"# 只换 NCCL，torch 保持 2.7.1 不动（1 卡已验证可用）。\n"
        f"{SECTION}\n"
        f'"nvidia-nccl-cu12=={ver}"\n'
        f"{MARK_END}\n"
    )


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, arg = sys.argv[1], sys.argv[2]

    src = open(path, encoding="utf-8").read()
    cleaned = _strip_existing(src)

    if arg == "--revert":
        open(path, "w", encoding="utf-8", newline="\n").write(cleaned)
        print("  已移除 NCCL override")
        return 0

    ver = arg
    block = _block(ver)

    # 插到 [tool.uv] 段之后（找不到就追加到文件末尾）
    anchor = re.search(r"(?m)^\[tool\.uv\]\s*$", cleaned)
    if anchor:
        insert_at = anchor.end() + 1  # 跳过该行的换行
        out = cleaned[:insert_at] + "\n" + block + cleaned[insert_at:]
    else:
        out = cleaned.rstrip() + "\n\n" + block

    # 校验：TOML 必须仍能解析，且 override 生效
    try:
        import tomllib
        parsed = tomllib.loads(out)
    except ImportError:  # py3.10 没 tomllib
        try:
            import tomli as tomllib  # type: ignore
            parsed = tomllib.loads(out)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] 无法校验 TOML：{e}", file=sys.stderr)
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] 注入后 TOML 非法：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    got = parsed.get("tool", {}).get("uv", {}).get("override-dependencies")
    if not got or f"nvidia-nccl-cu12=={ver}" not in got:
        print(f"  [FAIL] 注入后读不到 override-dependencies：{got!r}", file=sys.stderr)
        return 1

    open(path, "w", encoding="utf-8", newline="\n").write(out)
    print(f"  [ OK ] 已注入 {SECTION} -> nvidia-nccl-cu12=={ver}（TOML 校验通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
