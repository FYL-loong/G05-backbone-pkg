#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""把 HF 上的离线评测产物（eval/A、eval/B）拉到本地，并打印指标对照。

为什么单独写一个：
  - 平台任务容器是【临时的】，跑完即销毁，所以评测产物必须先传到 HF，
    想看时再拉下来。这个脚本就是"拉下来"这一步。
  - `hf download --include "eval/*"` 能用，但本机/内网常出现
    `Server disconnected` / DNS 抖动，中途断掉会留下半截文件。
    这里对【每个文件】单独重试 + 断点续传（Range 请求），失败只影响那一个文件。
  - 默认只用标准库 urllib，不依赖 huggingface_hub（本机 hub 版本曾出过
    `process() takes no keyword arguments`）。

用法：
    python tools/fetch_eval_results.py                     # 默认拉到 ./eval_results
    python tools/fetch_eval_results.py --dest D:\G05\eval_results
    python tools/fetch_eval_results.py --side A            # 只拉 A（微调）
    python tools/fetch_eval_results.py --list              # 只列文件，不下载
    python tools/fetch_eval_results.py --no-compare        # 不打印对照表

拉完之后本地会是这样：
    eval_results/
      eval/A/metrics.json
      eval/A/episodes/000000/actions.pkl
      eval/A/episodes/000000/result_ar_fm.html          ← 浏览器直接打开
      eval/A/episodes/000000/result_chunks_left_arm.png
      ...
      eval/B/metrics.json

退出码：0 成功；1 有文件最终下载失败（其余仍会完成）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_REPO = "Zenith6666/yam-backbone"
DEFAULT_REV = "main"
DEFAULT_DEST = "eval_results"
CHUNK = 1 << 20          # 1 MiB
RETRIES = 6


def _log(msg: str) -> None:
    print(msg, flush=True)


def _api_tree(repo: str, prefix: str, rev: str = DEFAULT_REV) -> list[dict]:
    """列出 repo 下 prefix 的【所有文件】（递归），带重试。"""
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}/{prefix}?recursive=true"
    last = None
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [it for it in data if it.get("type") == "file"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** i, 20))
    raise RuntimeError(f"列目录失败 {url}: {type(last).__name__}: {last}")


def _download(repo: str, path: str, dest: str, rev: str = DEFAULT_REV,
              expected_size: int | None = None) -> bool:
    """下载单个文件；已存在且大小一致则跳过；中断后从断点续传。"""
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    out = os.path.join(dest, *path.split("/"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if expected_size is not None and os.path.isfile(out):
        if os.path.getsize(out) == expected_size:
            _log(f"  = 跳过（已存在 {expected_size:,}B）  {path}")
            return True

    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(out) if os.path.isfile(out) else 0
        if expected_size is not None and have == expected_size and have > 0:
            _log(f"  = 完成（{have:,}B）  {path}")
            return True
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(out, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
            got = os.path.getsize(out)
            if expected_size is not None and got != expected_size:
                _log(f"  ! 大小不符（{got:,} != {expected_size:,}），重试 {attempt}/{RETRIES}  {path}")
                time.sleep(min(2 ** attempt, 20))
                continue
            _log(f"  ✓ {got:>12,}B  {path}")
            return True
        except Exception as e:  # noqa: BLE001
            _log(f"  ! 下载出错（{type(e).__name__}），重试 {attempt}/{RETRIES}  {path}")
            time.sleep(min(2 ** attempt, 20))
    _log(f"  ✗ 最终失败  {path}")
    return False


def _find_metrics(dest: str, side: str) -> str | None:
    p = os.path.join(dest, "eval", side, "metrics.json")
    return p if os.path.isfile(p) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--rev", default=DEFAULT_REV)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--side", choices=["A", "B", "both"], default="both")
    ap.add_argument("--list", action="store_true", help="只列文件，不下载")
    ap.add_argument("--no-compare", action="store_true", help="不打印指标对照表")
    a = ap.parse_args()

    prefixes = ["eval"] if a.side == "both" else [f"eval/{a.side}"]
    files: list[dict] = []
    for p in prefixes:
        try:
            files += _api_tree(a.repo, p, a.rev)
        except RuntimeError as e:
            _log(f"  ✗ {e}")
            return 1
    # 去重（both 模式下 eval 与 eval/A 不会重叠，这里只是保险）
    seen, uniq = set(), []
    for it in files:
        if it["path"] not in seen:
            seen.add(it["path"]); uniq.append(it)
    files = sorted(uniq, key=lambda x: x["path"])

    total = sum(int(it.get("size") or 0) for it in files)
    _log(f"  HF {a.repo} / eval 下共 {len(files)} 个文件，合计 {total / 1024 / 1024:.2f} MB")
    if a.list:
        for it in files:
            _log(f"    {int(it.get('size') or 0):>12,}  {it['path']}")
        return 0

    dest = os.path.abspath(os.path.expanduser(a.dest))
    _log(f"  下载到: {dest}")
    ok = fail = 0
    for i, it in enumerate(files, 1):
        _log(f"  [{i}/{len(files)}] {it['path']}")
        if _download(a.repo, it["path"], dest, a.rev, int(it.get("size") or 0)):
            ok += 1
        else:
            fail += 1
    _log(f"\n  完成：成功 {ok} 个，失败 {fail} 个")

    # ---- 附带打印指标对照（有 A/B 两份 metrics.json 时）
    if not a.no_compare:
        # 现在 HF 上会同时存在：
        #   eval/A/metrics.json          ← 默认入口（只在"A/B 同轮、同 episode 数"时更新）
        #   eval/A/{N}ep/metrics.json    ← 按 episode 数分开存，永不互相覆盖
        # 先把所有变体列出来，避免"拿 A 的 3 条结果去比 B 的 10 条结果"这种口径错配。
        variants = []
        for root, _dirs, fns in os.walk(os.path.join(dest, "eval")):
            if "metrics.json" in fns:
                rel = os.path.relpath(os.path.join(root, "metrics.json"), dest)
                variants.append(rel.replace(os.sep, "/"))
        if len(variants) > 1:
            _log("\n  注意：发现多份 metrics.json（不同 episode 数的结果分开存）：")
            for v in sorted(variants):
                _log(f"    {v}")
            _log("    默认入口是 eval/{A,B}/metrics.json；要指定别的口径，"
                 "直接用 tools/eval_compare.py 比较对应两个文件。")

        ma, mb = _find_metrics(dest, "A"), _find_metrics(dest, "B")
        cmp_tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_compare.py")
        if ma and mb and os.path.isfile(cmp_tool):
            _log("\n  === 指标对照（eval/{A,B}/metrics.json，默认入口）===")
            import subprocess
            subprocess.run([sys.executable, cmp_tool, ma, mb,
                            "--label-a", "A(微调)", "--label-b", "B(基座)"])
        elif ma or mb:
            side = "A" if ma else "B"
            _log(f"\n  === {side} 的绝对指标 ===")
            d = json.load(open(ma or mb, encoding="utf-8"))
            for k in sorted(d):
                _log(f"    {k:<34} {float(d[k]):.6f}")

    # ---- 提示怎么看可视化
    ep_dir = os.path.join(dest, "eval", "A", "episodes")
    if os.path.isdir(ep_dir):
        _log(f"\n  可视化（浏览器打开 .html 看 GT vs 预测曲线）：")
        for name in sorted(os.listdir(ep_dir))[:5]:
            for fn in sorted(os.listdir(os.path.join(ep_dir, name))):
                if fn.endswith(".html"):
                    _log(f"    {os.path.join(ep_dir, name, fn)}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
