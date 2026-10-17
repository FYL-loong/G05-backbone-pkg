#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""清理 HF 仓库里的中间 checkpoint，只保留交付物。

背景
----
正式训练每个 checkpoint 约 34.3 GB（model 11.44 + optimizer 22.88），
`upload_checkpoints.sh` 会把它们陆续推到 HF，1 个 epoch 就留下 10 份左右，
仓库体积可达 300+ GB —— 而真正有用的只有最终权重。

默认【只列出、不删除】（dry-run）。确认后加 --yes 真正删除。

保留
----
    model_state_dict.pt         最终微调权重（交付物）
    checkpoints/step_<FINAL>.pt 与 model_state_dict.pt 同尺寸的那份（最终 checkpoint）
    其余所有文件                action_tokenizer.pt / dataset_stats.json /
                                hydra_config.yaml / hf_processor/ / logs/

删除
----
    checkpoints/step_*.pt 中除上述以外的（每份约 34.3GB，含优化器状态）

⚠️ 为什么不用「编号最大的」当最终 checkpoint：
    实测 step_18000.pt（34.3GB，含优化器）编号比 step_17143.pt（11.4GB，纯推理）
    更大，但 last.pt 指向的是 17143，即真正的最终权重是后者。
    所以判据是【与 model_state_dict.pt 同文件尺寸】，不是最大编号。

用法
----
    export HF_TOKEN=hf_xxx          # 需要 write 权限
    python3 tools/clean_hf_checkpoints.py          # 预览（列出将删什么、释放多少）
    python3 tools/clean_hf_checkpoints.py --yes    # 真删

    # 额外保留编号最大的 N 份（一般不需要：它们是大文件）
    python3 tools/clean_hf_checkpoints.py --keep-steps 1 --yes

    # 换仓库
    python3 tools/clean_hf_checkpoints.py --repo someone/other --yes

本脚本优先用 huggingface_hub；若不可用则回退到纯 urllib 调用 HF 的 commit API
（这样在没装依赖的机器上也能跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

STEP_RE = re.compile(r"^checkpoints/step_(\d+)\.pt$")
HF = "https://huggingface.co"


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# --------------------------------------------------------------------------- HTTP 回退
def _http_json(url: str, token: str | None = None, payload: dict | None = None,
               method: str | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}


def _list_files_http(repo: str, token: str | None) -> dict[str, int]:
    d = _http_json(f"{HF}/api/models/{repo}/tree/main?recursive=true", token)
    out: dict[str, int] = {}
    for it in d if isinstance(d, list) else []:
        if it.get("type") == "file":
            out[it["path"]] = int(it.get("size") or 0)
    return out


def _delete_files_http(repo: str, paths: list[str], token: str, batch: int = 40) -> list[str]:
    """返回删除失败的路径列表。HF commit API 支持一次提交多个 delete 操作。"""
    failed: list[str] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ops = [{"op": "delete", "path": p} for p in chunk]
        payload = {"operations": ops, "summary": "清理中间 checkpoint（只保留最终权重）"}
        try:
            _http_json(f"{HF}/api/models/{repo}/commit/main", token, payload, "POST")
            for p in chunk:
                print(f"[clean] 已删除 {p}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            print(f"[clean] 批量删除失败（HTTP {e.code}）: {detail}", file=sys.stderr)
            # 逐条重试，定位具体失败项
            for p in chunk:
                try:
                    _http_json(f"{HF}/api/models/{repo}/commit/main", token,
                               {"operations": [{"op": "delete", "path": p}],
                                "summary": "清理中间 checkpoint"}, "POST")
                    print(f"[clean] 已删除 {p}")
                except Exception as e2:  # noqa: BLE001
                    failed.append(p)
                    print(f"[clean] 删除失败 {p}: {type(e2).__name__}: {e2}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failed.extend(chunk)
            print(f"[clean] 批量删除异常: {type(e).__name__}: {e}", file=sys.stderr)
    return failed


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO_ID", "Zenith6666/yam-backbone"))
    ap.add_argument("--yes", action="store_true", help="真正删除（默认只预览）")
    ap.add_argument("--keep-steps", type=int, default=0,
                    help="额外保留编号最大的 N 份（默认 0；最终 checkpoint 始终保留）")
    ap.add_argument("--http", action="store_true", help="强制走 HTTP（不依赖 huggingface_hub）")
    a = ap.parse_args()

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HF_TOKEN_WRITE"))
    repo = a.repo

    use_hub = False
    api = None
    if not a.http:
        try:
            from huggingface_hub import HfApi  # noqa: F401
            use_hub = True
        except Exception:  # noqa: BLE001
            use_hub = False

    if a.yes and not token:
        print("[clean] 删除需要 HF_TOKEN（write 权限）", file=sys.stderr)
        return 2

    print(f"[clean] 仓库: {repo}   方式: {'huggingface_hub' if use_hub else 'HTTP'}")

    # ---- 列出文件
    try:
        if use_hub:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
            files = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
        else:
            files = _list_files_http(repo, token)
    except Exception as e:  # noqa: BLE001
        print(f"[clean] 读取仓库文件失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    steps = []
    for f, sz in files.items():
        m = STEP_RE.match(f)
        if m:
            steps.append((int(m.group(1)), f, sz))
    steps.sort()

    # 最终 checkpoint = 与 model_state_dict.pt 同尺寸的那份（不是编号最大的！）
    ref = files.get("model_state_dict.pt")
    final_by_size = {f for _, f, sz in steps if ref and sz == ref}
    keep_recent = ({f for _, f, _ in steps[-a.keep_steps:]} if a.keep_steps > 0 else set())
    keep = final_by_size | keep_recent

    total = sum(files.values())
    to_delete = [(f, sz) for _, f, sz in steps if f not in keep]
    freed = sum(sz for _, sz in to_delete)

    print(f"[clean] 文件 {len(files)} 个，总体积 {_gb(total)}")
    print(f"[clean] step checkpoint {len(steps)} 份：")
    for s, f, sz in steps:
        print(f"        [{'保留' if f in keep else '删除'}] step {s:>6}  {_gb(sz):>9}  {f}")
    if ref and not final_by_size:
        print("[clean] ⚠️ 没有 step checkpoint 与 model_state_dict.pt 同尺寸，"
              "无法判定最终权重（本次不会删除任何 step 文件）", file=sys.stderr)
        return 1
    if not to_delete:
        print("[clean] 没有需要删除的文件。")
        return 0

    print(f"[clean] 将删除 {len(to_delete)} 个，释放 {_gb(freed)}")
    print(f"[clean] 仓库体积 {_gb(total)} -> {_gb(total - freed)}")

    if not a.yes:
        print("\n[clean] 预览模式。确认后重跑并加 --yes")
        return 0

    # ---- 执行删除
    ok = fail = 0
    if use_hub:
        for f, sz in to_delete:
            try:
                api.delete_file(path_in_repo=f, repo_id=repo, repo_type="model",
                                commit_message="清理中间 checkpoint（只保留最终权重）")
                ok += 1
                print(f"[clean] 已删除 {f}  ({_gb(sz)})")
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"[clean] 删除失败 {f}: {type(e).__name__}: {e}", file=sys.stderr)
    else:
        failed = _delete_files_http(repo, [f for f, _ in to_delete], token)
        fail = len(failed)
        ok = len(to_delete) - fail

    print(f"\n[clean] 完成：成功 {ok}，失败 {fail}")
    if fail:
        print("[clean] 有失败项；可重跑本脚本（已删的会自动跳过）", file=sys.stderr)
        return 1
    print(f"[clean] 现在仓库应约 {_gb(total - freed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
