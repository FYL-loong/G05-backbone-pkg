#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HF（gated 仓库 OpenGalaxea/G05）下载官方三件套，带断点续传 + 字节数硬断言。

为什么需要：
    实验室机器上 /mnt/sda/chenyuan/local_hf/ 只有【我们微调后】的产物，
    官方 g05-base 的 backbone / codec / stats 都没有。
    真机 2×2 的另外两格（官方 backbone）必须用这三个文件。

为什么不用 hf CLI：
    本机 runtime/env/bin/hf 的 shebang 指向不存在的解释器（坏的），
    且 huggingface.co 直连超时，只有 hf-mirror.com 直连可用。

用法：
    export HF_TOKEN=hf_xxx          # 或用 --token，或读 ~/.cache/huggingface/token
    python3 fetch_official_assets.py --dst /mnt/sda/chenyuan/official_hf

    # 只下某一个文件
    python3 fetch_official_assets.py --dst /mnt/sda/chenyuan/official_hf --only backbone

校验：
    严格按【字节数】断言（这就是区分官方与我们微调版最强的手段：
      官方 backbone 11,440,519,964  vs  我们的 11,440,527,634，差 7,670 字节）
    同时打印 sha256 供记录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = 'OpenGalaxea/G05'

# key -> (repo 内路径, 期望字节数, 说明)
ASSETS = {
    'backbone': ('g05-base/checkpoints/model_state_dict.pt', 11440519964,
                 '官方 g05-base backbone'),
    'codec':    ('action_tokenizer.pt',                       506886775,
                 '官方原版 action codec（训练 checkpoint，含 optimizer）'),
    'stats':    ('g05-base/dataset_stats.json',                 5258626,
                 '官方 dataset_stats.json（多本体统计）'),
    'hydra':    ('g05-base/.hydra/config.yaml',                   41437,
                 '官方 g05-base 的 hydra 配置（仅参考，可选）'),
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: str, total: int, token: str) -> None:
    """带 Range 断点续传 + 重试。"""
    tmp = dst + '.part'
    for attempt in range(1, 9):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have >= total:
            break
        headers = {'Authorization': 'Bearer ' + token}
        if have:
            headers['Range'] = 'bytes=%d-' % have
        req = urllib.request.Request(url, headers=headers)
        try:
            r = urllib.request.urlopen(req, timeout=180)
            mode = 'ab' if have and r.status == 206 else 'wb'
            if mode == 'wb':
                have = 0
            t0 = time.time()
            got = have
            with open(tmp, mode) as f:
                while True:
                    chunk = r.read(4 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (256 << 20) < (4 << 20):
                        el = max(time.time() - t0, 1e-6)
                        speed = (got - have) / 1048576.0 / el
                        print('      %5.1f%%  %.2f/%.2f GB  %.1f MB/s'
                              % (got * 100.0 / total, got / 1073741824.0,
                                 total / 1073741824.0, speed), flush=True)
            r.close()
        except urllib.error.HTTPError as e:
            if e.code == 416:          # 已经下完
                break
            print('      HTTP %s（第 %d 次），5 秒后重试' % (e.code, attempt), flush=True)
            time.sleep(5)
        except Exception as e:
            print('      %s（第 %d 次），5 秒后重试' % (type(e).__name__, attempt), flush=True)
            time.sleep(5)
        else:
            if os.path.getsize(tmp) >= total:
                break

    got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if got != total:
        raise RuntimeError('下载不完整：%d / %d 字节' % (got, total))
    os.replace(tmp, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', required=True, help='下载目录（建议 /mnt/sda/chenyuan/official_hf）')
    ap.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'),
                    help='默认 hf-mirror.com（huggingface.co 在本机直连超时）')
    ap.add_argument('--token', default=None)
    ap.add_argument('--only', default=None,
                    choices=list(ASSETS.keys()), help='只下某一个')
    ap.add_argument('--skip-hydra', action='store_true', help='跳过可选的 hydra 配置')
    a = ap.parse_args()

    token = (a.token or os.environ.get('HF_TOKEN')
             or (open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
                 if os.path.exists(os.path.expanduser('~/.cache/huggingface/token')) else ''))
    if not token:
        print('  ✗ 没有 HF_TOKEN（export HF_TOKEN=... 或 ~/.cache/huggingface/token）')
        return 2

    os.makedirs(a.dst, exist_ok=True)
    keys = [a.only] if a.only else [k for k in ASSETS
                                    if not (a.skip_hydra and k == 'hydra')]
    print('  目标目录: %s' % a.dst)
    print('  endpoint : %s' % a.endpoint)
    print()

    manifest = {}
    rc = 0
    for k in keys:
        rpath, want, desc = ASSETS[k]
        dst = os.path.join(a.dst, os.path.basename(rpath) if k != 'stats'
                           else 'official_dataset_stats.json')
        if k == 'codec':
            dst = os.path.join(a.dst, 'action_tokenizer_official.pt')
        if k == 'hydra':
            dst = os.path.join(a.dst, 'official_g05_base_hydra.yaml')
        print('  [%s] %s' % (k, desc))
        print('      repo : %s :: %s' % (REPO, rpath))
        print('      期望 : %d 字节' % want)

        if os.path.exists(dst) and os.path.getsize(dst) == want:
            print('      ✓ 已存在且大小正确，跳过下载')
        else:
            url = '%s/%s/resolve/main/%s' % (a.endpoint.rstrip('/'), REPO, rpath)
            try:
                download(url, dst, want, token)
            except Exception as e:
                print('      ✗ 下载失败: %s' % e)
                rc = 1
                continue

        size = os.path.getsize(dst)
        ok = (size == want)
        print('      ✓ 大小 %d %s' % (size, '（符合）' if ok else '（✗ 不符！）'))
        h = sha256_of(dst)
        print('      sha256 = %s' % h)
        manifest[k] = {'file': os.path.basename(dst), 'bytes': size,
                       'expected_bytes': want, 'sha256': h, 'repo_path': rpath}
        print()

    mpath = os.path.join(a.dst, 'MANIFEST_official.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('  清单已写: %s' % mpath)

    print()
    print('  === 识别哪个 backbone 是官方 vs 我们的 ===')
    print('    官方 = 11,440,519,964 B')
    print('    我们的 = 11,440,527,634 B（差 7,670 B）')
    return rc


if __name__ == '__main__':
    sys.exit(main())
