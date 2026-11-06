#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从现有部署包生成真机 2×2 对比用的 4 个包。

思路：4 个包**只差 3~4 个文件**（backbone / codec / dataset_stats / ar_codec_stats），
      其余（代码、启动脚本、live_configs、runtime/env 软链）全部共用。
      所以用 `cp -al`（硬链接复制）→ 瞬间完成、几乎不占额外空间；
      再把那几个文件换成指向对应来源的硬链接。

产物（默认输出到 --out 下）：
    g05_cell_ours_yam    我们的 backbone + YAM codec      ← 等价于现有包
    g05_cell_ours_off    我们的 backbone + 官方 codec
    g05_cell_off_off     官方 backbone   + 官方 codec
    g05_cell_off_yam     官方 backbone   + YAM codec

尺寸指纹（硬断言）：
    我们的 backbone  11,440,527,634    官方 backbone  11,440,519,964
    YAM codec           169,709,565    官方 codec        506,886,775
    YAM stats               226,311    官方 stats          5,258,626

用法：
    python3 tools/build_cell_pkgs.py \
        --src      /mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026 \
        --official /mnt/sda/chenyuan/official_hf \
        --out      /mnt/sda/chenyuan/motion

    # 先看要做什么，不动磁盘
    python3 tools/build_cell_pkgs.py ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

OURS_BB = 11440527634
OFF_BB = 11440519964
YAM_CODEC = 169709565
OFF_CODEC = 506886775
YAM_STATS = 226311
OFF_STATS = 5258626

# cell -> (backbone 来源标签, codec 标签, stats 标签)
CELLS = {
    'g05_cell_ours_yam': ('ours', 'yam', 'yam'),
    'g05_cell_ours_off': ('ours', 'off', 'yam'),
    'g05_cell_off_off':  ('off',  'off', 'off'),
    'g05_cell_off_yam':  ('off',  'yam', 'off'),
}


def hardlink_copytree(src, dst):
    """优先硬链接复制整包（同一文件系统下瞬间完成且几乎不占空间）。

    优先用系统 cp -al（Linux）；没有 cp（如 Windows）时用 Python 的 os.link 逐个建链，
    失败再退化成普通复制。
    """
    if os.name == 'posix' and shutil.which('cp'):
        r = subprocess.run(['cp', '-al', src, dst], capture_output=True, text=True)
        if r.returncode == 0:
            return None
        head = 'cp -al 失败: %s' % r.stderr.strip()[:160]

    def _cf(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    shutil.copytree(src, dst, copy_function=_cf, symlinks=True)
    return locals().get('head')


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def find_model_dir(pkg):
    base = os.path.join(pkg, 'models')
    if not os.path.isdir(base):
        raise RuntimeError('%s 下没有 models/ 目录' % pkg)
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isfile(os.path.join(d, 'checkpoints', 'model_state_dict.pt')):
            return d
    raise RuntimeError('%s/models/*/checkpoints/model_state_dict.pt 找不到' % pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='现有部署包（我们的 backbone + YAM codec）')
    ap.add_argument('--official', required=True, help='官方三件套所在目录')
    ap.add_argument('--out', required=True, help='输出父目录')
    ap.add_argument('--ours-backbone', default=None,
                    help='我们 backbone 路径（默认用 --src 里那个）')
    ap.add_argument('--yam-codec', default=None, help='YAM codec 路径（默认用 --src 里那个）')
    ap.add_argument('--yam-stats', default=None, help='YAM stats 路径（默认用 --src 里那个）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--relaxed-sizes', action='store_true',
                    help='仅用于冒烟测试：期望值改为实际文件大小（真实跑不要开）')
    a = ap.parse_args()

    try:
        src_model = find_model_dir(a.src)
    except Exception as e:
        print('  ✗ 无法定位 --src 里的模型目录: %s' % e)
        print('    期望结构: <src>/models/<name>/checkpoints/model_state_dict.pt')
        return 2
    ours_bb = a.ours_backbone or os.path.join(src_model, 'checkpoints', 'model_state_dict.pt')
    yam_codec = a.yam_codec or os.path.join(src_model, 'action_tokenizer.pt')
    yam_stats = a.yam_stats or os.path.join(src_model, 'dataset_stats.json')

    off_bb = os.path.join(a.official, 'model_state_dict.pt')
    off_codec = os.path.join(a.official, 'action_tokenizer_official.pt')
    off_stats = os.path.join(a.official, 'official_dataset_stats.json')

    print('  源包     : %s' % a.src)
    print('  模型目录 : %s' % src_model)
    print()

    # ---- 来源文件断言
    global OURS_BB, OFF_BB, YAM_CODEC, OFF_CODEC, YAM_STATS, OFF_STATS
    if a.relaxed_sizes:
        print('  ⚠️ --relaxed-sizes：期望值改为实际大小（只是冒烟测试，别用于真机实验）')
        OURS_BB = size(ours_bb); OFF_BB = size(off_bb)
        YAM_CODEC = size(yam_codec); OFF_CODEC = size(off_codec)
        YAM_STATS = size(yam_stats); OFF_STATS = size(off_stats)
    expect = {
        '我们的 backbone': (ours_bb, OURS_BB),
        '官方 backbone':   (off_bb, OFF_BB),
        'YAM codec':       (yam_codec, YAM_CODEC),
        '官方 codec':      (off_codec, OFF_CODEC),
        'YAM stats':       (yam_stats, YAM_STATS),
        '官方 stats':      (off_stats, OFF_STATS),
    }
    print('  === 来源文件检查 ===')
    bad = 0
    for name, (p, want) in expect.items():
        s = size(p)
        ok = (s == want)
        print('    %-16s %s' % (name, p))
        print('        %12d 期望 %12d  %s' % (s, want, '✅' if ok else '✗ 不符/缺失'))
        if not ok:
            bad += 1
    if bad:
        print()
        print('  ✗ 有 %d 个来源文件不对，先修好再继续' % bad)
        print('    （官方三件套用 tools/fetch_official_assets.py 下载）')
        return 1

    # backbone 与 codec 的标签空间不同（ours/off vs yam/off），按字段分别取
    def pick(field, tag):
        if field == 'backbone':
            return ours_bb if tag == 'ours' else off_bb
        if field == 'codec':
            return yam_codec if tag == 'yam' else off_codec
        return yam_stats if tag == 'yam' else off_stats

    print()
    print('  === 将生成 4 个包 ===')
    for cell, (b, c, s) in CELLS.items():
        print('    %-20s backbone=%-4s codec=%-3s stats=%s' % (cell, b, c, s))

    if a.dry_run:
        print()
        print('  [dry-run] 不写磁盘')
        return 0

    print()
    rows = []
    for cell, (b, c, s) in CELLS.items():
        dst = os.path.join(a.out, cell)
        if os.path.exists(dst):
            print('  ✗ %s 已存在，跳过（先手动删掉再重建）' % dst)
            continue
        print('  --- 构建 %s' % cell)

        # 硬链接整包（instant，几乎不占空间）
        try:
            warn_msg = hardlink_copytree(a.src, dst)
            if warn_msg:
                print('      ⚠ %s（已退化为 Python 硬链接复制）' % warn_msg)
        except Exception as e:
            print('      ✗ 复制失败: %s' % e)
            continue

        mdir = find_model_dir(dst)
        targets = [
            (os.path.join(mdir, 'checkpoints', 'model_state_dict.pt'), pick('backbone', b)),
            (os.path.join(mdir, 'action_tokenizer.pt'), pick('codec', c)),
            (os.path.join(mdir, 'dataset_stats.json'), pick('stats', s)),
        ]
        # 若包里还有 ar_codec_stats.json，按 stats 同源处理（它俩内容通常一致）
        acs = os.path.join(mdir, 'ar_codec_stats.json')
        if os.path.exists(acs):
            targets.append((acs, pick('stats', s)))

        okall = True
        for tgt, srcht in targets:
            try:
                if os.path.lexists(tgt):
                    os.remove(tgt)
                os.link(srcht, tgt)
            except Exception as e:
                print('      ✗ 链接失败 %s: %s' % (os.path.basename(tgt), e))
                okall = False
        if not okall:
            continue

        # 断言
        checks = [
            ('backbone', os.path.join(mdir, 'checkpoints', 'model_state_dict.pt'), OURS_BB if b == 'ours' else OFF_BB),
            ('codec',    os.path.join(mdir, 'action_tokenizer.pt'), YAM_CODEC if c == 'yam' else OFF_CODEC),
            ('stats',    os.path.join(mdir, 'dataset_stats.json'), YAM_STATS if s == 'yam' else OFF_STATS),
        ]
        print('      校验：')
        for label, p, want in checks:
            got = size(p)
            flag = '✅' if got == want else '✗'
            print('        %-9s %12d 期望 %12d %s' % (label, got, want, flag))
            if got != want:
                okall = False

        # 记录 ar.json 里指向的模型目录，确认一致
        arj = os.path.join(dst, 'live_configs', 'ar.json')
        ck = None
        if os.path.isfile(arj):
            try:
                cfg = json.load(open(arj, encoding='utf-8'))
                ck = cfg.get('checkpoint')
            except Exception:
                pass
        print('      ar.json checkpoint = %s' % ck)
        print('      %s %s' % ('✅ 完成' if okall else '✗ 有问题', dst))
        rows.append((cell, b, c, s, os.path.basename(mdir), okall))

    print()
    print('  === 汇总 ===')
    print('  %-20s %-8s %-7s %-6s %-6s' % ('包', 'backbone', 'codec', 'stats', 'OK'))
    for cell, b, c, s, mdir, ok in rows:
        print('  %-20s %-8s %-7s %-6s %-6s' % (cell, b, c, s, '✅' if ok else '✗'))

    print()
    print('  === 每个包怎么跑 ===')
    print('    D=/mnt/sda/chenyuan/motion/<包名>')
    print('    # 起服务端时把 --ckpt_path 指到新包的 model_state_dict.pt，')
    print('    # live_configs/ar.json 里的 checkpoint 也指向同一个包即可。')
    print('    # 客户端/执行层完全不变（同一套 bridge、限速、看门狗）。')
    return 0 if all(r[5] for r in rows) and len(rows) == 4 else 1


if __name__ == '__main__':
    sys.exit(main())
