#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 4 个 cell 包里的【元信息 + 硬编码路径】修正成各自实际内容。

要修的三处（都是 cell 包共有的坑）：
  ① models/ar_ft1ep/motion_package.json
       codec_sha256 / dataset_stats_sha256 / ar_codec_stats_sha256 /
       ckpt_sha256 / ckpt_bytes / source_checkpoint / initialization / step
     —— 这些原来全是【原包（ours+YAM）】的值，对 off_* 两格是错的。
  ② models/ar_ft1ep/motion_experiment.json 的 g05.checkpoint
  ③ live_configs/ar.json 的 g05.checkpoint
     —— 原来硬编码成 /…/g05_ft1ep_ours_20261026/models/ar_ft1ep。
        若有任何代码读了它而不是命令行 --ckpt_path，四格会全加载同一个模型，
        实验直接作废且数字看起来正常 —— 这是最危险的一处。

⚠️ 这些文件是【硬链接】（link count 5），必须"删除+重写"断开链接，
   否则改一个包会连带改掉源包和其它包。

用法：
    python3 tools/fix_cell_metadata.py --root /mnt/sda/chenyuan/motion
    python3 tools/fix_cell_metadata.py --root /mnt/sda/chenyuan/motion --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# 已知指纹（都是实测值，不用重新 hash 11GB）
FILES = {
    'ours_bb':   (11440527634, '31fa117df1bcf4cb6693b108ff27f55b0e1276cbf8ddb905cacc8faea50382c6'),
    'off_bb':    (11440519964, '072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327'),
    'yam_codec': (169709565,   'dfb9b19eb9b787a591556064b54eaf325b8b68df3027dbc2defb29d6f26f25ac'),
    'off_codec': (506886775,   '5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace'),
    'yam_stats': (226311,      'ddd12a9d1d87acea5d2fd615a0c95b177f14f71e4c1835917d6ebaaf30048d83'),
    'off_stats': (5258626,     'a336f126f846340503f7de705626422a5d9e2ef598de15e2fad14eb21a01dba2'),
}

CELLS = {
    'g05_cell_ours_yam': dict(bb='ours_bb', codec='yam_codec', stats='yam_stats',
                              init='official_g05_base + finetuned YAM codec + full 1epoch 10-task finetune',
                              step=17143),
    'g05_cell_ours_off': dict(bb='ours_bb', codec='off_codec', stats='yam_stats',
                              init='MISMATCH: ours backbone (trained with YAM codec) + OFFICIAL codec',
                              step=17143),
    'g05_cell_off_off':  dict(bb='off_bb', codec='off_codec', stats='off_stats',
                              init='official G0.5 g05-base release (backbone + codec + stats all original)',
                              step=None),
    'g05_cell_off_yam':  dict(bb='off_bb', codec='yam_codec', stats='off_stats',
                              init='MISMATCH: official backbone + finetuned YAM codec',
                              step=None),
}


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def rewrite_json(path, obj):
    """删除硬链接后重写（保证不污染其它包）。"""
    tmp = path + '.tmp_rewrite'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.remove(path)          # 断开硬链接
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='4 个 cell 包的父目录')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    ok_all = True
    for cell, spec in CELLS.items():
        pkg = os.path.join(a.root, cell)
        print('=' * 90)
        print('  %s' % cell)
        if not os.path.isdir(pkg):
            print('    ✗ 包不存在，跳过')
            ok_all = False
            continue

        mdir = os.path.join(pkg, 'models', 'ar_ft1ep')
        bb = os.path.join(mdir, 'checkpoints', 'model_state_dict.pt')
        codec = os.path.join(mdir, 'action_tokenizer.pt')
        stats = os.path.join(mdir, 'dataset_stats.json')
        acs = os.path.join(mdir, 'ar_codec_stats.json')

        # 先按实际字节数识别内容，和预期比对
        want_bbs, want_codes, want_sts = FILES[spec['bb']][0], FILES[spec['codec']][0], FILES[spec['stats']][0]
        got = (size(bb), size(codec), size(stats))
        print('    实际: backbone=%d  codec=%d  stats=%d' % got)
        if got != (want_bbs, want_codes, want_sts):
            print('    ✗ 实际内容与预期不符，跳过（期望 %d/%d/%d）'
                  % (want_bbs, want_codes, want_sts))
            ok_all = False
            continue
        print('    ✅ 内容与预期一致')

        new_ckpt_dir = os.path.join(pkg, 'models', 'ar_ft1ep')

        # ① motion_package.json
        mp = os.path.join(mdir, 'motion_package.json')
        if os.path.isfile(mp):
            obj = json.load(open(mp, encoding='utf-8'))
            before = (obj.get('ckpt_sha256', '')[:12], obj.get('codec_sha256', '')[:12],
                      obj.get('ckpt_bytes'))
            obj['checkpoint'] = 'checkpoints/model_state_dict.pt'
            obj['source_checkpoint'] = bb
            obj['ckpt_bytes'] = want_bbs
            obj['ckpt_sha256'] = FILES[spec['bb']][1]
            obj['codec_sha256'] = FILES[spec['codec']][1]
            obj['dataset_stats_sha256'] = FILES[spec['stats']][1]
            obj['ar_codec_stats_sha256'] = FILES[spec['stats']][1]
            obj['initialization'] = spec['init']
            if spec['step'] is not None:
                obj['step'] = spec['step']
            obj['cell_name'] = cell
            after = (obj['ckpt_sha256'][:12], obj['codec_sha256'][:12], obj['ckpt_bytes'])
            print('    ① motion_package.json  ckpt %s→%s  codec %s→%s  bytes %s→%s'
                  % (before[0], after[0], before[1], after[1], before[2], after[2]))
            if not a.dry_run:
                rewrite_json(mp, obj)

        # ② motion_experiment.json
        me = os.path.join(mdir, 'motion_experiment.json')
        if os.path.isfile(me):
            obj = json.load(open(me, encoding='utf-8'))
            g = obj.get('g05') or {}
            old = g.get('checkpoint')
            if isinstance(g, dict):
                g['checkpoint'] = new_ckpt_dir
                obj['g05'] = g
            print('    ② motion_experiment.json  checkpoint: %s → %s' % (old, new_ckpt_dir))
            if not a.dry_run:
                rewrite_json(me, obj)

        # ③ live_configs/ar.json
        ar = os.path.join(pkg, 'live_configs', 'ar.json')
        if os.path.isfile(ar):
            obj = json.load(open(ar, encoding='utf-8'))
            g = obj.get('g05') or {}
            old = g.get('checkpoint')
            if isinstance(g, dict):
                g['checkpoint'] = new_ckpt_dir
                obj['g05'] = g
            print('    ③ live_configs/ar.json    checkpoint: %s → %s' % (old, new_ckpt_dir))
            if not a.dry_run:
                rewrite_json(ar, obj)

        # 断开链接后确认新 inode 独立
        if not a.dry_run:
            for p in (mp, me, ar):
                if os.path.isfile(p):
                    st = os.stat(p)
                    if st.st_nlink > 1:
                        print('    ⚠ %s 仍有 %d 个硬链接' % (os.path.basename(p), st.st_nlink))

    print()
    print('=' * 90)
    if a.dry_run:
        print('  [dry-run] 未写磁盘')
    else:
        print('  %s' % ('✅ 全部修正完成' if ok_all else '⚠ 部分包未处理，见上'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
