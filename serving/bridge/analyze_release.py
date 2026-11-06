"""分析一次 trial 的释放时刻：模型想放多高 vs 实际放到多高。

回答的核心问题：
  「放太高」到底是模型自己提前松手，还是我们的低通/限速让手臂滞后导致的？

判定方法（看 lag = cmd_target - raw_target）：
  - lag 很小（<0.5°）  → 我们没拖后腿，是模型自己想在那里松手
  - lag 很大（>2°）    → 低通/限速把手臂拖在了高处，是执行层的问题

夹爪语义：**1.0 = 张开，0.0 = 闭合**
  - 抓取事件 = 从 >0.5 变到 <=0.5（闭合）
  - 释放事件 = 从 <=0.5 变到 >0.5（张开）

用法：
    python analyze_release.py                      # 自动取最新 CSV
    python analyze_release.py /path/to/ep.csv      # 指定文件
    python analyze_release.py /path/to/dir         # 取该目录最新 CSV
"""
from __future__ import annotations

import glob
import json
import os
import sys

# 关节顺序：[L0..L5, L_grip, R0..R5, R_grip]
NAMES = ['L_arm1', 'L_arm2', 'L_arm3', 'L_arm4', 'L_arm5', 'L_arm6', 'L_grip',
         'R_arm1', 'R_arm2', 'R_arm3', 'R_arm4', 'R_arm5', 'R_arm6', 'R_grip']
GRIP_IDX = (6, 13)


def newest_csv(arg: str | None) -> str:
    if arg and os.path.isfile(arg):
        return arg
    roots = [arg] if arg and os.path.isdir(arg) else ['/tmp/g05_runs/A', '/tmp/g05_runs/B', '/tmp/g05_runs']
    for r in roots:
        hits = glob.glob(os.path.join(r, '**', '*.csv'), recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    raise SystemExit('  找不到 CSV，请显式传路径')


def main() -> int:
    path = newest_csv(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit('  CSV 是空的')
    rows.sort(key=lambda r: r['step'])

    print('=' * 94)
    print('  释放时刻分析： %s' % path)
    print('  步数 %d   时长 %.2fs   换段次数 %d'
          % (len(rows), rows[-1]['t'], sum(1 for r in rows if r.get('sent_obs'))))
    print('=' * 94)

    # ── 抓取 / 释放事件（夹爪 1.0=张开, 0.0=闭合）──────────────────────────
    events = []
    for j in GRIP_IDX:
        prev = rows[0]['cmd_target'][j]
        for r in rows:
            cur = r['cmd_target'][j]
            if prev > 0.5 >= cur:
                events.append((r['step'], NAMES[j], '闭合(抓取)'))
            elif prev <= 0.5 < cur:
                events.append((r['step'], NAMES[j], '张开(释放)'))
            prev = cur
    events.sort()
    print()
    print('  ── 夹爪事件 ────────────────────────────────────────────────────')
    if not events:
        print('    （没有夹爪开合事件 —— 这一轮模型没抓任何东西）')
        return 1
    for step, name, kind in events:
        print('    step %4d  %-6s  %s' % (step, name, kind))

    # ── 每个释放事件前后：模型原始目标 vs 实际下发 ────────────────────────
    releases = [e for e in events if '释放' in e[2]]
    print()
    print('  ── 释放时刻：模型要放哪 vs 实际放到哪 ──────────────────────────')
    print('     lag = cmd_target - raw_target（正值 = 我们比模型指令【低】，负值 = 拖在高处）')
    print()
    print('    %-5s %-8s %9s %9s %8s %8s' % ('step', 'joint', 'raw', 'cmd', 'lag(°)', 'grip'))
    for step, name, _ in releases:
        r = next(x for x in rows if x['step'] == step)
        worst = None
        for i, nm in enumerate(NAMES):
            if nm.endswith('grip'):
                continue
            lag = r['cmd_target'][i] - r['raw_target'][i]
            if worst is None or abs(lag) > abs(worst[1]):
                worst = (nm, lag, r['raw_target'][i], r['cmd_target'][i])
        import math
        gi = NAMES.index(name)
        print('    %-5d %-8s %9.4f %9.4f %8.2f %8.2f'
              % (step, worst[0], worst[2], worst[3],
                 math.degrees(worst[1]), r['cmd_target'][gi]))

    # 汇总：全过程中的最大 lag（只看臂）
    arm_idx = [i for i, n in enumerate(NAMES) if 'arm' in n]
    import math
    max_lag = 0.0
    max_lag_step = -1
    for r in rows:
        for i in arm_idx:
            lag = abs(r['cmd_target'][i] - r['raw_target'][i])
            if lag > max_lag:
                max_lag, max_lag_step = lag, r['step']
    print()
    print('  ── 全程滤波/限速滞后（臂关节）─────────────────────────────────')
    print('    最大 |cmd - raw| = %.2f°（step %d）' % (math.degrees(max_lag), max_lag_step))
    if math.degrees(max_lag) < 0.5:
        print('    → ✅ 我们的滤波/限速几乎没滞后：**放太高是模型自己的决定**')
    elif math.degrees(max_lag) < 2.0:
        print('    → 🟡 有轻微滞后，但不是主因')
    else:
        print('    → 🔴 滞后明显：低通/限速把手臂拖在了高处，是执行层的问题')
        print('       对策：调小 LP_TAU（0.10 → 0.05）或放宽 ARM_SPEED')

    # ── 冻结统计 ─────────────────────────────────────────────────────────
    frozen = [r for r in rows if r.get('frozen')]
    print()
    print('  ── 释放冻结 ─────────────────────────────────────────────────────')
    if not frozen:
        print('    未触发（RELEASE_FREEZE=0 或没检测到张开）')
    else:
        steps = [r['step'] for r in frozen]
        print('    冻结 %d 步，范围 step %d ~ %d' % (len(steps), min(steps), max(steps)))

    # ── 夹爪限速统计 ─────────────────────────────────────────────────────
    gc = sum(1 for r in rows if r.get('clip_grip', 0) > 1e-6)
    ac = sum(1 for r in rows if r.get('clip_arm', 0) > 1e-6)
    print()
    print('  ── 限速统计 ─────────────────────────────────────────────────────')
    print('    夹爪被限速 %d 步   臂被限速 %d 步' % (gc, ac))
    if gc == 0:
        print('    ⚠️ 夹爪 0 步 = 模型从头到尾没命令夹爪动 → 这一轮它没在抓东西')
    return 0


if __name__ == '__main__':
    sys.exit(main())
