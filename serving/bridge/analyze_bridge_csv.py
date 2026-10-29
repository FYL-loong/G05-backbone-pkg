#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 bridge 的 --log-csv 输出，定位"换段跳变"到底有多大、发生在哪。

bridge 每步写一行 JSON（不是真 CSV，方便直接用）：
  step, t, sent_obs, prev[14], raw_target[14], cmd_target[14],
  d_raw[14], d_cmd[14], clip_grip, clip_arm, need_obs_next, lat_ms

用法：
    python analyze_bridge_csv.py run.csv
    python analyze_bridge_csv.py run.csv --out report.txt
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

DIMS = ["L_arm1", "L_arm2", "L_arm3", "L_arm4", "L_arm5", "L_arm6", "L_grip",
        "R_arm1", "R_arm2", "R_arm3", "R_arm4", "R_arm5", "R_arm6", "R_grip"]
ARM = list(range(6)) + list(range(7, 13))     # 12 个臂关节
GRIP = [6, 13]
RATE = 30.0


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] 第 {i+1} 行不是 JSON，跳过", file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    out_lines: list[str] = []

    def p(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    rows = load(args.csv)
    if not rows:
        print("  ❌ 没有数据")
        return 1

    n = len(rows)
    t = np.array([r["t"] for r in rows])
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    sent = np.array([bool(r.get("sent_obs")) for r in rows])
    n_swap = int(sent.sum())
    lat = np.array([r.get("lat_ms", 0.0) for r in rows])
    d_raw = np.array([r["d_raw"] for r in rows])       # 模型想走的一步
    d_cmd = np.array([r["d_cmd"] for r in rows])       # 实际下发的一步
    clip_arm = np.array([r.get("clip_arm", 0.0) for r in rows])
    clip_grip = np.array([r.get("clip_grip", 0.0) for r in rows])
    cmd = np.array([r["cmd_target"] for r in rows])

    p("=" * 96)
    p(f"  bridge 运行诊断：{args.csv}")
    p("=" * 96)
    p(f"  步数        : {n}")
    p(f"  时长        : {dur:.2f} s（有效 {n/dur if dur else 0:.1f} Hz）")
    p(f"  换段次数    : {n_swap}（每 {n/max(n_swap,1):.0f} 步一次）")
    p(f"  推理延迟    : avg={lat.mean():.0f}ms p50={np.percentile(lat,50):.0f}ms "
      f"p99={np.percentile(lat,99):.0f}ms max={lat.max():.0f}ms")
    p(f"  夹爪被限速  : {int((clip_grip > 1e-6).sum())} 步")
    p(f"  臂被限速    : {int((clip_arm > 1e-6).sum())} 步")

    # ── 1. 每步跳变幅度（模型想走多大）────────────────────────────────────
    p("")
    p("─" * 96)
    p("  1) 每步臂关节位移分布（|d_raw| 逐维，单位：度）")
    p("─" * 96)
    arm_idx = np.array(ARM)
    deg = np.rad2deg(np.abs(d_raw[:, arm_idx]))
    p(f"  {'dim':<10}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}{'max@step':>10}")
    for k, j in enumerate(arm_idx):
        col = deg[:, k]
        p(f"  {DIMS[j]:<10}{np.percentile(col,50):>8.2f}{np.percentile(col,95):>8.2f}"
          f"{np.percentile(col,99):>8.2f}{col.max():>8.2f}{int(col.argmax()):>10}")

    # ── 2. 换段 vs 非换段 的对比（这是核心）──────────────────────────────
    p("")
    p("─" * 96)
    p("  2) ★ 换段首帧 vs 普通步 的跳变对比（度）—— 差别大就是换段跳变的证据")
    p("─" * 96)
    for label, mask in (("换段首帧", sent), ("普通步", ~sent)):
        if mask.sum() == 0:
            p(f"  {label:<10}  无样本")
            continue
        vals = np.rad2deg(np.abs(d_raw[mask][:, arm_idx]))
        p(f"  {label:<10} n={int(mask.sum()):>4}  "
          f"p50={np.percentile(vals,50):>6.2f}  p95={np.percentile(vals,95):>6.2f}  "
          f"max={vals.max():>7.2f}")
    m_swap = np.rad2deg(np.abs(d_raw[sent][:, arm_idx])) if sent.sum() else np.zeros(1)
    m_norm = np.rad2deg(np.abs(d_raw[~sent][:, arm_idx])) if (~sent).sum() else np.zeros(1)
    ratio = (np.percentile(m_swap, 95) / np.percentile(m_norm, 95)
             if np.percentile(m_norm, 95) > 1e-9 else float("inf"))
    p(f"  → 换段 p95 / 普通 p95 = {ratio:.2f}×", )
    if ratio > 2.5:
        p("     🔴 换段首帧明显更大 → 确认存在换段跳变，需要做 re-anchor/平滑")
    elif ratio > 1.5:
        p("     🟡 换段稍大，属可接受范围，但仍可平滑")
    else:
        p("     ✅ 换段与普通步相当 → 换段不是位移尖峰问题（可能只是方向偏）")

    # ── 3. 最大的 N 次跳变发生在哪 ─────────────────────────────────────────
    p("")
    p("─" * 96)
    p(f"  3) 臂关节位移最大的 {args.top} 步")
    p("─" * 96)
    mag = np.rad2deg(np.abs(d_raw[:, arm_idx]).max(axis=1))
    for k in np.argsort(mag)[::-1][: args.top]:
        r = rows[int(k)]
        worst = int(np.argmax(np.abs(d_raw[int(k), arm_idx])))
        p(f"  step {r['step']:>4} t={r['t']:>6.2f}s {'【换段】' if sent[int(k)] else '        '}"
          f" 最大跳变 {mag[int(k)]:>6.2f}° ({DIMS[arm_idx[worst]]})"
          f"  clip_arm={r.get('clip_arm',0):.4f}"
          f"  d_raw={np.round(np.rad2deg(d_raw[int(k), arm_idx]),1).tolist()}")

    # ── 4. 夹爪行为 ────────────────────────────────────────────────────────
    p("")
    p("─" * 96)
    p("  4) 夹爪轨迹（看有没有 0→1 的猛跳；限速后应每步 ≤ 0.7/30 = 0.0233）")
    p("─" * 96)
    for j in GRIP:
        col = cmd[:, j]
        dcol = np.abs(np.diff(col))
        p(f"  {DIMS[j]:<8} 范围 {col.min():.3f}~{col.max():.3f}  "
          f"最大单步变化 {dcol.max() if dcol.size else 0:.4f}  "
          f"（限速阈值 {0.7/RATE:.4f}）"
          + ("  🔴 超了！限速没生效" if dcol.size and dcol.max() > 0.7 / RATE * 1.2 else
             "  ✅ 限速生效"))
        # 状态切换次数
        close_ev = int(((col[:-1] > 0.5) & (col[1:] <= 0.5)).sum()) if col.size > 1 else 0
        open_ev = int(((col[:-1] <= 0.5) & (col[1:] > 0.5)).sum()) if col.size > 1 else 0
        p(f"           闭合事件(>0.5→<=0.5) {close_ev} 次；张开事件 {open_ev} 次")

    # ── 5. 方向偏转（换段时目标方向是否突变）──────────────────────────────
    p("")
    p("─" * 96)
    p("  5) 换段处相邻两步的『运动方向』夹角（°）—— 大角度=方向突变=看起来乱动")
    p("─" * 96)
    angs = []
    for k in range(1, n):
        if not sent[k]:
            continue
        a = d_cmd[k - 1, arm_idx]
        b = d_cmd[k, arm_idx]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-4 or nb < 1e-4:
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
        angs.append(np.rad2deg(np.arccos(cos)))
    if angs:
        p(f"  换段处方向夹角：n={len(angs)}  p50={np.percentile(angs,50):.0f}°  "
          f"p95={np.percentile(angs,95):.0f}°  max={max(angs):.0f}°")
        p("     （>90° 表示方向反了半边；>135° 基本是反向）")
        if np.percentile(angs, 95) > 135:
            p("     🔴 换段处方向大幅反转 → 这是「乱动」的主因，必须做 re-anchor")
        elif np.percentile(angs, 95) > 90:
            p("     🟡 换段处有明显方向变化")
        else:
            p("     ✅ 换段方向连续")
    else:
        p("  无有效样本")

    p("")
    p("=" * 96)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"  报告已存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
