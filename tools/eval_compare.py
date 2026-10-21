#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""对比两次 open-loop 评测的 metrics.json，打印一张对照表并给出结论。

为什么需要它：
    单独看 ar_action_l1=0.05 没有意义 —— 不知道 0.05 是好还是差。
    必须和基线（官方基座）对比：A 明显更低才算微调有效。

用法：
    python3 tools/eval_compare.py A/metrics.json B/metrics.json
    python3 tools/eval_compare.py --label-a 微调 --label-b 基座 A.json B.json

退出码：0 = A 不差于 B；1 = A 明显差于 B（提示不要上真机）；2 = 参数/文件问题
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 指标方向：True = 越小越好（误差类），False = 越大越好（准确率类）
LOWER_BETTER_HINTS = ("_l1", "loss", "_mse", "_rmse", "_err")
HIGHER_BETTER_HINTS = ("_acc", "accuracy", "_success", "_score")

# A 比 B 差多少就判定为「明显更差」
BAD_REL = 0.05      # 相对差 > 5%
GOOD_REL = -0.10    # 相对改善 > 10%


def _direction(key: str) -> bool:
    k = key.lower()
    if any(h in k for h in LOWER_BETTER_HINTS):
        return True
    if any(h in k for h in HIGHER_BETTER_HINTS):
        return False
    return True     # 未知默认按误差处理


def _load(path: str) -> dict:
    # utf-8-sig：兼容带/不带 BOM 的 JSON（Windows 上手工生成的文件常带 BOM）
    with open(path, encoding="utf-8-sig") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path} 不是 dict：{type(d)}")
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A(ours)")
    ap.add_argument("--label-b", default="B(base)")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not os.path.isfile(p):
            print(f"[compare] 找不到 {p}", file=sys.stderr)
            return 2
    try:
        A = _load(args.a)
        B = _load(args.b)
    except Exception as e:  # noqa: BLE001
        print(f"[compare] 读取失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    keys = sorted(set(A) | set(B))
    if not keys:
        print("[compare] 两个 metrics.json 都是空的", file=sys.stderr)
        return 2

    print("=" * 88)
    print(f"open-loop 评测对比   {args.label_a} vs {args.label_b}")
    print("=" * 88)
    print(f"  {'指标':<38} {'A':>12} {'B':>12} {'相对变化':>12}  判定")
    print("-" * 88)

    verdicts = []
    for k in keys:
        a = A.get(k)
        b = B.get(k)
        if a is None or b is None:
            print(f"  {k:<38} {('%.6f' % a) if a is not None else '—':>12} "
                  f"{('%.6f' % b) if b is not None else '—':>12} {'—':>12}  仅一侧有")
            continue
        lower_better = _direction(k)
        if abs(b) < 1e-12:
            rel = 0.0
            rel_s = "n/a"
        else:
            rel = (a - b) / abs(b)
            rel_s = f"{rel * 100:+.1f}%"
        # 相对变化换算成「越大越好」的视角：improve > 0 表示 A 更好
        improve = (-rel) if lower_better else rel
        if improve >= -GOOD_REL:
            tag = "A 更好 ✅"
        elif improve <= -BAD_REL:
            tag = "A 更差 ❌"
        else:
            tag = "持平"
        print(f"  {k:<38} {a:>12.6f} {b:>12.6f} {rel_s:>12}  {tag}")
        if k.endswith("_l1") or "_l1" in k:
            verdicts.append((k, improve))

    print("-" * 88)
    if verdicts:
        avg = sum(v for _, v in verdicts) / len(verdicts)
        print(f"  L1 类指标平均改善: {avg * 100:+.1f}%  "
              f"（正值 = {args.label_a} 更好）")
        print()
        if avg > 0.10:
            print(f"  结论: ✅ 微调【明显有效】—— {args.label_a} 的 L1 比 {args.label_b} "
                  f"低 {avg * 100:.1f}%，可以进入推理/真机阶段。")
            rc = 0
        elif avg > 0.02:
            print(f"  结论: 🟡 微调【小幅有效】—— 改善 {avg * 100:.1f}%。"
                  f"建议再跑更多 episode 确认，或续训 2 epoch。")
            rc = 0
        elif avg > -0.05:
            print(f"  结论: ⚠️ 微调【几乎无改善】—— 相对变化 {avg * 100:+.1f}%。"
                  f"考虑续训（降 LR、2 epoch）。")
            rc = 0
        else:
            print(f"  结论: ❌ {args.label_a} 明显【差于】{args.label_b}"
                  f"（{avg * 100:+.1f}%）。不要上真机，先排查 codec 配套 / 相对动作还原。")
            rc = 1
    else:
        rc = 0

    print()
    print("  提示：L1 只是平均值，会掩盖病态。务必打开 episodes/ 下的 GT vs 预测图，")
    print("        逐关节检查【量级、符号、是否某关节不动】。")
    print("=" * 88)
    return rc


if __name__ == "__main__":
    sys.exit(main())
