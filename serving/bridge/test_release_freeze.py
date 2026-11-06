"""patch 26 释放冻结逻辑 + 夹爪方向限速的单元测试。

**夹爪真实语义：1.0 = 张开，0.0 = 闭合。**
依据：
  - home_robot.py 复位流程打印「阶段 2 会张开到 [1. 1.]」
  - 真机 CSV 实测「0.998 → 0.030」是闭合过程，analyze_bridge_csv.py 也把
    >0.5→<=0.5 判定为「闭合事件」

这组测试是为了锁死两个真机上暴露过的错：
  1. GRIPPER_JOINTS 是 tuple，直接 ndarray[tuple] 会 IndexError
  2. 张开/闭合的方向判断写反，导致闭合提速 4.3 倍直接把物块夹飞/夹不住

运行：
    python -m pytest serving/bridge/test_release_freeze.py -q
或：
    python serving/bridge/test_release_freeze.py
"""
from __future__ import annotations

import numpy as np

import vla_client as V
from vla_client import ARM_JOINTS, GRIPPER_JOINTS, NUM_JOINTS, release_freeze_update

GJ = list(GRIPPER_JOINTS)


def _state(arms=0.0, grips=1.0):
    """构造 14 维状态：[L0..L5, L_grip, R0..R5, R_grip]。"""
    s = np.full(NUM_JOINTS, float(arms), dtype=np.float64)
    for j in GJ:
        s[j] = float(grips)
    return s


# ── 回归测试 1：tuple 索引 ─────────────────────────────────────────────────
def test_gripper_joints_is_tuple_regression():
    """回归：GRIPPER_JOINTS 是 tuple，直接用 ndarray[tuple] 会 IndexError。"""
    arr = _state()
    try:
        arr[GRIPPER_JOINTS]
    except IndexError:
        pass
    else:
        raise AssertionError("ndarray[GRIPPER_JOINTS] 居然没报错？tuple 索引行为变了")
    # 纯函数内部必须用 list()，所以不能抛
    release_freeze_update(_state(), _state(grips=0.5), False, 10, -1, 9)


# ── 回归测试 2：方向限速（1.0=张开 / 0.0=闭合）────────────────────────────
def test_rate_limit_opening_is_fast():
    """张开（值变大 0→1）应走 open_rate：3.0/s → 每步 0.1。"""
    prev = _state(grips=0.0)
    target = _state(grips=1.0)
    out, _, _ = V.rate_limit_target(target, prev, 1 / 30,
                                    gripper_rate=0.7, arm_speed=1.0,
                                    gripper_open_rate=3.0)
    assert abs(out[GJ[0]] - 3.0 / 30) < 1e-6, out[GJ[0]]


def test_rate_limit_closing_is_slow():
    """闭合（值变小 1→0）必须仍走 close_rate：0.7/s → 每步 0.02333。"""
    prev = _state(grips=1.0)
    target = _state(grips=0.0)
    out, _, _ = V.rate_limit_target(target, prev, 1 / 30,
                                    gripper_rate=0.7, arm_speed=1.0,
                                    gripper_open_rate=3.0)
    assert abs(out[GJ[0]] - (1.0 - 0.7 / 30)) < 1e-6, out[GJ[0]]


def test_rate_limit_open_rate_disabled_falls_back():
    """open_rate<=0 → 张开也用 close_rate（向后兼容）。"""
    prev = _state(grips=0.0)
    target = _state(grips=1.0)
    out, _, _ = V.rate_limit_target(target, prev, 1 / 30,
                                    gripper_rate=0.7, arm_speed=1.0)
    assert abs(out[GJ[0]] - 0.7 / 30) < 1e-6, out[GJ[0]]


def test_rate_limit_arm_unaffected():
    """臂关节仍按 arm_speed 限速。"""
    prev = np.zeros(NUM_JOINTS)
    target = prev.copy()
    target[3] = 0.5
    out, _, clip_arm = V.rate_limit_target(target, prev, 1 / 30,
                                           gripper_rate=0.7, arm_speed=1.0,
                                           gripper_open_rate=3.0)
    assert abs(out[3] - 1.0 / 30) < 1e-6
    assert clip_arm > 0


# ── 冻结逻辑 ──────────────────────────────────────────────────────────────
def test_disabled_by_default():
    """freeze_steps=0 → 不冻结、不改 target。"""
    prev = _state(arms=0.1, grips=0.0)
    target = _state(arms=0.5, grips=1.0)     # 张开（0→1）
    before = target.copy()
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 0)
    assert (fu, frozen, trig) == (-1, False, False)
    assert np.array_equal(target, before)


def test_trigger_on_opening_and_pin_arm():
    """夹爪开始张开（值变大）→ 触发冻结，且臂被钉在上一帧位置。"""
    prev = _state(arms=0.0, grips=0.9)
    target = _state(arms=0.3, grips=1.0)     # 张开 + 臂想动
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 9)
    assert opening is True
    assert trig is True
    assert fu == 109
    assert frozen is True
    assert np.allclose(target[ARM_JOINTS], prev[ARM_JOINTS]), "臂应被钉住"
    assert target[GJ[0]] == 1.0, "夹爪目标本身不应被改"


def test_no_retrigger_while_still_opening():
    """持续张开时不应把冻结反复续期（边沿检测）。"""
    prev = _state(arms=0.0, grips=0.9)
    target = _state(arms=0.3, grips=1.0)
    fu1, _, _, trig1 = release_freeze_update(prev, target, False, 100, -1, 9)
    assert trig1 is True and fu1 == 109
    prev2 = _state(arms=0.0, grips=0.95)
    target2 = _state(arms=0.3, grips=1.0)
    fu2, frozen2, opening2, trig2 = release_freeze_update(prev2, target2, True, 101, fu1, 9)
    assert opening2 is True
    assert trig2 is False, "边沿检测失效：不应重复触发"
    assert fu2 == 109, "冻结截止步不应被续期"
    assert frozen2 is True


def test_freeze_expires():
    """冻结到期后不再钉住手臂。"""
    prev = _state(arms=0.0, grips=0.95)
    target = _state(arms=0.3, grips=1.0)
    fu, frozen, _, _ = release_freeze_update(prev, target, True, 110, 109, 9)
    assert frozen is False
    assert np.allclose(target[ARM_JOINTS], 0.3), "冻结过期后不应再改臂目标"


def test_closing_does_not_trigger():
    """闭合（值变小 1→0）不触发冻结。"""
    prev = _state(arms=0.0, grips=1.0)
    target = _state(arms=0.0, grips=0.5)
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 9)
    assert opening is False
    assert trig is False
    assert frozen is False
    assert fu == -1


def test_tiny_gripper_noise_ignored():
    """1e-4 级别的抖动不应触发（阈值 1e-3）。"""
    prev = _state(grips=1.0)
    target = _state(grips=1.0 + 1e-4)
    _, _, opening, trig = release_freeze_update(prev, target, False, 100, -1, 9)
    assert opening is False and trig is False


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print('  ✓ %s' % name)
        except Exception as e:
            bad += 1
            print('  ✗ %s: %s: %s' % (name, type(e).__name__, e))
    print()
    print('  %d 个测试，失败 %d 个' % (len(tests), bad))
    raise SystemExit(0 if bad == 0 else 1)
