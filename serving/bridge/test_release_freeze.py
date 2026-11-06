"""patch 26 释放冻结逻辑的单元测试。

这些测试存在的理由：该逻辑第一版直接写 ``last["t"][GRIPPER_JOINTS]``，
而 ``GRIPPER_JOINTS`` 是 tuple ``(6, 13)`` —— numpy 把 tuple 当多维索引，
真机上直接 IndexError，跑了一次才暴露。抽成纯函数后这里可以本地拦住。

运行：
    python -m pytest serving/bridge/test_release_freeze.py -q
或：
    python serving/bridge/test_release_freeze.py
"""
from __future__ import annotations

import numpy as np

from vla_client import ARM_JOINTS, GRIPPER_JOINTS, NUM_JOINTS, release_freeze_update


def _state(arms=0.0, grips=1.0):
    """构造 14 维状态：[L0..L5, L_grip, R0..R5, R_grip]。"""
    s = np.full(NUM_JOINTS, float(arms), dtype=np.float64)
    for j in GRIPPER_JOINTS:
        s[j] = float(grips)
    return s


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
    target = _state(grips=0.5)
    release_freeze_update(_state(), target, False, 10, -1, 9)


def test_disabled_by_default():
    """freeze_steps=0 → 不冻结、不改 target。"""
    prev = _state(arms=0.1, grips=1.0)
    target = _state(arms=0.5, grips=0.0)
    before = target.copy()
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 0)
    assert fu == -1
    assert frozen is False
    assert trig is False
    assert np.array_equal(target, before)


def test_trigger_on_opening_and_pin_arm():
    """夹爪开始张开 → 触发冻结，且臂被钉在上一帧位置。"""
    prev = _state(arms=0.0, grips=1.0)
    target = _state(arms=0.3, grips=0.9)     # 张开（1.0 → 0.9）+ 臂想动
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 9)
    assert opening is True
    assert trig is True
    assert fu == 109
    assert frozen is True
    # 臂必须等于 prev 的臂（钉住），夹爪保持不变
    assert np.allclose(target[ARM_JOINTS], prev[ARM_JOINTS])
    assert target[list(GRIPPER_JOINTS)][0] == 0.9


def test_no_retrigger_while_still_opening():
    """持续张开时不应把冻结反复续期（边沿检测）。"""
    prev = _state(arms=0.0, grips=1.0)
    target = _state(arms=0.3, grips=0.9)
    fu1, _, _, trig1 = release_freeze_update(prev, target, False, 100, -1, 9)
    assert trig1 is True and fu1 == 109
    # 下一步仍在张开，且 opening_prev=True
    prev2 = _state(arms=0.0, grips=0.9)
    target2 = _state(arms=0.3, grips=0.8)
    fu2, frozen2, opening2, trig2 = release_freeze_update(prev2, target2, True, 101, fu1, 9)
    assert opening2 is True
    assert trig2 is False, "边沿检测失效：不应该重复触发"
    assert fu2 == 109, "冻结截止步不应被续期"
    assert frozen2 is True


def test_freeze_expires():
    """冻结到期后不再钉住手臂。"""
    prev = _state(arms=0.0, grips=0.9)
    target = _state(arms=0.3, grips=0.8)
    fu, frozen, _, _ = release_freeze_update(prev, target, True, 110, 109, 9)
    assert frozen is False
    assert np.allclose(target[ARM_JOINTS], 0.3), "冻结过期后不应再改臂目标"


def test_closing_does_not_trigger():
    """闭合（0.0 → 1.0，值变大）不触发冻结。"""
    prev = _state(arms=0.0, grips=0.0)
    target = _state(arms=0.0, grips=0.5)
    fu, frozen, opening, trig = release_freeze_update(prev, target, False, 100, -1, 9)
    assert opening is False
    assert trig is False
    assert frozen is False
    assert fu == -1


def test_tiny_gripper_noise_ignored():
    """1e-4 级别的抖动不应触发（阈值 1e-3）。"""
    prev = _state(grips=1.0)
    target = _state(grips=1.0 - 1e-4)
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
