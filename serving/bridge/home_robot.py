#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YAM 双臂缓慢移回 home 姿态，并可选择随后缓慢张开夹爪释放物体。

为什么需要这个
--------------
真机试验结束后，机械臂常常停在"半空夹着东西"的姿态，守护进程处于 safe_hold
（抱死保持）。直接松开会让东西从空中掉落；直接下发 home 又可能一步跳过去。
本工具用**两段式 + 速度限幅**安全复位：

  阶段 1：双臂按 ``--speed``（默认 0.15 rad/s ≈ 8.6°/s）线性移向 home 角度，
          **夹爪保持不动** —— 途中不会掉东西。
  阶段 2：夹爪按 ``--gripper-speed``（默认 0.7/s，与 yam_motion 一致）缓慢张开，
          在低位释放物体。
  最后：确认所有关节都在 home 容差内，释放安全会话（自动 SAFE HOLD）。

安全保证
--------
* 走 ``ZMQClientRobot`` + ``SafetyClient`` + ``SupervisedMotionSession``
  （与 vla_client 同一条路径）→ 有 400ms 看门狗 + 软限位；
* 每步都用 ``rate_limit_target`` 限速，绝不跳变；
* 任一关节"反向变大"（说明被卡住/撞到）→ 立即停并 safe hold；
* 超过 ``--max-seconds`` 或收敛即退出，退出必回 SAFE HOLD。

用法
----
    # 只看计划，不动机械臂
    python home_robot.py --dry-run

    # 复位：先降手臂（夹爪不动），再张开夹爪释放
    python home_robot.py

    # 只降手臂，完全不碰夹爪（还想继续夹着）
    python home_robot.py --keep-gripper

    # 更慢/更快
    python home_robot.py --speed 0.10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vla_client as vc           # 复用 ZmqYamRobot / rate_limit_target / RATE


# yam_motion/live_configs/runtime.yaml 里的 reset.target（这是这台机器公认的 home）
HOME_TARGET = np.array([0, 0, 0, 0, 0, 0, 1.0,
                        0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)


def read_state(robot) -> np.ndarray:
    return np.asarray(robot.get_observations()["joint_positions"],
                      dtype=np.float32).reshape(-1)


def converge(robot, target: np.ndarray, speed: float, gripper_speed: float,
             tol_rad: float, max_seconds: float, label: str,
             move_gripper: bool) -> tuple[bool, np.ndarray]:
    """以限速方式把关节推向 target。返回 (是否收敛, 最终命令)。"""
    period = 1.0 / vc.RATE
    cur = read_state(robot).copy()
    t0 = time.monotonic()
    step = 0
    prev_err = None
    arm_idx = np.array(vc.ARM_JOINTS)

    print(f"  [{label}] 开始，初始最大误差 "
          f"{np.rad2deg(np.abs(cur[arm_idx] - target[arm_idx]).max()):.1f}°")
    while True:
        if time.monotonic() - t0 > max_seconds:
            print(f"  [{label}] ⏱ 超过 {max_seconds:.0f}s，停止")
            return False, cur

        robot.check()
        err = np.abs(cur[arm_idx] - target[arm_idx]).max()
        if err < tol_rad and (not move_gripper or
                              np.abs(cur[vc.GRIPPER_JOINTS] - target[vc.GRIPPER_JOINTS]).max()
                              < 0.02):
            print(f"  [{label}] ✅ 收敛（最大误差 {np.rad2deg(err):.2f}°，{step} 步 / "
                  f"{time.monotonic()-t0:.1f}s）")
            return True, cur

        # 卡住检测：误差比上一步还大 10% 以上 → 被挡住/撞到
        if prev_err is not None and err > prev_err * 1.10 + 1e-4:
            print(f"  [{label}] ❌ 误差从 {np.rad2deg(prev_err):.2f}° 增大到 "
                  f"{np.rad2deg(err):.2f}° —— 疑似被挡住/撞到，立即停止")
            return False, cur
        prev_err = err

        cmd, _, _ = vc.rate_limit_target(target, cur, period,
                                         gripper_rate=gripper_speed if move_gripper else 0.0,
                                         arm_speed=speed)
        robot.command_joint_state(cmd)
        cur = cmd
        step += 1
        if step % 30 == 0:
            print(f"  [{label}] {step:4d}步 {time.monotonic()-t0:5.1f}s  "
                  f"剩余最大误差 {np.rad2deg(err):6.2f}°")
        time.sleep(period)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=0.15,
                    help="臂关节速度上限 rad/s（默认 0.15 ≈ 8.6°/s，很慢很安全）")
    ap.add_argument("--gripper-speed", type=float, default=vc.GRIPPER_RATE_LIMIT_S,
                    help="夹爪张开速度 /s（默认 0.7，与 yam_motion 一致）")
    ap.add_argument("--tolerance-deg", type=float, default=1.5,
                    help="臂关节收敛容差（度）")
    ap.add_argument("--max-seconds", type=float, default=90,
                    help="单阶段超时（秒）")
    ap.add_argument("--keep-gripper", action="store_true",
                    help="完全不碰夹爪（还想继续夹着物体时用）")
    ap.add_argument("--target", default=None,
                    help="自定义目标（14 个逗号分隔数字）；默认 runtime.yaml 的 reset target")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不动机器人")
    args = ap.parse_args()

    target = (np.array([float(x) for x in args.target.split(",")], dtype=np.float32)
              if args.target else HOME_TARGET.copy())
    if target.shape[0] != vc.NUM_JOINTS:
        print(f"  ❌ --target 需要 {vc.NUM_JOINTS} 个数，给了 {target.shape[0]}")
        return 2
    tol = np.deg2rad(args.tolerance_deg)

    if args.dry_run:
        robot = vc.MockRobot()
        cur = read_state(robot)
    else:
        robot = vc.ZmqYamRobot(command_timing="legacy_30hz")
        st = robot.status()
        print(f"  守护进程: state={st.get('state')!r} "
              f"hold_confirmed={st.get('hold_confirmed')} "
              f"stale_clients={st.get('stale_clients')!r}")
        cur = read_state(robot)

    arm_idx = np.array(vc.ARM_JOINTS)
    print(f"\n  当前姿态 → 目标姿态（度）")
    print(f"  {'dim':<10}{'当前':>10}{'目标':>10}{'差值':>10}")
    for j in range(vc.NUM_JOINTS):
        if j in vc.GRIPPER_JOINTS:      # 夹爪是 0~1 归一化行程，不是弧度
            print(f"  {vc.DIM_NAMES_FOR_LOG[j]:<10}{cur[j]:>10.3f}"
                  f"{target[j]:>10.3f}{target[j]-cur[j]:>10.3f}   (0~1)")
        else:
            print(f"  {vc.DIM_NAMES_FOR_LOG[j]:<10}"
                  f"{np.rad2deg(cur[j]):>10.1f}{np.rad2deg(target[j]):>10.1f}"
                  f"{np.rad2deg(target[j]-cur[j]):>10.1f}   (°)")
    need = np.rad2deg(np.abs(cur[arm_idx] - target[arm_idx]).max())
    print(f"\n  臂关节最大需移动 {need:.1f}°；按 {np.rad2deg(args.speed):.1f}°/s "
          f"预计 {need / max(np.rad2deg(args.speed), 1e-6):.1f} 秒")
    if args.keep_gripper:
        print("  夹爪：保持不动（--keep-gripper）")
    else:
        print(f"  夹爪：阶段 2 会张开到 {target[list(vc.GRIPPER_JOINTS)]} "
              f"（速度 {args.gripper_speed}/s → 全行程 {1/args.gripper_speed:.2f}s）")

    if args.dry_run:
        print("\n  [dry-run] 用的是 MockRobot，所以「当前」全是 0 —— "
              "这里只验证计划计算，不动机器人")
        return 0

    ok_any = False
    try:
        with robot.motion_session():
            print("\n  ✅ 安全会话已激活")
            # 阶段 1：双臂回 home，夹爪不动
            phase1_target = target.copy()
            if not args.keep_gripper:
                phase1_target[list(vc.GRIPPER_JOINTS)] = cur[list(vc.GRIPPER_JOINTS)]
            ok1, cur = converge(robot, phase1_target, args.speed,
                                args.gripper_speed, tol, args.max_seconds,
                                "阶段1 手臂回位", move_gripper=False)
            ok_any = ok1
            # 阶段 2：张开夹爪
            if ok1 and not args.keep_gripper:
                print("\n  ⚠️  即将张开夹爪释放物体（如果还夹着东西，请确认下方是台面）")
                time.sleep(1.0)
                ok2, cur = converge(robot, target, args.speed, args.gripper_speed,
                                    tol, args.max_seconds, "阶段2 张开夹爪",
                                    move_gripper=True)
                ok_any = ok_any and ok2
    finally:
        print("\n  收尾：释放会话（自动 SAFE HOLD）+ 关闭连接")
        try:
            robot.close(disable_motors=False)
        except Exception as exc:                        # noqa: BLE001
            print(f"  ⚠️ close 失败：{exc}")
        try:
            st = robot.status()
            print(f"  最终状态: state={st.get('state')!r} "
                  f"hold_confirmed={st.get('hold_confirmed')}")
        except Exception:                               # noqa: BLE001
            pass

    print("\n  " + ("✅ 复位完成" if ok_any else "⚠️ 未完全收敛 —— 再看上面日志"))
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
