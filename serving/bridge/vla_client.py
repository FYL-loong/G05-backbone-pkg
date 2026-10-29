#!/usr/bin/env python3
"""Real-machine bridge: cameras + joint state → ``serve_policy`` (WS) → YAM bimanual.

Origin
------
Written by 张恩同 (2026-10-26) as the client half of the async-serving design.
This copy is his file with **8 reviewed patches** applied.  Upstream path on the
lab box: ``g05-yam-deployment/bridge/vla_client.py``.

PATCH TABLE (all verified against gello / i2rt / GalaxeaVLA sources + real runs)
-------------------------------------------------------------------------------
1. ``capture_images()`` was ``NotImplementedError`` → now reuses
   ``yam_motion.cameras.YamCameras`` (causal frames only, ``max_age_s`` guarded).
2. Added ``--mock-robot``.  ``YAMRobot.create_pair`` runs gripper calibration that
   **physically moves the arms**, so the "protocol smoke test" was NOT safe.
3. **Clock-deficit fix.**  Inference blocks ~0.5 s on the first request and on
   every chunk swap.  The original only did ``next_tick += period``, so after a
   block the loop fired cached steps back-to-back — compressing ~0.5 s of
   trajectory into tens of ms.  Now any request longer than one period
   re-baselines the clock: stalled time is **dropped, never replayed**.
4. ``ws.recv()`` had no timeout → wrapped in ``asyncio.wait_for``; timeout ⇒
   explicit safe hold.
5. ``websockets >= 14`` auto-reads ``HTTP_PROXY``/``HTTPS_PROXY``.  This box has a
   **dead** proxy at ``127.0.0.1:7897``, so a plain ``connect()`` to
   ``ws://127.0.0.1`` got hijacked (colleague's ``fake_policy_client.py`` fails
   exactly this way).  ``_ws_connect()`` passes ``proxy=None``.
6. **First-request timeout 180 s** (steady 10 s).  The first forward pass includes
   CUDA kernel autotune and takes ~12 s on this 4090; a flat 5 s timeout killed it.
7. **Non-zero exit codes** on safety aborts (2 unhealthy / 3 server error /
   4 timeout / 5 camera), so wrapper scripts can tell failure from success.
8. **``--robot-backend {zmq,direct,mock}``.**  ``zmq`` (default, real machine) goes
   through ziyu's ``yam_safety_daemon`` at :6001/:6002 via ``ZMQClientRobot`` +
   ``SafetyClient`` + ``SupervisedMotionSession`` — that is the only way to get
   the 400 ms watchdog and soft limits, and it does not fight over the CAN bus.
   ``direct`` keeps the original ``YAMRobot.create_pair`` path for debugging.

Data contract (verified on real runs — the four-group shape came back exactly)
-----------------------------------------------------------------------------
* **Action frame** = follower (i2rt) native frame (``applied_joint_state``).
* **Joint order/sign = identity**: 14-dim =
  ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
* **Action value = absolute** (server already did merger/normalizer/relative
  backward).  Do NOT add state back.
* **Server reply** (observed): ``{'left_arm': (6,), 'right_arm': (6,),
  'left_gripper': (1,), 'right_gripper': (1,)}`` — ``lower_body`` is dropped via
  ``_absent_keys`` and we hold the last commanded value for it.

Usage
-----
1) Read-only status probe (touches nothing, safe while someone else drives)::

    python vla_client.py --task x --check-robot

2) Safe protocol smoke test (no robot, no cameras)::

    python vla_client.py --task "Pick up the small square and place it in the groove" \\
        --fake-images --mock-robot --steps 96 --verbose

3) Real run through the safety daemon::

    python vla_client.py --task "Pick up the small square and place it in the groove" \\
        --robot-backend zmq --robot-port 6001 --safety-port 6002 \\
        --camera-config <deploy>/live_configs/robot.yaml \\
        --motion-src    <deploy>/code/motion/src \\
        --duration 20

Safety
------
Every exit path runs a safe hold then closes.  With ``--robot-backend zmq`` the
``SupervisedMotionSession`` context manager performs the hold and requires the
daemon to confirm it.  ``build_robot()``/``create_pair`` (direct mode) runs gripper
calibration, which **moves the arms**.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import sys
import time
import uuid
from contextlib import contextmanager

import msgpack
import numpy as np

logger = logging.getLogger("vla_client")

# ── tunables ─────────────────────────────────────────────────────────────────
# 首次推理含 CUDA kernel autotune + 显存池增长：实测在 4090 上要 ~12 s（稳态 ~0.5 s）。
# 所以首包单独给宽超时（patch 6）。2026-10-27 实测：统一 5 s 会把首次调用掐掉。
FIRST_RECV_TIMEOUT_S = 180.0
RECV_TIMEOUT_S = 10.0    # 稳态服务端 Recompute 实测 p50 ≈ 513 ms
JUMP_WARN_DEG = 5.0      # 一步内臂关节变化超过这个角度就告警
NUM_JOINTS = 14

# ★ patch 10/11：执行层限速（2026-10-29 真机症状复盘后加）
# 背景：真机上出现 (a) 第一次夹取把小方块"夹飞"，(b) 换段时乱动。
# 根因：桥接层把模型输出的**绝对目标原样下发**，而模型的夹爪输出可能在一步内
# 从 0 跳到 1 —— 30Hz 下就是 33 ms 走完全行程，夹爪"啪"一下弹合把物体弹飞。
# yam_motion 的标准配置（live_configs/runtime.yaml）是：
#     gripper_rate_limit: {opening_speed_s: 0.7, closing_speed_s: 0.7}
#     g05_joint_handoff:  {max_speed_rad_s: 0.5}
# 即夹爪 0→1 需要 1.43 s。我们照它来实现。
GRIPPER_RATE_LIMIT_S = 0.7      # 单位/秒（归一化夹爪行程），0→1 需 1.43 s
ARM_MAX_SPEED_RAD_S = 2.5       # 143°/s。实测正常步进 0.9~2.1 rad/s，
                                # 这个值只裁剪换段尖峰，不干扰正常跟随
GRIPPER_JOINTS = (6, 13)        # 左右夹爪在 14 维里的下标

# ★ patch 11：相机丢帧容忍
# 原来 max_age_s=0.1、且单帧取不到就直接 abort —— RealSense 偶发 100~300ms 卡顿
# 就会让整次运行提前结束。改成：先保持当前位置重试，持续超过阈值才真 abort。
CAMERA_MAX_AGE_S = 0.20         # 允许 6 帧（30Hz）的陈旧度
CAMERA_STALL_ABORT_S = 1.0      # 连续 1 s 拿不到可用帧才判定相机故障
CAMERA_SAMPLE_RETRY_S = 0.045   # 单次取帧内部重试窗口（>1 个帧周期 33ms）

# ★ patch 16：模型输出合理性看门狗（2026-10-29 事故后加）
# 事故：任务完成后模型输出崩坏，6 个右臂关节同一瞬间要求跳 40~62°，
#       30Hz 来回摆动造成剧烈震动，右侧电机控制线程保护性停机（daemon 进 fault）。
# 两道闸：
#   ① 偏差闸 —— 单关节目标离当前姿态 > OUTPUT_SANITY_DEG 视为异常，
#      连续 OUTPUT_SANITY_TRIP 步异常 → 立即安全停止。
#      （正常跟踪时该偏差实测 p95 < 3°，留 20 倍余量）
#   ② 振荡闸 —— 相邻两步的位移方向连续反向 OUTPUT_OSC_TRIP 次且幅度够大
#      → 判定"来回抖动" → 立即安全停止。
OUTPUT_SANITY_DEG = 60.0        # 单关节目标偏离当前姿态的上限（度）
OUTPUT_SANITY_TRIP = 3          # 连续多少步超限才算真异常
OUTPUT_OSC_DEG = 2.0            # 参与振荡判定的最小单步位移（度）
OUTPUT_OSC_TRIP = 8             # 连续多少次方向反转折算振荡

# ★ patch 20：命令低通滤波（"low gain"，治"回原位时震动"）
# 背景（2026-10-29）：任务完成后模型要回原位，但回程轨迹在换段频率
#   (~16 步 ≈ 0.53 s ≈ 1.9 Hz) 上反复反向 → 手臂震动而不是平滑移动。
#   限速器是逐关节投影，不会引入方向反转 → 抖动来自模型输出本身。
# 做法：一阶低通（EMA）  smoothed += alpha*(raw-smoothed),  alpha = dt/(tau+dt)
#   tau=0 关闭（保持原样）；建议 0.08~0.15 s。
#   注意：滤波在限速**之前**，所以物理速度/限位仍然严格成立。
CMD_LOWPASS_TAU_S = 0.0        # 0 = 关闭（默认不改变现有行为）

# ★ patch 21：稳定即停（不靠猜时长）
# 问题：训练 episode 约 16 秒，但 --duration 是硬切 —— 设短了切在任务中途，
#   设长了会跑进"任务完成后模型崩坏区"（实测崩坏在 t≈12.4s 之后，且会震动）。
# 做法：给一个宽松上限（如 40s），一旦模型原始输出连续 N 步几乎不变
#   （|Δraw| < 0.3°）就判定"已收敛/已到位"，正常退出（退出码 0）。
#   为防一开始就静止误判，必须先在 step>=60 且累计有过 >3° 的动作才启用。
STOP_WHEN_STILL_STEPS = 0       # 0 = 关闭；建议 60（=2 秒）
STILL_EPS_DEG = 0.3             # 判定"不动"的单步阈值（度）

# ★ patch 22：夹爪释放即停（任务感知的结束条件）
# 问题：--stop-when-still 是被动等模型安静 —— 实测任务约 18~20s 完成后，
#   模型还会"做一些动作" 13 秒才安静，于是 stop-when-still 到 33s 才触发。
#   这 13 秒的乱动既浪费又无意义（复位脚本会把手臂精确带回 home）。
# 做法：夹爪闭合(抓取) → 张开(放置) 的【释放事件】发生后再等 N 步就正常结束。
#   N 给一点余量让手臂先抬离物体，建议 45（1.5 秒）。
#   要求抓取阶段闭合至少 STOP_AFTER_RELEASE_MIN_CLOSED 步，避免初始张开被误判。
STOP_AFTER_RELEASE_STEPS = 0        # 0 = 关闭；建议 45
STOP_AFTER_RELEASE_MIN_CLOSED = 10  # 释放前至少闭合过多少步

# 退出码：0 = 正常跑完；>=2 = 被安全停机制中断（上层脚本据此判失败）
EXIT_OK = 0
EXIT_UNHEALTHY = 2
EXIT_SERVER_ERROR = 3
EXIT_TIMEOUT = 4
EXIT_CAMERA_STALE = 5
EXIT_PREFLIGHT = 6
EXIT_MODEL_INSANE = 7    # 模型输出异常（看门狗触发）

# ★ patch 18：会话接管前等待上一次会话真正失效的时间上限（秒）
# 守护进程用心跳超时判定 stale_clients，释放后约 1 秒才清空 session_id。
SESSION_READY_TIMEOUT_S = 12.0


# ─── msgpack + numpy codec (wire-compatible with g05.utils.websocket) ─────────

def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tobytes(),
                "dtype": obj.dtype.str, "shape": obj.shape}
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _bytes_to_str_keys(obj):
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj):
    obj = _bytes_to_str_keys(obj)
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]),
                          shape=tuple(obj["shape"]))
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


# ─── embodiment schema ───────────────────────────────────────────────────────

# 训练用的 10 个任务的**权威文本**（来自 HF Zenith6666/yam-lerobot 各
# meta/tasks.parquet 的 index）。模型的条件是这段文本，喂错就等于给错指令，
# 所以这里内置成表，真机用 --task-id TASK-YAM-0006 即可，免去手打出错。
TASK_TEXTS = {
    "TASK-YAM-0001": "Hit the ball with claws to score a goal",
    "TASK-YAM-0002": "Pick up the seal, dip it in the ink pad, and stamp.",
    "TASK-YAM-0004": "Grab the ham sausage to the foam box",
    "TASK-YAM-0006": "Pick up the small square and place it in the groove",
    "TASK-YAM-0007": "One gripper holds the bottle, and the inner side of the "
                     "other gripper rubs open the bottle cap.",
    "TASK-YAM-0008": "One gripper clamps the bottle, and the other gripper "
                     "rotates to open the cap.",
    "TASK-YAM-0010": "One gripper holds the gum bottle, and the other gripper "
                     "pops open the lid.",
    "TASK-YAM-0013": "Sort the blocks from left to right according to their size.",
    "TASK-YAM-0014": "Pick up the towel and wipe the whiteboard with it.",
    "TASK-YAM-0015": "Put items in the drawer.",
}
EMBODIMENT_TYPE = "galaxea_r1lite"
RATE = 30  # Hz — YAM 数据对齐在 30Hz 相机网格上

IMAGE_SPECS = {
    "head_rgb": (3, 480, 640),
    "left_wrist_rgb": (3, 480, 640),
    "right_wrist_rgb": (3, 480, 640),
}
CAMERA_NAME_MAP = {"main": "head_rgb", "left": "left_wrist_rgb",
                   "right": "right_wrist_rgb"}

# 14 维 = [左臂6, 左夹爪1, 右臂6, 右夹爪1]，恒等映射
STATE_SPLIT = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}
ACTION_KEYS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
ARM_JOINTS = np.r_[0:6, 7:13]   # 12 个臂关节下标（不含夹爪）

# 仅用于日志里的可读维度名
DIM_NAMES_FOR_LOG = [
    "L_arm1", "L_arm2", "L_arm3", "L_arm4", "L_arm5", "L_arm6", "L_grip",
    "R_arm1", "R_arm2", "R_arm3", "R_arm4", "R_arm5", "R_arm6", "R_grip",
]


# ─── pure joint mapping (unit-tested) ────────────────────────────────────────

def split_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """14 维关节向量 → 4 个 part（恒等映射，无重排/无符号翻转）。"""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS}-dim joint state, got {state.shape[0]}")
    return {key: state[lo:hi] for key, (lo, hi) in STATE_SPLIT.items()}


def action_to_target(action: dict, last_target: np.ndarray) -> np.ndarray:
    """服务端的 action dict 合并回 14 维绝对目标。

    缺失的 key（AR 头 ``_absent_keys`` 丢掉的）保持上一帧命令值，绝不填 0。
    """
    last_target = np.asarray(last_target, dtype=np.float32).reshape(-1)
    parts = []
    for key in ACTION_KEYS:
        lo, hi = STATE_SPLIT[key]
        val = action.get(key)
        if val is None:
            parts.append(last_target[lo:hi])
            continue
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        if arr.shape[0] != (hi - lo):
            raise ValueError(f"action['{key}'] has {arr.shape[0]} dims, expected {hi - lo}")
        parts.append(arr)
    return np.concatenate(parts)


def lowpass_target(raw: np.ndarray, smoothed, dt_s: float,
                   tau: float) -> tuple[np.ndarray, float]:
    """一阶低通（patch 20，"low gain"）。返回 ``(new_smoothed, alpha)``。

    用途：把模型输出在换段频率（~1.9 Hz）上的来回抖动滤成平滑运动。
    ``tau <= 0`` 时直接旁路（alpha=1，行为与不滤波完全一致）。
    ``smoothed is None`` 时用 raw 初始化（避免第 0 步从 0 猛冲）。
    """
    raw = np.asarray(raw, dtype=np.float64)
    if tau <= 0:
        return raw.astype(np.float32), 1.0
    if smoothed is None:
        return raw.astype(np.float32), 1.0
    if dt_s <= 0 or dt_s > 0.5:      # 异常间隔用标称周期
        dt_s = 1.0 / RATE
    a = float(np.clip(dt_s / (tau + dt_s), 0.0, 1.0))
    prev = np.asarray(smoothed, dtype=np.float64)
    return (prev + a * (raw - prev)).astype(np.float32), a

def rate_limit_target(target: np.ndarray, prev: np.ndarray, dt_s: float,
                      gripper_rate: float = GRIPPER_RATE_LIMIT_S,
                      arm_speed: float = ARM_MAX_SPEED_RAD_S):
    """按速度上限裁剪目标（patch 10）。返回 ``(new_target, clipped_grip, clipped_arm)``。

    为什么需要：模型输出的是**绝对目标**，夹爪维度可能一步之内 0→1。
    30 Hz 下那就是 33 ms 走完全行程 —— 夹爪"啪"一下弹合会把物体弹飞。
    ``yam_motion`` 的标准值（live_configs/runtime.yaml）是 0.7/s，即 0→1 需 1.43 s。

    臂用 2.5 rad/s：实测正常步进 0.9~2.1 rad/s，这个值只裁剪换段尖峰。
    ``0`` 表示该路限速关闭。
    """
    out = np.asarray(target, dtype=np.float64).copy()
    prev = np.asarray(prev, dtype=np.float64)
    if out.shape != prev.shape:
        raise ValueError(f"shape mismatch {out.shape} vs {prev.shape}")
    if dt_s <= 0 or dt_s > 0.5:      # 异常间隔不放大限速
        dt_s = 1.0 / RATE
    clip_grip = 0.0
    clip_arm = 0.0
    for j in range(out.shape[0]):
        cap = (gripper_rate if j in GRIPPER_JOINTS else arm_speed) * dt_s
        if cap <= 0:
            continue
        d = float(out[j] - prev[j])
        if abs(d) > cap:
            out[j] = prev[j] + np.sign(d) * cap
            if j in GRIPPER_JOINTS:
                clip_grip = max(clip_grip, abs(d) - cap)
            else:
                clip_arm = max(clip_arm, abs(d) - cap)
    return out.astype(np.float32), clip_grip, clip_arm

def build_obs(state: np.ndarray, images: dict[str, np.ndarray],
              task: str, frequency: float) -> dict:
    """构造服务端 ``_validate_obs`` 期望的观测字典。"""
    return {
        "embodiment_type": EMBODIMENT_TYPE,
        "images": images,
        "state": split_state(state),
        "task": task,
        # ★ YAM 必须是 30；不给服务端会回落到 15（SO-100）
        "frequency": float(frequency),
    }


# ─── cameras ─────────────────────────────────────────────────────────────────

class NullCameras:
    """合成零图。只用于协议冒烟。"""

    def start(self) -> None:
        pass

    def sample(self, _ts: float):
        return ({name: np.zeros(shape, dtype=np.uint8)
                 for name, shape in IMAGE_SPECS.items()}, 0.0)

    def close(self) -> None:
        pass


class RealCameras:
    """RealSense rig via ``yam_motion.cameras.YamCameras``（host 时钟对齐）。

    ``YamCameras.sample(t)`` 只返回 ``host_timestamp <= t`` 的因果帧，并拒绝超过
    ``max_age_s`` 的旧帧 —— 正是策略需要的对齐语义。
    """

    def __init__(self, robot_cfg_path: str, motion_src: str,
                 max_age_s: float = CAMERA_MAX_AGE_S) -> None:
        import yaml  # 只有真机相机路径才需要

        if motion_src not in sys.path:
            sys.path.insert(0, motion_src)
        from yam_motion.cameras import YamCameras

        cfg = yaml.safe_load(open(robot_cfg_path, encoding="utf-8"))
        for key in ("gello_root", "camera_config", "camera_names"):
            if key not in cfg:
                raise KeyError(f"{robot_cfg_path} 缺字段 {key!r}")

        self._cam = YamCameras({
            "gello_root": cfg["gello_root"],
            "camera_config": cfg["camera_config"],
            "camera_names": cfg["camera_names"],
            "max_frame_age_s": max_age_s,
        })
        self._names = list(cfg["camera_names"])

    def start(self) -> None:
        self._cam.start()

    def sample(self, ts: float, retry_s: float | None = None):
        """取一组因果帧。

        ★ patch 14：内层重试（合并张恩同的 5×5ms 重试，但修了两个问题）：
          ① 他重试用的是**同一个 ts** —— 而 ``YamCameras.sample(ts)`` 只接受
             ``host_timestamp <= ts`` 的帧，新到的帧时间戳更晚会被因果过滤拒掉，
             所以"那一刻还没帧"的情况用旧 ts 重试**永远拿不到**。
             这里每次重试用**实时推进的 ts**。
          ② 5×5ms=20ms 比一个帧周期（33ms @30Hz）还短，重试窗口不够。
             默认给 ``CAMERA_SAMPLE_RETRY_S``（45ms，>1 个帧周期）。
        失败（连续取不到）仍返回 None，由外层再做"原地保持 1 s"的宽容处理。
        """
        if retry_s is None:
            retry_s = CAMERA_SAMPLE_RETRY_S
        deadline = time.monotonic() + max(0.0, retry_s)
        out = None
        last_ts = float(ts)
        while True:
            # ★ 关键：每次重试都**推进**时间戳，否则新到帧的时间戳更晚会被
            #   ``host_timestamp <= ts`` 的因果过滤拒掉，重试永远无效。
            #   这里还要防 monotonic() 在本机分辨率较粗时两次返回同值（Windows
            #   实测出现过），所以低于上次就强制 +1ms 递增。
            now = max(ts, time.monotonic())
            if now <= last_ts:
                now = last_ts + 0.001
            last_ts = now
            out = self._cam.sample(now)
            if out is not None:
                break
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.004)
        frames, min_ts = out
        chw = {}
        for rig_name in ("main", "left", "right"):
            hwc = frames[rig_name]
            arr = np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))
            want = IMAGE_SPECS[CAMERA_NAME_MAP[rig_name]]
            if arr.shape != want:
                raise RuntimeError(f"camera {rig_name} frame {arr.shape} != {want}")
            chw[CAMERA_NAME_MAP[rig_name]] = arr.astype(np.uint8, copy=False)
        return chw, min_ts

    def close(self) -> None:
        self._cam.close()


# ─── robot backends ──────────────────────────────────────────────────────────
#
# 三个后端提供同一套接口，主循环不需要知道具体是哪种：
#   num_dofs() / get_joint_limits() / get_observations() / get_joint_state()
#   command_joint_state(t) / get_health_status() / enter_safe_hold(reason) / close()
# 另外两个可选接口：
#   check()          —— 每步调用，把后端内部故障（如心跳线程死掉）抛出来
#   motion_session() —— 上下文管理器，zmq 后端用它申请/释放安全会话

class MockRobot:
    """假机器人：不连 CAN、不做标定、不动任何东西。"""

    def num_dofs(self) -> int:
        return NUM_JOINTS

    def get_joint_limits(self) -> np.ndarray:
        lim = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        lim[:13, 0], lim[:13, 1] = -3.14, 3.14
        lim[13] = (0.0, 1.0)
        return lim

    def get_joint_state(self) -> np.ndarray:
        return np.zeros(NUM_JOINTS, dtype=np.float32)

    def get_observations(self) -> dict:
        z = np.zeros(NUM_JOINTS, dtype=np.float32)
        return {"joint_positions": z, "joint_velocities": z.copy(),
                "joint_efforts": z.copy()}

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        pass

    def get_health_status(self) -> dict:
        return {"ok": True, "reason": ""}

    def enter_safe_hold(self, reason: str | None = None):
        return np.zeros(NUM_JOINTS, dtype=np.float32)

    def close(self, disable_motors: bool = False) -> None:
        pass


class DirectYamRobot:
    """直连 CAN 后端（``YAMRobot.create_pair``）。

    ⚠️ 会做夹爪标定、会真动机器人；且**独占 CAN 总线**，与安全守护进程冲突。
    只在调试/守护进程未运行时使用，**没有 400ms 看门狗**。
    """

    def __init__(self, left_channel: str = "can_follower_l",
                 right_channel: str = "can_follower_r") -> None:
        from gello.robots.robot import BimanualRobot
        from gello.robots.yam import YAMRobot

        left, right = YAMRobot.create_pair(
            left_channel=left_channel, right_channel=right_channel)
        self.robot = BimanualRobot(left, right)
        logger.info("直连 CAN 就绪：num_dofs=%d", self.robot.num_dofs())

    def num_dofs(self) -> int:
        return int(self.robot.num_dofs())

    def get_joint_limits(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_limits(), dtype=np.float64)

    def get_joint_state(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_state(), dtype=np.float32)

    def get_observations(self) -> dict:
        return self.robot.get_observations()

    def command_joint_state(self, t: np.ndarray) -> None:
        self.robot.command_joint_state(np.asarray(t, dtype=np.float32))

    def get_health_status(self) -> dict:
        return self.robot.get_health_status()

    def enter_safe_hold(self, reason: str | None = None):
        return self.robot.enter_safe_hold(reason=reason)

    def close(self, disable_motors: bool = False) -> None:
        self.robot.close(disable_motors=disable_motors)


class ZmqYamRobot:
    """走安全守护进程（:6001 机器人 RPC / :6002 安全 RPC）。

    这是**唯一推荐的真机接入方式**：
      * 不抢 CAN —— 硬件由守护进程持有；
      * 有 400ms 看门狗 —— 我们停止心跳后守护进程自动 safe_hold；
      * 有软限位 —— 由守护进程端执行，所以 ``get_joint_limits()`` 返回 None；
      * 有安全会话 —— 必须先 ``begin_session`` 拿到独占控制权。

    ``count`` 常量与 gello 源码一致（gello/zmq_core/robot_node.py，
    gello/safety/rpc.py，gello/safety/motion_session.py）。
    """

    #: 守护进程认为"健康"的状态集合（其余如 fault/disarmed 立即中止）
    OK_STATES = ("active", "safe_hold")

    def __init__(self, host: str = "127.0.0.1", port: int = 6001,
                 safety_port: int = 6002, timeout_ms: int = 1000,
                 command_timing: str = "legacy_30hz", rate: float = RATE) -> None:
        from gello.safety.rpc import SafetyClient
        from gello.zmq_core.robot_node import ZMQClientRobot

        if command_timing not in ("timed", "legacy_30hz"):
            raise ValueError("command_timing 必须是 timed 或 legacy_30hz")
        self.robot = ZMQClientRobot(host=host, port=port, timeout_ms=timeout_ms)
        self.safety = SafetyClient(host=host, port=safety_port, timeout_ms=timeout_ms)
        self.command_timing = command_timing
        self.rate = float(rate)
        # legacy_30hz：守护进程自己按 30Hz 计时 → command_period_s 必须是 None
        # timed：我们显式告诉它周期
        self.robot.command_period_s = (
            None if command_timing == "legacy_30hz" else 1.0 / self.rate)
        self._session = None
        logger.info("ZMQ 后端就绪：robot=%s:%d safety=%s:%d timing=%s command_period_s=%s",
                    host, port, host, safety_port, command_timing,
                    self.robot.command_period_s)

    # ---- 会话管理 -----------------------------------------------------------
    def status(self) -> dict:
        return self.safety.status()

    def preflight(self) -> dict:
        """接管前自检：必须是 safe_hold 且没有活跃客户端。"""
        st = self.status()
        state = st.get("state")
        if state not in self.OK_STATES:
            raise RuntimeError(
                f"守护进程 state={state!r}，拒绝接管（可能 fault/disarmed）。"
                f"reason={st.get('reason')!r}")
        if state != "safe_hold":
            raise RuntimeError(
                f"机器人正被控制（state={state!r}），拒绝接管。"
                f"stale_clients={st.get('stale_clients')!r}")
        if not st.get("stale_clients"):
            raise RuntimeError(
                f"已有活跃控制会话，拒绝接管。status={st}")
        return st

    def wait_session_ready(self, timeout: float = SESSION_READY_TIMEOUT_S) -> dict:
        """★ patch 18：等守护进程"可以接管"再返回。

        问题（2026-10-29 实测）：刚释放会话后立刻再次申请，会得到
        ``RuntimeError: 已有活跃控制会话，拒绝接管``，因为守护进程判定
        ``stale_clients`` 需要约 1 秒（心跳超时），释放瞬间它还是空的。
        影响：连续跑两次 bridge 时第二次必失败。

        判据：``state == 'safe_hold'`` 且 ``stale_clients`` 非空。
        超时后不抛异常（交给 preflight 按原逻辑报错并给出清晰原因）。
        """
        deadline = time.monotonic() + max(0.0, timeout)
        waits = 0
        last = {}
        while True:
            try:
                last = self.status()
            except Exception as exc:  # noqa: BLE001
                logger.debug("读 status 失败：%s", exc)
                last = {}
            if last.get("state") == "safe_hold" and last.get("stale_clients"):
                if waits:
                    logger.info("会话可用（等了 %.1f s，stale_clients=%r）",
                                waits * 0.1, last.get("stale_clients"))
                return last
            if time.monotonic() >= deadline:
                if waits:
                    logger.warning("等了 %.1f s 仍不能接管：state=%r stale_clients=%r —— "
                                   "若刚跑过一次，请再等 1~2 秒", timeout,
                                   last.get("state"), last.get("stale_clients"))
                return last
            if waits == 0:
                logger.info("等待上一次会话释放（守护进程需约 1 秒判定心跳失效）...")
            waits += 1
            time.sleep(0.1)

    @contextmanager
    def motion_session(self):
        """申请独占受监督会话，退出时自动回到 safe_hold（并确认）。"""
        from gello.safety.motion_session import SupervisedMotionSession

        self.wait_session_ready()      # ★ patch 18：先等上一次会话真正失效
        self.preflight()
        sid = uuid.uuid4().hex
        logger.info("申请安全会话 session=%s operation=g05_vla_bridge ...", sid)
        try:
            with SupervisedMotionSession(self.safety, sid,
                                         operation="g05_vla_bridge") as sess:
                self._session = sess
                logger.info("✅ 安全会话已激活（含 parent/collector 两个 200ms 心跳线程）")
                try:
                    yield sess
                finally:
                    self._session = None
        except Exception as exc:  # noqa: BLE001
            # ★ patch 17：守护进程可能已经进 fault（例如电机链保护性停机），
            #   hold() 拿不到确认时会抛 HoldNotConfirmed。这时不要再往上抛成
            #   traceback 掩盖真正原因 —— 明确告诉操作员该查什么。
            name = type(exc).__name__
            logger.error("=" * 72)
            logger.error("安全会话退出异常：%s: %s", name, exc)
            try:
                st = self.status()
                hh = st.get("hardware_health") or {}
                logger.error("守护进程 state=%r reason=%r", st.get("state"), st.get("reason"))
                for side in ("left", "right"):
                    s = (hh.get("sides") or {}).get(side) or {}
                    logger.error("  %-5s ok=%s running=%s reason=%r", side,
                                 s.get("ok"), s.get("running"), s.get("reason"))
                if st.get("state") == "fault":
                    logger.error(">>> 机器人已进入 FAULT，需要【重启安全守护进程】才能恢复：")
                    logger.error(">>>   1) ps -p %s -o args=   # 查看当前启动命令",
                                 st.get("pid"))
                    logger.error(">>>   2) 停掉该进程后按原命令重启")
                    logger.error(">>>   3) 重启后跑: vla_client.py --task x --check-robot")
                    logger.error(">>>      必须看到 state='safe_hold' 且 hardware_health.ok=True")
            except Exception as exc2:  # noqa: BLE001
                logger.error("（读 status 也失败了：%s）", exc2)
            logger.error("=" * 72)
            raise
        logger.info("安全会话已释放 → 守护进程确认 SAFE HOLD")

    def check(self) -> None:
        """每步调用：心跳线程若发现 fault/disarmed 会在这里抛出。"""
        if self._session is not None:
            self._session.check()

    # ---- Robot 接口 ---------------------------------------------------------
    def num_dofs(self) -> int:
        return int(self.robot.num_dofs())

    def get_joint_limits(self):
        # 软限位由守护进程执行；返回 None 表示本地不做裁剪
        return None

    def get_joint_state(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_state(), dtype=np.float32)

    def get_observations(self) -> dict:
        return self.robot.get_observations()

    def command_joint_state(self, t: np.ndarray) -> None:
        self.robot.command_joint_state(np.asarray(t, dtype=np.float32))

    def get_health_status(self) -> dict:
        try:
            st = self.safety.status()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"safety.status() 失败: {exc}"}
        state = st.get("state")
        ok = state in self.OK_STATES
        return {"ok": ok,
                "reason": "" if ok else f"state={state!r} reason={st.get('reason')!r}",
                "raw": st}

    def enter_safe_hold(self, reason: str | None = None):
        try:
            return self.safety.safe_hold(reason or "g05_vla_bridge safe hold")
        except Exception as exc:  # noqa: BLE001
            logger.error("safety.safe_hold 失败：%s", exc)
            return None

    def close(self, disable_motors: bool = False) -> None:
        for name, fn in (("safety", self.safety.close), ("robot", self.robot.close)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.error("%s.close 失败：%s", name, exc)


def make_robot(backend: str, mock_robot: bool, robot_host: str, robot_port: int,
               safety_port: int, robot_timeout_ms: int, command_timing: str):
    """按 ``--robot-backend`` 构造后端；``--mock-robot`` 优先级最高。"""
    if mock_robot or backend == "mock":
        return MockRobot(), "mock"
    if backend == "zmq":
        return ZmqYamRobot(host=robot_host, port=robot_port,
                           safety_port=safety_port, timeout_ms=robot_timeout_ms,
                           command_timing=command_timing), "zmq"
    if backend == "direct":
        return DirectYamRobot(), "direct"
    raise ValueError(f"未知 backend: {backend}")


def print_robot_status(robot, backend: str) -> int:
    """只读探测：打印后端状态，不改任何东西。"""
    print(f"  backend      = {backend}")
    if backend == "zmq":
        import json
        st = robot.status()
        print(f"  safety.state = {st.get('state')!r}")
        print(f"  reason       = {st.get('reason')!r}")
        print(f"  clients      = {st.get('clients')!r}")
        print(f"  stale_clients= {st.get('stale_clients')!r}")
        print(f"  --- 完整 status ---\n{json.dumps(st, indent=2, default=str)}")
        try:
            n = robot.num_dofs()
            print(f"  num_dofs     = {n}")
            pos = robot.get_joint_state()
            print(f"  joint_state  = {np.round(pos, 4).tolist()}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 读机器人失败：{exc}")
        state = st.get("state")
        if state == "active":
            print("\n  ⚠️ 机器人正被控制（state=active）—— 现在**不能**接管，"
                  "会打断正在跑的客户端。")
            return EXIT_PREFLIGHT
        if state != "safe_hold":
            print(f"\n  ❌ state={state!r} 不是 safe_hold，不能接管。")
            return EXIT_PREFLIGHT
        print("\n  ✅ 处于 safe_hold，可以申请会话接管。")
        return EXIT_OK

    # mock / direct
    try:
        print(f"  num_dofs     = {robot.num_dofs()}")
        print(f"  joint_limits = {np.asarray(robot.get_joint_limits()).shape}")
        print(f"  joint_state  = {np.round(robot.get_joint_state(), 4).tolist()}")
        print(f"  health       = {robot.get_health_status()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ {exc}")
        return EXIT_PREFLIGHT
    return EXIT_OK


# ─── main loop ────────────────────────────────────────────────────────────────

def _ws_connect(uri: str):
    """``websockets.connect`` with localhost-safe options.

    websockets >= 14 auto-detects HTTP proxy env vars (``ALL_PROXY`` /
    ``HTTPS_PROXY``).  This box runs clash-verge and the configured proxy port is
    **dead**, so a plain ``websockets.connect("ws://127.0.0.1:...")`` fails with
    ``ConnectionRefusedError``.  Pass ``proxy=None`` where supported; older
    websockets reject the kwarg, so fall back without it.
    """
    import websockets

    kwargs = dict(max_size=None, ping_interval=30, ping_timeout=120,
                  proxy=None, compression=None)
    try:
        return websockets.connect(uri, **kwargs)
    except TypeError as exc:
        if "proxy" not in str(exc):
            raise
        kwargs.pop("proxy")
        logger.debug("websockets 不支持 proxy 参数，已回退")
        return websockets.connect(uri, **kwargs)


async def _loop(ws, robot, cameras, task, steps, duration, limits,
                jump_warn, max_step, latencies, t_run0, last,
                gripper_rate=GRIPPER_RATE_LIMIT_S, arm_speed=ARM_MAX_SPEED_RAD_S,
                cam_stall_abort=CAMERA_STALL_ABORT_S, csv_writer=None,
                freeze_logger=None,
                output_sanity_deg=OUTPUT_SANITY_DEG,
                output_sanity_trip=OUTPUT_SANITY_TRIP,
                output_osc_deg=OUTPUT_OSC_DEG,
                output_osc_trip=OUTPUT_OSC_TRIP,
                cmd_tau=CMD_LOWPASS_TAU_S,
                stop_after_release=STOP_AFTER_RELEASE_STEPS,
                stop_still_steps=STOP_WHEN_STILL_STEPS,
                still_eps_deg=STILL_EPS_DEG):
    """主循环。``last`` 是 ``{"t": ndarray}`` 的可变容器（上一帧命令的目标）。

    返回 (exit_code, step, behind, stats)。
    """
    period = 1.0 / RATE
    hello = unpackb(await ws.recv())
    if not isinstance(hello, dict):
        raise RuntimeError(f"server hello is {type(hello).__name__}, expected dict")
    if "error" in hello:
        raise RuntimeError(f"server rejected connection: {hello['error']}")
    action_steps = hello.get("action_steps")
    logger.info("connected  server action_steps=%s", action_steps)
    if action_steps is not None and int(action_steps) > 16:
        logger.warning("server action_steps=%s > 16: 前瞻窗口只有 "
                       "action_steps/RATE = %.0f ms", action_steps,
                       float(action_steps) * 1000.0 / RATE)
    logger.info("执行层限速：夹爪 %.2f/s（全行程 %.2f s）；臂 %.2f rad/s（%.0f°/s）%s",
                gripper_rate, (1.0 / gripper_rate) if gripper_rate > 0 else 0,
                arm_speed, np.rad2deg(arm_speed),
                "" if gripper_rate > 0 else "  ⚠️ 夹爪限速已关闭（容易把物体夹飞）")

    need_obs = True
    next_tick = time.monotonic()
    step = 0
    behind = 0
    exit_code = EXIT_OK
    last_cmd_t = time.monotonic()
    cam_misses_total = 0
    grip_clip_steps = 0
    arm_clip_steps = 0
    images_for_log = None     # 最近一次真实观测的画面（patch 15）
    sanity_bad = 0            # 连续超偏差步数（patch 16）
    prev_delta = None         # 上一步位移向量（patch 16 振荡检测）
    osc_count = 0             # 连续方向反转次数（patch 16）
    # patch 22：夹爪释放检测状态
    grip_closed_steps = {j: 0 for j in GRIPPER_JOINTS}   # 每个夹爪连续闭合步数
    release_step = None                                  # 释放事件发生的步号
    prev_raw = None           # 上一步的模型原始输出（patch 21）
    still_count = 0           # 连续"模型输出不动"步数（patch 21）
    moved_deg = 0.0           # 累计最大单步位移（用于启用稳定检测）
    sm = None                 # 低通滤波状态（patch 20）
    smoothed_steps = 0
    freeze_steps = 0          # 连续"几乎不动"的步数（patch 15 诊断）
    freeze_reports = 0

    while True:
        if steps > 0 and step >= steps:
            break
        if duration > 0 and (time.monotonic() - t_run0) >= duration:
            break

        # 后端内部故障（ZMQ：心跳线程发现 fault/disarmed）
        chk = getattr(robot, "check", None)
        if chk is not None:
            try:
                chk()
            except Exception as exc:  # noqa: BLE001
                logger.error("后端故障：%s", exc)
                robot.enter_safe_hold(reason=f"backend check failed: {exc}")
                exit_code = EXIT_UNHEALTHY
                break

        health = robot.get_health_status()
        if not health.get("ok", False):
            logger.error("robot unhealthy: %s", health.get("reason") or health)
            robot.enter_safe_hold(reason="unhealthy")
            exit_code = EXIT_UNHEALTHY
            break

        now = time.monotonic()
        state = np.asarray(
            robot.get_observations()["joint_positions"],
            dtype=np.float32).reshape(-1)
        if state.shape[0] != NUM_JOINTS:
            logger.error("joint_positions 维度 %d != %d", state.shape[0], NUM_JOINTS)
            robot.enter_safe_hold(reason="bad joint dim")
            exit_code = EXIT_PREFLIGHT
            break

        sent_obs = bool(need_obs)      # 本步是否发了真实观测（= 换段首帧）
        if need_obs:
            # ★ patch 11：相机丢帧容忍 —— 先保持当前位置重试，别一帧就 abort
            period = 1.0 / RATE
            cam = cameras.sample(now)
            if cam is None:
                stall_deadline = time.monotonic() + cam_stall_abort
                n_miss = 1
                while time.monotonic() < stall_deadline:
                    robot.command_joint_state(last["t"])      # 原地保持，不动
                    await asyncio.sleep(period)
                    cam = cameras.sample(time.monotonic())
                    if cam is not None:
                        break
                    n_miss += 1
                if cam is None:
                    logger.error("相机连续 %.1fs 无可用帧（丢 %d 次）→ safe hold",
                                 cam_stall_abort, n_miss)
                    robot.enter_safe_hold(reason="camera stalled")
                    exit_code = EXIT_CAMERA_STALE
                    break
                cam_misses_total += n_miss
                logger.warning("相机丢帧 %d 次后恢复（累计 %d），已原地保持等待",
                               n_miss, cam_misses_total)
                next_tick = time.monotonic()          # 等待期间时钟已滞后，重对齐
            images, _frame_ts = cam
            obs = build_obs(state, images, task, RATE)
            images_for_log = images      # 冻结时要把这份画面存下来（patch 15）
        else:
            obs = {}

        await ws.send(packb(obs))
        t_req = time.monotonic()
        # ★ patch 6：首包给宽超时（第一次 forward 含 CUDA autotune，实测 ~12 s）
        timeout = FIRST_RECV_TIMEOUT_S if step == 0 else RECV_TIMEOUT_S
        try:
            resp = unpackb(await asyncio.wait_for(ws.recv(), timeout=timeout))
        except asyncio.TimeoutError:
            logger.error("服务端 %.1fs 无响应（step %d, timeout=%s%.1f）→ safe hold",
                         time.monotonic() - t_req, step,
                         "first " if step == 0 else "steady ", timeout)
            robot.enter_safe_hold(reason="server timeout")
            exit_code = EXIT_TIMEOUT
            break
        t_got = time.monotonic()
        latencies.append(t_got - t_req)

        # ★ patch 3：阻塞过的推理不允许以"连发"形式补放
        if t_got - t_req > period:
            next_tick = t_got
            logger.debug("阻塞 %.0f ms → 时钟重对齐", (t_got - t_req) * 1e3)

        if "error" in resp:
            logger.error("服务端错误：%s", resp["error"])
            robot.enter_safe_hold(reason="server error")
            exit_code = EXIT_SERVER_ERROR
            break

        try:
            target = action_to_target(resp.get("action", {}), last["t"])
        except ValueError as exc:
            logger.error("动作解析失败：%s（keys=%s）", exc,
                         list(resp.get("action", {}).keys()))
            robot.enter_safe_hold(reason="bad action")
            exit_code = EXIT_SERVER_ERROR
            break
        raw_target = target.copy()     # 限速前的模型原始输出（用于 CSV 诊断）

        # ── 关节跳变诊断 ───────────────────────────────────────────────────
        delta = target[ARM_JOINTS] - last["t"][ARM_JOINTS]
        jump = float(np.max(np.abs(delta))) if delta.size else 0.0
        if jump > jump_warn:
            j = int(np.argmax(np.abs(delta)))
            dim_name = DIM_NAMES_FOR_LOG[int(ARM_JOINTS[j])]
            logger.warning("step %d: 关节跳变 %.1f°（%s，换段？）",
                           step, np.rad2deg(jump), dim_name)

        # ── ★ patch 10：执行层限速（这一条是"夹飞"/"换段乱动"的核心修复）──
        # 夹爪：0.7/s（0→1 需 1.43 s，对齐 yam_motion 的 gripper_rate_limit）
        # 臂  ：2.5 rad/s（143°/s，只裁剪换段尖峰，不干扰正常跟随）
        # ── ★ patch 22：夹爪释放即停（任务感知）──────────────────────────
        # 用【限速后的命令值】判断夹爪开合，它才反映真实物理状态。
        if stop_after_release > 0:
            for _j in GRIPPER_JOINTS:
                _v = float(target[_j])
                if _v < 0.5:                       # 闭合中
                    grip_closed_steps[_j] += 1
                else:                              # 张开了
                    if (release_step is None
                            and grip_closed_steps[_j] >= STOP_AFTER_RELEASE_MIN_CLOSED):
                        release_step = step
                        logger.info("检测到夹爪 #%d 释放（之前闭合了 %d 步）→ "
                                    "任务放置完成，再跑 %d 步（%.1fs）就结束",
                                    _j, grip_closed_steps[_j], stop_after_release,
                                    stop_after_release / RATE)
                    grip_closed_steps[_j] = 0
            if release_step is not None and (step - release_step) >= stop_after_release:
                logger.info("夹爪释放后已过 %d 步（%.2fs）→ 判定任务完成，正常结束"
                            "（剩余回位交给确定性的复位脚本）",
                            step - release_step, (step - release_step) / RATE)
                break
        # ── ★ patch 21：稳定即停（不靠猜时长）────────────────────────────
        # 判据用【模型原始输出】的步间变化，不受限速和滤波影响。
        # 启用条件：step >= 60（跑过 2 秒）且已出现过 >3° 的动作 ——
        # 防止一开始就静止（例如模型还在等待）被误判成"已完成"。
        if stop_still_steps > 0:
            _rj = float(np.rad2deg(np.max(np.abs(
                raw_target[ARM_JOINTS] - last["t"][ARM_JOINTS]))))
            if prev_raw is not None:
                _ds = float(np.rad2deg(np.max(np.abs(raw_target - prev_raw))))
            else:
                _ds = 999.0
            prev_raw = raw_target.copy()
            moved_deg = max(moved_deg, _rj)
            if step >= 60 and moved_deg > 3.0 and _ds < still_eps_deg:
                still_count += 1
            else:
                still_count = 0
            if still_count >= stop_still_steps:
                logger.info("模型输出连续 %d 步几乎不变（<%.1f°）且已运行 %.1fs "
                            "→ 判定已到位，正常结束（累计最大位移 %.1f°）",
                            still_count, still_eps_deg,
                            time.monotonic() - t_run0, moved_deg)
                break
        # ── ★ patch 20：命令低通滤波（"low gain"）────────────────────────
        # 放在限速之前：先滤掉模型输出在换段频率上的来回抖，再由限速保证
        # 物理速度上限。tau=0 时完全旁路（行为与之前一致）。
        if cmd_tau > 0:
            _was_none = sm is None
            target, _a = lowpass_target(target, sm,
                                        time.monotonic() - last_cmd_t, cmd_tau)
            sm = target
            if not _was_none:
                smoothed_steps += 1
        now_cmd = time.monotonic()
        target, clipped_grip, clipped_arm = rate_limit_target(
            target, last["t"], now_cmd - last_cmd_t,
            gripper_rate=gripper_rate, arm_speed=arm_speed)
        if clipped_grip > 1e-6:
            grip_clip_steps += 1
            if grip_clip_steps <= 5 or grip_clip_steps % 30 == 0:
                logger.info("夹爪限速生效：本步裁剪 %.3f（%.2f/s → 全行程需 %.2f s）"
                            "（第 %d 次）", clipped_grip, gripper_rate,
                            (1.0 / gripper_rate) if gripper_rate > 0 else 0,
                            grip_clip_steps)
        if clipped_arm > 1e-6:
            arm_clip_steps += 1
            if arm_clip_steps <= 5 or arm_clip_steps % 30 == 0:
                logger.warning("臂限速生效：本步裁剪 %.1f°（第 %d 次）—— "
                               "这就是换段跳变，被限速平滑掉",
                               np.rad2deg(clipped_arm), arm_clip_steps)

        # 兼容旧的硬钳位参数（--max-joint-step-deg，默认关闭）
        if max_step > 0:
            delta2 = target[ARM_JOINTS] - last["t"][ARM_JOINTS]
            jump2 = float(np.max(np.abs(delta2))) if delta2.size else 0.0
            if jump2 > max_step:
                target[ARM_JOINTS] = last["t"][ARM_JOINTS] + delta2 * (max_step / jump2)

                # ── ★ patch 16：模型输出合理性看门狗（先于限速与下发）──────────────
        _dev = np.rad2deg(np.abs(raw_target[ARM_JOINTS] - last["t"][ARM_JOINTS]))
        _dev_max = float(_dev.max()) if _dev.size else 0.0
        if _dev_max > output_sanity_deg:
            sanity_bad += 1
            logger.warning("⚠️ 模型输出异常：%s 目标偏离当前 %.1f°（上限 %.0f°），"
                           "连续 %d/%d 步", DIM_NAMES_FOR_LOG[int(ARM_JOINTS[int(_dev.argmax())])],
                           _dev_max, output_sanity_deg, sanity_bad, output_sanity_trip)
        else:
            sanity_bad = 0
        if sanity_bad >= output_sanity_trip:
            logger.error("🛑 模型输出持续异常 %d 步（最大偏离 %.1f°）→ 判定策略崩坏，"
                         "立即安全停止（防剧烈震动损坏硬件）", sanity_bad, _dev_max)
            robot.enter_safe_hold(reason="model output insane")
            exit_code = EXIT_MODEL_INSANE
            break

        # 振荡闸：连续方向反转
        _d = raw_target[ARM_JOINTS] - last["t"][ARM_JOINTS]
        if np.rad2deg(np.abs(_d).max()) > output_osc_deg:
            if prev_delta is not None and float(np.dot(prev_delta, _d)) < 0:
                osc_count += 1
            else:
                osc_count = 0
            prev_delta = _d.copy()
        else:
            osc_count = 0
            prev_delta = None
        if osc_count >= output_osc_trip:
            logger.error("🛑 检测到持续振荡：模型输出连续 %d 次方向反转（每步 >%.0f°）"
                         "→ 立即安全停止", osc_count, output_osc_deg)
            robot.enter_safe_hold(reason="model oscillating")
            exit_code = EXIT_MODEL_INSANE
            break
        # ── ★ patch 15：冻结检测（诊断"抓住后停在半空不动"）──────────────
        # 判定：本步模型原始输出相对上一步命令的位移 < 0.5° 且持续 ≥ 1 秒。
        # 命中时调用 freeze_logger（会存下"模型看到的三路画面"），每段只报一次。
        _raw_jump = float(np.max(np.abs(raw_target - last["t"])))
        if np.rad2deg(_raw_jump) < 0.5:
            freeze_steps += 1
        else:
            if freeze_steps >= int(1.0 * RATE):     # 刚结束一段冻结
                logger.warning("冻结结束：之前连续 %.1f 秒模型输出几乎不动（累计 %d 段）",
                               freeze_steps / RATE, freeze_reports)
            freeze_steps = 0
        if freeze_steps == int(1.0 * RATE) and freeze_logger is not None:
            freeze_reports += 1
            try:
                freeze_logger(step, images_for_log)
            except Exception as exc:                # noqa: BLE001
                logger.debug("freeze_logger 失败：%s", exc)

        if limits is not None:
            target = np.clip(target, limits[:, 0], limits[:, 1])
        target = target.astype(np.float32)
        robot.command_joint_state(target)
        if csv_writer is not None:
            try:
                import json as _json
                csv_writer.write(_json.dumps({
                    "step": step,
                    "t": round(now_cmd - t_run0, 4),
                    "sent_obs": bool(sent_obs),          # 本步是否发了真实观测（=换段首帧）
                    "prev": [round(float(x), 5) for x in last["t"]],
                    "raw_target": [round(float(x), 5) for x in raw_target],
                    "cmd_target": [round(float(x), 5) for x in target],
                    "d_raw": [round(float(x), 5) for x in (raw_target - last["t"])],
                    "d_cmd": [round(float(x), 5) for x in (target - last["t"])],
                    "clip_grip": round(float(clipped_grip), 5),
                    "clip_arm": round(float(clipped_arm), 5),
                    "need_obs_next": bool(resp.get("need_obs", True)),
                    "lat_ms": round((t_got - t_req) * 1e3, 1),
                }) + "\n")
            except Exception as exc:  # noqa: BLE001
                logger.debug("写 csv 失败：%s", exc)
        last["t"] = target
        last_cmd_t = now_cmd

        need_obs = bool(resp.get("need_obs", True))
        step += 1

        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            behind += 1
            if behind % 30 == 1:
                logger.warning("落后 %.0f ms（step %d）", -delay * 1e3, step)

    return exit_code, step, behind, {
        "cam_misses": cam_misses_total,
        "gripper_clipped_steps": grip_clip_steps,
        "arm_clipped_steps": arm_clip_steps,
        "freeze_reports": freeze_reports,
        "smoothed_steps": smoothed_steps,
        "stopped_by_still": still_count >= stop_still_steps if stop_still_steps > 0 else False,
    }


def _save_png(path: str, hwc_rgb: np.ndarray) -> bool:
    """保存一帧 RGB HWC uint8 到 PNG（cv2 优先，PIL 兜底）。"""
    try:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image
        Image.fromarray(hwc_rgb).save(path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("存 PNG 失败（cv2/PIL 都不可用）：%s", exc)
        return False


def camera_test(args) -> int:
    """相机只读测试：开 3 路、采 30 帧、打印形状/均值/耗时。

    ⭐ 全零帧 = 相机没真正出图（或曝光全黑），要单独排查。
    ``--save-frames DIR`` 会把最后一批帧存成 PNG（看现场用）。
    """
    cam = RealCameras(args.camera_config, args.motion_src)
    print(f"  打开相机：{args.camera_config}")
    cam.start()
    ok = 0
    last_images = None
    try:
        for i in range(30):
            t0 = time.monotonic()
            out = cam.sample(time.monotonic())
            dt = (time.monotonic() - t0) * 1e3
            if out is None:
                print(f"  [{i:02d}] ❌ 无因果帧 / 帧过旧（>max_frame_age_s）")
            else:
                images, ts = out
                shapes = {k: tuple(v.shape) for k, v in images.items()}
                means = {k: round(float(v.mean()), 1) for k, v in images.items()}
                bad = [k for k, v in images.items()
                       if v.shape != IMAGE_SPECS[k] or v.dtype != np.uint8]
                if bad:
                    print(f"  [{i:02d}] ❌ 形状/类型不符：{bad}  {shapes}")
                else:
                    ok += 1
                    last_images = images
                    print(f"  [{i:02d}] ✅ ts={ts:.3f} means={means} 耗时={dt:.1f}ms")
            time.sleep(max(0.0, 1.0 / RATE - (time.monotonic() - t0)))
    finally:
        cam.close()
    print(f"\n  通过 {ok}/30 帧")

    save_dir = getattr(args, "save_frames", None)
    if save_dir and last_images is not None:
        from pathlib import Path
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        saved = []
        for key, chw in last_images.items():
            hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))  # CHW→HWC
            p = d / f"{key}.png"
            if _save_png(str(p), hwc):
                saved.append(str(p))
        if saved:
            print("  📷 已存现场图：")
            for p in saved:
                print(f"       {p}")
        else:
            print("  ⚠️ 存图失败（cv2/PIL 都不可用）")

    if ok < 25:
        print("  ❌ 相机不合格（帧率或形状不过关），先修相机再做闭环")
        return EXIT_CAMERA_STALE
    print("  ✅ 相机通路正常（注意检查上面的 means 是否非 0 —— 全 0 表示没真出图）")
    return EXIT_OK


def session_test(args) -> int:
    """安全会话测试：申请会话 → 30Hz 命令到当前位置（≈不动）→ 释放 → 复查状态。

    这是上真机前最关键的一步：验证我们能不能拿到独占控制权、命令能不能下发、
    释放后守护进程是否确认回到 safe_hold。**全程机械臂只是原地保持**。
    """
    robot, backend = make_robot(args.robot_backend, args.mock_robot,
                                args.robot_host, args.robot_port,
                                args.safety_port, args.robot_timeout_ms,
                                args.command_timing)
    if backend != "zmq":
        print(f"  ❌ session-test 只支持 zmq 后端（当前 {backend}）")
        return EXIT_PREFLIGHT

    secs = float(args.session_test)
    print(f"  会话测试：{secs:.1f}s @ {RATE}Hz，命令到当前实际位置（机械臂不移动）")
    st0 = robot.status()
    print(f"  接管前：state={st0.get('state')!r} hold_confirmed={st0.get('hold_confirmed')} "
          f"stale_clients={st0.get('stale_clients')!r}")

    n = 0
    t_begin = time.monotonic()
    rc = EXIT_OK
    dry = bool(getattr(args, "session_dry", False))
    try:
        with robot.motion_session():
            period = 1.0 / RATE
            # ★ patch 19：用 deadline 对齐而不是"干完活再 sleep(period)"。
            #   旧写法实际周期 = 工作耗时 + 33ms（实测只有 19.9Hz），
            #   会误报"命令频率偏低"，让人以为是 ZMQ 太慢。
            next_tick = time.monotonic()
            behind = 0
            if dry:
                print(f"  ✅ 会话已激活（DRY 模式：{secs:.1f}s 内不下发任何指令）")
                while time.monotonic() - t_begin < secs:
                    robot.check()                   # 只维护心跳，不发命令
                    next_tick += period
                    d = next_tick - time.monotonic()
                    if d > 0:
                        time.sleep(d)
                    else:
                        behind += 1
            else:
                print("  ✅ 会话已激活，开始 30Hz 命令当前位置")
                while time.monotonic() - t_begin < secs:
                    robot.check()                   # 心跳线程若故障会在这里抛
                    pos = np.asarray(
                        robot.get_observations()["joint_positions"],
                        dtype=np.float32).reshape(-1)
                    robot.command_joint_state(pos)  # 命令到"当前位置" ≈ 不动
                    n += 1
                    next_tick += period
                    d = next_tick - time.monotonic()
                    if d > 0:
                        time.sleep(d)
                    else:
                        behind += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 会话中断：{type(exc).__name__}: {exc}")
        rc = EXIT_UNHEALTHY
    finally:
        dt = time.monotonic() - t_begin
        try:
            st1 = robot.status()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 释放后 status() 失败：{exc}")
            st1 = {}
        robot.close(disable_motors=False)

    _hz = (n / dt) if dt else 0.0
    print(f"\n  命令次数 = {n}   用时 = {dt:.1f}s   有效频率 = {_hz:.1f} Hz"
          + ("  (DRY：一条命令都没发)" if n == 0 else "")
          + (f"   落后 {behind} 次" if n else ""))
    print(f"  释放后：state={st1.get('state')!r} hold_confirmed={st1.get('hold_confirmed')} "
          f"reason={str(st1.get('reason'))[:80]!r}")
    if st1.get("state") != "safe_hold":
        print("  ❌ 释放后没有回到 safe_hold —— 手动确认机器人状态！")
        return EXIT_UNHEALTHY
    if not st1.get("hold_confirmed"):
        print("  ⚠️ safe_hold 但 hold_confirmed 不是 True，再查一次")
    if n > 0 and n < secs * RATE * 0.85:
        print(f"  ⚠️ 命令频率偏低（{n / dt:.1f}Hz，期望 ~{RATE}）")
        return EXIT_UNHEALTHY
    print("  ✅ 会话测试通过：能接管、能下发、释放后回到 safe_hold")
    return rc


async def run(args) -> int:
    robot, backend = make_robot(args.robot_backend, args.mock_robot,
                                args.robot_host, args.robot_port,
                                args.safety_port, args.robot_timeout_ms,
                                args.command_timing)
    logger.info("robot backend = %s", backend)

    if backend == "zmq":
        st = None
        try:
            st = robot.preflight()
            logger.info("preflight OK：state=%r stale_clients=%r",
                        st.get("state"), st.get("stale_clients"))
        except Exception as exc:  # noqa: BLE001
            logger.error("preflight 失败：%s", exc)
            robot.close()
            return EXIT_PREFLIGHT

    cameras = NullCameras() if args.fake_images else RealCameras(
        args.camera_config, args.motion_src,
        max_age_s=float(getattr(args, "camera_max_age_s", CAMERA_MAX_AGE_S)))
    cameras.start()
    logger.info("cameras: %s", "synthetic zeros" if args.fake_images else args.camera_config)

    # ★ patch 9：相机预热。RealSense 的自动曝光/白平衡要好几帧才收敛 ——
    #   2026-10-27 实测：head_rgb 均值 77→127 用了一整秒。开头就喂首帧会让模型
    #   看到"偏暗画面"，与训练分布不符。这里丢弃前 warm 秒的帧，不发指令。
    warm = float(getattr(args, "camera_warmup_s", 0.0) or 0.0)
    if warm > 0 and not args.fake_images:
        t_w = time.monotonic()
        n_w = 0
        while time.monotonic() - t_w < warm:
            try:
                cameras.sample(time.monotonic())      # 只丢弃，不保存
            except Exception as exc:  # noqa: BLE001
                logger.debug("预热期取帧失败（忽略）：%s", exc)
            n_w += 1
            time.sleep(1.0 / RATE)
        logger.info("相机预热完：%.1fs，丢弃 %d 帧（等自动曝光收敛）", warm, n_w)

    jump_warn = np.deg2rad(JUMP_WARN_DEG)
    max_step = np.deg2rad(args.max_joint_step_deg) if args.max_joint_step_deg > 0 else 0.0
    latencies: list[float] = []
    csv_fh = None
    csv_path = getattr(args, "log_csv", None)
    if csv_path:
        csv_fh = open(csv_path, "w", encoding="utf-8")
        logger.info("CSV 诊断日志：%s（每步一行，含 raw_target/cmd_target/跳变/换段标记）",
                    csv_path)
    # ★ patch 15：冻结时存画面
    freeze_logger = None
    snap_dir = getattr(args, "snapshot_on_freeze", None)
    if snap_dir:
        from pathlib import Path
        Path(snap_dir).mkdir(parents=True, exist_ok=True)

        def freeze_logger(step, images):                 # noqa: ANN001
            if images is None:
                logger.warning("冻结但还没有真实画面可存（step %d）", step)
                return
            saved = []
            for key, chw in images.items():
                hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))
                pth = Path(snap_dir) / f"freeze_step{step:04d}_{key}.png"
                if _save_png(str(pth), hwc):
                    saved.append(pth.name)
            logger.warning("❄️ 模型连续 1 秒几乎不动（step %d）→ 已存画面：%s",
                           step, ", ".join(saved) if saved else "（存图失败）")

    loop_kw = dict(
        gripper_rate=float(args.gripper_rate_limit),
        arm_speed=float(args.arm_max_speed_rad_s),
        cam_stall_abort=float(args.camera_stall_abort_s),
        csv_writer=csv_fh,
        freeze_logger=freeze_logger,
        output_sanity_deg=float(args.output_sanity_deg),
        output_osc_trip=int(args.output_osc_trip),
        cmd_tau=float(args.cmd_lowpass_tau),
        stop_still_steps=int(args.stop_when_still),
        stop_after_release=int(args.stop_after_release),
    )
    logger.info("执行层配置：夹爪限速 %.2f/s，臂限速 %.2f rad/s，相机停顿阈值 %.1fs",
                loop_kw["gripper_rate"], loop_kw["arm_speed"], loop_kw["cam_stall_abort"])

    last = {"t": np.asarray(
        robot.get_observations()["joint_positions"], dtype=np.float32).reshape(-1)}
    if last["t"].shape[0] != NUM_JOINTS:
        raise RuntimeError(f"joint_positions 维度 {last['t'].shape[0]}，期望 {NUM_JOINTS}")
    limits = robot.get_joint_limits()
    if limits is not None:
        limits = np.asarray(limits, dtype=np.float64)
        if limits.shape != (NUM_JOINTS, 2):
            raise RuntimeError(f"get_joint_limits() shape {limits.shape}，期望 "
                               f"({NUM_JOINTS}, 2)")
        logger.info("本地软限位已启用")
    else:
        logger.info("本地软限位关闭（由守护进程执行）")

    t_run0 = time.monotonic()
    exit_code = EXIT_OK
    step = 0
    behind = 0
    stats: dict = {}
    try:
        ws_ctx = _ws_connect(f"ws://{args.host}:{args.port}")
        async with ws_ctx as ws:
            if backend == "zmq":
                with robot.motion_session():
                    exit_code, step, behind, stats = await _loop(
                        ws, robot, cameras, args.task, args.steps, args.duration,
                        limits, jump_warn, max_step, latencies, t_run0, last,
                        **loop_kw)
            else:
                exit_code, step, behind, stats = await _loop(
                    ws, robot, cameras, args.task, args.steps, args.duration,
                    limits, jump_warn, max_step, latencies, t_run0, last,
                    **loop_kw)
    except Exception as exc:  # noqa: BLE001
        logger.error("运行异常：%s: %s", type(exc).__name__, exc)
        exit_code = EXIT_SERVER_ERROR
        raise
    finally:
        logger.info("收尾：safe hold + close")
        if csv_fh is not None:
            try:
                csv_fh.close()
                logger.info("CSV 诊断日志已写：%s", csv_path)
            except Exception:  # noqa: BLE001
                pass
        try:
            robot.enter_safe_hold(reason="bridge shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.error("enter_safe_hold 失败：%s", exc)
        try:
            cameras.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("cameras.close 失败：%s", exc)
        try:
            robot.close(disable_motors=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("close 失败：%s", exc)

        if latencies:
            arr = np.asarray(latencies) * 1e3
            logger.info("latency: n=%d avg=%.1fms p50=%.1fms p99=%.1fms max=%.1fms",
                        arr.size, arr.mean(), np.percentile(arr, 50),
                        np.percentile(arr, 99), arr.max())
        logger.info("steps completed=%d behind_schedule=%d", step, behind)
        if stats:
            logger.info("执行层统计：相机丢帧恢复 %d 次；夹爪被限速 %d 步；臂被限速 %d 步；冻结段 %d 段",
                        stats.get("cam_misses", 0),
                        stats.get("gripper_clipped_steps", 0),
                        stats.get("arm_clipped_steps", 0),
                        stats.get("freeze_reports", 0))
            if stats.get("arm_clipped_steps", 0):
                logger.warning("臂限速生效了 %d 步 —— 说明换段确实有跳变，"
                               "已被平滑；若仍觉动作生硬可调小 --arm-max-speed-rad-s",
                               stats["arm_clipped_steps"])

    return exit_code


def main():
    ap = argparse.ArgumentParser(description="Real-machine YAM bridge client")
    ap.add_argument("--host", default="127.0.0.1", help="serve_policy host")
    ap.add_argument("--port", type=int, default=18879,
                    help="serve_policy port（本项目 18879；上游默认 8765）")
    ap.add_argument("--task", default=None,
                    help="VLM 任务指令（自由文本）。必须与训练文本完全一致。"
                         "推荐改用 --task-id 免手打出错")
    ap.add_argument("--task-id", default=None, metavar="TASK-YAM-XXXX",
                    help="用内置权威表取任务文本，可选："
                         + ", ".join(sorted(TASK_TEXTS)))
    ap.add_argument("--list-tasks", action="store_true",
                    help="打印内置的 10 个任务文本后退出")
    ap.add_argument("--steps", type=int, default=0, help="步数上限；0 = 不限")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="运行秒数；0 = 不限（与 --steps 0 一起用）")

    # 机器人后端（patch 8）
    ap.add_argument("--robot-backend", choices=("zmq", "direct", "mock"),
                    default="zmq",
                    help="zmq=走安全守护进程（真机推荐）；direct=直连 CAN（调试）；"
                         "mock=不动硬件")
    ap.add_argument("--mock-robot", action="store_true",
                    help="等价于 --robot-backend mock（不连 CAN、不标定、不动）")
    ap.add_argument("--robot-host", default="127.0.0.1")
    ap.add_argument("--robot-port", type=int, default=6001, help="机器人 RPC 端口")
    ap.add_argument("--safety-port", type=int, default=6002, help="安全 RPC 端口")
    ap.add_argument("--robot-timeout-ms", type=int, default=1000)
    ap.add_argument("--command-timing", choices=("legacy_30hz", "timed"),
                    default="legacy_30hz",
                    help="与部署包 runtime.yaml 的 command_timing 对齐")
    ap.add_argument("--check-robot", action="store_true",
                    help="只读探测：打印机器人/安全守护状态后退出，不发任何指令")
    ap.add_argument("--session-test", type=float, default=0.0, metavar="SECONDS",
                    help="安全会话测试：申请会话→30Hz 命令到当前位置（机械臂不移动）→"
                         "释放→复查 safe_hold。上真机前的关键一步")
    ap.add_argument("--session-dry", action="store_true",
                    help="配合 --session-test：只申请/持有/释放安全会话，"
                         "**一条运动指令都不下发**（最保守的一步）")
    ap.add_argument("--output-sanity-deg", type=float, default=OUTPUT_SANITY_DEG,
                    help="看门狗：单关节目标偏离当前姿态超过这个角度视为模型输出异常"
                         "（默认 60°，正常跟踪实测 p95<3°）；0 = 关闭")
    ap.add_argument("--stop-after-release", type=int, default=STOP_AFTER_RELEASE_STEPS,
                    help="夹爪「闭合→张开」（放置释放）后再跑 N 步就正常结束，N 建议 45（1.5s）。"
                         "这是任务感知的结束条件，比等模型自己安静可靠。0=关闭")
    ap.add_argument("--stop-when-still", type=int, default=STOP_WHEN_STILL_STEPS,
                    help="模型输出连续 N 步几乎不变（<0.3°）就正常结束，不用猜 --duration。"
                         "建议 60（2 秒）。0=关闭。配合较宽松的 --duration（如 40）使用")
    ap.add_argument("--cmd-lowpass-tau", type=float, default=CMD_LOWPASS_TAU_S,
                    help="命令一阶低通时间常数（秒）。0=关闭；建议 0.08~0.15 治"
                         "「回原位时震动」——把模型输出的来回抖滤成平滑运动，"
                         "且滤波在限速之前，物理限位依然严格")
    ap.add_argument("--output-osc-trip", type=int, default=OUTPUT_OSC_TRIP,
                    help="看门狗：连续多少次方向反转折算振荡（默认 8）；0 = 关闭")
    ap.add_argument("--snapshot-on-freeze", default=None, metavar="DIR",
                    help="运行中检测到模型连续 1 秒几乎不动时，把当时的三路画面存成 PNG"
                         "（诊断「夹住后停住」用）")
    ap.add_argument("--save-frames", default=None, metavar="DIR",
                    help="配合 --camera-test：把最后一批帧存成 PNG（看现场用）")
    ap.add_argument("--camera-test", action="store_true",
                    help="相机只读测试：开 3 路、采 30 帧、打印形状/均值/耗时")

    # 相机
    ap.add_argument("--fake-images", action="store_true",
                    help="合成零图（只用于协议冒烟）")
    ap.add_argument("--allow-fake-images-on-robot", action="store_true",
                    help="允许真机 + 合成图（模型会输出乱动作，危险）")
    ap.add_argument("--camera-warmup-s", type=float, default=1.5,
                    help="开跑前丢弃前 N 秒相机帧（等自动曝光/白平衡收敛，默认 1.5s）")
    ap.add_argument("--camera-config", default=None,
                    help="含 gello_root/camera_config/camera_names 的 robot.yaml")
    ap.add_argument("--motion-src", default=None,
                    help="yam_motion 的 src 目录（提供 YamCameras）")

    # ★ patch 10/11：执行层限速
    ap.add_argument("--log-csv", default=None, metavar="PATH",
                    help="把每步的目标/跳变/换段标记写成 CSV（诊断换段跳变用）")
    ap.add_argument("--gripper-rate-limit", type=float, default=GRIPPER_RATE_LIMIT_S,
                    help="夹爪限速（单位/秒）。0.7 表示 0→1 需 1.43 s，"
                         "对齐 yam_motion 的 gripper_rate_limit；0 = 关闭（易把物体夹飞）")
    ap.add_argument("--arm-max-speed-rad-s", type=float, default=ARM_MAX_SPEED_RAD_S,
                    help="臂关节速度上限（rad/s）。默认 2.5（≈143°/s），"
                         "只裁剪换段尖峰、不干扰正常跟随；0 = 关闭")
    ap.add_argument("--camera-stall-abort-s", type=float, default=CAMERA_STALL_ABORT_S,
                    help="相机连续取不到帧超过这个秒数才 abort（期间原地保持）")
    ap.add_argument("--camera-max-age-s", type=float, default=CAMERA_MAX_AGE_S,
                    help="允许的相机帧陈旧度上限（秒），默认 0.20（≈6 帧 @30Hz）")
    ap.add_argument("--max-joint-step-deg", type=float, default=0.0,
                    help="可选的每步臂关节斜率限制（度）；0 = 关")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # ---- 只读探测模式 ----
    if args.check_robot:
        robot, backend = make_robot(args.robot_backend, args.mock_robot,
                                    args.robot_host, args.robot_port,
                                    args.safety_port, args.robot_timeout_ms,
                                    args.command_timing)
        try:
            return print_robot_status(robot, backend)
        finally:
            try:
                robot.close(disable_motors=False)
            except Exception:  # noqa: BLE001
                pass

    # ---- 解析任务文本（--task-id 优先，内置权威表）----
    if args.list_tasks:
        print("  内置权威任务文本（来自 HF Zenith6666/yam-lerobot）：")
        for k in sorted(TASK_TEXTS):
            print(f"    {k}  {TASK_TEXTS[k]!r}")
        return 0
    if args.task_id:
        key = args.task_id.strip().upper()
        if key not in TASK_TEXTS:
            ap.error(f"未知任务 {args.task_id!r}；可选：{', '.join(sorted(TASK_TEXTS))}")
        txt = TASK_TEXTS[key]
        if args.task and args.task.strip() != txt:
            logger.warning("--task 与 --task-id 冲突，以 --task-id 为准（%r → %r）",
                           args.task, txt)
        args.task = txt
        logger.info("任务文本（--task-id %s）：%r", key, txt)
    if not args.task:
        ap.error("必须给 --task 或 --task-id（推荐 --task-id）")
    # ---- 相机只读测试 ----
    if args.camera_test:
        if not args.camera_config or not args.motion_src:
            ap.error("--camera-test 需要 --camera-config 和 --motion-src")
        return sys.exit(camera_test(args))

    # ---- 安全会话测试（上真机前最关键的一步）----
    if args.session_test > 0:
        return sys.exit(session_test(args))

    # ---- 防护 ----
    if args.fake_images and not args.mock_robot and not args.allow_fake_images_on_robot:
        ap.error("--fake-images 只用于协议测试；真机上合成图会让模型输出乱动作。"
                 "请同时加 --mock-robot，或显式加 --allow-fake-images-on-robot")
    if not args.fake_images:
        missing = [n for n, v in (("--camera-config", args.camera_config),
                                  ("--motion-src", args.motion_src)) if not v]
        if missing:
            ap.error(f"真机模式需要 {' 和 '.join(missing)}")
    else:
        logger.warning("fake images：模型输出没有意义 —— 仅协议测试")
    if args.robot_backend == "direct" and not args.mock_robot:
        logger.warning("direct 模式：直连 CAN，**没有 400ms 看门狗**，且会抢占总线。"
                       "真机推荐 --robot-backend zmq")

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
