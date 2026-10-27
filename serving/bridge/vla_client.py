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

# 退出码：0 = 正常跑完；>=2 = 被安全停机制中断（上层脚本据此判失败）
EXIT_OK = 0
EXIT_UNHEALTHY = 2
EXIT_SERVER_ERROR = 3
EXIT_TIMEOUT = 4
EXIT_CAMERA_STALE = 5
EXIT_PREFLIGHT = 6


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
                 max_age_s: float = 0.1) -> None:
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

    def sample(self, ts: float):
        out = self._cam.sample(ts)
        if out is None:
            return None
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

    @contextmanager
    def motion_session(self):
        """申请独占受监督会话，退出时自动回到 safe_hold（并确认）。"""
        from gello.safety.motion_session import SupervisedMotionSession

        self.preflight()
        sid = uuid.uuid4().hex
        logger.info("申请安全会话 session=%s operation=g05_vla_bridge ...", sid)
        with SupervisedMotionSession(self.safety, sid,
                                     operation="g05_vla_bridge") as sess:
            self._session = sess
            logger.info("✅ 安全会话已激活（含 parent/collector 两个 200ms 心跳线程）")
            try:
                yield sess
            finally:
                self._session = None
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
                jump_warn, max_step, latencies, t_run0, last):
    """主循环。``last`` 是 ``{"t": ndarray}`` 的可变容器（上一帧命令的目标）。

    返回 (exit_code, step, behind)。
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

    need_obs = True
    next_tick = time.monotonic()
    step = 0
    behind = 0
    exit_code = EXIT_OK

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

        if need_obs:
            cam = cameras.sample(now)
            if cam is None:
                logger.error("相机帧缺失/过旧 → safe hold")
                robot.enter_safe_hold(reason="camera stale")
                exit_code = EXIT_CAMERA_STALE
                break
            images, _frame_ts = cam
            obs = build_obs(state, images, task, RATE)
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

        # 关节跳变诊断 + 可选斜率限幅
        delta = target[ARM_JOINTS] - last["t"][ARM_JOINTS]
        jump = float(np.max(np.abs(delta))) if delta.size else 0.0
        if jump > jump_warn:
            logger.warning("step %d: 关节跳变 %.1f°（换段？）", step, np.rad2deg(jump))
        if max_step > 0 and jump > max_step:
            target[ARM_JOINTS] = last["t"][ARM_JOINTS] + delta * (max_step / jump)

        if limits is not None:
            target = np.clip(target, limits[:, 0], limits[:, 1])
        target = target.astype(np.float32)
        robot.command_joint_state(target)
        last["t"] = target

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

    return exit_code, step, behind


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
        args.camera_config, args.motion_src)
    cameras.start()
    logger.info("cameras: %s", "synthetic zeros" if args.fake_images else args.camera_config)

    jump_warn = np.deg2rad(JUMP_WARN_DEG)
    max_step = np.deg2rad(args.max_joint_step_deg) if args.max_joint_step_deg > 0 else 0.0
    latencies: list[float] = []

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
    try:
        ws_ctx = _ws_connect(f"ws://{args.host}:{args.port}")
        async with ws_ctx as ws:
            if backend == "zmq":
                with robot.motion_session():
                    exit_code, step, behind = await _loop(
                        ws, robot, cameras, args.task, args.steps, args.duration,
                        limits, jump_warn, max_step, latencies, t_run0, last)
            else:
                exit_code, step, behind = await _loop(
                    ws, robot, cameras, args.task, args.steps, args.duration,
                    limits, jump_warn, max_step, latencies, t_run0, last)
    except Exception as exc:  # noqa: BLE001
        logger.error("运行异常：%s: %s", type(exc).__name__, exc)
        exit_code = EXIT_SERVER_ERROR
        raise
    finally:
        logger.info("收尾：safe hold + close")
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

    return exit_code


def main():
    ap = argparse.ArgumentParser(description="Real-machine YAM bridge client")
    ap.add_argument("--host", default="127.0.0.1", help="serve_policy host")
    ap.add_argument("--port", type=int, default=18879,
                    help="serve_policy port（本项目 18879；上游默认 8765）")
    ap.add_argument("--task", required=True,
                    help="VLM 任务指令 —— 必须与训练文本完全一致。"
                         "本项目 10 任务模型用："
                         "'Pick up the small square and place it in the groove'（大写 P）")
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

    # 相机
    ap.add_argument("--fake-images", action="store_true",
                    help="合成零图（只用于协议冒烟）")
    ap.add_argument("--allow-fake-images-on-robot", action="store_true",
                    help="允许真机 + 合成图（模型会输出乱动作，危险）")
    ap.add_argument("--camera-config", default=None,
                    help="含 gello_root/camera_config/camera_names 的 robot.yaml")
    ap.add_argument("--motion-src", default=None,
                    help="yam_motion 的 src 目录（提供 YamCameras）")

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
