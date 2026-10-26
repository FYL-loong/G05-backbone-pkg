#!/usr/bin/env python3
"""Real-machine bridge: cameras + joint state → ``serve_policy`` (WS) → YAM bimanual.

Origin
------
Written by 张恩同 (2026-10-26) as the client half of the async-serving design.
This copy is his file with **four reviewed fixes** applied (see CHANGES below).
Upstream path on the lab box: ``g05-yam-deployment/bridge/vla_client.py``.

CHANGES vs the original (all reviewed against gello/i2rt on 2026-10-26)
----------------------------------------------------------------------
1. ``capture_images()`` was ``NotImplementedError``.  Now implemented by reusing
   ``yam_motion.cameras.YamCameras`` (already validated on the real rig: it only
   returns **causal** frames whose host timestamp <= the query time, and rejects
   frames older than ``max_age_s``).  Frame is HWC uint8 → transposed to CHW.
2. Added ``--mock-robot``.  The original always called ``build_robot()``, and
   ``YAMRobot.create_pair`` performs **gripper calibration that physically moves
   the arms** — so the "protocol smoke test" was NOT safe.  With ``--mock-robot``
   nothing is connected and nothing moves.
3. **Clock-deficit fix (the important one).**  Inference blocks ~520 ms on the
   first request and on every chunk swap.  The original only advanced
   ``next_tick += period``, so after a block the loop fell ~15 steps behind and
   then fired those cached steps back-to-back — compressing ~0.5 s of trajectory
   into a few tens of ms.  We now re-baseline the clock whenever a request took
   longer than one period, so stalled time is **dropped, never replayed**.
   (Same idea as ``yam_motion.g05_execution.align_chunk``: "discard elapsed
   predictions".)
4. ``ws.recv()`` had no timeout.  If the server hangs the loop blocked forever
   and the ``finally`` cleanup never ran.  Now wrapped in ``asyncio.wait_for``;
   a timeout triggers an explicit safe hold.
   Extras: per-step joint-jump logging, optional ``--max-joint-step-deg`` clamp,
   ``--duration`` seconds, and an end-of-run timing summary.

Data contract (verified, unchanged from the original — it was correct)
---------------------------------------------------------------------
* **Action frame** = follower (i2rt) native frame.  Recorded actions are
  ``applied_joint_state``; leader raw values live separately in ``leader_action``.
* **Joint order/sign = identity**: 14-dim =
  ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
  ``BimanualRobot.get_observations()["joint_positions"]`` already concatenates
  left(7) + right(7) in exactly this order.  No reorder, no sign flip.
* **Action value = absolute**: ``serve_policy`` postprocess already does
  ``merger.backward`` (unpad) → ``normalizer.backward`` (denorm) →
  ``RelativeJointTransform.backward`` (relative→absolute).  Do NOT add state back.
* **Obs to send**: ``{embodiment_type, images, state, task, frequency=30.0}`` with
  3 images ``(3,480,640) uint8`` and 4 state vectors ``(6,1,6,1) float32``.
* **Action received**: ``resp["action"]`` is
  ``{left_arm, left_gripper, right_arm, right_gripper}`` of 1-D ndarrays.  Keys
  the AR head dropped (``_absent_keys``) are absent — we hold the last commanded
  value for them.

Usage
-----
Server (GalaxeaVLA 3.10 env; AR-only, bf16)::

    cd <GalaxeaVLA>
    PYTHONPATH=src python ../g05-async-serving/serve_policy.py \
        --ckpt_path <run_dir>/checkpoints/model_state_dict.pt \
        --host 127.0.0.1 --port 18879 \
        eval_embodiment=galaxea_r1lite \
        model.model_weights_to_bf16=true \
        model.model_arch.continuous_action=false

Client — **safe offline protocol smoke test** (no robot, no cameras)::

    python bridge/vla_client.py --host 127.0.0.1 --port 18879 \
        --task "Pick up the small square and place it in the groove" \
        --fake-images --mock-robot --steps 96 --verbose

Client — real run (requires E-stop ready; see the safety note below)::

    python bridge/vla_client.py --host 127.0.0.1 --port 18879 \
        --task "Pick up the small square and place it in the groove" \
        --duration 20

Safety
------
The loop stops on: unhealthy robot status, server ``error`` reply, receive
timeout, or Ctrl+C.  Every exit path runs ``enter_safe_hold`` then
``close(disable_motors=True)``.  ``command_joint_state`` targets are clipped to
``get_joint_limits()`` (arm 6×2 + gripper [0,1] per side).

⚠️ ``build_robot()`` runs gripper calibration, which **moves the arms**.  Confirm
the E-stop and soft limits are live before any non-``--mock-robot`` run.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import time

import msgpack
import numpy as np

logger = logging.getLogger("vla_client")

# ── tunables ─────────────────────────────────────────────────────────────────
RECV_TIMEOUT_S = 5.0     # must be >> T_infer (~0.52 s); exceeding it = server hung
JUMP_WARN_DEG = 5.0      # log a warning when one step commands more than this
NUM_JOINTS = 14


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


# ─── embodiment schema (matches configs/data/yam.yaml + fake_policy_client) ───

EMBODIMENT_TYPE = "galaxea_r1lite"
RATE = 30  # Hz — YAM data is resampled onto the 30 Hz camera grid

IMAGE_SPECS = {
    "head_rgb": (3, 480, 640),
    "left_wrist_rgb": (3, 480, 640),
    "right_wrist_rgb": (3, 480, 640),
}
# Camera-rig names → observation keys (from live_configs/robot.yaml camera_names
# and the rig config ~/.config/yam/realsense_bimanual.yaml).
CAMERA_NAME_MAP = {"main": "head_rgb", "left": "left_wrist_rgb",
                   "right": "right_wrist_rgb"}

# 14-dim joint layout → per-part slices. Identity mapping confirmed by
# tools/yam/mcap_to_lerobot.py: left_arm=act[:6], left_gripper=act[6],
# right_arm=act[7:13], right_gripper=act[13].
STATE_SPLIT = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}
ACTION_KEYS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
ARM_JOINTS = np.r_[0:6, 7:13]   # indices of the 12 arm joints (grippers excluded)


# ─── pure joint mapping (unit-tested without gello / cameras / websockets) ────

def split_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """14-dim joint vector → ``{left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)}``.

    Identity mapping — no reorder or sign flip.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS}-dim joint state, got {state.shape[0]}")
    return {key: state[lo:hi] for key, (lo, hi) in STATE_SPLIT.items()}


def action_to_target(action: dict, last_target: np.ndarray) -> np.ndarray:
    """Merge the server's action dict back into a 14-dim absolute target.

    Missing keys (dropped by the AR head via ``_absent_keys``) fall back to the
    previous commanded value in ``last_target``, so the robot holds that part.
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
    """Build the raw observation dict the server's ``_validate_obs`` expects."""
    return {
        "embodiment_type": EMBODIMENT_TYPE,
        "images": images,
        "state": split_state(state),
        "task": task,
        # ★ must be 30 for YAM — the server falls back to 15 (SO-100) if omitted.
        "frequency": float(frequency),
    }


# ─── cameras ─────────────────────────────────────────────────────────────────

class NullCameras:
    """Synthetic zero images. Protocol smoke tests only."""

    def start(self) -> None:
        pass

    def sample(self, _ts: float):
        return ({name: np.zeros(shape, dtype=np.uint8)
                 for name, shape in IMAGE_SPECS.items()}, 0.0)

    def close(self) -> None:
        pass


class RealCameras:
    """RealSense rig via ``yam_motion.cameras.YamCameras`` (host-clock adapter).

    ``YamCameras.sample(t)`` returns only frames whose ``host_timestamp <= t``
    (causal) and rejects anything older than ``max_age_s`` — exactly the
    alignment semantics the policy needs.
    """

    def __init__(self, robot_cfg_path: str, motion_src: str,
                 max_age_s: float = 0.1) -> None:
        import sys

        import yaml  # only needed on the real-camera path

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
            if arr.shape != IMAGE_SPECS[CAMERA_NAME_MAP[rig_name]]:
                raise RuntimeError(
                    f"camera {rig_name} frame {arr.shape} != "
                    f"{IMAGE_SPECS[CAMERA_NAME_MAP[rig_name]]}")
            chw[CAMERA_NAME_MAP[rig_name]] = arr.astype(np.uint8, copy=False)
        return chw, min_ts

    def close(self) -> None:
        self._cam.close()


# ─── robot ───────────────────────────────────────────────────────────────────

class MockRobot:
    """Stand-in robot: no CAN, no motors, no calibration. Never moves anything."""

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


def build_robot():
    """Initialize the bimanual follower pair and return it.

    ⚠️ ``create_pair`` runs gripper calibration, which physically moves the robot.
    Confirm the E-stop / soft limits are live before calling.
    """
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    left, right = YAMRobot.create_pair(
        left_channel="can_follower_l", right_channel="can_follower_r"
    )
    robot = BimanualRobot(left, right)
    logger.info("Robot ready: num_dofs=%d, joint_limits=%s",
                robot.num_dofs(), robot.get_joint_limits().shape)
    return robot


# ─── main loop ────────────────────────────────────────────────────────────────

async def run(host, port, task, steps, duration, fake_images, mock_robot,
              camera_cfg, motion_src, max_joint_step_deg, verbose):
    import websockets

    period = 1.0 / RATE
    robot = MockRobot() if mock_robot else build_robot()
    limits = np.asarray(robot.get_joint_limits(), dtype=np.float64)
    if limits.shape != (NUM_JOINTS, 2):
        raise RuntimeError(f"get_joint_limits() shape {limits.shape}, expected "
                           f"({NUM_JOINTS}, 2)")

    cameras = NullCameras() if fake_images else RealCameras(camera_cfg, motion_src)
    cameras.start()
    logger.info("cameras: %s", "synthetic zeros" if fake_images else camera_cfg)

    jump_warn = np.deg2rad(JUMP_WARN_DEG)
    max_step = np.deg2rad(max_joint_step_deg) if max_joint_step_deg > 0 else 0.0
    latencies: list[float] = []
    behind = 0
    step = 0                      # defined before try: the finally summary uses it

    last_target = np.asarray(
        robot.get_observations()["joint_positions"], dtype=np.float32).reshape(-1)
    if last_target.shape[0] != NUM_JOINTS:
        raise RuntimeError(f"joint_positions dim {last_target.shape[0]}, expected {NUM_JOINTS}")

    t_run0 = time.monotonic()
    try:
        async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
            hello = unpackb(await ws.recv())
            if not isinstance(hello, dict):
                raise RuntimeError(f"server hello is {type(hello).__name__}, expected dict")
            if "error" in hello:
                raise RuntimeError(f"server rejected connection: {hello['error']}")
            action_steps = hello.get("action_steps")
            logger.info("connected %s:%d  server action_steps=%s", host, port, action_steps)
            if action_steps is not None and int(action_steps) > 16:
                logger.warning("server action_steps=%s > 16: look-ahead window is "
                               "action_steps/RATE = %.0f ms", action_steps,
                               float(action_steps) * 1000.0 / RATE)

            need_obs = True
            next_tick = time.monotonic()

            while True:
                if steps > 0 and step >= steps:
                    break
                if duration > 0 and (time.monotonic() - t_run0) >= duration:
                    break

                health = robot.get_health_status()
                if not health.get("ok", False):
                    logger.error("robot unhealthy: %s", health.get("reason") or health)
                    robot.enter_safe_hold(reason="unhealthy")
                    break

                now = time.monotonic()
                state = np.asarray(
                    robot.get_observations()["joint_positions"],
                    dtype=np.float32).reshape(-1)

                if need_obs:
                    cam = cameras.sample(now)
                    if cam is None:
                        logger.error("camera frame missing/stale → safe hold")
                        robot.enter_safe_hold(reason="camera stale")
                        break
                    images, _frame_ts = cam
                    obs = build_obs(state, images, task, RATE)
                else:
                    obs = {}

                await ws.send(packb(obs))
                t_req = time.monotonic()
                try:
                    resp = unpackb(await asyncio.wait_for(
                        ws.recv(), timeout=RECV_TIMEOUT_S))
                except asyncio.TimeoutError:
                    logger.error("server silent for %.1fs → safe hold", RECV_TIMEOUT_S)
                    robot.enter_safe_hold(reason="server timeout")
                    break
                t_got = time.monotonic()
                latencies.append(t_got - t_req)

                # ★ FIX 3: a blocking inference must NOT be replayed as a burst.
                if t_got - t_req > period:
                    next_tick = t_got
                    logger.debug("blocked %.0f ms → clock re-based", (t_got - t_req) * 1e3)

                if "error" in resp:
                    logger.error("server error: %s", resp["error"])
                    robot.enter_safe_hold(reason="server error")
                    break

                target = action_to_target(resp.get("action", {}), last_target)

                # joint-jump diagnostics (and optional slew clamp)
                delta = target[ARM_JOINTS] - last_target[ARM_JOINTS]
                jump = float(np.max(np.abs(delta))) if delta.size else 0.0
                if jump > jump_warn:
                    logger.warning("step %d: joint jump %.1f° (chunk swap?)",
                                   step, np.rad2deg(jump))
                if max_step > 0 and jump > max_step:
                    target[ARM_JOINTS] = last_target[ARM_JOINTS] + delta * (max_step / jump)

                target = np.clip(target, limits[:, 0], limits[:, 1]).astype(np.float32)
                robot.command_joint_state(target)
                last_target = target

                need_obs = bool(resp.get("need_obs", True))
                step += 1

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    behind += 1
                    if behind % 30 == 1:
                        logger.warning("behind schedule by %.0f ms at step %d",
                                       -delay * 1e3, step)
    finally:
        logger.info("shutting down: safe hold + close(disable_motors=True)")
        try:
            robot.enter_safe_hold(reason="shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.error("enter_safe_hold failed: %s", exc)
        try:
            cameras.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("cameras.close failed: %s", exc)
        try:
            robot.close(disable_motors=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("close failed: %s", exc)

        if latencies:
            arr = np.asarray(latencies) * 1e3
            logger.info("latency: n=%d avg=%.1fms p50=%.1fms p99=%.1fms max=%.1fms",
                        arr.size, arr.mean(), np.percentile(arr, 50),
                        np.percentile(arr, 99), arr.max())
        logger.info("steps completed=%d behind_schedule=%d", step, behind)


def main():
    ap = argparse.ArgumentParser(description="Real-machine YAM bridge client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18879,
                    help="serve_policy port (project default 18879; upstream 8765)")
    ap.add_argument("--task", required=True,
                    help="VLM task instruction — MUST match the training text exactly. "
                         "For our 10-task model: "
                         "'Pick up the small square and place it in the groove' (capital P)")
    ap.add_argument("--steps", type=int, default=0, help="bounded run; 0 = unlimited")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run; 0 = unlimited (takes effect with --steps 0)")
    ap.add_argument("--fake-images", action="store_true",
                    help="synthetic zero images (protocol smoke test only)")
    ap.add_argument("--mock-robot", action="store_true",
                    help="do NOT connect the robot (no CAN, no calibration, no motion)")
    ap.add_argument("--allow-fake-images-on-robot", action="store_true",
                    help="override the guard that forbids --fake-images without "
                         "--mock-robot (the model will command garbage)")
    ap.add_argument("--camera-config", default=None,
                    help="robot.yaml holding gello_root/camera_config/camera_names")
    ap.add_argument("--motion-src", default=None,
                    help="path to yam_motion's src dir (for YamCameras)")
    ap.add_argument("--max-joint-step-deg", type=float, default=0.0,
                    help="optional per-step arm slew clamp in degrees; 0 = off")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.fake_images and not args.mock_robot and not args.allow_fake_images_on_robot:
        ap.error("--fake-images 只用于协议测试；真机上合成图会让模型输出乱动作。"
                 "请同时加 --mock-robot，或显式加 --allow-fake-images-on-robot")
    if not args.fake_images:
        missing = [n for n, v in (("--camera-config", args.camera_config),
                                  ("--motion-src", args.motion_src)) if not v]
        if missing:
            ap.error(f"真机模式需要 {' 和 '.join(missing)}")
    if args.fake_images:
        logger.warning("fake images: model output is NOT meaningful — protocol test only")

    asyncio.run(run(args.host, args.port, args.task, args.steps, args.duration,
                    args.fake_images, args.mock_robot, args.camera_config,
                    args.motion_src, args.max_joint_step_deg, args.verbose))


if __name__ == "__main__":
    main()
