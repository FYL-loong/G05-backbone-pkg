#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``offline_replay_needobs.py`` 的本地回归测试（不需要模型、视频、机器人）。

覆盖：
  * ``state_at`` 的三种情形（正常命中 / t_ref 早于所有命令 → clamped / 精确边界）
  * ``pick_frame`` 的 latest_not_after_reference 语义
  * ``check_limits`` 对夹爪 [0,1] 与臂关节上下限的判断
  * **``replay()`` 的 --dry-run 全流程**（用合成日志 + 打桩解码器），
    这条能抓到"变量未定义 / 字段名写错 / 循环漏赋值"这类只在真环境才暴露的错

跑法：
    python test_offline_replay.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(mod_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load(HERE / "offline_replay_needobs.py", "offline_replay_needobs")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


# ─── 合成日志 ───────────────────────────────────────────────────────────────
T0 = 1000.0          # 会话起点（host_monotonic 秒）
N_CMD = 200          # 200 条命令 × 33ms ≈ 6.6s
N_QRY = 12
FPS = 30
N_FRAMES = 200


def make_fake_log() -> dict:
    rng = np.random.default_rng(7)
    commands = []
    pos = np.zeros(14, dtype=np.float64)
    for i in range(N_CMD):
        pos = pos + rng.normal(0, 0.004, 14)
        t = T0 + 0.05 + i * (1.0 / FPS)        # 第一条命令在第一个帧之后
        commands.append({
            "time_s": t,
            "state_time_s": t,
            "image_time_s": t - 0.01,
            "actual": pos.tolist(),
            "requested": pos.tolist(),
            "policy_target": pos.tolist(),
            "accepted": pos.tolist(),
        })
    queries = []
    for k in range(N_QRY):
        i = k * (N_CMD // N_QRY)
        q_t = T0 + i * (1.0 / FPS)             # 从第一个帧时刻开始（会触发 clamped）
        chunk = []
        base = np.asarray(commands[min(i, N_CMD - 1)]["actual"], dtype=np.float64)
        for s in range(32):
            chunk.append((base + rng.normal(0, 0.01, 14)).tolist())
        queries.append({
            "starts_new_chunk": True,
            "image_time_s": q_t,
            "state_time_s": q_t + 0.01,
            "returned_steps": 32,
            "predicted_actions": chunk,
            "requested_steps": 32,
        })
    return {
        "task": "pick up the small square and place it in the groove",
        "contract": {"dt_s": 0.04, "horizon_steps": 32, "start_offset_steps": 0,
                     "representation": "absolute_joint_position",
                     "action_model": "test",
                     "cameras": ["main", "left", "right"],
                     "image_alignment": "latest_not_after_reference",
                     "state_alignment": "latest_not_after_reference"},
        "video_recording": {
            "directory": str(HERE / "_fake_videos"),
            "cameras": {"main": {"observed": N_FRAMES},
                        "left": {"observed": N_FRAMES},
                        "right": {"observed": N_FRAMES}},
        },
        "commands": commands,
        "queries": queries,
    }


def make_fake_index() -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for cam, off in (("main", 0.0), ("left", 0.004), ("right", 0.012)):
        idx[cam] = [
            {"camera": cam, "frame": i,
             "capture_time_s": T0 + off + i * (1.0 / FPS), "pts": int(i * 1e6 / FPS)}
            for i in range(N_FRAMES)
        ]
    return idx


# ─── 单元测试 ───────────────────────────────────────────────────────────────
def test_state_at() -> None:
    log = make_fake_log()
    cmds = log["commands"]

    st, i, clamped = m.state_at(cmds, T0 + 0.5)
    check("state_at 正常命中", (not clamped) and i >= 0 and st.shape == (14,))

    st, i, clamped = m.state_at(cmds, T0 - 100.0)
    check("state_at t_ref 早于所有命令 → clamped 退回首条",
          clamped and i == 0, f"clamped={clamped} i={i}")

    # 精确边界：t 正好等于某条命令的 state_time_s
    j = 50
    st, i, clamped = m.state_at(cmds, cmds[j]["state_time_s"])
    check("state_at 边界（等于某条 state_time_s）应命中该条",
          (not clamped) and i == j, f"i={i} 期望 {j}")


def test_pick_frame() -> None:
    idx = make_fake_index()
    frames = idx["main"]
    r = m.pick_frame(frames, frames[10]["capture_time_s"])
    check("pick_frame 精确命中第 10 帧", r is not None and r["frame"] == 10)

    r = m.pick_frame(frames, frames[10]["capture_time_s"] + 1e-6)
    check("pick_frame 略晚仍是第 10 帧", r is not None and r["frame"] == 10)

    r = m.pick_frame(frames, frames[10]["capture_time_s"] - 1e-6)
    check("pick_frame 略早应退到第 9 帧（因果语义）",
          r is not None and r["frame"] == 9)

    r = m.pick_frame(frames, frames[0]["capture_time_s"] - 1.0)
    check("pick_frame 早于首帧 → None", r is None)


def test_check_limits() -> None:
    ok = np.zeros((5, 14), dtype=np.float64)
    ok[:, 6] = 0.5
    ok[:, 13] = 0.5
    check("check_limits 合规数据无告警", m.check_limits(ok, m.DEF_LIMITS) == [])

    bad = ok.copy()
    bad[:, 6] = 57.0            # 夹爪飞出 [0,1]
    check("check_limits 抓到夹爪越界", len(m.check_limits(bad, m.DEF_LIMITS)) >= 1)

    bad2 = ok.copy()
    bad2[:, 0] = -10.0          # 臂关节越界
    check("check_limits 抓到臂关节越界", len(m.check_limits(bad2, m.DEF_LIMITS)) >= 1)


def test_replay_dry_run() -> None:
    """跑通 replay() 的 --dry-run 全流程（打桩日志/索引/解码器）。"""
    fake_log = make_fake_log()
    fake_idx = make_fake_index()
    real_decode = m.decode_frames

    calls = {"decode": 0}

    def fake_decode(path, wanted):
        calls["decode"] += 1
        return {i: np.full((480, 640, 3), 60 + i % 100, dtype=np.uint8)
                for i in wanted}

    m.load_log = lambda p: fake_log
    m.load_frame_index = lambda vdir: fake_idx
    m.decode_frames = fake_decode
    try:
        args = SimpleNamespace(
            deploy=str(HERE), source_log="(fake)", bridge=str(HERE),
            task="Pick up the small square and place it in the groove",
            samples=N_QRY, port=0, endpoint=None, no_start_server=True,
            seed=0, dry_run=True, self_test=False)
        vc = m.load_vla_client(HERE)
        rc = asyncio.run(m.replay(args, vc))
        check("replay(--dry-run) 返回 0", rc == 0, f"rc={rc}")
        check("replay(--dry-run) 解了 3 路视频", calls["decode"] == 3,
              f"decode 调用 {calls['decode']} 次")
    finally:
        m.decode_frames = real_decode


def main() -> int:
    print("=" * 70)
    print("  offline_replay_needobs 本地回归测试")
    print("=" * 70)
    for fn in (test_state_at, test_pick_frame, test_check_limits,
               test_replay_dry_run):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"  ❌ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("  ✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
