#!/usr/bin/env python3
"""Self-test for ``vla_client``'s pure functions — no gello, no cameras, no GPU.

Runs as a plain script (``python3 test_vla_client.py``) and also under pytest.

Covers exactly the parts that are cheap to get wrong and expensive to discover on
the robot:
  * 14-dim identity layout (left_arm 6 / left_gripper 1 / right_arm 6 / right_gripper 1)
  * missing-key hold semantics (AR head drops groups; must hold, never zero-fill)
  * per-part dim validation
  * obs dict shape/keys/frequency
  * MockRobot satisfies the same surface the loop uses
  * msgpack ndarray round-trip (skipped if msgpack is absent)
"""

from __future__ import annotations

import sys

import numpy as np

import vla_client as vc


def _fail(msg: str) -> None:
    raise AssertionError(msg)


# ── split_state ──────────────────────────────────────────────────────────────

def test_split_state_identity():
    s = np.arange(14, dtype=np.float32)
    parts = vc.split_state(s)
    assert list(parts) == vc.ACTION_KEYS, list(parts)
    np.testing.assert_array_equal(parts["left_arm"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(parts["left_gripper"], [6])
    np.testing.assert_array_equal(parts["right_arm"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_array_equal(parts["right_gripper"], [13])
    for k, v in parts.items():
        if v.dtype != np.float32:
            _fail(f"{k} dtype {v.dtype} != float32")


def test_split_state_bad_len():
    try:
        vc.split_state(np.zeros(13, dtype=np.float32))
    except ValueError:
        return
    _fail("split_state accepted a 13-dim vector")


# ── action_to_target ─────────────────────────────────────────────────────────

def test_action_to_target_full():
    last = np.full(14, -1.0, dtype=np.float32)
    action = {
        "left_arm": np.arange(6, dtype=np.float32),
        "left_gripper": np.array([0.5], np.float32),
        "right_arm": np.arange(6, 12, dtype=np.float32),
        "right_gripper": np.array([0.9], np.float32),
    }
    out = vc.action_to_target(action, last)
    assert out.shape == (14,), out.shape
    np.testing.assert_allclose(out[:6], np.arange(6))
    np.testing.assert_allclose(out[6], 0.5, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[7:13], np.arange(6, 12))
    np.testing.assert_allclose(out[13], 0.9, rtol=0, atol=1e-6)


def test_action_to_target_holds_missing_group():
    """The critical safety rule: a dropped group must HOLD, not become 0."""
    last = np.zeros(14, dtype=np.float32)
    last[13] = np.float32(0.42)   # last commanded right gripper
    last[6] = np.float32(0.77)    # last commanded left gripper
    action = {
        "left_arm": np.ones(6, np.float32),
        "right_arm": np.ones(6, np.float32),
        # both grippers absent, mimicking _absent_keys
    }
    out = vc.action_to_target(action, last)
    # must equal the held float32 values bit-for-bit (not re-rounded, not zeroed)
    if out[13] != last[13]:
        _fail(f"right_gripper held {out[13]}, expected {last[13]} (must HOLD, not zero)")
    if out[6] != last[6]:
        _fail(f"left_gripper held {out[6]}, expected {last[6]}")
    np.testing.assert_allclose(out[:6], 1.0)
    np.testing.assert_allclose(out[7:13], 1.0)


def test_action_to_target_rejects_bad_dim():
    last = np.zeros(14, dtype=np.float32)
    try:
        vc.action_to_target({"left_arm": np.zeros(5, np.float32)}, last)
    except ValueError:
        return
    _fail("action_to_target accepted a 5-dim left_arm")


def test_action_to_target_empty_dict_holds_everything():
    last = np.arange(14, dtype=np.float32)
    out = vc.action_to_target({}, last)
    np.testing.assert_allclose(out, last)


# ── build_obs ────────────────────────────────────────────────────────────────

def test_build_obs():
    state = np.arange(14, dtype=np.float32)
    images = {k: np.zeros(s, np.uint8) for k, s in vc.IMAGE_SPECS.items()}
    obs = vc.build_obs(state, images, "Pick up the small square", 30)
    assert obs["embodiment_type"] == "galaxea_r1lite"
    assert obs["frequency"] == 30.0 and isinstance(obs["frequency"], float)
    assert obs["task"] == "Pick up the small square"
    assert set(obs["images"]) == {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"}
    for k, v in obs["images"].items():
        if v.shape != (3, 480, 640):
            _fail(f"{k} shape {v.shape} != (3,480,640)")
        if v.dtype != np.uint8:
            _fail(f"{k} dtype {v.dtype} != uint8")
    assert list(obs["state"]) == vc.ACTION_KEYS
    assert obs["state"]["right_gripper"].shape == (1,)


# ── MockRobot surface ────────────────────────────────────────────────────────

def test_mock_robot_surface():
    r = vc.MockRobot()
    assert r.num_dofs() == 14
    lim = r.get_joint_limits()
    assert lim.shape == (14, 2), lim.shape
    assert lim[13, 0] == 0.0 and lim[13, 1] == 1.0
    obs = r.get_observations()
    assert obs["joint_positions"].shape == (14,)
    assert r.get_health_status()["ok"] is True
    r.command_joint_state(np.zeros(14, np.float32))
    r.enter_safe_hold(reason="test")
    r.close(disable_motors=True)


def test_null_cameras_shape():
    c = vc.NullCameras()
    c.start()
    out = c.sample(0.0)
    assert out is not None
    images, _ts = out
    for k, v in images.items():
        if v.shape != (3, 480, 640) or v.dtype != np.uint8:
            _fail(f"{k}: {v.shape} {v.dtype}")
    c.close()


# ── msgpack codec (optional) ─────────────────────────────────────────────────

def test_codec_roundtrip():
    payload = {
        "action": {
            "left_arm": np.arange(6, dtype=np.float32),
            "left_gripper": np.array([0.25], np.float32),
        },
        "need_obs": True,
        "n": 3,
    }
    back = vc.unpackb(vc.packb(payload))
    assert back["need_obs"] is True and back["n"] == 3
    np.testing.assert_array_equal(back["action"]["left_arm"],
                                  payload["action"]["left_arm"])
    if back["action"]["left_arm"].dtype != np.float32:
        _fail("dtype lost in msgpack round-trip")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
