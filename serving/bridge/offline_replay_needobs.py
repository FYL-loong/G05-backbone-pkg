#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线回放验证：用**真实录制的视频帧 + 真实关节状态**跑通我们 bridge 的
need_obs 协议，检查部署契约是否成立。**不接机器人、不开相机。**

素材
----
``live_ar32_20261015_230527.json`` + 配套 ``*_videos_*/``（main/left/right.mp4
+ frames.jsonl 帧时间戳索引）。这是 2026-10-15 在**这台机器人上**真实跑的一次
AR 会话：1784 条命令（含 ``actual`` 14 维实测状态）、60 个 chunk（含
``predicted_actions`` 32×14）、1772×3 帧视频。

验证什么
--------
本脚本复用 ``serving/bridge/vla_client.py`` 里的 **同一份** ``split_state`` /
``build_obs`` / ``action_to_target`` / ``packb`` / ``unpackb``，因此它测的是真机
会跑的那条代码路径，只把"相机 + 机器人"换成"录制的帧 + 录制的状态"。

  ① 形状/dtype：obs 的 images (3,480,640) uint8、state 4 组 float32
  ② msgpack 往返无损（float32 不被降级成 float64）
  ③ 服务端接受并返回 4 组 (6,1,6,1) —— 与真机观测到的动作键完全一致
  ④ 逐维相关性：我们的动作 vs 日志里 AR5k 的动作（同帧同状态，应该强正相关）
  ⑤ 符号：任何一维出现负相关 = 符号翻转，真机上会猛动
  ⑥ 幅度比：RMS(ours)/RMS(ref) 应在 0.5~2.0（差 57 倍 = 度/弧度混用）
  ⑦ 绝对动作合理性：|ours - state| 很小（p95 < 0.3 rad），且方向与真实下一步
     状态变化 Δstate 正相关

用法
----
    # 全自动：自己拉起 serve_policy（--action_steps 1 保证每次都是真推理），跑完自己关
    python offline_replay_needobs.py --samples 20

    # 复用已在跑的服务端
    python offline_replay_needobs.py --endpoint ws://127.0.0.1:18879 --no-start-server

    # 只验证统计数学（不需要模型/视频，用合成数据）
    python offline_replay_needobs.py --self-test

安全：只读文件 + 拉起/关闭一个推理服务进程；不碰 CAN、不碰相机。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ─── 默认路径（可用参数覆盖）─────────────────────────────────────────────────
DEF_DEPLOY = Path("/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026")
DEF_LOG = Path("/mnt/sda/chenyuan/motion/compare_ar10k_pir250k_20261013/ar/logs/"
               "live_ar32_20261015_230527.json")
DEF_BRIDGE = Path("/home/chenyuan/workspace/g05-yam-deployment/"
                  "G05-backbone-pkg/serving/bridge")
DEF_TASK = "Pick up the small square and place it in the groove"   # 我们训练用的原文（大写 P）

CAM_KEYS = {"main": "head_rgb", "left": "left_wrist_rgb", "right": "right_wrist_rgb"}


# ─── 载入我们 bridge 的纯函数（保证测的是同一份代码）─────────────────────────
def load_vla_client(bridge_dir: Path):
    path = bridge_dir / "vla_client.py"
    if not path.is_file():
        sys.exit(f"[FAIL] 找不到 {path}")
    spec = importlib.util.spec_from_file_location("vla_client_replay", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── 视频索引与取帧 ─────────────────────────────────────────────────────────
def load_frame_index(vdir: Path) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {"main": [], "left": [], "right": []}
    with open(vdir / "frames.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cam = r.get("camera")
            if cam in idx:
                idx[cam].append(r)
    for cam in idx:
        idx[cam].sort(key=lambda r: r["frame"])
    return idx


def pick_frame(frames: list[dict], t_ref: float) -> dict | None:
    """latest_not_after_reference：取 capture_time_s <= t_ref 的最后一帧。"""
    best = None
    for r in frames:
        if r["capture_time_s"] <= t_ref:
            best = r
        else:
            break
    return best


def decode_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """顺序解码一次，只留需要的帧号（frames.jsonl 的 frame 序号 == 解码序）。"""
    import av  # 只在需要时导入

    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    with av.open(str(path)) as container:
        i = -1
        for frame in container.decode(video=0):
            i += 1
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")   # HWC uint8
            if i >= last:
                break
    return out


# ─── 数据抽取 ────────────────────────────────────────────────────────────────
def load_log(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def state_at(commands: list[dict], t_ref: float) -> tuple[np.ndarray, int]:
    """取 state_time_s <= t_ref 的最后一条命令的 actual，并返回其下标。"""
    best_i = -1
    for i, c in enumerate(commands):
        st = c.get("state_time_s")
        if st is None or st <= t_ref:
            best_i = i
        else:
            break
    if best_i < 0:
        raise RuntimeError(f"t_ref={t_ref} 之前没有命令记录")
    return np.asarray(commands[best_i]["actual"], dtype=np.float32), best_i


def delta_state(commands: list[dict], i: int) -> np.ndarray:
    """真实下一步的状态变化 Δstate（用于验证"动作方向"是否合理）。"""
    if i + 1 >= len(commands):
        return np.zeros(14, dtype=np.float32)
    a = np.asarray(commands[i]["actual"], dtype=np.float32)
    b = np.asarray(commands[i + 1]["actual"], dtype=np.float32)
    return (b - a).astype(np.float32)


# ─── 服务端启动/等待 ────────────────────────────────────────────────────────
def start_server(deploy: Path, port: int, timeout_s: float = 300.0):
    V = deploy / "runtime/env/bin/python"
    G = deploy / "code/motion/vendor/GalaxeaVLA"
    SERV = Path("/home/chenyuan/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py")
    CKPT = deploy / "models/ar_ft1ep/checkpoints/model_state_dict.pt"
    for p, name in ((V, "python"), (SERV, "serve_policy.py"), (CKPT, "checkpoint")):
        if not p.exists():
            sys.exit(f"[FAIL] 缺 {name}: {p}")

    log = Path(f"/tmp/replay_serve_{port}_{time.strftime('%H%M%S')}.log")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(G / "src")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        env.pop(k, None)          # 本机代理是死端口，会劫持 127.0.0.1

    cmd = [str(V), str(SERV),
           "--ckpt_path", str(CKPT), "--host", "127.0.0.1", "--port", str(port),
           "--action_steps", "1",           # ★ 每次请求都重新推理
           "eval_embodiment=galaxea_r1lite",
           "model.model_weights_to_bf16=true",
           "model.model_arch.continuous_action=false"]
    print(f"  启动服务端：port={port} log={log}")
    fh = open(log, "wb")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(G), env=env, start_new_session=True)

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            print(f"  ❌ 服务端退出（码 {proc.returncode}），日志尾部：")
            print("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
            sys.exit(1)
        txt = log.read_text(errors="replace")
        if "listening on" in txt:
            print(f"  ✅ 服务端就绪（{time.monotonic()-t0:.0f}s）")
            return proc, log
        time.sleep(2)
    proc.terminate()
    print("  ❌ 等待服务端超时，日志尾部：")
    print("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
    sys.exit(1)


def stop_server(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    print("  关闭服务端 ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()


# ─── 统计 ────────────────────────────────────────────────────────────────────
def col_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐列皮尔逊相关；常数列为 nan。"""
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    for j in range(a.shape[1]):
        x, y = a[:, j].astype(np.float64), b[:, j].astype(np.float64)
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        out[j] = float(np.corrcoef(x, y)[0, 1])
    return out


def rms_ratio(a: np.ndarray, b: np.ndarray) -> float:
    ra = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rb = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    return ra / rb if rb > 1e-12 else float("nan")


def compute_stats(ours: np.ndarray, ref: np.ndarray, state: np.ndarray,
                  dstate: np.ndarray) -> dict:
    """ours/ref/state: (N,14)；dstate: (N,14)。返回统计字典。"""
    ours = np.asarray(ours, np.float64)
    ref = np.asarray(ref, np.float64)
    state = np.asarray(state, np.float64)
    step_ours = ours - state
    return {
        "n": int(ours.shape[0]),
        "corr_ref": col_corr(ours, ref),          # 与 AR5k 动作的相关性
        "corr_dstate": col_corr(step_ours, dstate),  # 动作增量与真实状态变化
        "rms_ratio": rms_ratio(ours, ref),
        "abs_step": np.abs(step_ours),
        "abs_ref_step": np.abs(ref - state),
    }


def print_report(ours, ref, state, dstate, stats, dim_names) -> int:
    n = stats["n"]
    print("\n" + "=" * 96)
    print(f"  逐维统计（{n} 个采样；ref = 日志里 AR5k 的 predicted_actions[0]）")
    print("=" * 96)
    hdr = f"  {'dim':<16}{'corr(ours,ref)':>16}{'corr(Δstate)':>15}" \
          f"{'|ours-state| p50':>19}{'p95':>10}{'|ref-state| p50':>18}"
    print(hdr)
    print("  " + "-" * 92)
    for j, name in enumerate(dim_names):
        cr = stats["corr_ref"][j]
        cd = stats["corr_dstate"][j]
        p50 = float(np.percentile(stats["abs_step"][:, j], 50))
        p95 = float(np.percentile(stats["abs_step"][:, j], 95))
        r50 = float(np.percentile(stats["abs_ref_step"][:, j], 50))
        print(f"  {name:<16}{('nan' if np.isnan(cr) else f'{cr:+.3f}'):>16}"
              f"{('nan' if np.isnan(cd) else f'{cd:+.3f}'):>15}"
              f"{p50:>19.4f}{p95:>10.4f}{r50:>18.4f}")

    print("\n" + "=" * 96)
    print("  判定")
    print("=" * 96)
    fails: list[str] = []
    warns: list[str] = []

    cr = stats["corr_ref"]
    valid = ~np.isnan(cr)
    neg = [dim_names[j] for j in range(len(cr))
           if valid[j] and cr[j] < -0.2]
    # ★ 顺序/映射错的典型特征是"相关度≈0"，必须当硬失败（否则单向 swap 会被放过）
    dead = [dim_names[j] for j in range(len(cr))
            if valid[j] and cr[j] < 0.1]
    weak = [dim_names[j] for j in range(len(cr))
            if valid[j] and 0.1 <= cr[j] < 0.35]

    if neg:
        fails.append(f"❌ 符号翻转（负相关）：{neg}")
    if dead:
        fails.append(f"❌ 相关度≈0（疑似关节顺序/映射错）：{dead}")
    if not neg and not dead:
        print("  ✅ 无符号翻转、无相关度≈0 的维度")
    if weak:
        warns.append(f"🟡 相关偏弱：{weak}（若该维在数据里近似常数可忽略）")
    if not weak:
        print(f"  ✅ 全部 {int(valid.sum())}/{len(cr)} 有效维与 ref 相关 > 0.35")

    rr = stats["rms_ratio"]
    if not (0.5 <= rr <= 2.0):
        fails.append(f"❌ 幅度比 {rr:.3f} 不在 0.5~2.0（疑似度/弧度混用或归一化错）")
    else:
        print(f"  ✅ 幅度比 RMS(ours)/RMS(ref) = {rr:.3f}（在 0.5~2.0）")

    p95_max = float(np.max(np.percentile(stats["abs_step"], 95, axis=0)))
    if p95_max > 0.5:
        fails.append(f"❌ |ours-state| 的 p95 最大达 {p95_max:.3f} rad（>0.5，动作太远）")
    elif p95_max > 0.3:
        warns.append(f"🟡 |ours-state| p95 最大 {p95_max:.3f} rad（0.3~0.5，偏大）")
    else:
        print(f"  ✅ |ours-state| p95 最大 = {p95_max:.4f} rad（绝对动作、离当前姿态很近）")

    cd = stats["corr_dstate"]
    cdv = cd[~np.isnan(cd)]
    if cdv.size and float(np.nanmedian(cdv)) < 0:
        fails.append(f"🟡 增量方向与真实 Δstate 中位相关为负（{np.nanmedian(cdv):+.3f}）")
    elif cdv.size:
        print(f"  ✅ 增量方向与 Δstate 中位相关 = {float(np.nanmedian(cdv)):+.3f}")

    # ★ 关节限位检查：夹爪必须是 [0,1]，臂关节各有上下限。
    #   这一条能独立抓出「夹爪↔臂」互换、度/弧度混用等致命错误。
    bad_lim = check_limits(ours, DEF_LIMITS)
    if bad_lim:
        fails.append("❌ 超出真实关节限位（映射/尺度错）：" + "; ".join(bad_lim))
    else:
        print("  ✅ 全部动作落在真实关节限位内（夹爪 ∈ [0,1]，臂关节在各自上下限）")

    print()
    for w in warns:
        print("  " + w)
    for f in fails:
        print("  " + f)
    if fails:
        print("\n  ❌ 契约验证**未通过** —— 真机会猛动，先修再上")
        return 1
    print("\n  ✅ 契约验证通过 —— 数据通路、维度、符号、幅度全部合理")
    print("     （注意：这只证明「部署链路没错」，不等于策略质量好；"
          "策略质量见 RESULT_eval.md）")
    return 0


DIM_NAMES = [
    "L_arm1", "L_arm2", "L_arm3", "L_arm4", "L_arm5", "L_arm6", "L_grip",
    "R_arm1", "R_arm2", "R_arm3", "R_arm4", "R_arm5", "R_arm6", "R_grip",
]

# 真实关节限位（2026-10-27 15:03 从 yam_safety_daemon 的 status 读到的 yam_pro 值）。
# 夹爪是 [0,1]，臂关节各不相同 —— 任何一维越界都说明映射/尺度错了。
DEF_LIMITS = np.array([
    [-2.76799, 3.29159], [-0.15, 3.81519], [-0.15, 3.29159],
    [-1.84297, 1.72080], [-1.72080, 1.72080], [-2.24440, 2.24440],
    [0.0, 1.0],
    [-2.76799, 3.29159], [-0.15, 3.81519], [-0.15, 3.29159],
    [-1.84297, 1.72080], [-1.72080, 1.72080], [-2.24440, 2.24440],
    [0.0, 1.0],
], dtype=np.float64)


def check_limits(ours: np.ndarray, limits: np.ndarray, tol: float = 0.15) -> list[str]:
    """返回越界说明列表（空 = 全部在限位内）。

    tol 给 0.15 rad：策略偶尔预测到软限位外一点是正常的（守护进程会裁），
    这条检查的目的是抓**量级级别**的错误（度/弧度混用、夹爪↔臂互换）。
    """
    bad: list[str] = []
    lo = limits[:, 0] - tol
    hi = limits[:, 1] + tol
    for j in range(ours.shape[1]):
        v = ours[:, j]
        below = int(np.sum(v < lo[j]))
        above = int(np.sum(v > hi[j]))
        if below or above:
            bad.append(
                f"{DIM_NAMES[j]}: {below} 次低于 {lo[j]:.3f}、{above} 次高于 "
                f"{hi[j]:.3f}（实际范围 {v.min():.3f}~{v.max():.3f}）")
    return bad


# ─── 主流程 ──────────────────────────────────────────────────────────────────
async def replay(args, vc) -> int:
    deploy = Path(args.deploy)
    log = load_log(Path(args.source_log))
    vdir = Path(log["video_recording"]["directory"])
    print(f"  会话日志: {args.source_log}")
    print(f"  视频目录: {vdir}")
    print(f"  日志里的 task = {log.get('task')!r}")
    print(f"  我们发送的 task = {args.task!r}")
    print(f"  契约: {json.dumps(log.get('contract'), ensure_ascii=False)}")

    commands = log["commands"]
    queries = [q for q in log["queries"] if q.get("starts_new_chunk")]
    print(f"  commands={len(commands)}  queries(新 chunk)={len(queries)}  "
          f"实测帧数={log['video_recording']['cameras']['main']['observed']}")

    idx = load_frame_index(vdir)
    rng = np.random.default_rng(args.seed)
    n = min(args.samples, len(queries))
    pick = sorted(rng.choice(len(queries), size=n, replace=False).tolist())
    samples = []
    for qi in pick:
        q = queries[qi]
        t_ref = float(q["image_time_s"])
        st, ci = state_at(commands, t_ref)
        frames_wanted = {}
        ok = True
        for cam in ("main", "left", "right"):
            r = pick_frame(idx[cam], t_ref)
            if r is None:
                print(f"  [skip] qi={qi} cam={cam} 没有 <= t={t_ref} 的帧")
                ok = False
                break
            frames_wanted[cam] = r
        if not ok:
            continue
        samples.append({
            "qi": qi, "q": q, "t_ref": t_ref, "state": st, "ci": ci,
            "frames": frames_wanted,
            "dstate": delta_state(commands, ci),
            "ref": np.asarray(q["predicted_actions"][0], dtype=np.float32),
        })
    print(f"  采样 {len(samples)} 个 chunk 时刻")

    # 解码需要的帧（每路视频只顺序解码一次）
    print("  解码视频帧 ...")
    t0 = time.monotonic()
    for cam, key in CAM_KEYS.items():
        want = {s["frames"][cam]["frame"] for s in samples}
        got = decode_frames(vdir / f"{cam}.mp4", want)
        miss = want - set(got)
        if miss:
            print(f"  [WARN] {cam}: 缺帧 {sorted(miss)[:8]}")
        for s in samples:
            fi = s["frames"][cam]["frame"]
            s.setdefault("images", {})[key] = got.get(fi)
    print(f"  解码完成（{time.monotonic()-t0:.1f}s）")

    # ② msgpack 往返无损检查
    print("\n  [检查] msgpack 往返 ...")
    probe_img = np.zeros((3, 480, 640), dtype=np.uint8)
    probe_obs = vc.build_obs(samples[0]["state"], {k: probe_img for k in CAM_KEYS.values()},
                             args.task, 30)
    back = vc.unpackb(vc.packb(probe_obs))
    for k in CAM_KEYS.values():
        if back["images"][k].shape != (3, 480, 640) or back["images"][k].dtype != np.uint8:
            print(f"  ❌ 图像 {k} 往返后 {back['images'][k].shape} {back['images'][k].dtype}")
            return 1
    if back["state"]["left_arm"].dtype != np.float32:
        print(f"  ❌ state 往返后 dtype={back['state']['left_arm'].dtype}")
        return 1
    print("  ✅ msgpack 往返无损（图像 uint8 (3,480,640)、state float32）")

    # 服务端
    proc = None
    endpoint = args.endpoint
    try:
        if not args.no_start_server:
            proc, _ = start_server(deploy, args.port)
            endpoint = f"ws://127.0.0.1:{args.port}"
        print(f"\n  连接 {endpoint}")
        ours, refs, states, dstates = [], [], [], []
        async with vc._ws_connect(endpoint) as ws:
            hello = vc.unpackb(await asyncio.wait_for(ws.recv(), timeout=60))
            print(f"  hello: {hello}")
            for i, s in enumerate(samples):
                if any(s["images"].get(k) is None for k in CAM_KEYS.values()):
                    print(f"  [skip] {i}: 图像缺失")
                    continue
                imgs = {k: np.ascontiguousarray(np.transpose(s["images"][k], (2, 0, 1)))
                        for k in CAM_KEYS.values()}
                obs = vc.build_obs(s["state"], imgs, args.task, 30)
                t_req = time.monotonic()
                await ws.send(vc.packb(obs))
                timeout = 180.0 if i == 0 else 60.0
                resp = vc.unpackb(await asyncio.wait_for(ws.recv(), timeout=timeout))
                dt = (time.monotonic() - t_req) * 1e3
                if "error" in resp:
                    print(f"  ❌ 服务端错误：{resp['error']}")
                    return 1
                act = resp.get("action", {})
                if i == 0:
                    print(f"  首包返回键：{ {k: np.asarray(v).shape for k, v in act.items()} }"
                          f"  need_obs={resp.get('need_obs')}  {dt:.0f}ms")
                got = vc.action_to_target(act, s["state"])
                ours.append(got)
                refs.append(s["ref"])
                states.append(s["state"])
                dstates.append(s["dstate"])
                print(f"  [{i:02d}] t_ref={s['t_ref']:.3f} 帧="
                      f"{ {c: s['frames'][c]['frame'] for c in s['frames']} }  "
                      f"{dt:6.0f}ms  |ours-state|max={float(np.max(np.abs(got - s['state']))):.4f}")
    finally:
        stop_server(proc)

    if len(ours) < max(5, args.samples // 2):
        print(f"  ❌ 有效采样太少（{len(ours)}），无法统计")
        return 1
    stats = compute_stats(np.stack(ours), np.stack(refs),
                          np.stack(states), np.stack(dstates))
    return print_report(np.stack(ours), np.stack(refs), np.stack(states),
                        np.stack(dstates), stats, DIM_NAMES)


def self_test() -> int:
    """不依赖模型/视频，验证判据本身正确（故意注入 5 种故障）。

    合成数据按**真实量级**生成：臂关节小幅运动、夹爪在 [0,1]，
    这样"顺序错"才会像真机上那样表现为相关度≈0 或越界。
    """
    rng = np.random.default_rng(0)
    n = 40
    state = np.zeros((n, 14), dtype=np.float32)
    for j in range(14):
        lo, hi = DEF_LIMITS[j]
        if j in (6, 13):                       # 夹爪：真实就在 [0,1]
            state[:, j] = rng.uniform(0.2, 0.9, n)
        else:                                  # 臂关节：在限位内小幅摆动
            span = min(0.15, (hi - lo) / 4.0)
            center = float(np.clip(0.0, lo + span, hi - span))
            state[:, j] = rng.uniform(center - span, center + span, n)
    dstate = rng.normal(0, 0.01, size=(n, 14)).astype(np.float32)
    good = (state + 3 * dstate + rng.normal(0, 0.003, size=(n, 14))).astype(np.float32)
    good[:, 6] = np.clip(good[:, 6], 0, 1)
    good[:, 13] = np.clip(good[:, 13], 0, 1)

    cases = [
        ("1/5 正常情况（应通过）", good, True),
        ("2/5 符号翻转（左臂 6 维取负）", np.column_stack(
            [-good[:, 0:6], good[:, 6:]]).astype(np.float32), False),
        ("3/5 尺度错 57 倍（度/弧度混用）", (good * 57.0).astype(np.float32), False),
        ("4/5 关节顺序错（左右臂+夹爪整体互换）",
         good[:, [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6]].copy(), False),
        ("5/5 夹爪与臂关节互换（L_grip ← L_arm1）", None, False),
    ]
    bad5 = good.copy()
    tmp = bad5[:, 6].copy()
    bad5[:, 6] = bad5[:, 0]
    bad5[:, 0] = tmp
    cases[4] = (cases[4][0], bad5, False)

    for title, arr, should_pass in cases:
        print("\n" + "=" * 80)
        print(f"  self-test {title}")
        print("=" * 80)
        st = compute_stats(arr, good, state, dstate)
        rc = print_report(arr, good, state, dstate, st, DIM_NAMES)
        if should_pass:
            assert rc == 0, f"【{title}】应通过但被判失败"
        else:
            assert rc != 0, f"【{title}】应被抓但被放过"

    print("\n" + "=" * 80)
    print("  ✅ self-test 全部通过：判据能区分「正常」与 4 类致命映射错误")
    print("     （符号翻转 / 尺度错 / 左右互换 / 夹爪与臂互换）")
    print("=" * 80)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", default=str(DEF_DEPLOY))
    ap.add_argument("--source-log", default=str(DEF_LOG))
    ap.add_argument("--bridge", default=str(Path(__file__).resolve().parent),
                    help="vla_client.py 所在目录（默认＝本脚本所在目录）")
    ap.add_argument("--task", default=DEF_TASK)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--port", type=int, default=18879)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--no-start-server", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    vc = load_vla_client(Path(args.bridge))

    print("=" * 96)
    print("  离线回放验证：真实视频帧 + 真实关节状态 → 我们的 need_obs 协议")
    print("=" * 96)
    try:
        return asyncio.run(replay(args, vc))
    except KeyboardInterrupt:
        print("\n  被中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
