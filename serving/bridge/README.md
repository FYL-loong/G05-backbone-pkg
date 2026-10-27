# serving/bridge/ —— 真机桥接客户端

把 **相机 + 关节状态** 送进 `serve_policy`（WebSocket + msgpack），再把返回的
**绝对关节目标** 下发给 YAM 双臂。

---

## 1. 来源与改动

| 文件 | 来源 |
|---|---|
| `vla_client.py` | **张恩同原作**（实验机 `g05-yam-deployment/bridge/vla_client.py`，2026-10-26）+ 我们打的 **8 个补丁** |
| `test_vla_client.py` | 我们补的纯函数自测（10 项，无需 gello / 相机 / GPU） |
| `smoke_test.sh` | 一键端到端冒烟（起服务端 → fake client → 我们的 bridge → 自动收尾） |

### 打过的 8 个补丁

| # | 原问题 | 修法 | 严重性 |
|---|---|---|---|
| **1** | `capture_images()` 是 `NotImplementedError`，真机必崩 | 复用 `yam_motion.cameras.YamCameras`（已验证：**只取 host_ts ≤ 查询时刻的因果帧**，超 `max_age_s` 拒绝） | 🔴 阻塞 |
| **2** | `--fake-images` 仍会 `build_robot()`，而 `create_pair` **做夹爪标定会真的动机械臂** | 新增 `--mock-robot`；并且**禁止**「合成图 + 真机器人」组合（除非显式 `--allow-fake-images-on-robot`） | 🔴 安全 |
| **3** | 推理阻塞 ~520 ms 后 `next_tick` 落后，客户端会把缓存步**在几十毫秒内连发**（≈把 0.5 s 轨迹压缩到几十 ms） | 请求耗时 > 1 个周期就 `next_tick = t_got` **重对齐**，停住的时间**丢弃而非补放**（同 `yam_motion.g05_execution.align_chunk` 的思路） | 🔴 会导致剧烈动作 |
| **4** | `await ws.recv()` 无超时，服务端挂住会永久阻塞且不走清理 | `asyncio.wait_for(...)`；超时 → `enter_safe_hold` + 退出 | 🟡 安全 |
| **5** | `websockets.connect(uri, max_size=None)` 直连 —— 但 websockets ≥14 **会自动读 `ALL_PROXY`/`HTTPS_PROXY`**，本机跑着 clash-verge，连 `127.0.0.1` 会被劫持 | 抽出 `_ws_connect()`，显式 `proxy=None, compression=None, ping_interval=30, ping_timeout=120`，老版本不支持 `proxy` 时自动回退 | 🔴 **实测会连不上** |
| **6** | 统一 5 s 收包超时，**把首次推理掐死** —— 首次 forward 含 CUDA kernel autotune，实测 ~12 s | 首包 `FIRST_RECV_TIMEOUT_S=180`，稳态 `RECV_TIMEOUT_S=10` | 🔴 实测误触发 safe hold |
| **7** | 安全停机却返回 **退出码 0**，上层脚本把中断误判成成功 | 安全停机返回非零：2=机器人异常 / 3=服务端错误 / 4=收包超时 / 5=相机丢帧 / 6=preflight 失败 | 🟡 自动化判据 |
| **8** | `build_robot()` 直连 CAN：**抢守护进程的 CAN 总线、没有 400 ms 看门狗、没有软限位** | 新增 `--robot-backend {zmq,direct,mock}`。`zmq`（默认）走 `ZMQClientRobot`(:6001) + `SafetyClient`(:6002) + `SupervisedMotionSession`；`direct` 保留原 `create_pair` 路径供调试 | 🔴 真机安全 |

> **补丁 5 的实测证据**（本地起 WS 服务 + 把 `ALL_PROXY` 指向 `127.0.0.1:1`）：
> ```
> ① naive   websockets.connect(uri, max_size=None)  → ConnectionRefusedError（走代理）
> ② patched vla_client._ws_connect(uri)             → ✅ 直连成功，echo 收到 b'hi'
> ```
> `yam_motion/policies/g05.py` 里也显式传了 `proxy=None` —— 同一台机器上踩过的同一个坑。

### 顺带增强

- `--duration <秒>`（和 `--steps` 二选一限制时长）
- 每个 chunk 切换处**打印关节跳变**（>5° 警告），可选 `--max-joint-step-deg` 做斜率限幅
- 结束时打印延迟统计（avg / p50 / p99 / max）与 `behind_schedule` 计数
- `hello` 非 dict / 带 `error` 时给出明确报错；`action_steps > 16` 时警告窗口过窄
- 默认端口改为 **18879**（本项目部署包端口；上游默认 8765 仍可 `--port` 指定）

### **未改动**（原实现正确，经 gello 源码核实）

- msgpack 编解码与 `g05.utils.websocket` 兼容
- 14 维 **恒等映射** `left_arm(6) | left_gripper(1) | right_arm(6) | right_gripper(1)`
  （`BimanualRobot.get_observations()["joint_positions"]` 本身就是 left(7)+right(7) 拼接）
- `frequency=30.0` 显式发送
- 缺 key → **保持上一帧**（AR 头 `_absent_keys` 时不能填 0）
- 绝对动作，**不重复加 state**（`serve_policy` 的 postprocess 已做过 relative→absolute）
- `BimanualRobot` 的 7 个方法全部存在且签名吻合：
  `num_dofs`(14) / `get_joint_limits`((14,2)) / `get_observations` / `command_joint_state` /
  `get_health_status`→`{"ok","reason"}` / `enter_safe_hold(reason=)` / `close(disable_motors=)`

---

## 2. 跑之前

```bash
# 3.11 YAM 环境（gello）只差 websockets
V=/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python
$V -m pip install -q websockets
$V -c "import websockets,msgpack,numpy;print('ok')"
$V -c "import pyrealsense2 as rs;print('pyrealsense2', rs.__version__)"   # 相机需要
```

**自测（不需要机器人 / 相机 / GPU）**：

```bash
cd serving/bridge
$V test_vla_client.py        # 期望 10/10 passed
```

---

## 3. 安全冒烟（不动机械臂）

服务端（GalaxeaVLA 3.10 环境，AR-only + bf16）：

```bash
cd <GalaxeaVLA>
PYTHONPATH=src python ../g05-async-serving/serve_policy.py \
    --ckpt_path <run_dir>/checkpoints/model_state_dict.pt \
    --host 127.0.0.1 --port 18879 \
    eval_embodiment=galaxea_r1lite \
    model.model_weights_to_bf16=true \
    model.model_arch.continuous_action=false
```

客户端（**合成图 + 假机器人 = 零风险协议测试**）：

```bash
$V bridge/vla_client.py --host 127.0.0.1 --port 18879 \
    --task "Pick up the small square and place it in the groove" \
    --fake-images --mock-robot --steps 96 --verbose
```

通过标准：`need_obs` 交替出现、动作 14 维有限、`latency` 统计里能看到 chunk 切换处
的 ~520 ms 尖峰但**不出现连续几十步的无 sleep 连发**（日志里 `behind schedule` 不应疯狂刷）。

---

## 4. 真机运行（先确认急停可用）

### 4.0 先只读探测（**不改变任何状态**，别人正在用时也能跑）

```bash
$V bridge/vla_client.py --task x --check-robot
```

期望：
```
  backend      = zmq
  safety.state = 'safe_hold'
  stale_clients= [...]
  ✅ 处于 safe_hold，可以申请会话接管。
```
若 `state='active'` → **正被别人控制，不要接管**（返回码 6）。

### 4.1 走安全守护进程（推荐）

```bash
$V bridge/vla_client.py --host 127.0.0.1 --port 18879 \
    --task "Pick up the small square and place it in the groove" \
    --robot-backend zmq --robot-port 6001 --safety-port 6002 \
    --command-timing legacy_30hz \
    --camera-config <deploy>/live_configs/robot.yaml \
    --motion-src    <deploy>/code/motion/src \
    --duration 20
```

会话生命周期（由 `SupervisedMotionSession` 保证）：

```
preflight: state 必须 == 'safe_hold' 且 stale_clients 非空
  ↓
begin_session + 起 parent/collector 两个 200ms 心跳线程
  ↓
等待 stale_clients 变空 → 自动 activate()（safe_hold → active）
  ↓
主循环：每步 session.check()（心跳线程发现 fault/disarmed 立刻抛）
  ↓
退出：hold() → 守护进程确认 SAFE HOLD → 关连接
```

守护进程端有 **400 ms 看门狗**：我们的两个心跳线程只要 200 ms 内没发心跳，
守护进程自动抱死 —— 我们的进程被 kill -9 也不会留下失控的机械臂。

### 4.2 直连 CAN（仅调试，独占总线、无看门狗）

```bash
$V bridge/vla_client.py --task "..." --robot-backend direct ...
```

⚠️ 会做夹爪标定、**会真动机器人**，且与守护进程抢 CAN。只有守护进程停掉时才可用。

**安全边界**：机器人不健康 / 后端 check 抛错 / 服务端 `error` / 收包超时 / 相机丢帧
→ 一律 `enter_safe_hold()` 后 `close(disable_motors=True)`，并以非零码退出。
zmq 模式下本地不做裁剪（**软限位由守护进程执行**）；direct 模式按
`get_joint_limits()` 裁剪（每臂 6×2 + 夹爪 [0,1]）。

---

## 5. 仍需在真机上确认（代码之外的未知）

| # | 事项 |
|---|---|
| 1 | `pyrealsense2` 是否在该 venv 里（相机必需） |
| 2 | `YamCameras.start()` 能否打开 3 路（`/dev/video*` 未被 6001/6002 进程占用） |
| 3 | 相机帧名与 `realsense_bimanual.yaml` 的 `main/left/right` 一一对应 |
| 4 | 实测 `T_infer` 与 `action_steps` 的搭配（判据 `T_infer ≤ action_steps/30`） |
| 5 | 真实 `get_joint_limits()` 数值是否覆盖任务所需行程 |
