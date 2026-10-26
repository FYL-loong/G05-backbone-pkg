# serving/bridge/ —— 真机桥接客户端

把 **相机 + 关节状态** 送进 `serve_policy`（WebSocket + msgpack），再把返回的
**绝对关节目标** 下发给 YAM 双臂。

---

## 1. 来源与改动

| 文件 | 来源 |
|---|---|
| `vla_client.py` | **张恩同原作**（实验机 `g05-yam-deployment/bridge/vla_client.py`，2026-10-26）+ 我们打的 **4 个补丁** |
| `test_vla_client.py` | 我们补的纯函数自测（10 项，无需 gello / 相机 / GPU） |

### 打过的 4 个补丁

| # | 原问题 | 修法 | 严重性 |
|---|---|---|---|
| **1** | `capture_images()` 是 `NotImplementedError`，真机必崩 | 复用 `yam_motion.cameras.YamCameras`（已验证：**只取 host_ts ≤ 查询时刻的因果帧**，超 `max_age_s` 拒绝） | 🔴 阻塞 |
| **2** | `--fake-images` 仍会 `build_robot()`，而 `create_pair` **做夹爪标定会真的动机械臂** | 新增 `--mock-robot`；并且**禁止**「合成图 + 真机器人」组合（除非显式 `--allow-fake-images-on-robot`） | 🔴 安全 |
| **3** | 推理阻塞 ~520 ms 后 `next_tick` 落后，客户端会把缓存步**在几十毫秒内连发**（≈把 0.5 s 轨迹压缩到几十 ms） | 请求耗时 > 1 个周期就 `next_tick = t_got` **重对齐**，停住的时间**丢弃而非补放**（同 `yam_motion.g05_execution.align_chunk` 的思路） | 🔴 会导致剧烈动作 |
| **4** | `await ws.recv()` 无超时，服务端挂住会永久阻塞且不走清理 | `asyncio.wait_for(..., RECV_TIMEOUT_S=5)`；超时 → `enter_safe_hold` + 退出 | 🟡 安全 |

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

```bash
$V bridge/vla_client.py --host 127.0.0.1 --port 18879 \
    --task "Pick up the small square and place it in the groove" \
    --camera-config <deploy>/live_configs/robot.yaml \
    --motion-src    <deploy>/code/motion/src \
    --duration 20
```

- `--camera-config` 指向含 `gello_root / camera_config / camera_names` 的 robot.yaml
  （本项目即 `live_configs/robot.yaml`）
- 相机名映射：`main→head_rgb`、`left→left_wrist_rgb`、`right→right_wrist_rgb`，
  HWC `(480,640,3)` → CHW `(3,480,640)`

**安全边界**：机器人不健康 / 服务端 `error` / 收包超时 / Ctrl+C
→ 一律 `enter_safe_hold()` 后 `close(disable_motors=True)`；
命令后按 `get_joint_limits()` 裁剪（每臂 6×2 + 夹爪 [0,1]）。

⚠️ `build_robot()` 里 `create_pair` 会**做夹爪标定并移动机械臂** —— 上真机前务必确认
急停和安全守护（`gello.safety`，400 ms watchdog，6002 端口）可用。

---

## 5. 仍需在真机上确认（代码之外的未知）

| # | 事项 |
|---|---|
| 1 | `pyrealsense2` 是否在该 venv 里（相机必需） |
| 2 | `YamCameras.start()` 能否打开 3 路（`/dev/video*` 未被 6001/6002 进程占用） |
| 3 | 相机帧名与 `realsense_bimanual.yaml` 的 `main/left/right` 一一对应 |
| 4 | 实测 `T_infer` 与 `action_steps` 的搭配（判据 `T_infer ≤ action_steps/30`） |
| 5 | 真实 `get_joint_limits()` 数值是否覆盖任务所需行程 |
