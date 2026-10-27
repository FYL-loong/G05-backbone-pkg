# RUNBOOK —— G0.5 VLA × I2RT YAM 真机部署

> 最后更新：2026-10-27
> 适用机器：yam1（`chenyuan@ps`，RTX 4090 24GB，Ubuntu 22.04）
> **这份文档的目标：任何人只看这一份就能把这套东西跑起来、出了问题能自己定位。**

---

## 0. 一句话现状

| 项 | 状态 |
|---|---|
| backbone 微调 | ✅ 完成（10 任务 / 991 episodes / 1 epoch = **17,143 steps** / 4×GPU / 8h12m） |
| 开环 A/B 评测 | ✅ `ar_action_l1` **0.0545 vs 0.4992（+89.7%）** → 详见 `RESULT_eval.md` |
| 部署包 | ✅ `/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026`（11G） |
| 服务端 `T_infer` | ✅ **p50 = 513 ms**（预算 533 ms，余量 20 ms） |
| 端到端 serving 冒烟 | ✅ 96/96 步，`behind_schedule=0` |
| **部署契约离线验证** | ✅ 真实数据回放：右臂 6 维 `corr 0.966~0.999`，幅度比 0.995 |
| 相机（3 路 RealSense） | ✅ 30/30 帧，30 Hz，图像非零 |
| bridge 客户端 | ✅ 9 个补丁，单测 10/10 + 11/11 |
| **真机动作** | ⬜ **尚未执行**（下一步，见 §5） |

---

## 1. 资产清单

### 1.1 部署包 `$D = /mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026`

```
$D/
├── code/motion/                     yam_motion 源码 + 已打 AR 解码加速补丁的 GalaxeaVLA 副本
├── runtime/env -> /mnt/sda/chenyuan/motion/g05_ar5000_20261013/runtime/env   ★ 符号链接
├── launch.sh                        改好模型目录(ar_ft1ep)与端口(18879)
├── live_configs/{ar.json,robot.yaml,runtime.yaml}
├── models/ar_ft1ep/
│   ├── checkpoints/model_state_dict.pt   11,440,527,634 B
│   ├── action_tokenizer.pt                 169,709,565 B
│   ├── dataset_stats.json                      226,311 B
│   ├── ar_codec_stats.json                     226,311 B   (= yam_stats.json，两者同源)
│   ├── hf_processor/                          8 个文件（官方 qwen3_5_2b_base_processor）
│   ├── .hydra/config.yaml                       22,459 B（已展开链式插值）
│   └── motion_{contract,package,experiment,versions,split}.json
└── reports/, logs/
```

⚠️ **`runtime/env` 是符号链接**，指向 `g05_ar5000_20261013/runtime/env`。
**删掉那个包会连带弄坏我们这个包。** 要独立就把 env 实体复制过来（约 8G）。

### 1.2 关键路径

| 用途 | 路径 |
|---|---|
| 部署包 | `/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026` |
| 骨架来源（可重建） | `/mnt/sda/chenyuan/motion/g05_ar_newcodec5000_20261016` |
| 本地产物缓存 | `/mnt/sda/chenyuan/local_hf` |
| 我们的代码仓库 | `~/workspace/g05-yam-deployment/G05-backbone-pkg` |
| 同事的 serving 仓库 | `~/workspace/g05-yam-deployment/g05-async-serving` |
| Python 3.10（推理） | `$D/runtime/env/bin/python` |
| Python 3.11（gello/相机） | `/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python` |
| 真实回放素材 | `/mnt/sda/chenyuan/motion/compare_ar10k_pir250k_20261013/ar/logs/live_ar32_20261015_230527.json` |

### 1.3 产物校验值（从 HF 下载后核对）

| 文件 | 字节 | sha256 |
|---|---|---|
| `model_state_dict.pt` | 11,440,527,634 | `31fa117df1bcf4cb6693b108ff27f55b0e1276cbf8ddb905cacc8faea50382c6` |
| `action_tokenizer.pt` | 169,709,565 | `dfb9b19eb9b787a591556064b54eaf325b8b68df3027dbc2defb29d6f26f25ac` |
| `dataset_stats.json` | 226,311 | `ddd12a9d1d87acea5d2fd615a0c95b177f14f71e4c1835917d6ebaaf30048d83` |
| `hydra_config.yaml` | 18,337 | `54c7452a48b15d83fd7118165f0b53f2186d2f11dc43f8641c44a0192986a0e8` |
| `yam_stats.json` | 226,311 | 同 `dataset_stats.json`（两者内容相同） |

> `action_tokenizer.pt` 与 `ar_codec_stats` 的 sha256 **与同事 `ar_newcodec5000` 包里
> `motion_package.json` 记录的一致** —— 说明我们和他们共用同一个 YAM codec。

---

## 2. 三个进程

```
┌─────────────────────────────┐
│ yam_safety_daemon           │  :6001 机器人 RPC   :6002 安全 RPC
│ （持有 CAN 总线 + 400ms 看门狗）│  通常是 ziyu 启动的：
└──────────┬──────────────────┘  .../RhOSPolicy/.../experiments/yam_safety_daemon.py
           │                      --hardware-port 6001 --safety-port 6002
     ┌─────┴─────┐
     │           │
┌────┴─────┐ ┌───┴──────────────┐
│ vla_client│ │ 别人的客户端      │   ← 同一时刻只能有一个持会话
│ (3.11)    │ │ (ziyu Pi_0.5 等) │
└────┬─────┘ └──────────────────┘
     │ WebSocket + msgpack（need_obs 协议）
┌────┴──────────────────┐
│ serve_policy.py :18879 │  GalaxeaVLA 3.10 环境
└───────────────────────┘
```

**CAN 只能有一个主人。** 我们的 bridge 走 `--robot-backend zmq`（默认）连守护进程，
不抢 CAN；`--robot-backend direct` 是直连 CAN 的调试路径，**没有看门狗**，别用在真任务上。

---

## 3. 启动顺序

### 3.0 环境变量（每个新 shell 都要）

```bash
D=/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
V=$D/runtime/env/bin/python                 # 3.10 推理
V311=/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python   # 3.11 桥接
G=$D/code/motion/vendor/GalaxeaVLA
S=/home/chenyuan/workspace/g05-yam-deployment/g05-async-serving
B=/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
export HF_ENDPOINT=https://hf-mirror.com
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy   # 见 §7.1
```

### 3.1 起服务端（终端 A，前台常驻）

```bash
cd "$G"
PYTHONPATH="$G/src" "$V" "$S/serve_policy.py" \
    --ckpt_path "$D/models/ar_ft1ep/checkpoints/model_state_dict.pt" \
    --host 127.0.0.1 --port 18879 \
    eval_embodiment=galaxea_r1lite \
    model.model_weights_to_bf16=true \
    model.model_arch.continuous_action=false
```

等到出现：`Policy server listening on ... (mode=chunk(16), device=cuda)`

> `--action_steps` 默认 16（= 前瞻窗口 533 ms）。做离线回放时用 `--action_steps 1`
> 让每次请求都真推理。

### 3.2 确认守护进程在跑、且空闲

```bash
"$V311" "$B/vla_client.py" --task x --check-robot
```

**必须看到**：
```
  safety.state = 'safe_hold'
  hold_confirmed = True
  stale_clients = ['collector', 'parent']     ← 非空才能接管
  hardware_health.ok = True
  ✅ 处于 safe_hold，可以申请会话接管。
```

若 `state='active'` → **有人正在控制，不要接管**（返回码 6）。

### 3.3 起 bridge（终端 B）

```bash
"$V311" "$B/vla_client.py" \
    --host 127.0.0.1 --port 18879 \
    --task-id TASK-YAM-0006 \
    --robot-backend zmq --robot-port 6001 --safety-port 6002 \
    --command-timing legacy_30hz \
    --camera-config "$D/live_configs/robot.yaml" \
    --motion-src    "$D/code/motion/src" \
    --camera-warmup-s 1.5 \
    --duration 20 --verbose
```

---

## 4. 不动机械臂的三种离线检查（随时可跑）

```bash
# ① 协议冒烟：合成图 + 假机器人（零风险）
"$V311" "$B/vla_client.py" --task-id TASK-YAM-0006 \
    --fake-images --mock-robot --steps 96 --port 18879

# ② 相机只读测试
"$V311" "$B/vla_client.py" --task x --camera-test \
    --camera-config "$D/live_configs/robot.yaml" --motion-src "$D/code/motion/src"

# ③ 部署契约离线回放（真实录制帧 + 真实关节状态，会自己起停服务端）
"$V" "$B/offline_replay_needobs.py" --samples 20
"$V" "$B/offline_replay_needobs.py" --samples 20 --dry-run    # 30s，不加载模型
```

③ 的通过标准（当前已通过）：
```
✅ 被判定的 N 个「确实在动的」臂关节维全部正相关，无 ≈0、无负相关
✅ 幅度比 RMS(ours)/RMS(ref) = 0.995
✅ |ours-state| p95 最大 = 0.1203 rad
✅ 全部动作落在真实关节限位内
```

---

## 5. 真机分级流程（**每级都要人工盯**）

> **前置**：急停按钮在手边能按到；机械臂工作范围内无人、无松动物。

### Level 0 · DRY（**一条运动指令都不发**）

```bash
"$V311" "$B/vla_client.py" --task x --session-test 3 --session-dry
```

通过标准：`命令次数 = 0`，`释放后 state='safe_hold'`，`hold_confirmed=True`。
**这一步只验证"能不能接管 + 释放后一定回到抱死"。**

### Level 1 · 原地保持（30 Hz 命令"当前位置"）

```bash
"$V311" "$B/vla_client.py" --task x --session-test 5
```

通过标准：`有效频率 ≈ 30 Hz`，机械臂基本不动（有轻微保持声），退出后回 `safe_hold`。

### Level 2 · 闭环 10 秒（**第一次真动**）

```bash
"$V311" "$B/vla_client.py" \
    --host 127.0.0.1 --port 18879 --task-id TASK-YAM-0006 \
    --robot-backend zmq --command-timing legacy_30hz \
    --camera-config "$D/live_configs/robot.yaml" \
    --motion-src    "$D/code/motion/src" \
    --max-joint-step-deg 2 \
    --duration 10 --verbose
```

`--max-joint-step-deg 2` 是**斜率保护**（每步臂关节最多转 2°，约 60°/s），
第一次跑务必加上；确认动作合理后可以去掉。

失败时的退出码：`2`=机器人异常 `3`=服务端错误 `4`=收包超时 `5`=相机丢帧 `6`=preflight。

### Level 3 · 真实任务

去掉 `--duration`/`--max-joint-step-deg`，按任务布置场景，逐步加时长。

---

## 6. 换任务 / 停止 / 回滚

### 6.1 换任务

bridge 侧：用 `--task-id TASK-YAM-XXXX`（内置权威表，见 `serving/bridge/TASK_TEXTS.md`）。

`yam_motion` 那条路径（`ar.json` 驱动）需要同步：手工改
`$D/live_configs/ar.json` 与 `$D/models/ar_ft1ep/motion_experiment.json` 的 `"task"` 字段。
完整映射与批量脚本见 `serving/bridge/TASK_TEXTS.md`。

### 6.2 安全停止

| 场景 | 操作 |
|---|---|
| 正常结束 | `Ctrl+C` → bridge 自动 `hold()` → 守护进程确认 SAFE HOLD |
| bridge 崩了 | 守护进程 **400 ms 看门狗**自动抱死（两个心跳线程停发即触发） |
| 要立刻停 | **拍急停**（硬件） |
| 只停服务端 | `Ctrl+C` 终端 A；机器人不受影响（仍在 safe_hold） |

### 6.3 回滚

```bash
# 交还控制权给守护进程（bridge 退出后本来就会自动做，这里是手动兜底）
"$V311" -c "
import sys; sys.path.insert(0,'/home/chenyuan/workspace/YAM/gello_software')
from gello.safety.rpc import SafetyClient
c=SafetyClient(host='127.0.0.1', port=6002, timeout_ms=2000)
print(c.safe_hold('manual rollback')); c.close()"

# 删掉我们的部署包（不影响别人，但会释放 11G）
rm -rf /mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
```

---

## 7. 排错表（全是**实际踩过的**）

### 7.1 🔴 代理劫持：连不上 `127.0.0.1`

**现象**：`ConnectionRefusedError: Connect call failed ('127.0.0.1', 7897)`（和 WebSocket 有关）

**原因**：`HTTPS_PROXY=http://localhost:7897` 是个**死端口**（clash 实际听 7890/7891），
而 `websockets ≥14` 会自动读代理变量，连本机也被劫持。同事的 `fake_policy_client.py`
没传 `proxy=None`，**必挂**；我们的 `vla_client.py` 已修（补丁 5）。

**处理**：
```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

### 7.2 🔴 首次推理 12 秒，5 秒超时把它掐死

**现象**：`server silent for 5.0s → safe hold`，但服务端没事

**原因**：第一次 forward 含 CUDA kernel autotune，实测 **~12 s**；
稳态只要 ~0.5 s。统一 5 s 超时会误杀（补丁 6 已修：首包 180 s / 稳态 10 s）。

### 7.3 🔴 安全停机却返回 0，脚本误判成功

补丁 7 已修（返回 2/3/4/5/6）。**如果你的包装脚本还按 0/1 判，要改。**

### 7.4 🔴 `runtime/env/bin/hf` 是坏脚本

**现象**：`bad interpreter: /mnt/pfs/.../bin/python: No such file or directory`

**原因**：`hf` 的 shebang 指向**训练机**路径，搬到 yam1 就废了。
**不要用 `hf` CLI**，用 `$V -c "from huggingface_hub import snapshot_download; ..."`
（`build_deploy_pkg.sh` 已经这么做了）。

### 7.5 🟡 HF 下载：直连超时 / 代理把 TLS 掐断

| 线路 | 结果 |
|---|---|
| 直连 `huggingface.co` | 超时 |
| 走 clash 7890 / 7891 | `error:0A000126:SSL routines::unexpected eof while reading` |
| **直连 `hf-mirror.com`（带 token）** | ✅ **可用** |

```bash
export HF_ENDPOINT=https://hf-mirror.com
curl -L --noproxy '*' -H "Authorization: Bearer $HF_TOKEN" \
  https://hf-mirror.com/Zenith6666/yam-backbone/resolve/main/dataset_stats.json \
  -o /tmp/x.json           # 期望 226311 字节
```

> 更稳的办法：在自己电脑上下好再传（`/mnt/sda/chenyuan/local_hf` + `LOCAL_HF=` 装配）。

### 7.6 🟡 8765 端口被占

`ss -tln | grep 8765` 若有输出，用 **18879**（本项目端口）。

### 7.7 🟡 相机自动曝光要 ~1 秒才收敛

实测 `head_rgb` 均值 `77 → 127`。开跑就喂首帧会让模型看到偏暗画面。
补丁 9 已加 `--camera-warmup-s`（默认 1.5 s，丢弃前 1.5 s 的帧）。

### 7.8 🟡 相机被别的进程占着

`RealSense` 一个设备通常只能一个进程打开。若 `--camera-test` 报 device busy，
说明 ziyu 的相机栈在跑，需要协调。

### 7.9 🟡 装配时 `InterpolationKeyError: model.tokenizer._target_`

**原因**：run dir 里放了 `action_tokenizer.pt` 会触发 sidecar 路径，
而 `.hydra/config.yaml` 里的链式插值没展开。
**处理**：`tools/build_deploy_pkg.sh` 已内建"展开链式插值 + 不放 codec 进 run dir"，
不要手工改 config。

### 7.10 🟡 任务文本大小写/句号错 → 动作模式串

见 `serving/bridge/TASK_TEXTS.md`。**统一用 `--task-id`**。

---

## 8. 📌 尚未验证的假设（诚实清单）

| # | 假设 | 风险 | 怎么验证 |
|---|---|---|---|
| 1 | **真机上动作质量与离线一致** | 离线只证明"链路对"，不等于"策略好" | Level 2 闭环 10 s + 人工看 |
| 2 | 左臂会配合 | 那次回放素材里**左臂全程没动**，可能训练数据里左臂动作少 | Level 3 观察左臂响应 |
| 3 | 10% 余量够用（513 vs 533 ms） | 相机取帧 + 网络抖动可能再吃几十 ms | Level 2 看 `behind_schedule` 是否增长 |
| 4 | 长时稳定性 | 只跑过 96 步（3.2 s）和 19 次采样 | Level 3 逐步加时长 |
| 5 | 守护进程不会被别人抢走 | ziyu 可能随时回来跑他的策略 | 每次开跑前 `--check-robot` |
| 6 | 夹爪输出可用 | AR5k 的右夹爪是退化的（恒 0），我们的没单独验过 | Level 3 专门看夹爪 |

---

## 9. 后续优化建议（**按性价比排序**）

| # | 优化 | 预期 | 代价 |
|---|---|---|---|
| 1 | **TensorRT** | `T_infer` 513 → 可能 300 ms 以下 | 大（隔壁 RHOS 线已在做，可复用） |
| 2 | 装 `flash-attn` FA2 预编译轮 | 视神经塔占比，可能 20~50 ms | 中（只有视觉注意力受益） |
| 3 | ~~跳过 `lower_body` 解码省 20%~~ | ❌ **不可行** | 已核实：160 token 是 `tokenizer_meta` 的 5 组布局决定的，**训练时就这么学的**，砍 token 需要重训 |
| 4 | 继续训练 2~3 epoch | 开环已证 1 epoch 足够（+89.7%），预计收益小 | 8h/epoch × 4 卡 |
| 5 | 多任务切换的 prompt 缓存 | 换任务不用重启服务端 | 小 |

---

## 10. 一分钟自检（新接手者先跑这个）

```bash
D=/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
V=$D/runtime/env/bin/python
V311=/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python
B=/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

echo "① 部署包完整性"; ls -la "$D/models/ar_ft1ep/" | head -12
echo "② 3.10 环境";   "$V" -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "③ 3.11 环境";   "$V311" -c "import websockets,msgpack,pyrealsense2;print('  ok')"
echo "④ bridge 单测"; cd "$B" && "$V311" test_vla_client.py | tail -3
echo "⑤ 任务文本";    "$V311" "$B/vla_client.py" --list-tasks | head -3
echo "⑥ 机器人状态";  "$V311" "$B/vla_client.py" --task x --check-robot | tail -5
echo "⑦ GPU";         nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```
