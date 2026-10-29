# G0.5 VLA × YAM 日常启动手册

> 每次要跑任务时照着做。前提：`serving/bridge/` 下已有本文档配套的脚本。
> 三个进程：**安全守护进程 → 推理服务端 → 客户端**（顺序不能乱）

---

## 0. 一次性准备（只做一次）

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
chmod +x run_episode.sh
```

需要的文件（md5 对不上就从 Windows 的 `D:\G05\_upload\` 重新拖）：

| 文件 | md5 |
|---|---|
| `vla_client.py` | `25038294814fac6cdd6296ca3c257366` |
| `home_robot.py` | `018bf6f9a2520ffcd111a23bb24fe720` |
| `run_episode.sh` | `88c75f4b6278296744963116bd38dd6c` |
| `analyze_bridge_csv.py` | `714f829c8079b75e26a5e4eead2e5717` |

```bash
md5sum vla_client.py home_robot.py run_episode.sh analyze_bridge_csv.py
```

---

## 1. 打开一个新终端，先设环境

```bash
D=/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
B=/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
V311=/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python
GELLO=/home/chenyuan/workspace/YAM/gello_software

# ★ 必须：本机代理是死端口 7897，会劫持 websockets，导致连 127.0.0.1 都被拦
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

---

## 2. 安全守护进程（端口 6001 / 6002）

**先查，在跑就跳过：**

```bash
ss -tln | grep -E ':6001 |:6002 ' && echo "✅ 守护进程在跑，跳过步骤2" || echo "❌ 需要启动"
```

**要启动时：**

```bash
cd "$GELLO"
nohup "$V311" -u experiments/yam_safety_daemon.py \
  --left-config-path  "$HOME/.config/yam/arms/yam_left.yaml" \
  --right-config-path "$HOME/.config/yam/arms/yam_right.yaml" \
  --hardware-port 6001 --safety-port 6002 \
  > /tmp/daemon.log 2>&1 &
echo "daemon pid=$!"

for i in $(seq 1 30); do ss -tln | grep -q ':6002 ' && { echo "✅ 就绪 $((i*2))s"; break; }; sleep 2; done
```

**必须验证两侧硬件都健康**（这是最重要的一步，别跳）：

```bash
"$V311" "$B/vla_client.py" --task x --check-robot | head -12
```

期望：
```
state = 'safe_hold'
✅ 处于 safe_hold，可以申请会话接管
hardware_health.ok = True
  left  running=True
  right running=True      ← 右侧必须在
```

> ⚠️ **绝对不要停别人起的守护进程**（ziyu / 张恩同可能也在用）。只在你启动的才能停。

---

## 3. 推理服务端（端口 18879）

**先查：**

```bash
ss -tln | grep ':18879 ' && echo "✅ 服务端在跑，跳过步骤3" || echo "❌ 需要启动"
```

**要启动时：**

```bash
D=/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
cd "$D/code/motion/vendor/GalaxeaVLA"

PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup "$D/runtime/env/bin/python" -u \
  ~/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py \
  --ckpt_path "$D/models/ar_ft1ep/checkpoints/model_state_dict.pt" \
  --host 127.0.0.1 --port 18879 \
  eval_embodiment=galaxea_r1lite \
  model.model_weights_to_bf16=true \
  model.model_arch.continuous_action=false \
  > /tmp/serve.log 2>&1 &
echo "serve pid=$!"

# ★ 判据只看端口，不要等日志（输出重定向后是块缓冲，会误判成"起不来"）
for i in $(seq 1 90); do
  ss -tln | grep -q ':18879 ' && { echo "✅ 就绪 $((i*2))s"; break; }
  sleep 2
done
ss -tln | grep ':18879 ' || echo "❌ 3 分钟内没起来，看 /tmp/serve.log"
```

**冷启动要 1~3 分钟**（要从机械盘读 11.4 GB 权重）。看到 `LISTEN` 才算成功。

**可选但强烈建议：预热**（否则第一次推理要 12~18 秒）

```bash
"$V311" "$B/vla_client.py" --task-id TASK-YAM-0006 \
    --fake-images --mock-robot --steps 4 --port 18879 2>&1 | tail -5

grep "Recompute:" /tmp/serve.log | tail -2
# 期望看到 4xx~5xx ms（不再是 12000+ ms）
```

---

## 4. 放道具 + 确认现场

- **小方块**：右臂前方 20~30 cm，头相机视野内
- **凹槽板**：方块前方

拍图确认（可选，出问题时很有用）：

```bash
"$V311" "$B/vla_client.py" --task x --camera-test \
    --camera-config "$D/live_configs/robot.yaml" \
    --motion-src "$D/code/motion/src" \
    --save-frames /tmp/scene 2>&1 | tail -6
```

---

## 5. 跑任务 —— **一条命令**

```bash
cd "$B"

# 先空跑看计划（不动机器人）
DRY=1 ./run_episode.sh

# 真跑：15 秒任务 + 自动精确复位
./run_episode.sh
```

它自动做 5 步：**依赖检查 → 服务端检查 → 机器人预检 → 跑策略 → 精确复位**

这次跑完会输出：
```
汇总
  策略退出码 : 0
  日志       : /tmp/g05_runs/ep_20261029_220000.log
  CSV        : /tmp/g05_runs/ep_20261029_220000.csv
  复位日志   : /tmp/g05_runs/home_20261029_220000.log
```

### 常用变体

```bash
DURATION=20 ./run_episode.sh              # 跑 20 秒
ARM_SPEED=0.8 ./run_episode.sh            # 更慢更稳（<1.0 更不容易震）
RESET=0 ./run_episode.sh                  # 跑完不复位（要连试几次时用）
TASK_ID=TASK-YAM-0013 ./run_episode.sh    # 换任务
```

### 退出码含义

| 码 | 含义 | 说明 |
|---|---|---|
| **0** | 正常跑完 | ✅ |
| 2 | 被安全停机制中断 | 守护进程 unhealthy |
| 3 | 服务端错误 | 看 `/tmp/serve.log` |
| 4 | 服务端超时 | — |
| 5 | 相机故障 | 重新插拔相机 |
| 6 | 预检失败 | 机器人被占用 / 状态不对 |
| **7** | **模型输出异常，看门狗触发** | **这是好事** —— 在损坏硬件前停住了 |

> **退出码 7 不是故障**。它表示模型跑到了"崩坏区"（通常在任务完成后），看门狗主动刹停。
> 处理：缩短 `DURATION` 重试，或不管它（已复位）。

---

## 6. 事后分析

```bash
cd "$B"
python3 analyze_bridge_csv.py /tmp/g05_runs/ep_<时间戳>.csv
```

---

## 7. 关停

| 进程 | 要不要停 |
|---|---|
| `run_episode.sh` / `vla_client.py` | 自然结束，不用管 |
| **`serve_policy.py`** | **可以留着**（下次省 1~3 分钟）。要停：`pkill -f serve_policy.py` |
| **`yam_safety_daemon.py`** | **不要停**（除非是你起的，且确认别人不用） |

---

## 8. 关键参数（别乱改）

| 参数 | 值 | 为什么 |
|---|---|---|
| `--arm-max-speed-rad-s` | **1.0**（57°/s） | 系统有 0.533 s 观测延迟，速度快了必然震荡。实测 >120°/s 时反向率 33% |
| `--cmd-lowpass-tau` | **0.10** | 压掉换段频率（1.875 Hz）的抖动 88% |
| 夹爪限速 | 0.7/s（内置默认） | 对齐 `yam_motion` 标准，防止夹飞物体 |
| 看门狗 | 60° / 8 次（内置默认） | 模型输出崩坏时在 0.1 秒内刹停，防止烧电机 |

**`--arm-max-speed-rad-s` 不要超过 1.5**（86°/s），否则回原位时会震荡。

---

## 9. 排错速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: websockets` | 用错 python | 用 `$V311`（gello 的 venv） |
| `ConnectionRefusedError: 127.0.0.1:7897` | 代理劫持 | `unset ALL_PROXY HTTPS_PROXY ...` |
| `已有活跃控制会话，拒绝接管` | 上一次会话还没失效（需约 1 s） | 等 2 秒重试（新版本会自动等） |
| `vla_client.py 是旧版` | 没拖新文件 | 拖 md5 `25038294…` |
| 服务端"起不来" | 块缓冲误导 / 还在读盘 | **只看 `ss -tln \| grep 18879`** |
| `state='fault'` + `RIGHT follower ... motor chain stopped` | 电机保护性停机 | **重启守护进程**；仍不行则给右臂断电 10 秒 |
| 手臂剧烈震动 | 速度太快 | `ARM_SPEED=0.8`，加 `LP_TAU=0.15` |
| 抓不稳/反复重抓 | 物体形状问题 | 换更容易夹的物体，或调整夹持姿态 |
| 回原位位置偏了 | 模型学的 home ≠ `runtime.yaml` 的全零 | 正常现象，`run_episode.sh` 会自动精确复位 |

---

## 10. 一句话版本

```bash
# 每次要跑：
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

ss -tln | grep -q ':6002 ' || echo "⚠ 先起守护进程（见步骤2）"
ss -tln | grep -q ':18879 ' || echo "⚠ 先起服务端（见步骤3）"

./run_episode.sh
```
