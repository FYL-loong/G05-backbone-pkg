# 真机 A/B 对比实验操作手册（codec 微调 vs 官方 codec）

> **一句话**：同一个微调 backbone、同一套执行层、同一份 stats，**只把 `action_tokenizer.pt` 换成官方版**，在真机上各跑 10 次，比成功率。
>
> 这是离线 2×2 矩阵里**同一行**的两个格子，真机结果可以直接和离线 `ar_action_l1`（0.0545 vs 0.1869）对照。

---

## 一、实验设计

| | **A 组（我们的）** | **B 组（对照）** |
|---|---|---|
| backbone | 11,440,527,634 B | **同一个文件** |
| `dataset_stats.json` | 226,311 B | **同一个文件** |
| **`action_tokenizer.pt`** | **169,709,565 B（张恩同微调 YAM codec）** | **506,886,775 B（官方原版）** |
| 执行层 / 参数 / 任务文本 | 完全相同 | 完全相同 |

**唯一变量 = codec。** 只要这两点成立，对比就有效：
1. 服务端日志确认加载的是**本组**的包（`g05_cell_ours_yam` / `g05_cell_ours_off`）
2. 两个 codec 的**字节数**分别为 169,709,565 / 506,886,775

---

## 二、前置条件

### 1. 四个实验包已建好

```
/mnt/sda/chenyuan/motion/g05_cell_ours_yam      ← A 组用
/mnt/sda/chenyuan/motion/g05_cell_ours_off      ← B 组用
/mnt/sda/chenyuan/motion/g05_cell_off_off       （备用，本次不用）
/mnt/sda/chenyuan/motion/g05_cell_off_yam       （备用，本次不用）
```

### 2. 道具（必须和训练数据一致）

- **绿色梯形物块**（顶部斜面，约 3cm）
- **绿色带方孔的垫子**
- 摆放：**物块在画面右侧，垫子在中间**（与训练画面一致）

参考训练画面（从 `TASK-YAM-0006` 的 MCAP 里解出来的）：
- 物块在右、垫子在中间 → 模型用**右臂**去够
- 物块在左 → 模型会改用左臂（不是 bug，是它在响应场景）

### 3. 代码版本

`serving/bridge/` 下三个文件的 md5 必须是：

| 文件 | 字节数 | md5 |
|---|---|---|
| `vla_client.py` | 89,332 | `0a0be09ce2550b6919b428287947f5d2` |
| `run_episode.sh` | 11,269 | `f68d1cd11ae3b486f0be6e9c45bcd48f` |
| `test_release_freeze.py` | 6,960 | `361e25be212ebcb9950a28f7fc0aa7f3` |

核对：
```bash
cd ~/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && md5sum vla_client.py run_episode.sh test_release_freeze.py
```

跑一遍单元测试确认逻辑没坏：
```bash
cd ~/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && ~/workspace/YAM/gello_software/.venv/bin/python test_release_freeze.py
```
期望：`11 个测试，失败 0 个`

### 4. 环境干净

**必须先停掉所有占用相机/机器人的服务**（否则左腕相机 0 帧、直接跑不起来）：

```bash
ps -eo pid,args | grep -iE 'roboapi|eval-web' | grep -v grep
```
有输出就 kill 掉（这些是相机预览/测评台服务，会独占相机）。

---

## 三、固定配置（A/B 两组必须完全一致）

`serving/bridge/run_episode.sh` 里的默认值，**不要用环境变量改**：

| 参数 | 值 | 作用 |
|---|---|---|
| `ARM_SPEED` | 1.0 rad/s | 臂限速（>1.5 会因 0.53s 延迟震荡） |
| `LP_TAU` | 0.10 s | 命令低通，压高频抖动 |
| `GRIP_CLOSE_RATE` | **0.7 /s** | 夹爪闭合（慢，防把物块夹飞）|
| `GRIP_OPEN_RATE` | **3.0 /s** | 夹爪张开（快，防释放时把物块甩飞）|
| `RELEASE_FREEZE` | **9 步** | 张开后冻结手臂 0.3s，让物块自然落下 |
| `RELEASE_STOP` | 45 步 | 释放后 1.5s 结束 |
| `STILL` | 60 步 | 兜底：模型输出 2 秒不变就结束 |
| `DURATION` | 40 s | 硬上限 |
| 任务文本 | `Pick up the small square and place it in the groove` | 与训练一致 |

---

## 四、跑 A 组（10 次）

### 4.1 起服务端

```bash
pkill -f serve_policy.py; sleep 3
```

```bash
D=/mnt/sda/chenyuan/motion/g05_cell_ours_yam; cd "$D/code/motion/vendor/GalaxeaVLA" && PYTHONPATH="$PWD/src" nohup "$D/runtime/env/bin/python" -u "$HOME/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py" --ckpt_path "$D/models/ar_ft1ep/checkpoints/model_state_dict.pt" --host 127.0.0.1 --port 18879 eval_embodiment=galaxea_r1lite model.model_weights_to_bf16=true model.model_arch.continuous_action=false > /tmp/serve_A.log 2>&1 &
```

```bash
for i in $(seq 1 90); do ss -tln | grep -q ':18879 ' && { echo "READY $((i*2))s"; break; }; sleep 2; done
```

### 4.2 【必做】确认加载的是 A 格 + 正确 codec

```bash
tr -d '\n' < /tmp/serve_A.log | grep -oE 'g05_cell_ours_yam' | head -1
```

```bash
ls -l /mnt/sda/chenyuan/motion/g05_cell_ours_yam/models/ar_ft1ep/action_tokenizer.pt
```

**必须**：
```
g05_cell_ours_yam
169709565 ... action_tokenizer.pt
```

### 4.3 预热（省掉首次推理的 12~18 秒冷启动）

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && /home/chenyuan/workspace/YAM/gello_software/.venv/bin/python vla_client.py --task-id TASK-YAM-0006 --fake-images --mock-robot --steps 4 --port 18879 2>&1 | tail -2
```
看到 `steps completed=4` 即可。

### 4.4 跑一次 trial（重复 10 次）

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && OUTDIR=/tmp/g05_runs/A ./run_episode.sh
```

**每次跑之前**：把物块摆回原位（物块在右、垫子在中间）。

---

## 五、判定标准（三档，当场记）

| 判定 | 定义 |
|---|---|
| ✅ **成功** | 梯形块**平躺嵌入方孔**、与垫面齐平（像训练数据最后一帧） |
| 🟡 **部分** | 进入方孔区域，但**立起 / 斜卡 / 明显偏位** |
| ❌ **失败** | 没进方孔区域：没夹起 / 夹飞 / 中途停 / 乱动 / 超时 |

**记录格式**（一行一条）：
```
A1  ✅
A2  🟡 立起来
A3  ❌ 超时没动
```

---

## 六、切到 B 组

### 6.1 换服务端（只改 `D=` 那一行）

```bash
pkill -f serve_policy.py; sleep 3
```

```bash
D=/mnt/sda/chenyuan/motion/g05_cell_ours_off; cd "$D/code/motion/vendor/GalaxeaVLA" && PYTHONPATH="$PWD/src" nohup "$D/runtime/env/bin/python" -u "$HOME/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py" --ckpt_path "$D/models/ar_ft1ep/checkpoints/model_state_dict.pt" --host 127.0.0.1 --port 18879 eval_embodiment=galaxea_r1lite model.model_weights_to_bf16=true model.model_arch.continuous_action=false > /tmp/serve_B.log 2>&1 &
```

```bash
for i in $(seq 1 90); do ss -tln | grep -q ':18879 ' && { echo "READY $((i*2))s"; break; }; sleep 2; done
```

### 6.2 【必做】确认加载的是 B 格 + 官方 codec

```bash
tr -d '\n' < /tmp/serve_B.log | grep -oE 'g05_cell_ours_off' | head -1
```

```bash
ls -l /mnt/sda/chenyuan/motion/g05_cell_ours_off/models/ar_ft1ep/action_tokenizer.pt
```

**必须**：
```
g05_cell_ours_off
506886775 ... action_tokenizer.pt
```

⚠️ **这一步是实验有效性的唯一保证** —— 两格变成同一格的话，整个实验作废且数据看起来正常。

### 6.3 预热 + 跑 10 次（`OUTDIR` 换成 B）

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && /home/chenyuan/workspace/YAM/gello_software/.venv/bin/python vla_client.py --task-id TASK-YAM-0006 --fake-images --mock-robot --steps 4 --port 18879 2>&1 | tail -2
```

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && OUTDIR=/tmp/g05_runs/B ./run_episode.sh
```

---

## 七、自检规则（每 5 次查一次）

### 7.1 快速看步数（最省事）

```bash
cd /tmp/g05_runs/A && for f in ep_*.csv; do printf "%-28s %5d 步\n" "$f" "$(wc -l < $f)"; done
```

| 步数 | 含义 |
|---|---|
| 350~600 | 正常（抓到 → 释放 → 45 步后结束）|
| 700~950 | 偏晚结束 |
| ~1180 | 跑满 40 秒 = **没完成任务** |
| <50 | 立刻中断 → **作废** |

### 7.2 关键指标（确认模型真的在做事）

```bash
ls -t /tmp/g05_runs/A/ep_*.log | head -1 | xargs grep -E 'steps completed|夹爪被限速|累计最大位移' | tail -3
```

| 指标 | 正常 | 异常 |
|---|---|---|
| `夹爪被限速` | 40~90 步 | **0 步**（模型没抓东西）|
| `累计最大位移` | 20~40° | <10° |
| `steps completed` | 350~600 | >1000 |

**A 组出现"夹爪 0 步 / 位移 <10°"= 该轮作废**（模型没在做事，多半是场景/姿态/服务端状态问题）。
**B 组出现同样现象 → 不要判废**，这是 A/B 差异本身，照实记 ❌。

### 7.3 单轮深度分析（有 `analyze_release.py` 时）

```bash
cd ~/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge && ~/workspace/YAM/gello_software/.venv/bin/python analyze_release.py /tmp/g05_runs/A
```

输出：夹爪开合事件步号、释放时刻"模型想要 vs 实际下发"的滞后（判断是模型提前松手还是执行层拖后腿）、冻结步数、限速步数。

---

## 八、已踩过的坑（照这个排查）

| 现象 | 原因 | 处理 |
|---|---|---|
| `camera warmup timed out (main=..., left=0, ...)` | ① `roboapi` 预览服务占相机 ② **左腕 D405 USB 接触不良** | 先 `ps \| grep roboapi` kill；仍 0 帧就**插拔左腕相机 USB** |
| 第 2 步报"服务端没在跑"但其实在 | `set -o pipefail` + `grep -q` 的 SIGPIPE（已在 patch 里根治） | 已修复；若重现，`ss -tln \| grep 18879` 直接确认 |
| 夹爪"啪"一下猛夹、夹不住 | 夹爪语义搞反（1.0=张开，0.0=闭合）→ 闭合提速 4.3 倍 | 已修复（patch26c），`test_release_freeze.py` 里有回归测试锁死 |
| `IndexError: too many indices for array` | `GRIPPER_JOINTS` 是 tuple，`ndarray[tuple]` 被当多维索引 | 已修复，同上 |
| 双臂都不动、夹爪 0 步、跑满超时 | 起跑姿态不在 home / 场景与训练不一致 / 服务端跑久退化 | ① `home_robot.py --speed 0.12` ② 重启服务端 ③ 对照训练画面摆道具 |
| 左臂在动、右臂不动 | 物块被摆到了画面左侧 | 物块摆到右侧（与训练一致）|
| 每次 `run_episode.sh` 都从异常姿态开始 | 上一轮中断没复位 | 手动 `home_robot.py --speed 0.12` 后再跑 |

---

## 九、收尾与存档

```bash
mkdir -p /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d) && cp -r /tmp/g05_runs/* /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d)/ && cp /tmp/serve_A.log /tmp/serve_B.log /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d)/ 2>/dev/null; du -sh /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d)
```
`/tmp` 重启会清空，**数据必须搬走**。

```bash
pkill -f serve_policy.py; sleep 3; ss -tln | grep 18879 || echo "服务端已停"
```

**守护进程（端口 6001/6002）不要 kill** —— 它维持 `safe_hold` 和硬件监控。只有断电/放假才关。

---

## 十、当前进度（2026-11-06）

| 组 | 已完成 | 还需 | 备注 |
|---|---|---|---|
| **A**（`g05_cell_ours_yam`） | **6 / 10** | 4 | 有效轮：16:31:55 / 16:33:11 / 16:35:13 / 16:36:01 / 16:37:10 / 16:37:46 |
| **B**（`g05_cell_ours_off`） | **3 / 10** | 7 | 有效轮：16:43:04 / 16:44:05 / 16:45:48 |

**初步信号**（样本太小，不能写结论）：A 有 4/6 在 13~18 秒早停（完成了抓放周期），B 只有 1/3；A 有 1/6 跑满超时，B 有 1/3。

**作废记录**（不要计入）：
- 14:36 ~ 15:28：原始代码 / 六边形道具 / 诊断期
- 15:44 ~ 15:56：IndexError + 相机故障（没跑起来）
- 16:02 ~ 16:06：夹爪语义反了（夹不住）
- 16:08 ~ 16:24（`/tmp/g05_runs/A_test1`）：`RELEASE_FREEZE=0` 配置不同

**下一步**：起 A 格服务端 → 补 4 轮 A → 切 B 格 → 补 7 轮 B。
