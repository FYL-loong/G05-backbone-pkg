# 真机 2×2 对比实验协议（TASK-YAM-0006）

> 目的：验证离线 2×2 矩阵的结论能否外推到真机 ——
> **两个微调（backbone、codec）是否都必要，且 codec 必须与 backbone 配套训练。**
>
> 离线结果（开环，10 episodes）：
>
> | | 官方 codec | YAM codec |
> |---|---|---|
> | 官方 backbone | 0.499219 | 0.506075 |
> | 我们 backbone | 0.186910 | **0.054483** |

---

## 一、四个包

| 包名 | backbone | codec | stats | 离线 L1 | 预期真机表现 |
|---|---|---|---|---|---|
| `g05_cell_ours_yam` | 我们的 | YAM | YAM | **0.0545** | ✅ 能完成任务 |
| `g05_cell_ours_off` | 我们的 | 官方 | YAM | 0.1869 | 🟡 能动、夹不准 |
| `g05_cell_off_off` | 官方 | 官方 | 官方 | 0.4992 | ❌ 不动 / 乱动 |
| `g05_cell_off_yam` | 官方 | YAM | 官方 | 0.5061 | ❌ 同上 |

**四格的执行层完全相同**：同一套 bridge、同一限速、同一看门狗、同一任务文本。
唯一的差别是模型目录里那 3~4 个文件。

---

## 二、固定参数（四格必须一致，中途不许改）

```
ARM_SPEED  = 1.0        # rad/s，57°/s
LP_TAU     = 0.10       # 命令低通
RELEASE_STOP = 45       # 夹爪释放后 45 步结束
STILL      = 60         # 兜底：连续 60 步不动
DURATION   = 30         # 上限
任务文本    = Pick up the small square and place it in the groove
```

---

## 三、每格开始前（拍图存档）

1. 把方块摆到**记号笔标记的位置**，凹槽板位置不动
2. `--camera-test --save-frames /tmp/cell_<包名>/` 存三路图（作为该格的摆放凭证）
3. `--check-robot` 确认 `state='safe_hold'` + 两侧 `running=True`
4. `home_robot.py --speed 0.12` 复位到同一姿态

```bash
D=/mnt/sda/chenyuan/motion
B=/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
V311=/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

"$V311" "$B/vla_client.py" --task x --check-robot | tail -4
"$V311" "$B/home_robot.py" --speed 0.12 | tail -6
```

---

## 四、每次 trial

### 起服务端（换格时才需要重启）

```bash
D=/mnt/sda/chenyuan/motion/g05_cell_ours_yam     # ← 换格时改这里
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
cd "$D/code/motion/vendor/GalaxeaVLA"
PYTHONPATH="$PWD/src" nohup "$D/runtime/env/bin/python" -u \
  ~/workspace/g05-yam-deployment/g05-async-serving/serve_policy.py \
  --ckpt_path "$D/models/ar_ft1ep/checkpoints/model_state_dict.pt" \
  --host 127.0.0.1 --port 18879 eval_embodiment=galaxea_r1lite \
  model.model_weights_to_bf16=true model.model_arch.continuous_action=false \
  > /tmp/serve_<格名>.log 2>&1 &
for i in $(seq 1 90); do ss -tln | grep -q ':18879 ' && { echo "✅ 就绪 $((i*2))s"; break; }; sleep 2; done
```

> ⚠️ 判据**只看端口**，不要等日志（块缓冲会误判）。
> ⚠️ 换格前先 `pkill -f serve_policy.py`，确认端口释放。

### 跑一次

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
./run_episode.sh
```

约 25 秒（策略 + 自动复位）。

---

## 五、成功判据（三档，必须当场判定）

| 判定 | 定义 |
|---|---|
| ✅ **成功** | 方块被夹起 → 移动到凹槽 → 松手后**停留在凹槽内**（可容许轻微偏移，但不得掉出） |
| 🟡 **部分** | 夹起并到达凹槽上方 / 放入后掉出 / 只完成对准未松手 |
| ❌ **失败** | 没夹起 / 夹飞 / 中途掉落 / 中途停住不动 / 看门狗刹停 / 超时未完成 |

**失败模式细分**（记下来，后面分析用）：

| 代号 | 含义 |
|---|---|
| F1 | 夹飞（方块被弹开） |
| F2 | 够不到 / 没夹起 |
| F3 | 夹起后中途停住不动 |
| F4 | 夹起后乱动 / 大幅摆动 |
| F5 | 放了但掉出凹槽 |
| F6 | 看门狗刹停（日志有 `🛑`） |
| F7 | 程序异常退出（退出码非 0） |

---

## 六、顺序与停止规则

- **四格交替**：`ours_yam → ours_off → off_off → off_yam →` 循环，共 10 轮 = 40 次
- **提前停止**：某格**连续 5 次失败**（且不是 F6/F7 这种程序问题）→ 该格停止，记为 0/5
- **重新开始**：换格后**第一次**必须重新 `home_robot.py` 复位

---

## 七、记录表（复制到文本文件，逐行填）

```
格名              trial  结果   失败码  用时(s)  看门狗  备注
ours_yam            1
ours_yam            2
...
ours_yam           10
ours_off            1
...
off_off             1
...
off_yam             1
...
```

Markdown 版本（40 行模板）：

| 格 | # | 结果 | 失败码 | 用时(s) | 看门狗 | 备注 |
|---|---|---|---|---|---|---|
| ours_yam | 1 | | | | | |
| ours_yam | 2 | | | | | |
| ours_yam | 3 | | | | | |
| ours_yam | 4 | | | | | |
| ours_yam | 5 | | | | | |
| ours_yam | 6 | | | | | |
| ours_yam | 7 | | | | | |
| ours_yam | 8 | | | | | |
| ours_yam | 9 | | | | | |
| ours_yam | 10 | | | | | |
| ours_off | 1 | | | | | |
| ours_off | 2 | | | | | |
| ours_off | 3 | | | | | |
| ours_off | 4 | | | | | |
| ours_off | 5 | | | | | |
| ours_off | 6 | | | | | |
| ours_off | 7 | | | | | |
| ours_off | 8 | | | | | |
| ours_off | 9 | | | | | |
| ours_off | 10 | | | | | |
| off_off | 1 | | | | | |
| off_off | 2 | | | | | |
| off_off | 3 | | | | | |
| off_off | 4 | | | | | |
| off_off | 5 | | | | | |
| off_off | 6 | | | | | |
| off_off | 7 | | | | | |
| off_off | 8 | | | | | |
| off_off | 9 | | | | | |
| off_off | 10 | | | | | |
| off_yam | 1 | | | | | |
| off_yam | 2 | | | | | |
| off_yam | 3 | | | | | |
| off_yam | 4 | | | | | |
| off_yam | 5 | | | | | |
| off_yam | 6 | | | | | |
| off_yam | 7 | | | | | |
| off_yam | 8 | | | | | |
| off_yam | 9 | | | | | |
| off_yam | 10 | | | | | |

---

## 八、汇总口径

| 指标 | 算法 |
|---|---|
| **成功率** | ✅ 次数 / 实际试验次数 |
| **部分完成率** | 🟡 次数 / 实际试验次数 |
| **平均用时** | 仅统计 ✅ 的 trial |
| **主要失败模式** | 出现次数最多的失败码 |

**判读**：

| 真机结果 | 结论 |
|---|---|
| 只有 `ours_yam` 成功，其余三格 ≤1 次成功 | ✅ **与离线完全一致** —— 两个微调都必要，codec 必须配套 |
| `ours_yam` 也基本失败 | ❌ 离线 → 真机的鸿沟没跨过去，要查执行层/场景 |
| `ours_off` 也能成 | ⚠️ codec 的必要性被高估，需复核 |
| 官方两格也能偶尔成 | ⚠️ 说明任务偏简单，区分度不够，考虑换更难的任务 |

---

## 九、安全（每次都要遵守）

1. **急停在手边**，尤其是官方 backbone 那两格（预期输出崩坏）
2. 看门狗必须开着（默认 60°/3 步 + 8 次振荡）—— 防止 10-29 那次烧电机事故重演
3. 出现 `🛑` 日志 = 看门狗正常刹停，**不用重启 daemon**，只是该 trial 记 F6
4. 只有 `state='fault'` 才需要重启守护进程
5. 不要停别人起的守护进程（ziyu / 张恩同可能在用）

---

## 十、收尾

```bash
pkill -f serve_policy.py
ss -tln | grep 18879 || echo "✅ 服务端已停"
"$V311" "$B/vla_client.py" --task x --check-robot | tail -3
```

把日志和记录表一起存档：
```bash
mkdir -p /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d)
cp /tmp/serve_*.log /tmp/g05_runs/ep_*.csv /tmp/g05_runs/ep_*.log \
   /mnt/sda/chenyuan/motion/real_robot_2x2_$(date +%Y%m%d)/
```
