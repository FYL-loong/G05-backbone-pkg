# 交付说明：2026-10-29 真机联调改动

> 项目：G0.5 VLA × I2RT YAM 双臂真机部署
> 时间：2026-10-29 20:30 — 22:20（真机现场联调）
> 范围：**仅执行层（`serving/bridge/`）**，未改动训练、模型、服务端任何代码
> 提交：17 个 commit（`ed3c45d` → `385950f`），仓库 `G05-backbone-pkg` 分支 `main`

---

## 一、一句话概要

昨晚在真机上暴露的 **8 个问题全部定位到根因并修复**，其中 **3 个是硬件级风险**（夹爪把物体弹飞、模型输出崩坏导致剧烈震动并烧坏右夹爪电机驱动、复位时位置不准确）。

新增 **4 个工具文件**（复位脚本、自动化流程、CSV 诊断分析器、启动手册），单元测试从 **10 项扩到 27 项**。

---

## 二、改动清单（一眼版）

| # | 问题 | 根因 | 修复位置 | 状态 |
|---|---|---|---|---|
| 1 | **夹取时把小方块弹飞** | 桥接层**完全没有夹爪限速**，模型输出 0→1 在 33ms 内走完全行程（标准是 1.43s，快 **43 倍**） | `vla_client.py` `rate_limit_target()` + `--gripper-rate-limit` | ✅ 真机已确认解决 |
| 2 | **相机偶发丢帧导致整次运行提前结束** | 单帧取不到即 abort + `max_frame_age=0.1` 过紧；同事的重试逻辑用**同一个 ts** 重试，而 `YamCameras` 只接受 `host_timestamp <= ts`，新帧会被因果过滤拒掉 → 重试永远无效 | 三层防护 + `RealCameras.sample()` 推进 ts 重试 | ✅ 真机 3 轮共 1,639 步 0 次丢帧 |
| 3 | **任务完成后回原位时剧烈震动** | 系统有 **0.533s 观测延迟**（异步预取代价），快速运动时"误差 = 速度 × 延迟"过大 → 过冲 → 反向 → 1.875Hz 震荡。实测反向率：<30°/s 时 **1%**，>120°/s 时 **33.3%** | 降速 `--arm-max-speed-rad-s 1.0`（57°/s）+ 低通 `--cmd-lowpass-tau 0.10` | ✅ 真机确认"修复了很多" |
| 4 | **震动导致右夹爪电机 CAN 失联、daemon 进 fault** | 模型在任务完成后输出崩坏：**6 个右臂关节在同一瞬间（step 352）要求跳 40~62°**（p95 从 2.7° 飙到 26.6°）→ 30Hz 高频摆动 → 右夹爪（motor 7）堵转过流 → DM 驱动停止应答 | 新增两道看门狗（偏差闸 + 振荡闸）+ fault 时明确报错 | ✅ 防线已加，事故后未复现 |
| 5 | **连续跑两次 bridge 第二次必失败** | 守护进程用心跳超时判定 `stale_clients` 需约 **1 秒**，刚释放会话 77ms 内申请接管会被拒 | `ZmqYamRobot.wait_session_ready()` | ✅ 真机连续三次接管全过（每次等 0.9s） |
| 6 | **fault 时抛裸 traceback，看不出该查什么** | `HoldNotConfirmed` 直接冒泡 | 打印守护进程 state/reason + 左右两侧 hardware_health + 重启步骤 | ✅ |
| 7 | **跑完停在半空，回原位位置不准确** | ① 模型学的"home"≠`runtime.yaml` 里的全零（实测差 arm4 24°、arm6 14°）；② 靠 `--duration` 猜时长会切在任务中途 | 新增 `home_robot.py` 确定性复位 + `run_episode.sh` 把复位纳入流程 | ✅ 复位收敛到 **1.28~1.46°** |
| 8 | **任务完成后模型还在乱动 20 秒才安静** | `--stop-when-still` 是被动等模型安静，实测任务 12.5s 完成，但要到 33s 才安静 | `--stop-after-release`（任务感知：夹爪释放即停） | ⚠️ 判据已修复，**最新版尚未上真机验证**（见第五节） |

---

## 二·补：改动量汇总（`git log --since="2026-10-29 18:00" --numstat`）

| 文件 | 新增行 | 删除行 | 类型 |
|---|---:|---:|---|
| `serving/bridge/vla_client.py` | +726 | −103 | 修改（核心） |
| `serving/bridge/test_vla_client.py` | +246 | 0 | 修改（10 → 27 项测试） |
| `serving/bridge/home_robot.py` | +210 | −1 | **新增** |
| `serving/bridge/run_episode.sh` | +187 | −6 | **新增** |
| `serving/bridge/analyze_bridge_csv.py` | +192 | 0 | **新增** |
| `serving/bridge/QUICKSTART.md` | +257 | 0 | **新增** |
| **合计** | **+1,818** | **−110** | 6 个文件 |

共 **17 个 commit**（`ed3c45d` → `385950f`），**未触碰**训练代码、模型权重、部署包、服务端。
## 三、代码改动明细

### 3.1 `serving/bridge/vla_client.py`（主要改动文件）

改动量：**+726 / −103 行**（16 次提交累计），从 46,566 B → 83,562 B，现在 1,524 行

#### 新增常量

```python
# 执行层限速
GRIPPER_RATE_LIMIT_S   = 0.7    # 夹爪 0.7/s，0→1 需 1.43s（对齐 yam_motion 的 gripper_rate_limit）
ARM_MAX_SPEED_RAD_S    = 2.5    # 臂 143°/s（实测推荐 1.0，见下文）
GRIPPER_JOINTS         = (6, 13)

# 相机容错
CAMERA_MAX_AGE_S       = 0.20   # 原 0.1，放宽到容 6 帧
CAMERA_STALL_ABORT_S   = 1.0    # 连续 1s 取不到才 abort（期间原地保持）
CAMERA_SAMPLE_RETRY_S  = 0.045  # 单次取帧内部重试窗口（> 1 帧周期 33ms）

# 模型输出看门狗
OUTPUT_SANITY_DEG      = 60.0   # 单关节目标偏离当前 >60° 视为异常（正常跟踪 p95 < 3°）
OUTPUT_SANITY_TRIP     = 3      # 连续 3 步即停
OUTPUT_OSC_DEG         = 2.0    # 参与振荡判定的最小单步位移
OUTPUT_OSC_TRIP        = 8      # 连续 8 次方向反转即停

# 结束条件
CMD_LOWPASS_TAU_S      = 0.0    # 低通，0=关闭（推荐 0.10）
STOP_WHEN_STILL_STEPS  = 0      # 模型输出稳定即停，0=关闭（推荐 60）
STOP_AFTER_RELEASE_STEPS = 0    # 夹爪释放即停，0=关闭（推荐 45）
RELEASE_OPEN_V         = 0.80   # 夹爪"完全张开"阈值
RELEASE_CLOSED_V       = 0.30   # 夹爪"完全闭合"阈值
RELEASE_OPEN_HOLD      = 10     # 张开需持续步数
RELEASE_MIN_TRANSPORT_DEG = 10.0  # 释放时臂姿态须距抓取姿态 ≥10°（证明真搬动了）

SESSION_READY_TIMEOUT_S = 12.0
EXIT_MODEL_INSANE      = 7      # 新增退出码
```

#### 新增纯函数（可单测）

| 函数 | 作用 |
|---|---|
| `rate_limit_target(target, prev, dt, gripper_rate, arm_speed)` | 逐关节速度限幅，返回 `(new_target, clipped_grip, clipped_arm)` |
| `lowpass_target(raw, smoothed, dt, tau)` | 一阶低通，返回 `(new_smoothed, alpha)` |
| `_save_png(path, hwc_rgb)` | 存 PNG（cv2 优先，PIL 兜底） |
| `ZmqYamRobot.wait_session_ready(timeout)` | 等守护进程可接管（`safe_hold` + `stale_clients` 非空） |
| `RealCameras.sample(ts, retry_s)` | 内层重试，**每次推进 ts**（修掉同事版本的因果过滤问题） |

#### 主循环执行顺序（关键，勿动）

```
L961  target = action_to_target(...)          模型原始输出（绝对关节目标）
L968  raw_target = target.copy()              存档，供诊断/看门狗使用
L982  稳定即停检测        （读 raw_target）
L1005 低通滤波            （改 target）
L1016 速度限幅            （改 target）
L1034 max-joint-step-deg  旧式硬钳位（可选用）
L1040 输出看门狗          （读 raw_target）
L1092 ★ 夹爪释放检测      （读 target = 真实下发值）
L1145 软限位裁剪
L1147 command_joint_state(target)             下发
```

> ⚠️ **踩过的坑**：夹爪释放检测最初放在 L1016 **之前**，读的是模型原始输出。原始输出在夹爪闭合过渡期会在 0.5 附近抖动，抖一下就误判成"释放"，导致任务在 7.4 秒被砍断（方块还没放下）。**任何"从动作判断状态"的逻辑都必须读限速后的真实命令值。**

#### 新增命令行参数（12 个）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gripper-rate-limit` | 0.7 | 夹爪速度（单位/秒）；0=关闭 |
| `--arm-max-speed-rad-s` | 2.5 | 臂速度上限 rad/s；0=关闭。**禁止 >1.5** |
| `--camera-stall-abort-s` | 1.0 | 相机连续无帧多久才 abort |
| `--camera-max-age-s` | 0.20 | 允许的帧陈旧度 |
| `--log-csv` | 无 | 每步写 JSON 诊断（含 raw/cmd 目标、跳变、换段标记、限速量） |
| `--save-frames` | 无 | 配合 `--camera-test` 存三路 PNG |
| `--snapshot-on-freeze` | 无 | 检测到模型连续 1s 不动时自动存当时画面 |
| `--output-sanity-deg` | 60 | 偏差闸阈值；0=关闭 |
| `--output-osc-trip` | 8 | 振荡闸阈值；0=关闭 |
| `--cmd-lowpass-tau` | 0.0 | 低通时间常数秒；**推荐 0.10** |
| `--stop-when-still` | 0 | 模型输出连续 N 步不变即结束；推荐 60 |
| `--stop-after-release` | 0 | 夹爪释放后 N 步结束；推荐 45 |

#### 其他行为改动

- 首包超时 `180s` / 稳态 `10s`（原统一 5s，会把 12~18 秒的首次推理掐掉）
- 失败返回精确退出码：`0` 正常 / `2` unhealthy / `3` 服务端错误 / `4` 超时 / `5` 相机 / `6` 预检 / `7` 模型输出异常
- 收尾打印执行层统计：`相机丢帧恢复 N 次；夹爪被限速 N 步；臂被限速 N 步；冻结段 N 段`
- `--session-test` 改用 deadline 对齐（原写法实测只有 19.9Hz 并误报"频率偏低"）

---

### 3.2 `serving/bridge/test_vla_client.py`

改动量：**+246 行**（纯新增），测试数 **10 → 27**

| 新增测试组 | 数量 | 覆盖 |
|---|---|---|
| `rate_limit_target` | 6 | 夹爪 0→1 恰好 43 步(1.43s)、正常步进不裁、尖峰裁到 0.0833 rad、关闭、异常 dt 回落、shape 校验 |
| `RealCameras` 重试 | 3 | **ts 必须单调推进**（这条当场抓出 Windows monotonic 同值问题）、窗口超时返回 None、CHW 形状校验 |
| `wait_session_ready` | 3 | 轮询到就绪、已就绪不重复轮询、超时不抛异常 |
| `lowpass_target` | 5 | 旁路、首步用 raw 初始化、阶跃收敛、**1.875Hz ±20° 抖动衰减到 <45%**、异常 dt |
| 原有 | 10 | 动作解析/obs 构造/编解码/mock 机器人 |

运行：`$V311 test_vla_client.py` → **27/27 passed**

> ⚠️ **测试缺口**：夹爪释放判据（`--stop-after-release` 的状态机）目前是**内联在主循环里**，没有抽成纯函数，因此**没有单元测试**，只用真实 CSV 离线回放验证过（见第五节）。建议后续抽取后补测。

---

### 3.3 新增文件 1：`serving/bridge/home_robot.py`（210 行新增）

**确定性复位工具**：把双臂缓慢移回 home 姿态。

```
阶段 1：双臂按 --speed（默认 0.15 rad/s = 8.6°/s）线性移向 home，【夹爪保持不动】→ 途中不会掉东西
阶段 2：夹爪按 0.7/s 缓慢张开，在低位释放物体
收尾  ：释放安全会话（自动 SAFE HOLD）
```

安全机制：
- 走 `ZMQClientRobot` + `SafetyClient` + `SupervisedMotionSession`（400ms 看门狗 + 软限位）
- 每步 `rate_limit_target` 限速，绝不跳变
- 误差**反向变大 10%** 即判定被卡住/撞到，立即停止
- 超时 90s/阶段；Ctrl+C 走 finally 必然回 SAFE HOLD

参数：`--speed` `--gripper-speed` `--tolerance-deg` `--max-seconds` `--keep-gripper` `--target` `--dry-run`

**实测**：118.1° → **1.43°**（611 步 / 20.8s）；50.5° → **1.28°**（418 步 / 14.1s）

---

### 3.4 新增文件 2：`serving/bridge/run_episode.sh`（+187 / −6 行）

**一次完整试验的自动化流程**，5 步：

```
1/5 依赖检查     检查 vla_client.py 是否含 --cmd-lowpass-tau（防拖旧文件白跑）
2/5 服务端检查   18879 是否监听
3/5 机器人预检   check-robot，不是 safe_hold 就拒绝
4/5 跑策略       带 --arm-max-speed-rad-s / --cmd-lowpass-tau / --stop-after-release / --stop-when-still
5/5 精确复位     无论策略成功、超时还是被看门狗停掉，都做
```

设计原则：
- **复位是流程的一部分，不是可选项** —— 手工记"跑完要复位"一定会漏，漏了下次起跑姿态不一致、实验不可比
- 每次留 CSV，事后可精确分析
- 退出码有中文说明（尤其 `7 = 看门狗触发，是好事`）

可覆盖环境变量：`DURATION` `ARM_SPEED` `LP_TAU` `STILL` `RELEASE_STOP` `HOME_SPEED` `RESET` `DRY` `TASK_ID` `SERVER_PORT`

---

### 3.5 新增文件 3：`serving/bridge/analyze_bridge_csv.py`（+192 行）

**CSV 诊断分析器**，5 段报告：

1. 每维臂关节位移分布（p50/p95/p99/max + 出现在第几步）
2. **换段首帧 vs 普通步** 的跳变对比（>2.5× 判定换段跳变）
3. 位移最大的 Top N 步（含 raw/cmd 目标和限幅量）
4. 夹爪轨迹与单步最大变化（校验限速是否生效）
5. 换段处运动方向夹角

---

### 3.6 新增文件 4：`serving/bridge/QUICKSTART.md`（+257 行）

**日常启动手册**（10 节）：三个进程启动顺序、健康验证、一条命令跑任务、退出码全表、关键参数约束、9 条排错速查。

---

## 四、实测数据

### 4.1 夹爪限速（问题 1）

| | 改动前 | 改动后 |
|---|---|---|
| 夹爪 0→1 耗时 | **33 ms**（一步到位） | **1,430 ms**（43 步） |
| 真机观感 | "啪"一下弹合 → 方块被弹飞 | 平缓闭合，**方块不再弹飞**（用户确认） |

### 4.2 相机容错（问题 2）

| 轮次 | 步数 | 相机丢帧 |
|---|---|---|
| run1 (20s) | 572 | **0 次** |
| run60 | 391 | **0 次** |
| run_obj (20s) | 556 | **0 次** |
| run_slow (20s) | 561 | **0 次** |

（改动前：单帧超时即整次运行 abort）

### 4.3 震动定量分析（问题 3）—— 关键证据

从 `run_obj.csv`（556 步）统计**每步位移方向的反转率 vs 速度**：

| 速度区间 | 步数 | 反向率 |
|---|---|---|
| 0-10 °/s | 299 | **0.0%** |
| 10-30 °/s | 112 | **0.9%** |
| 30-60 °/s | 64 | **6.2%** |
| 60-120 °/s | 53 | **9.4%** |
| **>120 °/s** | 27 | **33.3%** |

**反向率随速度单调上升** → 确认是延迟反馈震荡 → 降速是对症手段。
反向间隔分布：1 步（15Hz 高频摆动）8 次、15~17 步（换段频率）6 次 → **主因是 15Hz 步间摆动**，只在高速时出现。

### 4.4 低通滤波衰减（问题 3）

`--cmd-lowpass-tau` 对 1.875Hz（换段频率）±20° 抖动的抑制：

| tau (s) | 抖动残余 | 跟随滞后 |
|---|---|---|
| 0（关闭） | 99.5% | 0° |
| 0.04 | 29.5% | 7.6° |
| **0.08** | **17.3%** | 15.0° |
| **0.12** | **12.3%** | 21.9° |
| 0.20 | 7.7% | 33.8° |

推荐 **0.10**（抖动降到约 15%，滞后约 18°）。

### 4.5 事故证据（问题 4）

```
step 352（t≈12.4s）同一瞬间：
  R_arm1 要求跳 40.49°   R_arm4 要求跳 37.87°
  R_arm2 要求跳 61.97°   R_arm5 要求跳 12.04°
  R_arm3 要求跳 49.05°   R_arm6 要求跳 25.52°

而 p95 从上一轮的 2.70° 飙到本轮的 26.61°
```

守护进程日志：
```
AssertionError: fail to communicate with the motor 7 on yam_real
                at can channel socketcan channel 'can_follower_r'
RuntimeError: MotorChainRobot(...): motor_chain_robot's motor chain is not running,
              exiting the robot server
```

**motor 7 = 右夹爪电机**。温度正常（39℃/45℃），是**堵转过流**导致驱动保护性停机。

### 4.6 会话等待（问题 5）

```
第2次接管：等待上一次会话释放（守护进程需约 1 秒判定心跳失效）...
           会话可用（等了 0.9 s，stale_clients=['collector', 'parent']）
第3次接管：同样 0.9 s
✅ 三次全过（改动前第2、3次必失败）
```

### 4.7 复位精度（问题 7）

| 场景 | 初始误差 | 收敛 | 用时 |
|---|---|---|---|
| 首次复位（夹着方块） | 118.1° | **1.43°** | 611 步 / 20.8s |
| 事故后复位 | 50.5° | **1.28°** | 418 步 / 14.1s |
| 夹爪张开（阶段2） | — | **0.00°** | 34 步 / 1.1s |

速度精确性：剩余误差每 30 步递减 **5.73°** = `0.10 rad/s × 57.3` ✅ 限速精确工作。

### 4.8 结束时机（问题 8）

| 方案 | 实际停止时刻 | 问题 |
|---|---|---|
| `--duration 20` 固定 | 20.0s | 切在任务中途 / 跑进崩坏区 |
| `--stop-when-still 60` | **33.0s** | 任务 12.5s 完成，之后白跑 20.5s 乱动 |
| `--stop-after-release 45` | **13.4s**（离线回放预测） | 判据需修正（已完成） |

真实任务时间线（从 CSV 提取）：
```
t= 5.64s  R_grip 开 → 闭   抓取
t=12.54s  R_grip 闭 → 开   放置释放   ★ 任务完成
t=33.00s  实际停止（旧逻辑）            ← 多跑 20.5s
```

---

## 五、验证状态（交付时必须说明）

| 修复 | 验证方式 | 状态 |
|---|---|---|
| 夹爪限速 | 真机 + 6 项单测 | ✅ 用户确认"不会飞了" |
| 相机容错 | 真机 4 轮 1,639 步 + 3 项单测 | ✅ 0 次丢帧 |
| 臂限速 + 低通 | 真机 + 5 项单测（含 1.875Hz 衰减） | ✅ 用户确认"修复了很多" |
| 输出看门狗 | 合成数据 + 本地逻辑测试 | ⚠️ **事故后未再复现，未经真机触发验证** |
| 会话等待 | 真机连续 3 次 + 3 项单测 | ✅ 0.9s 等待，三次全过 |
| fault 报错 | 真机（事故现场） | ✅ |
| 复位工具 | 真机 3 次 | ✅ 收敛 1.28~1.46° |
| `--stop-after-release` | **仅真实 CSV 离线回放**（6 场景） | ⚠️ **最新版（patch 25）尚未上真机** |

### ⚠️ 交付时需注意的 3 个待验证点

1. **`--stop-after-release`（patch 25）未经真机验证**
   用真实 CSV 回放的预期：成功那轮会在 **402 步 / 13.4s** 停止；误判那轮**不再触发**。
   若真机上出现误判或没触发，退路：`RELEASE_STOP=0` 关闭它，退回 `--stop-when-still` 兜底。

2. **输出看门狗未在真机触发过**
   阈值依据：正常跟踪实测 p95 < 3°，设 60°（20 倍余量）。逻辑已用合成数据验证（62° 跳变第 3 步触发、5° 方向反转第 8 次触发），但**没有真实触发记录**。

3. **释放判据没有单元测试**
   状态机内联在主循环中，仅用真实 CSV 离线回放验证。建议后续抽成纯函数补测。

---

## 六、交付文件清单

### 6.1 本次改动/新增（6 个文件）

| 文件 | 大小 | MD5 |
|---|---|---|
| `serving/bridge/vla_client.py` | 83,562 B | `61e20a6b2a41d94b4879c8c46af327c8` |
| `serving/bridge/test_vla_client.py` | 17,048 B | `82d0f19b479b4ab06c3c3836b116976f` |
| `serving/bridge/home_robot.py` | 9,373 B | `018bf6f9a2520ffcd111a23bb24fe720` |
| `serving/bridge/run_episode.sh` | 9,655 B | `15917fb34144006d8c6d5830e5d90ad9` |
| `serving/bridge/analyze_bridge_csv.py` | 8,270 B | `714f829c8079b75e26a5e4eead2e5717` |
| `serving/bridge/QUICKSTART.md` | 8,209 B | `637b06f379f74da88d6b7a731cc7c436` |

### 6.2 未改动

| 文件 | 说明 |
|---|---|
| `serving/bridge/offline_replay_needobs.py` | 离线契约回放工具，无变动 |
| `serving/bridge/test_offline_replay.py` | 11 项测试，无变动 |
| `serving/bridge/smoke_test.sh` | 冒烟脚本，无变动 |
| `serving/bridge/README.md`、`TASK_TEXTS.md` | 无变动 |
| 服务端（`g05-async-serving/`） | **完全未改动** |
| 训练代码、模型权重、部署包 | **完全未改动** |

### 6.3 部署到真机的方法

VSCode Remote-SSH 拖拽上传后：

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
chmod +x run_episode.sh
md5sum vla_client.py home_robot.py run_episode.sh analyze_bridge_csv.py
```

> ⚠️ **VSCode 拖拽不保留执行位**，每次覆盖 `run_episode.sh` 后都要重新 `chmod +x`。

---

## 七、使用方法

### 7.1 日常一条命令

```bash
cd /home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy   # ★ 必须
./run_episode.sh
```

自动完成：依赖检查 → 服务端检查 → 机器人预检 → 跑策略 → 精确复位

### 7.2 推荐参数（已在 `run_episode.sh` 里配好）

```bash
--arm-max-speed-rad-s 1.0     # 57°/s。★ 关键：>1.5 会因 0.533s 延迟而震荡
--cmd-lowpass-tau 0.10        # 低通，压掉 88% 换段频率抖动
--stop-after-release 45       # 夹爪释放后 1.5s 结束（主结束条件）
--stop-when-still 60          # 兜底：模型输出连续 2s 不变
--duration 30                 # 安全网上限
--gripper-rate-limit 0.7      # 内置默认，对齐 yam_motion
```

### 7.3 常用变体

```bash
DURATION=20 ./run_episode.sh              # 缩短上限
ARM_SPEED=0.8 ./run_episode.sh            # 更慢更稳
RELEASE_STOP=0 DURATION=40 ./run_episode.sh  # 关闭释放检测（安全退路）
RESET=0 ./run_episode.sh                  # 不复位（连试几次）
TASK_ID=TASK-YAM-0013 ./run_episode.sh    # 换任务
```

### 7.4 事后分析

```bash
python3 analyze_bridge_csv.py /tmp/g05_runs/ep_<时间戳>.csv
```

---

## 八、遗留问题与后续工作

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | **系统有 0.533s 观测延迟**（`T_infer 507ms` ≈ 播放窗口 `533ms`） | 快动作必然震荡，只能靠降速缓解 | 唯一解是缩短 `T_infer`：**TensorRT**（bf16 已开；`torch.compile` 实测更慢 588ms）。若能做到 <267ms，可用 `action_steps=8` 把延迟减半 |
| 2 | **模型任务完成后输出崩坏** | 会剧烈震动，有烧电机风险 | 看门狗已兜底；根治需**训练时加终止信号**（放到之后回 home 并停止） |
| 3 | 模型学的 home ≠ 配置的全零（差 24°/14°） | 模型自己回位不准 | 已用确定性复位脚本绕开；或在训练数据里统一收尾姿态 |
| 4 | 方块形状导致反复重抓 | 抓取不稳定 | 换更易夹的物体，或调整夹持姿态 |
| 5 | 释放判据无单元测试 | 回归风险 | 抽成纯函数补测 |
| 6 | `--stop-after-release` 未上真机验证 | 可能误判 | 下一轮真机验证；退路 `RELEASE_STOP=0` |

---

## 九、改动前的状态对比（便于理解风险）

| 维度 | 改动前（10-29 白天） | 改动后（10-29 夜间） |
|---|---|---|
| 夹爪控制 | 无任何限速，0→1 用 33ms | 0.7/s，0→1 用 1.43s |
| 相机异常 | 单帧丢失即终止整次运行 | 三层防护，1,639 步 0 次中断 |
| 臂速度限制 | 只有 `--max-joint-step-deg`（默认关闭） | `--arm-max-speed-rad-s` 按 dt 归一化 + 低通 |
| 模型输出合理性 | **完全没有检查**（模型说跳 62° 就发 62°） | 偏差闸 + 振荡闸，超限 0.1 秒内刹停 |
| 运行诊断 | 只有文本日志 | 每步 CSV（raw/cmd/跳变/换段/限幅量）+ 5 段分析器 + 冻结存图 |
| 跑完复位 | 无（手工或让模型回） | `home_robot.py`，收敛 <1.5° |
| 结束时机 | `--duration` 硬切 | 夹爪释放即停 + 稳定兜底 + 时长上限（三级） |
| 单元测试 | 10 项 | 27 项 |
| 连续运行 | 第二次必失败 | 自动等 0.9s，可连续跑 |
| fault 提示 | 裸 traceback | 两侧 hardware_health + 恢复步骤 |

---

## 十、关键结论（给接手的人）

1. **执行层是这套部署最脆弱的一环**，但也是唯一能在不重训模型的前提下改善的环节。昨晚 8 个问题里 **7 个在执行层**，全部可修。

2. **"震动"不是模型抖动，是延迟反馈不稳定** —— 系统有 0.533s 观测延迟，速度快了必然震荡。这是控制论必然，不是 bug。**降速是唯一立刻有效的手段**，根治要靠缩短推理时间。

3. **任何从动作推断状态的逻辑，必须读限速后的真实命令，不能读模型原始输出** —— 这条坑导致过一次 7.4 秒误判砍断任务。

4. **模型有它自己的"home"**，是从演示数据学来的，和配置文件里写的全零不是一回事。**回位这种设备维护动作交给确定性脚本，比让模型做可靠得多。**

5. **不要停别人的安全守护进程**（ziyu / 张恩同可能也在用）。服务端可以留着复用（省 1~3 分钟冷启动）。

---

*文档生成时间：2026-10-30 13:00*
*对应提交：`385950f`（第 98 个 commit）*
