# TODO —— G0.5 VLA × I2RT YAM 真机部署（进度与交接）

> **更新**：2026-10-27 15:45　|　**维护**：陈远 + AI assistant
> **计划来源**：张恩同的 `g05-async-serving/TODO.md`（2026-10-26，225 行，仍在他的仓库里未提交）
> **技术手册**：`RUNBOOK.md`（324 行，启动/排错/回滚全在里面）
> **本文件用途**：交付用进度台账 —— 一眼看清"做到了哪、还差什么、谁接手怎么继续"。

---

## 0. 一页总览

| 阶段 | 内容 | 状态 | 证据 |
|---|---|---|---|
| 前置 | 环境 | ✅ | 3.10 推理 / 3.11 桥接双环境跑通 |
| **第 1 步** | 模型 + `T_infer`（**决策关口**） | ✅ | **p50 = 513 ms** < 预算 533 ms |
| **第 2 步** | 端到端 serving 冒烟 | ✅ | 96/96 步，`behind_schedule=0` |
| **第 3 步** | 机器人控制打通 | 🟡 代码完成，**真机未验证** | ZMQ + 安全会话，待 Level 0/1 |
| **第 4 步** | 相机接入 | ✅ | 30/30 帧、30 Hz、三路非零 |
| **第 5 步** | bridge 客户端 | ✅ | 9 个补丁，单测 10/10 + 11/11 |
| **第 6 步** | 联调验收 | ⬜ **未开始** | 依赖第 3 步 |
| 加餐 A | 部署包装配 | ✅ | 11 G，`/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026` |
| 加餐 B | 离线契约验证 | ✅ | 右臂 6 维 `corr 0.966~0.999` |
| 加餐 C | 10 个任务文本 | ✅ | 991 episodes 对齐，内置 `--task-id` |

**一句话**：**离线全部做完并验证通过，只差真机动作（第 3 步收尾 + 第 6 步）。**

---

## 1. 原 TODO 的「待补信息」—— 两个空项已闭环

| 原条目 | 状态 | 结论 |
|---|---|---|
| 关节对齐 | ✅（原作者已定） | 恒等映射 `[左臂6,左夹爪1,右臂6,右夹爪1]`，无重排/无符号翻转；我们实测 6 维 corr 0.966~0.999 **再次独立证实** |
| 相机 shape_meta | ✅（原作者已定） | 3 路 `(3,480,640) uint8`；`main→head_rgb / left→left_wrist_rgb / right→right_wrist_rgb` |
| **任务名样例** | ✅ **我们已填** | 10 条权威文本，见 `serving/bridge/TASK_TEXTS.md`；来源 HF `Zenith6666/yam-lerobot` 各 `meta/tasks.parquet` **的 index**（不是列！） |
| **安全接入方式** | ✅ **我们已定并实现** | 走 `yam_safety_daemon`：`ZMQClientRobot(:6001)` + `SafetyClient(:6002)` + `SupervisedMotionSession`（**400 ms 看门狗 + 软限位 + 独占会话**）。**不直连 CAN**，因为直连会抢总线且没有看门狗 |

---

## 2. 已完成 —— 逐项带证据

### 2.1 第 1 步 `T_infer`（决策关口）✅

| 测法 | 结果 | 说明 |
|---|---|---|
| `offline_chunk_harness.py`（RTC，`--action_steps 1 --rate 0`，19 步中位数） | **571 ms** | 含每步重建 obs 的开销，偏高 |
| **真实服务端 `Recompute:` 日志（19 次，p50）** | **513 ms** | ★ **这才是部署时的真实值** |
| 预算（`action_steps=16 @30Hz`） | 533 ms | 余量 **20 ms** |
| 同事同卡同 codec 基线 | 518.7 ms | 我们与他相差 1% |

- 首包特例：第一次 forward 含 CUDA autotune，实测 **~12 s** → bridge 已做**首包 180 s / 稳态 10 s** 的双超时
- **20% 优化已评估并否决**：160 token = `tokenizer_meta` 的 5 组 `(left_control,left_gripper,right_control,right_gripper,lower_body)` × 4 codebook × 8，**训练时就这么学的**，砍 token 需重训 → 记入「不做」，加速改走 TensorRT

### 2.2 第 2 步 serving 冒烟 ✅

```
fake_policy_client : 通过（need_obs 握手、chunk 触发每 16 步）
我们的 bridge      : steps completed=96  behind_schedule=0  退出码 0
动作契约           : {'left_arm':(6,), 'right_arm':(6,), 'left_gripper':(1,), 'right_gripper':(1,)}
服务端 Recompute   : 490~513 ms（稳态）
```

### 2.3 第 3 步 机器人控制 🟡 代码完成

已实现（`vla_client.py --robot-backend zmq`）：

| 机制 | 实现 |
|---|---|
| 观测/下发 | `ZMQClientRobot(host, 6001)`；`command_period_s = None`（对应 `command_timing: legacy_30hz`） |
| 安全会话 | `with SupervisedMotionSession(safety, uuid, operation="g05_vla_bridge")` — 内部自动 `begin_session` → 两个 200 ms 心跳线程 → 等 `stale_clients` 清空 → `activate()` |
| 接管前自检 | `state` 必须 `== safe_hold` 且 `stale_clients` 非空，否则拒绝（返回码 6） |
| 释放 | `__exit__` 自动 `hold()` 并要求守护进程确认 SAFE HOLD |
| 退出码 | 2 机器人异常 / 3 服务端错误 / 4 收包超时 / 5 相机丢帧 / 6 preflight 失败 |

**已实测的只读探测**（`--check-robot`）：
```
state='safe_hold'  hold_confirmed=True  stale_clients=['collector','parent']
hardware_health: 双侧 ok，feedback_age 0.2/1.0 ms，comm_freq 253/260 Hz，温度 39–50°C
真实关节限位（yam_pro）：[-2.768,3.292] [-0.15,3.815] [-0.15,3.292]
                        [-1.843,1.721] [-1.721,1.721] [-2.244,2.244] [0,1]   ×左右
```

**⬜ 未做**：Level 0（DRY）/ Level 1（原地保持）/ Level 2（闭环）—— 见 §4。

### 2.4 第 4 步 相机 ✅

```
--camera-test：通过 30/30 帧
ts 严格 +0.0333 s（精确 30 Hz）
means = head 77~127 / left 82~89 / right 73~78   ← 三路都真实出图
单帧耗时 0.5~1.6 ms（帧已在缓存）
```
**由此发现并修复**：自动曝光要 ~1 秒才收敛（head 均值 77→127）→ 新增 `--camera-warmup-s`（默认 1.5 s，丢弃前 1.5 s 帧，不发指令）

### 2.5 第 5 步 bridge ✅ —— 9 个补丁

| # | 问题 | 修法 | 级别 |
|---|---|---|---|
| 1 | `capture_images()` 是 `NotImplementedError` | 复用 `yam_motion.cameras.YamCameras`（因果帧 + max_age） | 🔴 |
| 2 | `--fake-images` 仍会 `create_pair` → **夹爪标定真动机械臂** | 新增 `--mock-robot`；禁止「合成图+真机器人」 | 🔴 安全 |
| 3 | 阻塞 0.5 s 后把缓存步**几十毫秒内连发** | 请求耗时 > 1 周期即 `next_tick = t_got` 重对齐（丢弃而非补放） | 🔴 |
| 4 | `ws.recv()` 无超时 | `asyncio.wait_for` + 超时 safe hold | 🟡 |
| 5 | `websockets ≥14` 读 `HTTPS_PROXY=:7897`（**死端口**）→ 连本机被劫持 | `_ws_connect(proxy=None)`（同事的 `fake_policy_client` 正因这个**必挂**） | 🔴 实测 |
| 6 | 统一 5 s 超时把**首次 12 s** 推理掐死 | 首包 180 s / 稳态 10 s | 🔴 实测 |
| 7 | 安全停机却返回 **0**，上层误判成功 | 返回 2/3/4/5/6 | 🟡 |
| 8 | 直连 CAN 抢总线、**无 400 ms 看门狗** | `--robot-backend {zmq,direct,mock}`，默认 zmq | 🔴 真机安全 |
| 9 | 相机自动曝光未收敛就开跑 | `--camera-warmup-s`（默认 1.5 s） | 🟡 |

**测试**：`test_vla_client.py` 10/10；`test_offline_replay.py` 11/11；`offline_replay_needobs.py --self-test` 5/5（注入符号翻转/尺度×57/左右互换/夹爪↔臂互换全部正确识别）

### 2.6 加餐 A：部署包 ✅

```
/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026   11 G
```
装配脚本 `tools/build_deploy_pkg.sh`（**双模式**）：
- 在线：`snapshot_download` + `HF_ENDPOINT=hf-mirror` + 清空代理（本机直连 HF 超时、走代理 TLS 被掐）
- **离线**：`LOCAL_HF=<dir>` 直接用本地文件（走"自己电脑下载 → 传"的稳妥路径）

内建断言：5 个产物**逐字节校验**、`hf_processor` 8 文件、链式插值展开、`${run_dir}/...` 路径改写、`launch.sh` 模型目录/端口替换、`task` 用**大写 P**。

> ⚠️ `runtime/env` 是**符号链接** → 指向 `g05_ar5000_20261013/runtime/env`。**删那个包会连带弄坏本包。**

### 2.7 加餐 B：离线契约验证 ✅（上真机前最重要的一道闸门）

用 **2026-10-15 真实会话录制**（`live_ar32_20261015_230527.json` + 3 路 mp4 + `frames.jsonl`）：
真实视频帧 + 真实 `actual` 关节状态 → 走**我们 bridge 的同一份** `build_obs/packb/action_to_target`
→ `need_obs` 协议 → serve_policy（`--action_steps 1` 保证每次真推理）→ 与日志里 AR5k 的 `predicted_actions[0]` 逐维对比

**结果（19 个采样）**：
```
R_arm1..6  corr = +0.999 +0.999 +0.998 +0.966 +0.978 +0.995      ← 决定性证据
幅度比 RMS(ours)/RMS(ref) = 0.995
|ours-state| p95 最大 = 0.1203 rad        （绝对动作语义正确）
增量方向与 Δstate 中位相关 = +0.399
全部动作落在真实关节限位内（夹爪 ∈ [0,1]）
msgpack 往返无损（uint8/float32）
→ ✅ 契约验证通过
```
**逐项排除的故障**：关节顺序错 / 符号翻转 / 度弧度混用 / 绝对-相对搞错 / 相机通道或命名错 / msgpack dtype 丢失 / 归一化错。

**豁免说明（不是掩盖）**：左臂 6 维 + 左夹爪在该次会话里**静止**（`std(state)` ≤ 0.004 rad），
两个模型在静止关节上各自自由浮动、相关性无意义 → 标为「静止(豁免)」，不参与判定。

### 2.8 加餐 C：10 个任务权威文本 ✅

来源：HF `Zenith6666/yam-lerobot` 各子数据集 `meta/tasks.parquet` **的 index**（LeRobot v3 把文本存 index，不是列），
合计 **991 episodes**，与训练数据一致。已内置进 `vla_client.TASK_TEXTS`，提供 `--task-id` / `--list-tasks`。

**易错点（实测踩过）**：0002/0008/0010/0013 **结尾有句号**，其余没有；全部首字母大写；0007/0008/0010 是 90+ 字符长句。
0006 曾因小写 `p` 跑错 → 现在 `--task-id` 与 `--task` 冲突时**以 task-id 为准并告警**。

---

## 3. 交付物清单

| 文件 | 位置 | 作用 |
|---|---|---|
| `RUNBOOK.md` | 仓库根 | **技术手册**：启动顺序 / 真机分级 / 排错表（10 条实际坑）/ 回滚 |
| `serving/bridge/vla_client.py` | | **桥接客户端**（9 补丁 + `--robot-backend` + `--task-id` + `--session-test` + `--camera-test`） |
| `serving/bridge/offline_replay_needobs.py` | | **离线契约验证**（含 `--dry-run` / `--from-npz` / `--self-test`） |
| `serving/bridge/test_vla_client.py` | | 纯函数单测 10 项 |
| `serving/bridge/test_offline_replay.py` | | 回放脚本本地回归 11 项 |
| `serving/bridge/smoke_test.sh` | | 一键冒烟（起服务端→fake client→我们的 bridge→自动收尾） |
| `serving/bridge/TASK_TEXTS.md` | | 10 个任务文本 + 易错点 + 部署包同步脚本 |
| `serving/bridge/README.md` | | 补丁表与用法 |
| `tools/build_deploy_pkg.sh` | | 部署包装配（在线/离线双模式，内建字节断言） |
| `verify_bridge_files.sh` | 家目录 | 校验 9 个交付文件 md5（自报哪个不符） |
| `RESULT_eval.md` | 仓库根 | 开环 A/B 评测结论（`ar_action_l1` +89.7%） |

**仓库**：`https://github.com/FYL-loong/G05-backbone-pkg`（`main`，78 commits）

---

## 4. 剩余工作（**只剩真机**）

### 4.1 真机四级（每级都要人工盯，急停在手边）

| 级别 | 命令 | 判据 | 状态 |
|---|---|---|---|
| **L0 DRY** | `--session-test 3 --session-dry` | `命令次数=0`、释放后回 `safe_hold` + `hold_confirmed=True` | ⬜ |
| **L1 保持** | `--session-test 5` | 有效频率 ≈ 30 Hz、机械臂基本不动 | ⬜ |
| **L2 闭环 10 s** | 完整 bridge + `--max-joint-step-deg 2 --duration 10` | 动作量级合理、`behind_schedule` 不增长 | ⬜ |
| **L3 真实任务** | 去掉限幅/时长限制，TASK-YAM-0006 场景 | 能抓住小方块放进凹槽 | ⬜ |

### 4.2 前置条件 / 依赖

| 依赖 | 现状 | 风险 |
|---|---|---|
| `yam_safety_daemon`（:6001/:6002） | **ziyu 启的进程**，当前 `safe_hold` 空闲 | ⚠️ 他随时可能回来跑 Pi_0.5 → 每次开跑前 `--check-robot` |
| 服务端（:18879） | 需手动起 | 端口 8765 被别人占着 |
| 相机 RealSense | 可用 | 若 ziyu 相机栈在跑会 device busy |
| GPU | 24 GB，ziyu 的 Pi_05 常驻 8.6 GB（util 4%） | 显存够（我们 bf16 ~7 GB） |

### 4.3 真机阶段要回答的 6 个假设（RUNBOOK §8）

- [ ] 真机动作质量与离线一致？
- [ ] **左臂会配合吗**（回放素材里左臂全程未动，可能训练数据左臂动作少）
- [ ] 20 ms 余量够用吗（相机取帧+抖动可能再吃几十 ms）
- [ ] 长时稳定性（离线只跑过 3.2 s）
- [ ] 守护进程会不会被抢
- [ ] 夹爪输出可用吗（AR5k 的右夹爪是退化的，我们的没单独验过）

---

## 5. 明确「不做」并说明理由

| 项 | 结论 | 理由 |
|---|---|---|
| 跳过 `lower_body` 省 20%（513→410 ms） | ❌ **不做** | 160 token 由 `tokenizer_meta` 的 5 组布局决定，**训练时就按这个学的**；砍 token 会让 `decode_ar` 的 `expected_lengths` 反查失败（5×nr×8=128 → nr=3.2 非整数）→ 必须重训 |
| 继续训练 2–3 epoch | ❌ **暂不做** | 开环已证 1 epoch 足够（`ar_action_l1` +89.7%），预计收益小、成本 8 h/epoch × 4 卡 |
| 官方 codec 基线对比 | ❌ 不做 | 项目已定"只用微调后的 codec" |
| `torch.compile` | ❌ 不做 | 同事实测更慢（587 vs 519 ms） |

---

## 6. 交接：接手的人怎么继续

```bash
# 1) 确认文件都是最新版
bash ~/verify_bridge_files.sh          # 期望 9/9 ✓

# 2) 通读技术手册
less ~/workspace/g05-yam-deployment/G05-backbone-pkg/RUNBOOK.md

# 3) 一分钟自检（RUNBOOK §10 可整段复制）
# 4) 从 §4.1 的 Level 0 开始推进
```

**三条铁律**：
1. **每次开跑前先 `--check-robot`** —— 确认 `state=safe_hold` 且 `stale_clients` 非空，否则会被别人抢
2. **先 `--session-dry`（零指令）→ 再 `--session-test`（原地保持）→ 才闭环**
3. **不允许绕过守护进程直连 CAN**（`--robot-backend direct` 仅调试用，无看门狗）

---

## 7. 遗留疑问（不阻塞交付）

| # | 疑问 | 现状 |
|---|---|---|
| 1 | 训练时 `total_dataloader_batches=338` ≈ 5,408 帧，而数据有 991 episodes / 数十万帧 —— 疑似只加载了约 1/25 | A/B 评测两边同数据，结论仍成立；但若真有 bug，模型见过约 1/25 数据即已 +89.7% → 修了可能更好 |
| 2 | AR5k 模型的右夹爪输出退化（恒 0，后期跳 1） | 是他们那侧的问题，不影响我们；但说明"夹爪预测"在这个 codec 上偏弱，真机需重点看 |
| 3 | 左臂在录制会话里全程静止 | 可能是任务本身只用右臂；但需真机确认左臂响应 |
| 4 | 部署包依赖 `g05_ar5000_20261013` 的 env 符号链接 | 该包被删则本包失效；要独立就把 env 实体复制过来（~8 G） |

---

## 8. 变更记录

| 日期 | 事项 |
|---|---|
| 2026-10-17 | backbone 4×GPU 微调完成（17,143 steps，ce_loss 1656→2.099） |
| 2026-10-26 | 开环 A/B 评测完成（`ar_action_l1` 0.0545 vs 0.4992） |
| 2026-10-27 | 部署包装配完成；`T_infer`=513 ms；serving 冒烟通过；bridge 9 补丁；离线契约验证通过；相机验证通过；10 任务文本收齐；RUNBOOK 完成 |
