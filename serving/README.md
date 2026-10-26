# serving/ —— 异步推理服务端

离线验证已接入 `MODE=serve-check`（见仓库根 `start.sh`）。**不接真机、不需要 `gello_software`。**

---

## 1. 这些文件是谁写的（重要，便于日后合并/追溯）

| 文件 | 来源 | 说明 |
|---|---|---|
| `serve_policy.py` | **官方 `GalaxeaVLA/scripts/serve_policy.py` + 张恩同修改** | 官方 530 行 → 他的 503 行（新增 48 / 删除 75） |
| `serve_policy_official_ref.py` | **官方原版**（sha256 `77b942ac…`，19,913 B） | ⚠️ **仅作追溯参考，不参与运行**。以后官方更新 `serve_policy.py` 时可用来做三方合并 |
| `chunked_policy.py` | 🟢 **张恩同原创**（官方无此文件） | 他的核心贡献：把官方内嵌的 `ChunkedPolicyWrapper` 抽出为独立模块，并改为**异步** |
| `fake_policy_client.py` | 🟢 **张恩同原创** | 合成客户端，走完整协议 |
| `offline_chunk_harness.py` | 🟢 **张恩同原创** | 不起 WS、直接驱动 wrapper 测时序 |
| `tests/` + `conftest.py` | 🟢 **张恩同原创** | 3 个无 GPU 单测 |
| `README.md`（本文件） | 我们 | — |

### 他到底改了什么（diff 实测）

```
官方 serve_policy.py 里有一个内嵌类：
    class ChunkedPolicyWrapper:
        """Wraps model inference with chunk caching and step-by-step serving."""

他做的两件事：
① 把这个类抽成 chunked_policy.py（-75 行）
② 把推理调用改成异步，避免阻塞 WebSocket 事件循环（+48 行）：
       async def compute(raw_obs):
           actions = await asyncio.to_thread(inferencer.infer, [obs_dict])
```

**② 是"异步推理"真正落地的地方**：WebSocket 服务端是单线程事件循环，若在循环里直接跑
1~2 秒的推理，整条连接会冻结那么久。`asyncio.to_thread` 把推理甩到线程池，事件循环继续收发。

### 他 README 里自述的三个"未验证"点

1. 从未在真实 ckpt 上跑过（开发机磁盘不够）
2. **AR slot 对齐**（YAM 6+2 关节 vs R1-Lite 27 维的顺序/符号）
3. `frequency` 参数必须显式发（默认 15 是 SO-100 的，对 YAM@30Hz 是分布漂移）

> 第 2 点 **已在第 4 步（`MODE=eval`）被间接验证**：评测走的是同一套
> `processor` / `PolicyInferencer` / `postprocess`，逐关节相关系数 0.86~1.00、0 符号反转。
> `MODE=serve-check` 补的是"网络传输 + chunk 缓存"这层。

---

## 2. 协议（客户端必须照此实现）

**连接后服务端先发：**
```python
{"action_steps": 16}          # 每 16 步重算一次 chunk
```

**客户端发观测（msgpack，编码见 `g05.utils.websocket`）：**
```python
{
  "embodiment_type": "galaxea_r1lite",
  "images": {                 # 注意是 [C,H,W] uint8，不是 [H,W,C]
    "head_rgb":       ndarray[3,480,640] uint8,
    "left_wrist_rgb": ndarray[3,480,640] uint8,
    "right_wrist_rgb":ndarray[3,480,640] uint8,
  },
  "state": {
    "left_arm":      ndarray[6] float32,
    "left_gripper":  ndarray[1] float32,
    "right_arm":     ndarray[6] float32,
    "right_gripper": ndarray[1] float32,
  },
  "task": "Pick up the small square and place it in the groove",
  "frequency": 30.0,          # ★ 必须发！默认 15 会导致分布漂移
}
```

**服务端回（一次一步，已反归一化成绝对关节目标）：**
```python
{
  "action": {part: ndarray[dim]},   # 可能缺 key（AR 头不置信的部件）
  "need_obs": True/False,
}
```

**客户端必须遵守的 4 条：**

| # | 规则 | 原因 |
|---|---|---|
| 1 | `need_obs == False` 时发空 `{}`，不要发真实观测 | 服务端从缓存直接返回，这条路径要**瞬间**完成 |
| 2 | `action` 里缺的 key → **保持上一帧的值**，不要填 0 | AR 头偶尔不输出某些部件；填 0 = 突然把关节命令到 0，危险 |
| 3 | episode 开始时发 `{"__reset__": True}` | 清空 chunk 缓存，否则会拿上一段的残余动作 |
| 4 | 图像是 **`[C,H,W]`** uint8 | 服务端按此做 resize/normalize |

---

## 3. 怎么跑

```bash
# 前置：MODE=setup 已跑过（.venv / 权重 / processor 就位）
GPUS=4 MODE=serve-check bash start.sh
```

四个子步：

| 子步 | 内容 | 产出 |
|---|---|---|
| **S1** | `pytest serving/tests -q` | 验证 chunked 状态机 / obs 构建 / 编解码（无 GPU） |
| **S2** | `offline_chunk_harness.py --action_steps 1 --steps 6 --rate 0` | **`T_infer`**（RTC 模式每步都重算） |
| **S3** | `offline_chunk_harness.py --action_steps 16 --steps 48 --rate 30` | **换段是否阻塞**、实测有效频率 |
| **S4** | 起 `serve_policy.py` + `fake_policy_client.py` | WebSocket 链路 + 协议正确性 |

**环境变量：**

| 变量 | 默认 | 含义 |
|---|---|---|
| `SERVE_ACTION_STEPS` | 16 | 每多少步重算一次 |
| `SERVE_RATE` | 30 | 控制频率 |
| `SERVE_PORT` | 8765 | 端口 |
| `SERVE_TINFER_STEPS` | 6 | S2 步数 |
| `SERVE_PLANB_STEPS` | 48 | S3 步数 |
| `SERVE_WS_STEPS` | 64 | S4 步数 |
| `SERVE_SKIP_WS` | 0 | 设 1 跳过 S4 |

---

## 4. 判据

```
前瞻窗口 = action_steps / rate × 1000 = 16 / 30 × 1000 = 533 ms
```

| `T_infer` | 判定 | 应对 |
|---|---|---|
| **≤ 533 ms** | ✅ 异步前瞻能完全吸收，实时可行 | 进真机阶段 |
| 533 ms ~ 1.5 s | ⚠️ 每 16 步「饿」一下 | ① 确认 bf16（`model.model_weights_to_bf16`，代码默认 True）② `model.use_torch_compile=true` ③ 降 `SERVE_RATE` |
| > 1.5 s | 🔴 明显卡顿 | 同上，且需重新评估控制节奏 |

**注意**：`action_steps` 调大**不是**解法 —— `chunked_policy.py:122` 要求 `chunk_len >= 2*action_steps`，
我们 `chunk_len=32`、`action_steps=16` 已恰好是上限。要更长窗口必须改模型 `horizon` 并重训 codec+backbone。

---

## 5. 还没做的（阻塞项）

| # | 缺什么 | 阻塞什么 |
|---|---|---|
| 1 | **`gello_software`**（实验室 YAM 驱动/桥接代码） | 客户端**硬件层**（读关节/读相机/写动作/急停 4 个方法） |
| 2 | YAM 14 维关节顺序 + 符号 + 夹爪量纲 | 同上，且符号错会直接撞机 |
| 3 | 相机型号/路数/分辨率/取流方式 | 图像采集 |
| 4 | 30 Hz 能否达到、动作是位置还是增量 | `action_steps` 取值 |
| 5 | 安全限制（软限位/速度上限/急停接口） | 首次上机安全 |

**协议层可以现在写**（对着 `fake_policy_client.py` 测），只把硬件层留空 —— 见项目计划。
