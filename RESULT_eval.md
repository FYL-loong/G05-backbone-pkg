# 离线评测结果（open-loop，第 4 步）

> 目的：回答「微调后的 backbone 到底值不值得上真机」。
> 结论：**值得。** 但只在 10 条 episode / 部分加载数据上验证过，见"限制"。

---

## 1. A/B 设置

| | **A（微调后）** | **B（官方原版）** |
|---|---|---|
| backbone | `runs/yam-ft-1ep/checkpoints/model_state_dict.pt`（4 卡 1 epoch / 17143 步） | `checkpoints/g05-base/checkpoints/model_state_dict.pt` |
| codec | `checkpoints/action_tokenizer_yam.pt`（微调 YAM 版） | `checkpoints/action_tokenizer.pt`（官方版） |
| 归一化 stats | `runs/yam-ft-1ep/dataset_stats.json` | `checkpoints/g05-base/dataset_stats.json` |
| 数据管线 | `configs/task/yam.yaml` | 同左（保证可比） |
| 验证集 | TASK-YAM-0014/0015（**完整留出的两个任务**） | 同左 |

**代码**：`scripts/eval_open_loop.py` 与官方 `GalaxeaVLA` **字节级完全相同**（SHA256 一致，未改）。

---

## 2. 结果（10 条 episode）

| 指标 | A（微调） | B（官方基座） | 相对变化 |
|---|---|---|---|
| `rollout/ar_action_l1` | **0.054483** | 0.499219 | **−89.1%** |
| `rollout/fm_action_l1` | **0.096251** | 0.325423 | **−70.4%** |
| `rollout/ar_action_token_acc` | **0.469991** | 0.078054 | **+502%**（6.0×） |
| `rollout/fm_action_acc` | **0.207494** | 0.020370 | **+919%**（10.2×） |
| `rollout/ar_action_acc` | −1（该指标未计算） | −1 | — |

**L1 类平均改善 +79.8%** → `✅ 微调明显有效`。

**AR 优于 FM**（0.054 vs 0.096）——与部署选择 AR 一致。

**产物**（HF `Zenith6666/yam-backbone`）：
- `eval/A/metrics.json`、`eval/B/metrics.json`
- `eval/A/episodes/{000000,000001,000002}/`：`actions.pkl`（原始 GT/预测数组）+ `result_ar_fm.html` + 4 张 per-part chunk 图

---

## 3. 逐关节体检（数值 + 图，3 条 episode）

用 `actions.pkl` 里的 `raw_gt (T,14)` / `raw_pd (T,32,14)` 直接算，不靠肉眼：

| 检查项 | 结果 |
|---|---|
| **符号反转**（corr < −0.5） | **0 个** ✅ |
| **死关节**（预测 std≈0） | **0 个** ✅ |
| **量级异常**（幅度比 >3 或 <1/3） | **0 个** ✅；幅度比中位数 **1.0009** |
| 相关系数 | 中位数 **0.915**，均值 0.734 |
| **右臂 6 关节 + 右夹爪** | corr **0.86–1.00**，误差/GT变化幅度 **0.19–0.37** → **跟踪优秀** |
| **左臂 L_arm0/L_arm3/L_arm5** | corr 0.75/0.94/0.78 → 跟踪好 |
| **左臂 L_arm1/L_arm2** | GT 本身近似常数（归一化 std 仅 0.0166/0.0109）→ 低相关属正常，均值正确 |
| **左臂 L_arm4** | GT 呈阶梯状，预测有轻微滞后 |
| chunk 内平滑性 | 相邻步平均跳变 左 0.004–0.009 / 右 0.03–0.07，无尖刺 |

**视觉复核**：右臂 6 维预测曲线几乎完全贴合 GT；左臂静态关节预测为带轻微抖动的常数（正确）。

**注意**：这几条 episode 里**右臂承担主要动作，左臂大部分时间近似静止**。若真机任务需要双臂协同，需另找左臂参与度高的任务复测。

---

## 4. 限制（务必一起读）

1. **只评了 10 条 episode** —— 差距 89% 远超噪声，但不足以给"成功率"级结论。
2. **数据只加载了约 1/25**：`total_dataloader_batches=338`（≈5,408 帧，约 21 条 val episode），而数据集元信息是 991 episode。A/B 用同一份数据所以对比有效，但数字只代表已加载部分。**待查**。
3. **open-loop ≠ 闭环**：用数据集里的真实观测喂模型，不验证"误差累积"。真正的判据是真机成功率。

---

## 5. 复现命令

```bash
cd /workspace 2>/dev/null || cd "$HOME"
rm -rf G05-backbone-pkg
git clone https://github.com/FYL-loong/G05-backbone-pkg.git
cd G05-backbone-pkg
GPUS=4 MODE=setup bash start.sh
EVAL_EPISODES=10 GPUS=4 MODE=eval bash start.sh          # A vs B
EVAL_SKIP_BASE=1 EVAL_EPISODES=3 EVAL_UPLOAD_VIZ=1 GPUS=4 MODE=eval bash start.sh   # 只要可视化
```

环境变量：`HF_TOKEN`（平台配置）、`GPUS=4`。
