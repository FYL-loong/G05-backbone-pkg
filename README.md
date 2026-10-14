# G05-backbone-pkg

G0.5 复现第二步 —— **backbone（VLA 全模型）微调**代码。从上游
[OpenGalaxea/GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 抽取微调 + 评测所需
部分，供算力平台 clone 后直接微调。

> 配套：ActionCodec 码本微调在 `ActionCodec-finetune-pkg`（产物
> `action_tokenizer_yam.pt` 已上传 HF，本仓库 `download_weights.sh` 会拉取）。

## 目录

```
├── scripts/
│   ├── finetune.py               # 训练入口
│   ├── eval_open_loop.py         # open-loop 评测入口
│   ├── run/finetune.sh           # 训练 launcher：bash scripts/run/finetune.sh <n_gpu> yam ...
│   ├── run/eval_open_loop.sh     # 评测 launcher
│   └── utils/                    # finetune.py 依赖（metric/preflight/eval_snapshot/train_eval）
├── src/g05/                      # 模型 + 数据 + processor 全部源码
├── configs/                      # train.yaml / model/g05.yaml / task/yam.yaml / data/yam.yaml
├── pyproject.toml + uv.lock      # uv 依赖（Python 3.10 + cu128 torch）
├── setup_env.sh                  # 装 uv + uv sync
├── download_weights.sh           # 下 backbone + processor + codec + stats
├── download_dataset.sh           # 下 data/yam/TASK-YAM-*_lerobot
└── upload_output.sh              # 上传微调产物到 HF
```

## 算力平台 runbook

```bash
# 0. 克隆 + 装环境（uv sync）
git clone https://github.com/Zenith9320/G05-backbone-pkg && cd G05-backbone-pkg
bash setup_env.sh
source .venv/bin/activate

# 1. 设 token + 输出目录
export HF_TOKEN="hf_xxxxxxxx"                      # 读写权限 token
export G05_OUTPUT_DIR=/workspace/output            # 见下方「输出目录」说明

# 2. 下载权重 + 数据集（脚本自带字节数校验，能查出截断）
bash download_weights.sh        # g05-base + processor + action_tokenizer_yam.pt + yam_stats.json
bash download_dataset.sh        # 全部 10 个任务（data/yam/TASK-YAM-*_lerobot）
#    只要单任务可加：ONLY_TASK=TASK-YAM-0006 bash download_dataset.sh

# 3. 五级验证（逐级通过再正式训练）
#    L1 配置组装（不加载权重、不需要 GPU 显存）
bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
#    L2 数据管线 + forward/backward（显式置空权重，秒级起，不验证权重加载）
bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
    model.pretrained_ckpt=null model.max_steps=10
#    L3 ★ 权重加载验证（这一级才有 11GB 加载）
bash verify_weight_load.sh
#    L4 单批过拟合（loss 应明显下降）
bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500 \
    logger.mode=offline
#    L5 少量数据真实训练，确认存盘与上传通路
bash scripts/run/finetune.sh 1 yam --max_datasets 1 model.max_steps=20 \
    logger.mode=offline

# 4. 正式训练（GPU 数按平台定；wandb 离线）
nohup bash scripts/run/finetune.sh 8 yam logger.mode=offline \
    > /workspace/train.stdout.log 2>&1 &
echo "PID=$!"

# 5. ★ 训练期间定期打这个（容器临时，产物必须落 HF）
EXP_DIR=$G05_OUTPUT_DIR/yam/<exp> bash upload_checkpoints.sh
#    先看会传什么：DRY_RUN=1 EXP_DIR=... bash upload_checkpoints.sh
#    只传最后一个：MODE=last EXP_DIR=... bash upload_checkpoints.sh

# 6. open-loop 评测（指向产出 ckpt）
bash scripts/run/eval_open_loop.sh \
  --ckpt_path $G05_OUTPUT_DIR/yam/<exp>/last.pt \
  --task_config configs/task/yam.yaml

# 7. 训练结束后的最终收尾上传
EXP_DIR=$G05_OUTPUT_DIR/yam/<exp> bash upload_output.sh
```

### 为什么 L2 和 L3 要分开

L2 用 `model.pretrained_ckpt=null`，会让 `scripts/finetune.py` 走
`else: model = instantiate(cfg.model.model_arch)`（finetune.py:572-573）
—— **权重随机初始化，完全不加载那 11GB**。它只验证「数据管线 + forward/backward 不崩」。

**权重加载是最容易静默出问题的一步**：`load_state_dict_safely` 遇到维度不匹配时
只打 warning 就继续（partial load），不会报错。所以必须有 L3 显式验证：

- 日志出现 `Loaded N/N` 且 `Missing (rand init): 0`
- 2 步跑完 loss 是有限值（非 nan/inf）

### 数据规模与验证集（重要，容易被误解）

**训练数据**：`Zenith6666/yam-lerobot` 上共 **10 个任务、991 条 episode**，全部 30 fps、
`robot_type = yam_bimanual`：

| 任务 | episodes | 指令 |
|---|---|---|
| TASK-YAM-0001 | 101 | Hit the ball with claws to score a goal |
| TASK-YAM-0002 | 100 | Pick up the seal, dip it in the ink pad, and stamp. |
| TASK-YAM-0004 | 100 | Grab the ham sausage to the foam box |
| TASK-YAM-0006 | 100 | Pick up the small square and place it in the groove |
| TASK-YAM-0007 | 100 | One gripper holds the bottle, and the inner side of the other gripper rubs open the bottle cap. |
| TASK-YAM-0008 | 91 | One gripper clamps the bottle, and the other gripper rotates to open the cap. |
| TASK-YAM-0010 | 99 | One gripper holds the gum bottle, and the other gripper pops open the lid. |
| TASK-YAM-0013 | 100 | Sort the blocks from left to right according to their size. |
| TASK-YAM-0014 | 100 | Pick up the towel and wipe the whiteboard with it. |
| TASK-YAM-0015 | 100 | Put items in the drawer. |
| **合计** | **991** | |

**⚠️ 验证集是「留出的两个完整任务」，不是每个任务各取 20%**

代码先把各 `dataset_dir` 的 episode 列表**按顺序拼接**，再在拼接结果上切一刀
（`base_lerobot_dataset.py:281-297`）：

```
split_ep = int(991 × (1 - 0.2)) = 792
train = 前 8 个任务（0001…0013）   = 792 条
val   = 后 2 个任务（0014, 0015）  = 199 条
```

**含义**：
- validation 衡量的是「**held-out 任务的零样本表现**」，不是同任务分布的泛化
- **验证 loss 偏高 ≠ 训练失败**，只代表模型没见过这两个任务
- 按 validation loss 选 best checkpoint 会**偏向** 0014/0015

如果想让验证集覆盖全部 10 个任务，需要改 `base_lerobot_dataset.py`（**官方代码**）
做每数据集分别切分 —— 本包**有意不改**，以保持与官方一致、便于排查问题。

### 输出目录 `G05_OUTPUT_DIR`

`configs/train.yaml` 用它拼 Hydra 输出目录，**且无默认值**，不设置会直接报错：

```yaml
dir: "${oc.env:G05_OUTPUT_DIR}/${hydra:runtime.choices.task}/${exp_name}"
```

**平台容器是临时的、没有持久盘**，所以写到容器内路径（如 `/workspace/output`）即可，
靠 **HF 落地**。注意磁盘占用：单个 checkpoint 约 **11.44GB**，
`checkpointing_steps: 500` 下 10 epoch 约产生 2–3 个（≈29GB）。

### 通过标准

- L3/L4 日志：`Loaded N/N`、`Missing (rand init): 0`，loss 稳定下降。
- 输出目录 `$G05_OUTPUT_DIR/yam/<exp>/`：`checkpoints/`、`dataset_stats.json`、
  `.hydra/config.yaml`、`action_tokenizer.pt`。
- HF 仓库 `Zenith6666/yam-backbone` 上有 `checkpoints/step_*.pt` 与各 sidecar。

## 权重路径对齐（脚本已保证）

| 文件 | 来源 | 配置引用 |
|---|---|---|
| `checkpoints/g05-base/checkpoints/model_state_dict.pt` | `OpenGalaxea/G05` | `configs/model/g05.yaml` → `pretrained_ckpt` |
| `checkpoints/qwen3_5_2b_base_processor/` | `OpenGalaxea/G05` | `configs/model/g05.yaml` → `hf_processor_path` |
| `checkpoints/action_tokenizer_yam.pt` | `Zenith6666/yam-actioncodec` | `configs/task/yam.yaml` → `tokenizer.vq_config.ckpt_dir` |
| `data/stats/yam_stats.json` | `Zenith6666/yam-actioncodec` | `configs/task/yam.yaml` → `datastatics_path` |
| `data/yam/TASK-YAM-XXXX_lerobot` | `Zenith6666/yam-lerobot` | `configs/data/yam.yaml` → `dataset_dirs` |

## 说明

1. **任务指令文本**：backbone 是 VLA，需要每条 episode 的真实自然语言指令。数据集已按
   `ActionCodec-finetune-pkg/tools/yam/tasks_yam.json` 的真实英文指令转换，parquet `task` 字段已就绪。
2. **全参微调**：`configs/task/yam.yaml` 已配 `vla_training_strategy: vla-full-train`
   + differential LR（`vision_lr_multiplier: 0.1` / `backbone_lr_multiplier: 1.0`）。
