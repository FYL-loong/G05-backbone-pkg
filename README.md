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

> **平台约束（实测）**：镜像只能用官方的、容器**每次任务都是全新的**、**没有交互式终端**。
> 所以流程做成了「一条命令全自动跑完」，中间的 L3 权重加载门禁由 `start.sh` **自动判定**
> （见 `verify_weight_load.sh` 的日志解析），不通过就停下、不会带着坏权重去训练。

### 镜像与 Python 版本（**选错必失败**）

| 项 | 要求 | 原因 |
|---|---|---|
| 镜像 | **`pytorch-ngc`，CUDA ≥ 12.8** | `pyproject.toml` 把 torch 钉在 `pytorch-cu128` index；`deepspeed` 等需 `nvcc` 就地编译 |
| 推荐 tag | **`25.04-py3`**（`2.7.0-cu12.8.0-py3.12-ubuntu24.04`） | 与 `torch==2.7.1` 同代，cu128 精确匹配 |
| 可选 tag | `25.05-py3` / `25.06-py3`（均为 cu12.8.x） | 同 CUDA 12.8 线 |
| ❌ 不要用 | `2.5.0-cu12.6.2-py3.10`（codec 用的那个） | CUDA 12.6 < 12.8 |
| ❌ 不要用 | `2.4.0` 及更早（≤ CUDA 12.4） | 太旧 |

**⚠️ 为什么 Python 版本是关键**：

```
.python-version             = 3.10.16
pyproject requires-python   = ">=3.10.16,<3.11"
NGC 镜像自带                 = Python 3.12      ← 不满足约束
```

→ **`uv` 必须自己下载一个独立的 Python 3.10**，默认源是 **GitHub Releases**
（`astral-sh/python-build-standalone`）。

**实测网络（2026-03，国内）**：

| 域名 | 用途 | 可达 |
|---|---|---|
| `mirrors.aliyun.com` | `uv.lock` 里 1049 个包（293 个 registry 条目） | ✅ |
| `download.pytorch.org` | torch cu128 轮子（69 条） | ✅ |
| `pypi.nvidia.com` | nvidia 依赖 | ✅ |
| `astral.sh` | 安装 `uv` | ✅ |
| `gh-proxy.com` | Python 3.10 下载镜像 | ✅ |
| **`github.com`** | **uv 下载 Python 的默认源** | ❌ **不可达** |

**所以 `start.sh` / `setup_env.sh` 默认启用 Python 下载镜像**：

```bash
UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
UV_HTTP_TIMEOUT=300     # torch 轮子数百 MB，默认超时不够
```

**如果该镜像也不通**，回退官方源（需要能访问 GitHub）：

```bash
UV_PYTHON_INSTALL_MIRROR= bash start.sh
```

### 依赖里哪些要编译

`uv.lock` 已锁死版本，**大部分是预编译 wheel**：

| 包 | 形态 | 需编译 |
|---|---|---|
| `flash-attn-4` | `py3-none-any.whl`（纯 Python） | ❌ |
| `flash-linear-attention` | `py3-none-any.whl` | ❌ |
| `liger-kernel` | `py3-none-any.whl` | ❌ |
| **`deepspeed` 0.15.4** | **只有 sdist** | ✅ **需要 `nvcc`** |
| 另有 11 个 | 仿真/几何依赖（`gym` / `robomimic` / `bddl` / `toppra` / `easydict` / `egl-probe` / `antlr4-python3-runtime` / `sharedarray` / `asciitree` / `future` / `timeout-decorator`） | ✅ 纯 C/C++，不需 CUDA |

**这 12 个包里任何一个编译失败，整个 `uv sync` 都会失败**（即使其中多数是仿真依赖、实际训练用不到）。

### 一条命令（全自动，无人值守）

平台任务的「执行命令」框里填这一条 —— **不需要任何 MODE 参数**：

```bash
set -e
export HF_TOKEN="hf_你的token"
git clone https://github.com/FYL-loong/G05-backbone-pkg.git G05-backbone-pkg
cd G05-backbone-pkg
bash start.sh
```

`HF_TOKEN` 写在执行命令里（平台环境变量栏存不了），**不要写进任何文件** —— 那会被 `git push` 上去。

`bash start.sh`（等价 `MODE=all`）会依次自动执行：

| 阶段 | 内容 | 失败行为 |
|---|---|---|
| preflight | token / GPU 显存 / 磁盘 / **网络探测** | 非零退出 |
| setup | `uv sync` → 拉 11.5GB 权重（带字节校验）→ 拉 10 个任务数据集 → stats 自检 | 非零退出 |
| L1 | `--dry-run` 配置组装 | 非零退出 |
| L2 | `--test` 数据管线 + forward/backward（权重随机初始化） | 非零退出 |
| **L3 ★** | **加载 11GB 权重跑 2 步**，脚本解析 `Loaded a/b` 与 `Missing (rand init)` | **不通过则非零退出，不进训练** |
| L4 | 单批过拟合（`--overfit_batch 5`，loss 应明显下降） | 非零退出 |
| train | 正式训练（后台，卡数自动探测） | — |
| 守护上传 | 训练启动后自动拉起 `upload_checkpoints.sh MODE=watch`，每 10 分钟把新 checkpoint 推到 HF | — |
| 等待 + 收尾 | 阻塞等训练结束 → 等 120s 落盘 → 停守护 → **自动上传权重/日志/配置/环境快照** | — |

**任何阶段失败都会自动把诊断上传到 HF**（`diagnostics/<阶段>_<时间戳>/`），
容器销毁后仍可事后排查。下载与安装步骤都带重试（各 3 次，间隔 60s）。

### 只跑部分阶段（调试用）

```bash
MODE=setup  bash start.sh     # 只装环境 + 拉权重/数据 + 校验
MODE=verify bash start.sh     # 只跑 L1/L2/L3/L4
MODE=train  bash start.sh     # 只启动训练（环境/数据须已就绪）
```

### 可选环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `GPUS` | 自动探测 | 训练卡数（`NPROC_PER_NODE` → `MLP_WORKER_GPU` → `MLP_WORKER_NUM` → `nvidia-smi`） |
| `G05_OUTPUT_DIR` | `/workspace/output` | 输出目录（容器内，靠 HF 落地） |
| `SKIP_L4` | `0` | `=1` 跳过单批过拟合直接训练 |
| `STRICT_L3` | `1` | `=0` 时 L3 无法判定也继续（**不推荐**） |
| `AUTO_UPLOAD` | `1` | `=0` 关闭守护上传 |
| `UPLOAD_INTERVAL` | `600` | 守护上传轮询秒数 |
| `ONLY_TASK` | 空 | 设了就只拉单任务，如 `TASK-YAM-0006` |

### 训练跑起来之后

训练在后台，`start.sh` 会打印 exp 目录。容器销毁后一切丢失，**所以收尾要在同一容器内做**：

```bash
EXP=$(cat /workspace/exp_dir.txt)

# 看训练进度
tail -f /workspace/train.stdout.log
# 看上传进度
tail -f /workspace/upload_checkpoints.log

# open-loop 评测（跑完再执行）
bash scripts/run/eval_open_loop.sh \
  --ckpt_path $EXP/last.pt --task_config configs/task/yam.yaml

# 最终收尾上传
EXP_DIR=$EXP bash upload_output.sh
```

手动上传 checkpoint（守护进程没起来时用）：

```bash
EXP_DIR=$EXP bash upload_checkpoints.sh            # 传全部
DRY_RUN=1 EXP_DIR=$EXP bash upload_checkpoints.sh  # 先看会传什么
MODE=last EXP_DIR=$EXP bash upload_checkpoints.sh  # 只传最后一个
```

### 为什么 L2 和 L3 必须分开

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
