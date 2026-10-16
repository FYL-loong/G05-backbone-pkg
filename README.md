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

### ⚠️ 视频解码：FFmpeg 与 `G05_VIDEO_BACKEND`

数据集里的 MP4 需要解码。上游代码的选择逻辑**有缺陷**（已在本包修复）：

```python
# 上游 video_utils.py
if importlib.util.find_spec("torchcodec"):   # 只检查 python 包能否 import
    return "torchcodec"                      # 没检查系统 FFmpeg 共享库
```

`torchcodec` 是 FFmpeg 的绑定，**运行期需要系统 ffmpeg**（官方 `README.md` 也把它列为安装前提）。
容器里没装时会退化成一堆 WARNING **但不报错、不退出**：

```
libtorchcodec: we tried to load ... Likely causes: 1. FFmpeg is not properly installed ...
libavutil.so.56/57/58/59: cannot open shared object file: No such file or directory
WARNING | >> Error loading sample  base_lerobot_dataset.py:904
  (dataset_dir=.../TASK-YAM-0001_lerobot, local_idx=27039) (attempt 803). Retrying with a random index
```

实测这一种情况会刷出 60 万行日志、**训练实质毫无进展**，极难发现。

**本包的修复**（`src/g05/data/lerobot/datasets/video_utils.py`）：
`get_safe_default_codec()` 现在会**真实探测** torchcodec（写一个 8 帧小视频、取 metadata），
失败就回退到 **`pyav`** —— 而 `av==14.2.0` 已在依赖里，**wheel 自带 FFmpeg 库，不需要系统 ffmpeg**。

| 环境变量 | 作用 |
|---|---|
| `G05_VIDEO_BACKEND=pyav` | 强制用 pyav（最稳，无需系统 ffmpeg） |
| `G05_VIDEO_BACKEND=torchcodec` | 强制用 torchcodec（需系统 ffmpeg 已装好） |
| 不设 | 自动探测；torchcodec 不可用则回退 pyav |

**若想用 torchcodec**（更快），在镜像里装系统 ffmpeg 即可：`apt-get install -y ffmpeg`。

### ⚠️ 视觉注意力后端：`flash-attn-4` 与 `cutlass-dsl` 版本不匹配

**另一个"import 成功但运行期崩"的坑，与上面视频解码同一模式。**

`vision.py` 的 import 链本来是 `FA4 → FA2 → SDPA`，靠 `except ImportError` 兜底。
但 `flash-attn-4` 是 **CuTe DSL** 实现 —— **forward 时才调 `cute.compile()` 现场编译 kernel**，
编译失败抛的异常 import 层抓不到：

```
vision.py:216  _attend_spatial → flash_attn_varlen_func
flash_attn/cute/interface.py:981 → cute.compile(...)
flash_attn/cute/flash_fwd.py:392
    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx]
cutlass/cute/core.py:2673 slice_ → cute_ops_gen.py:3551
ValueError: Operation creation failed
```

**根因是版本配对**（`uv.lock` 现有组合）：

| flash-attn-4 版本 | 声明的 cutlass-dsl 要求 | 实际装上 |
|---|---|---|
| **4.0.0b15**（uv.lock 里） | `>=4.4.2` | 4.5.2 |
| 4.0.0b16 | `>=4.5.2` | — |
| 4.0.0b20 – b25 | `==4.6.0.dev0` | — |
| **4.0.0b30+** | `>=4.6.2` | — |

即 **b15 声明兼容 `>=4.4.2`，但与 4.5.2 的 CuTe layout API 实际不兼容** —— 上游声明有误。

**本包的修复**（`src/g05/models/g05/qwen35/vision.py`）：加开关 `G05_VISION_ATTN`。

| 取值 | 行为 |
|---|---|
| **`sdpa`（本包默认）** | **不加载 flash-attn**，视觉注意力用 `F.scaled_dot_product_attention`（代码本来就有这条路径）。慢一些，但不会崩 |
| `auto` | 上游行为：FA4 → FA2 → SDPA |
| `fa4` / `fa2` | 强制指定（需已装匹配版本） |

`start.sh` 默认导出 `G05_VISION_ATTN=sdpa`。**想用 flash-attn 提速**，需要把
`flash-attn-4` 升到 **`>=4.0.0b30`** 并让 `nvidia-cutlass-dsl>=4.6.2` 一起解析
（改 `pyproject.toml` 的 pin 后重跑 `uv lock`），再设 `G05_VISION_ATTN=auto`。

### 日志量控制（平台下载有 100 万行上限）

平台日志下载**上限 100 万行**。实测一次坏掉的运行产生了 **3000 万行**以上，
真正的报错被彻底淹没。本包做了三层控制：

| 措施 | 效果 |
|---|---|
| **① 修视频解码**（上面那节） | 消除 ~98% 的日志 —— 那些行全部来自同一个 FFmpeg 错误 |
| **② 样本加载失败告警节流** | 单样本 10,000 次打点 → 21 次（**降 99.79%**）；第 1 次与最后一次仍完整打印 |
| **③ 压掉库噪音** | `NCCL_DEBUG=WARN`（去掉几百行拓扑 INFO）、`TQDM_MININTERVAL=10`、`PYTHONWARNINGS=ignore` |

**节流代码**在 `src/g05/data/base_lerobot_dataset.py` 的 `_warn_sample_retry()`：
失败是系统性的时候（例如缺 ffmpeg），`MAX_GETITEM_ATTEMPT=10000` 会让每个样本
刷 1 万条、每条还内嵌一大段异常。现在改为**第 1 次完整打印、之后每 500 次打一条摘要、
最后一次完整打印** —— 既省日志又不丢关键信息。

### 快速自检 `MODE=smoke`（**强烈建议：正式跑之前先来一次**）

**为什么需要它**：正式任务要先花 **~12 分钟**下载 11.44GB 权重 + ~20GB 数据集才进
`finetune.py`。但踩过的坑里有好几个是**配置/参数级**的，本来几秒就能发现 ——
例如 `Cannot set both max_epochs and max_steps`，那个断言在 `finetune.py:877`，
**发生在任何下载之前**，却被下载埋到第 17 分钟才暴露。

```bash
MODE=smoke bash start.sh      # 约 1 分钟，不下载权重/数据，不跑训练
```

它检查三件事：

| # | 检查 | 能抓到的坑 | 是否需要 `.venv` |
|---|---|---|---|
| 1 | **包完整性**（155 条 `g05.*` import 是否可解析） | 抽包漏文件、`.gitignore` 误吞源码目录 | ❌ 不需要 |
| 2 | **配置自检**（Hydra 组装 → `validate_train_config` → `max_epochs`/`max_steps` 互斥） | override key 拼错、参数冲突、缺数据目录。**默认 / L2 `--test` / L4 `--overfit` 三条路径都验** | ✅ **需要** |
| 3 | **视频解码**（真造一个小 MP4，用代码里同一套判据决定后端） | `torchcodec` 缺系统 FFmpeg 共享库（会回退 `pyav`） | ❌ 不需要（镜像自带 torch） |

> ⚠️ **第 2 项需要 `hydra`/`omegaconf`/`g05`，这些只在 `.venv` 里。**
> 容器刚起来时没有 `.venv`，smoke 会**明确跳过并提示**（不是失败）。
> 要现在就验配置：`SMOKE_AUTO_SETUP=1 MODE=smoke bash start.sh`（会先装环境，4-6 分钟）。

**但更重要的一点**：`MODE=all` 里配置自检**已经自动插在正确位置** ——

```
1.   安装环境（uv sync）        ~4 min
1.5  配置自检 ★                ~10 s   ← 环境已就绪、但还没下载
2.   下载权重（11.44GB）        ~6 min
3.   下载数据集（~20GB）        ~2 min
```

**所以正式任务不会再"下载 12 分钟才发现配置写错"。** 这是修 bug #5（`Cannot set both
max_epochs and max_steps`）暴露太晚的根本解法。

**没验的**（只能靠真实运行，但已收在一次任务里）：
11.44GB 权重完整加载（L3）、flash-attn/CuTe kernel 编译（已默认绕开走 SDPA）、
显存是否够、DDP 是否正常。

单独跑配置自检也可以（在装了环境的容器里）：

```bash
source .venv/bin/activate
python config_check.py                                    # 默认 task=yam
python config_check.py --test-mode model.max_steps=10 model.max_epochs=null
python config_check.py --overfit   model.max_steps=500 model.max_epochs=null
```

### 一条命令（全自动，无人值守）

平台任务的「执行命令」框里填这一条 —— **不需要任何 MODE 参数**：

```bash
set -e
export HF_TOKEN="hf_把你的真实token填这里"
export G05_VIDEO_BACKEND=pyav
git clone https://github.com/FYL-loong/G05-backbone-pkg.git G05-backbone-pkg
cd G05-backbone-pkg
bash start.sh
```

**关于这两行 `export`（只有这两个需要你写）**：

| 变量 | 必需？ | 说明 |
|---|---|---|
| **`HF_TOKEN`** | ★ **必需** | 留空/不设 → `start.sh` 前置检查**立刻失败**（否则训练完权重传不上去，容器一销毁全丢）。写在执行命令里，**不要写进任何文件**（会被 push 上去） |
| `G05_VIDEO_BACKEND` | 可选 | `pyav` 可省掉自动探测、直接走 pyav。**不写也能跑**（自动探测失败会回退 pyav） |

**不要**再手动设 `G05_VISION_ATTN` —— `start.sh` 已默认导出 `sdpa`（原因见上面那节）。
其余变量（`MODE` / `AUTO_UPLOAD` / `UPLOAD_INTERVAL` / `KEEPALIVE` / `ONLY_TASK` /
`SKIP_L4` / `STRICT_L3` / `HF_ENDPOINT` / `G05_OUTPUT_DIR`）都在脚本内有默认值，
见「可选环境变量」一节，一般不用指定。

`bash start.sh`（等价 `MODE=all`）会依次自动执行：

| 阶段 | 内容 | 失败行为 |
|---|---|---|
| preflight | token / **包完整性检查** / GPU 显存 / 磁盘 | 非零退出 |
| setup | `uv sync` → 拉 11.5GB 权重（带字节校验）→ 拉 10 个任务数据集 → stats 自检 | 非零退出 |
| L1 | `--dry-run` 配置组装 | 非零退出 |
| L2 | `--test` 数据管线 + forward/backward（权重随机初始化） | 非零退出 |
| **L3 ★** | **加载 11GB 权重跑 2 步**，脚本解析 `Loaded a/b` 与 `Missing (rand init)` | **不通过则非零退出，不进训练** |
| L4 | 单批过拟合（`--overfit_batch 5`，loss 应明显下降） | 非零退出 |
| train | 正式训练（后台，卡数自动探测） | — |
| 守护上传 | 训练启动后自动拉起 `upload_checkpoints.sh MODE=watch`，每 10 分钟把新 checkpoint 推到 HF | — |
| 等待 + 收尾 | 阻塞等训练结束 → 等 120s 落盘 → 停守护 → **自动上传最终权重到 HF** | — |

**日志**：⚠️ **平台自带下载只对「容器 stdout」有效**，而训练 stdout 被重定向到
`/workspace/train.stdout.log` 这个**文件**里 —— **平台采不到它**。所以本包会在收尾时
把训练日志一起上传到 HF（`logs/train.stdout.log`、`logs/train.exit_code`、
`logs/exp/**`）。实测踩过：训练跑 5 分钟就退出、没传日志，
崩溃现场随容器销毁永久丢失，只能靠猜。

下载与安装步骤都带重试（各 3 次，间隔 60s）。

**职责边界**：

| 产物 | 去处 | 原因 |
|---|---|---|
| **权重**（`model_state_dict.pt` / `checkpoints/step_*.pt`） | **HF** | 容器临时销毁，平台只留日志、不留权重；11.44GB 也超过 GitHub 限制 |
| **训练日志**（`train.stdout.log` / 退出码 / `exp/logs/**`） | **HF** | 训练 stdout 重定向进了文件，平台采不到；容器销毁即丢 |
| **平台 stdout**（start.sh 的输出） | **平台** | 平台自带日志下载 |
| **代码/脚本/配置** | **GitHub** | — |

### 只跑部分阶段（调试用）

```bash
MODE=setup  bash start.sh     # 只装环境 + 拉权重/数据 + 校验（不训练）
MODE=verify bash start.sh     # 只跑 L1/L2/L3/L4（不含 L3 之后的训练）
MODE=train  bash start.sh     # 只启动训练（环境/数据须已就绪）
MODE=smoke  bash start.sh     # ★ 快速自检（约 1 分钟，不下载不训练）
```

#### 建议：首次先用 **1 卡** 验证环境

`uv sync` 是整个流程唯一没被验证过的环节（`deepspeed` 要就地编译、`uv` 要联网
下载 Python 3.10）。**用它失败一次的成本，远低于 8 卡白等一小时。**

> ⚠️ 但这一步**不能代替 8 卡的完整跑**：1 卡的容器**销毁后不保留任何东西**，
> 8 卡任务仍需重新 `uv sync` + 重新下载。它的价值是**提前暴露环境问题**，
> 不是"预热"。

```bash
# 1 卡任务，执行命令：
set -e
export HF_TOKEN="hf_把你的真实token填这里"
git clone https://github.com/FYL-loong/G05-backbone-pkg.git G05-backbone-pkg
cd G05-backbone-pkg
MODE=setup bash start.sh
```

**它会验证 / 产出**：

| 检查 | 期望输出 |
|---|---|
| 包完整性 | `[ OK ] 包完整性通过`（155 条 g05 import 全部可解析） |
| Python 3.10 获取 | `[env] ✅ 已找到` 或 联网下载成功 |
| `uv sync`（约 10–30 分钟） | `[env] venv python: Python 3.10.x` |
| torch 可用性 | `torch 2.7.1+cu128  cuda=12.8  available=True` |
| 关键库导入 | `flash_attn` / `fla` / `liger_kernel` / `deepspeed` 全部有版本号 |
| 权重下载 + **字节校验** | `[ OK ] 全部权重校验通过` |
| 数据集 | 10 个任务，合计 `episodes=991` |
| stats 自检 | `[ OK ] stats 维度与退化检测全部通过` |

**不验证**（要完整那轮才验）：L1/L2/L3 权重加载、L4 过拟合、训练本身。

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
| `G05_VIDEO_BACKEND` | 自动探测 | 视频解码后端：`pyav`（不需系统 ffmpeg）或 `torchcodec`（需系统 ffmpeg） |
| `G05_VISION_ATTN` | **`sdpa`** | 视觉注意力后端：`sdpa`（稳）/ `auto`（FA4→FA2→SDPA）/ `fa4` / `fa2` |
| `KEEPALIVE` | `0` | 收尾后挂起 N 秒再退出，便于进容器查看 |

### 训练跑起来之后

`start.sh` 会**阻塞等训练结束并自动上传权重**，所以正常情况下你不需要手动做什么。

```bash
EXP=$(cat /workspace/exp_dir.txt)          # start.sh 写好的 exp 目录

tail -f /workspace/train.stdout.log        # 跟训练进度（或直接用平台日志下载）
tail -f /workspace/upload_checkpoints.log  # 跟守护上传进度

# 手动上传 checkpoint（守护进程没起来时用）
EXP_DIR=$EXP bash upload_checkpoints.sh            # 传全部
DRY_RUN=1 EXP_DIR=$EXP bash upload_checkpoints.sh  # 先看会传什么
MODE=last EXP_DIR=$EXP bash upload_checkpoints.sh  # 只传最后一个

# 手动上传最终权重
EXP_DIR=$EXP bash upload_output.sh

# open-loop 评测（趁容器还在）
bash scripts/run/eval_open_loop.sh \
  --ckpt_path $EXP/last.pt --task_config configs/task/yam.yaml
```

### 排查用的开关

| 变量 | 作用 |
|---|---|
| `KEEPALIVE=1800` | 收尾后挂起 N 秒再退出 —— 方便在任务结束前进容器查看 |

**日志**：平台自带下载功能；容器还活着时也可 `tail -f /workspace/train.stdout.log`。

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
