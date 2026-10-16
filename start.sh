#!/usr/bin/env bash
# ============================================================================
# start.sh — G0.5 backbone 微调一键启动（百度百舸 AIHC，无人值守）
#
# 职责：把微调跑起来 → 跑完 → 把【权重和训练日志】存到 HF。
#   - 权重：容器临时，必须上传 HF，否则训练白跑。
#   - 日志：⚠️ 必须上传。训练 stdout 被重定向到 /workspace/train.stdout.log 这个
#     【文件】，而平台日志采的是容器 stdout —— **采不到重定向进文件的内容**。
#     实测踩过：训练跑 5 分钟就退出、什么都没保存，因为没传这个文件，
#     崩溃现场随容器销毁永久丢失，只能靠猜。现在退出码 + 日志都会传到 HF。
#     （平台自带日志下载对【平台 stdout】有效，对重定向文件无效 —— 两者互补。）
#
# 平台「执行命令」框 —— 复制这段，只改 HF_TOKEN：
#
#   set -e
#   export HF_TOKEN="hf_把你的真实token填这里"     # setup/train/all 必需；smoke 不需要
#   export G05_VIDEO_BACKEND=pyav                 # 可选（省掉自动探测；不写也能跑）
#   git clone https://github.com/FYL-loong/G05-backbone-pkg.git G05-backbone-pkg
#   cd G05-backbone-pkg
#   bash start.sh
#
# 【不要】额外设 G05_VISION_ATTN —— 本脚本 L75 已默认导出 sdpa。
# 其余带默认值的变量见下方「环境变量」段，一般不需要手动指定。
#
# 分阶段（调试用）：
#   MODE=smoke   bash start.sh    ★快速自检（~1 分钟，不下载权重/数据、不训练）
#                                建议正式跑之前先来一次，抓配置级错误
#   MODE=setup   bash start.sh    装环境 + 拉权重/数据 + 自检
#   MODE=verify  bash start.sh    只跑 L1/L2/L3/L4
#   MODE=train   bash start.sh    只启动训练（环境/数据须已就绪）
#   MODE=all     bash start.sh    默认，全自动
#
# 环境变量：
#   HF_TOKEN          setup / train / all 必需（下载权重、上传结果）；
#                     smoke / smoke-ddp 不需要（它们不下载任何东西）
#   G05_OUTPUT_DIR    可选，默认 /workspace/output
#   GPUS              可选，训练卡数；默认取 NPROC_PER_NODE / MLP_WORKER_GPU /
#                     MLP_WORKER_NUM / nvidia-smi 探测
#   HF_ENDPOINT       可选，默认 https://huggingface.co
#                     实测百度百舸【连不上 hf-mirror】，不要设成它
#   ONLY_TASK         可选，只拉单任务，如 TASK-YAM-0006
#   SKIP_L4           可选，=1 跳过单批过拟合
#   STRICT_L3         默认 1；=0 时 L3 无法判定也继续（不推荐）
#   AUTO_UPLOAD       默认 1；=0 关闭 checkpoint 守护上传
#   UPLOAD_INTERVAL   守护上传轮询秒数，默认 600
#   KEEPALIVE         可选，收尾后挂起 N 秒再退出（便于进容器查看）
#   G05_DEBUG_CUDA    可选，=1 打开 CUDA_LAUNCH_BLOCKING + TORCH_USE_CUDA_DSA
#                     （定位 CUDA illegal memory access；会显著变慢）
#   G05_DEBUG_DDP     可选，=1 在 DDP 构造前打印 param/buffer 的设备/dtype
# ============================================================================
set -euo pipefail
# 故意不用 pipefail：`cmd | head -1` 这类写法里，下游 head 读够就退出，上游
# 收到 SIGPIPE（退出码 141），pipefail 会把整条管道判为失败 → set -e 中止脚本。
# 8 卡时 `nvidia-smi | head` 必现、1 卡不出现 —— 极难发现。靠 set -e 捕获主命令即可。
set +o pipefail

MODE="${MODE:-all}"
G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-/workspace/output}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ONLY_TASK="${ONLY_TASK:-}"
SKIP_L4="${SKIP_L4:-0}"
STRICT_L3="${STRICT_L3:-1}"
AUTO_UPLOAD="${AUTO_UPLOAD:-1}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-600}"
KEEPALIVE="${KEEPALIVE:-0}"
export G05_OUTPUT_DIR HF_ENDPOINT

# ------------------------------------------------- 压制日志噪音（平台有行数上限）
# 平台日志下载有 100 万行上限，而 NCCL 初始化默认会打出几百行拓扑 INFO
# （Tree/Pattern/GPU-... 之类），完全无用。设为 WARN 后只在真出问题时说话。
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
# tqdm 进度条每步刷新会产生大量行；降低刷新频率（只影响观感，不影响训练）
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TQDM_MINITERS="${TQDM_MINITERS:-50}"
# ------------------------------------------------- NCCL：绕开不匹配的 HPC-X 插件
# ★ 这是 4 卡训练报 "CUDA error: an illegal memory access" 的根因。
#
# 证据（train.stdout.log）：
#   NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v10 symbol.
#   NCCL INFO NET/Plugin: Loaded net plugin NCCL RDMA Plugin v9 (v9)      <- 退回 v9
#   NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
#   NCCL INFO P2P plugin v9 IBext_v9
#   NCCL INFO Using network NCCL RDMA Plugin v9
#
# 容器自带的 NCCL 是 2.26.2（期望插件实现 v10 接口），而 /opt/hpcx 里的
# RDMA/SHARP 插件只提供 v9 —— **ABI 不匹配的原生插件被加载并使用**，
# 在第一次通信（DDP broadcast + P2P 建链）时越界访问设备内存。
# 这解释了全部现象：
#   - 1 卡正常 / 4 卡崩       （P2P 插件只在多卡参与时被调用）
#   - 崩在 DDP broadcast      （第一次真正通信）
#   - 参数/buffer 全部干净    （与模型无关）
#
# 单机多卡【不需要】IB/RDMA：同机通信走 NVLink/P2P + 共享内存更快。
# 所以直接禁掉插件与 IB。这些都可被外部环境变量覆盖。
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"      # 不加载外部 net 插件
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"         # 关 IB/RoCE
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"         # 同机 P2P 走 NVLink
export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"   # 关 SHARP（collnet 插件）
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"  # 关 collnet

# 关掉 Python 的 DeprecationWarning / FutureWarning 刷屏
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

# G05_DEBUG_DDP=1：在 DDP 构造前打印所有 param/buffer 的 设备/dtype/连续性
export G05_DEBUG_DDP="${G05_DEBUG_DDP:-0}"
# G05_GRAD_BUCKET_VIEW=1：开启 DDP 的零拷贝梯度桶优化（默认 0 = 关闭，稳定优先）
#   实测 4 卡在 DDP 建好后、第一个 step 前报 illegal memory access，1 卡正常。
#   该优化在"共享参数(weight tying) + 混合 dtype"下容易构造错误的桶 view。
export G05_GRAD_BUCKET_VIEW="${G05_GRAD_BUCKET_VIEW:-0}"
# G05_DEBUG_DDP_BCAST=1：在 DDP 构造前逐个参数 broadcast（含 synchronize），
#   用于指认"到底是哪个参数让 NCCL 越界"。慢几分钟，只在排错时开。
export G05_DEBUG_DDP_BCAST="${G05_DEBUG_DDP_BCAST:-0}"

# ------------------------------------------------- 视觉注意力后端（重要）
# flash-attn-4 是 CuTe DSL 实现：**import 成功不代表能用** —— forward 时才现场编译
# kernel，编译失败抛 `ValueError: Operation creation failed`。实测组合：
#   flash-attn-4.0.0b15（uv.lock 现有）+ nvidia-cutlass-dsl 4.5.2
#   → flash_fwd.py:392 offset_batch_Q 的 layout 与 view 形状不弱一致，训练直接崩。
# b15 的依赖声明写的是 `nvidia-cutlass-dsl>=4.4.2`，即**上游声明与实际不兼容**。
#
# 默认走 SDPA：不用 flash-attn，视觉注意力用 torch 的
# F.scaled_dot_product_attention，代码里本来就有这条路径（vision.py 的 else 分支）。
# 慢一些，但不会因第三方 kernel 编译失败而让整个训练崩掉。
# 想试 flash-attn（装好匹配版本后）：G05_VISION_ATTN=auto 或 fa4
export G05_VISION_ATTN="${G05_VISION_ATTN:-sdpa}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- 工具函数
C_R='\033[31m'; C_G='\033[32m'; C_Y='\033[33m'; C_C='\033[36m'; C_0='\033[0m'
info() { printf "${C_C}[INFO]${C_0} %s\n" "$*"; }
ok()   { printf "${C_G}[ OK ]${C_0} %s\n" "$*"; }
warn() { printf "${C_Y}[WARN]${C_0} %s\n" "$*"; }
fail() { printf "${C_R}[FAIL]${C_0} %s\n" "$*" >&2; exit 1; }
step() { printf "\n${C_C}========================================================${C_0}\n${C_C} %s${C_0}\n${C_C}========================================================${C_0}\n" "$*"; }

# ------------------------------------------------- CUDA 调试开关（按需打开）
# ⚠️ 必须放在工具函数【之后】—— 它用到 warn()。曾经放在前面，导致
#    "start.sh: line 89: warn: command not found"，脚本第 4 行就死了，
#    而 `bash -n` 语法检查查不出"函数未定义"这种错误。
# G05_DEBUG_CUDA=1：
#   CUDA_LAUNCH_BLOCKING=1  让 CUDA 调用同步执行，"illegal memory access" 会停在
#                           真正犯错的 kernel 上（默认异步报，栈指向无关 API）
#   TORCH_USE_CUDA_DSA=1    打开 device-side assert，能指出越界下标
if [[ "${G05_DEBUG_CUDA:-0}" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    warn "G05_DEBUG_CUDA=1 —— 已开 CUDA_LAUNCH_BLOCKING + TORCH_USE_CUDA_DSA（会变慢）"
fi

# 激活 venv。activate 脚本在 set -u 下可能引用未定义变量而退出，故临时关 -u。
activate_venv() {
    [[ -f .venv/bin/activate ]] || fail "找不到 .venv/bin/activate，请先跑 MODE=setup"
    set +u
    # shellcheck disable=SC1091
    source .venv/bin/activate
    set -u
    info "python = $(command -v python)  ($(python -V 2>&1))"
}

# 带重试执行。无人值守时网络抖动不该让整个任务失败。
retry() {
    local tries="$1" delay="$2"; shift 2
    local i=1 rc=0
    while true; do
        rc=0
        "$@" || rc=$?
        if [[ $rc -eq 0 ]]; then
            return 0
        fi
        if [[ $i -ge $tries ]]; then
            fail "重试 $tries 次仍失败（退出码 $rc）：$*"
        fi
        warn "第 $i/$tries 次失败（退出码 $rc），${delay}s 后重试：$*"
        i=$((i+1))
        sleep "$delay"
    done
}

EXP_DIR_FOUND=""

# 需要 HF_TOKEN 的阶段才调用它。⚠️ 不要放在全局前置检查里 ——
# smoke / smoke-ddp 不下载任何东西，不该被 token 卡住（曾经因此白跑一次任务）。
require_hf_token() {
    [[ -n "${HF_TOKEN:-}" ]] || fail "未设置 HF_TOKEN。请把它写在平台任务的执行命令里：export HF_TOKEN=\"hf_...\""
    export HF_TOKEN
}

# ---------------------------------------------------------------- 前置检查
step "0. 前置检查"

info "MODE           = $MODE"
info "PROJECT_ROOT   = $PROJECT_ROOT"
info "G05_OUTPUT_DIR = $G05_OUTPUT_DIR"
info "HF_ENDPOINT    = $HF_ENDPOINT"
info "ONLY_TASK      = ${ONLY_TASK:-<全部 10 个任务>}"
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$MODE" == "smoke" || "$MODE" == "smoke-ddp" ]]; then
        info "HF_TOKEN       = <未设置，本 MODE 不需要>"
    else
        warn "HF_TOKEN       = <未设置> —— 需要下载/上传的阶段会失败"
    fi
else
    info "HF_TOKEN       = <已设置，${#HF_TOKEN} 字符>"
fi

# ---- 包完整性门禁 ----
# 便宜（几秒）且抓到过真 bug：.gitignore 写了裸目录名 `eval/`，把官方源码目录
# src/g05/utils/eval/ 一起忽略 → 文件从未 push → 容器里 import 失败，
# 而 setup 阶段只下载权重和数据、不 import 模型代码，发现不了，要跑到 L1 才崩。
if [[ -f check_package.py ]]; then
    info "包完整性检查（g05.* import 是否都能解析）..."
    if python check_package.py .; then
        ok "包完整性通过"
    else
        fail "包完整性检查失败 —— 见上方缺失模块清单（这是【代码缺文件】，不是环境问题）"
    fi
else
    warn "没有 check_package.py，跳过包完整性检查"
fi

# ---- 配置自检（专抓"配置/参数级"错误，几秒出结果）----
# 曾经踩过：`Cannot set both max_epochs and max_steps`（task=yam 把 max_epochs
# 覆盖成 10，又与命令行 max_steps 冲突）—— 该断言在 finetune.py:877，
# **发生在任何下载之前**，却被 12 分钟的下载埋到第 17 分钟才暴露。
#
# ⚠️ 它需要 hydra/omegaconf，而这两者只在 .venv 里 —— 所以：
#    - setup 之前（容器刚起）：跳过，不报错
#    - setup 之后：**必跑**（这才是真正能拦住问题的地方）
# 返回 0=通过（或跳过），非 0=配置有问题（致命）
config_check_now() {
    [[ -f config_check.py ]] || return 0
    if [[ ! -x .venv/bin/python ]]; then
        info "尚无 .venv，跳过配置自检（setup 之后会自动跑）"
        return 0
    fi
    if .venv/bin/python -c "import hydra, omegaconf" 2>/dev/null; then
        :
    else
        warn ".venv 里没有 hydra/omegaconf，跳过配置自检"
        return 0
    fi
    info "配置自检（几秒，不需要 GPU/权重/数据）..."
    if .venv/bin/python config_check.py "$@" ; then
        ok "配置自检通过"
        return 0
    fi
    return 1
}
# 注意：这里【只告警不退出】—— 如果 .venv 还没建（容器刚起），它在 setup
# 完成后还会再跑一次；在 setup 之后调用时才是致命检查。
config_check_now || warn "配置自检未通过（若此时尚无 .venv 属正常，setup 后会再验一次）"

if nvidia-smi >/dev/null 2>&1; then
    info "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/       /'
    # 不要写 `nvidia-smi ... | head -1`（见文件顶部关于 SIGPIPE 的说明）
    GPU_TOTAL_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
        | awk 'NR==1{gsub(/[^0-9]/,""); print; exit}')"
    if [[ "${GPU_TOTAL_MB:-0}" -lt 70000 ]]; then
        warn "单卡显存 ${GPU_TOTAL_MB} MiB < 70 GB，全参微调可能 OOM"
    else
        ok "单卡显存 ${GPU_TOTAL_MB} MiB，满足全量微调要求"
    fi
else
    warn "没有检测到 GPU（nvidia-smi 不可用）。MODE=train 会失败"
fi

mkdir -p "$G05_OUTPUT_DIR"
df -h "$G05_OUTPUT_DIR" | tail -1 | sed 's/^/       /'

# ============================================================================
# setup：装环境 + 拉权重 + 拉数据 + 自检
# ============================================================================
do_setup() {
    require_hf_token        # 下载权重/数据需要 HF 读权限
    step "1. 安装环境（uv sync，首次约 10-30 分钟）"
    # NGC 镜像自带 Python 3.12，但项目要求 3.10.16（.python-version / requires-python），
    # 所以 uv 必须联网下载独立 Python 3.10 —— 源默认是 GitHub Releases，国内不可靠。
    if [[ -z "${UV_PYTHON_INSTALL_MIRROR+x}" ]]; then
        export UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
        info "启用 Python 下载镜像: $UV_PYTHON_INSTALL_MIRROR"
    fi
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

    if [[ -x .venv/bin/python ]]; then
        ok ".venv 已存在，跳过 setup_env.sh（要重装先 rm -rf .venv）"
    else
        bash setup_env.sh
    fi
    activate_venv
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

    # ★ 配置自检放在这里：环境已就绪、但【还没下载 11.5GB 权重 + 20GB 数据】。
    #   这是能拦住"配置级"错误的最早时机。失败即致命，不再往下走。
    step "1.5 配置自检（几秒；在下载权重/数据【之前】拦下配置错误）"
    if ! config_check_now; then
        fail "配置自检失败 —— 见上方原因。这是【配置/传参】问题，不必下载权重。"
    fi

    step "2. 下载权重（约 11.5GB，失败自动重试 3 次）"
    retry 3 60 bash download_weights.sh

    step "3. 下载数据集（失败自动重试 3 次）"
    if [[ -n "$ONLY_TASK" ]]; then
        retry 3 60 env ONLY_TASK="$ONLY_TASK" bash download_dataset.sh
    else
        retry 3 60 bash download_dataset.sh
    fi

    step "4. 数据完整性自检"
    local n_ds
    n_ds="$(find data/yam -maxdepth 1 -type d -name 'TASK-YAM-*_lerobot' 2>/dev/null | wc -l | tr -d ' ')"
    info "发现 $n_ds 个 _lerobot 数据集"
    [[ "$n_ds" -gt 0 ]] || fail "data/yam 下没有 TASK-YAM-*_lerobot，数据集没拉下来"
    python - <<'PY'
import glob, json, os
dirs = sorted(glob.glob("data/yam/TASK-YAM-*_lerobot"))
tot = 0
for d in dirs:
    info_p = os.path.join(d, "meta", "info.json")
    try:
        j = json.load(open(info_p, encoding="utf-8"))
        ep, tk, fps = j.get("total_episodes"), j.get("total_tasks"), j.get("fps")
        tot += int(ep or 0)
        print(f"       {os.path.basename(d):30s} episodes={ep}  tasks={tk}  fps={fps}")
    except Exception as e:
        print(f"       {os.path.basename(d):30s} [读取失败] {type(e).__name__}: {e}")
print(f"       {'合计':30s} episodes={tot}")
if tot == 0:
    raise SystemExit("[FAIL] 所有数据集 episode 数都是 0，数据有问题")
PY

    step "5. stats 自检（维度 + 退化检测）"
    python - <<'PY'
import json, os, sys
p = "data/stats/yam_stats.json"
if not os.path.isfile(p):
    print(f"[FAIL] 缺 {p}"); sys.exit(1)
d = json.load(open(p, encoding="utf-8"))
TOL = 1e-4
want = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
bad = 0
for emb, v in d.items():
    print(f"  embodiment: {emb}")
    for kind in ("action", "state"):
        if kind not in v:
            print(f"    [FAIL] 缺 {kind}"); bad += 1; continue
        got = set(v[kind].keys())
        if got != set(want):
            print(f"    [FAIL] {kind} key 不匹配: {sorted(got)}"); bad += 1
        for k, st in v[kind].items():
            exp = want.get(k)
            q01, q99 = st.get("global_q01"), st.get("global_q99")
            if q01 is None or q99 is None:
                print(f"      {kind}.{k:16s} [无 q01/q99]"); continue
            if isinstance(q01, list):
                dim = len(q01)
                deg = sum(1 for i in range(dim) if (q99[i] - q01[i]) < TOL)
            else:
                dim, deg = 1, (1 if (q99 - q01) < TOL else 0)
            if exp is not None and dim != exp:
                print(f"    [FAIL] {kind}.{k} 维度 {dim} != {exp}"); bad += 1
            flag = f"退化维度={deg}" if deg else "ok"
            print(f"      {kind}.{k:16s} dim={dim} {flag}")
print()
if bad:
    print(f"[FAIL] stats 自检有 {bad} 处问题"); sys.exit(1)
print("[ OK ] stats 维度与退化检测全部通过")
PY
    ok "setup 阶段完成"
}

# ============================================================================
# verify：L1/L2/L3(门禁)/L4
# ============================================================================
do_verify() {
    activate_venv

    # ⚠️ max_epochs 与 max_steps 互斥（finetune.py:877 有 assert）：
    #    configs/task/yam.yaml 里 max_epochs=10（task 覆盖 model），所以任何想用
    #    max_steps 限步的地方，都必须同时把 max_epochs 置 null，否则报
    #    "Cannot set both `max_epochs` and `max_steps`!"。
    local NO_EPOCH="model.max_epochs=null"

    step "L1 配置组装（不加载权重、不占显存）"
    bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
    ok "L1 通过"

    step "L2 数据管线 + forward/backward（权重随机初始化，不验证权重加载）"
    bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
        model.pretrained_ckpt=null model.max_steps=10 "$NO_EPOCH" logger.mode=offline
    ok "L2 通过（数据管线没崩）"

    step "L3 ★ 权重加载验证（11GB 加载；脚本自动判定并返回门禁退出码）"
    STRICT="$STRICT_L3" bash verify_weight_load.sh
    ok "L3 通过（权重完整加载）"

    if [[ "$SKIP_L4" == "1" ]]; then
        warn "SKIP_L4=1，跳过 L4"
    else
        step "L4 单批过拟合（loss 应明显下降）"
        bash scripts/run/finetune.sh 1 yam --overfit_batch 5 \
            model.max_steps=500 "$NO_EPOCH" logger.mode=offline
        ok "L4 跑完（请确认上面 loss 明显下降）"
    fi
}

# ============================================================================
# train：正式训练 + 守护上传 + 等待结束 + 收尾上传权重
# ============================================================================
do_train() {
    require_hf_token        # 权重/checkpoint 要上传到 HF
    activate_venv

    local n="${GPUS:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-${MLP_WORKER_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}}}"
    [[ "${n:-0}" -ge 1 ]] || n=1
    info "GPU 数: $n（来源: GPUS / NPROC_PER_NODE / MLP_WORKER_GPU / MLP_WORKER_NUM / nvidia-smi）"
    info "注意：batch_size 是每卡的，全局 batch = batch_size × $n"

    step "启动正式训练（后台 -> /workspace/train.stdout.log）"
    nohup bash scripts/run/finetune.sh "$n" yam logger.mode=offline \
        > /workspace/train.stdout.log 2>&1 &
    local pid=$!
    echo "$pid" > /workspace/train.pid
    ok "训练 PID=$pid（已写入 /workspace/train.pid）"

    # 等 exp 目录出现（最多 5 分钟）。不用 `ls | head -1`（会触发 SIGPIPE）
    local exp="" i=0 _dirs=()
    while [[ $i -lt 30 ]]; do
        sleep 10; i=$((i+1))
        _dirs=()
        mapfile -t _dirs < <(ls -1dt "$G05_OUTPUT_DIR"/yam/*/ 2>/dev/null || true)
        if [[ ${#_dirs[@]} -gt 0 ]]; then
            exp="${_dirs[0]}"
            break
        fi
    done
    if [[ -n "$exp" ]]; then
        ok "本次 exp 目录: $exp"
        EXP_DIR_FOUND="$exp"
        echo "$exp" > /workspace/exp_dir.txt
    else
        warn "5 分钟内没识别出 exp 目录。稍后手动：ls $G05_OUTPUT_DIR/yam/"
    fi

    # 守护上传：训练中陆续把 checkpoint 推到 HF（容器销毁后的唯一备份）
    if [[ "$AUTO_UPLOAD" == "1" && -n "$exp" ]]; then
        step "启动 checkpoint 守护上传（每 ${UPLOAD_INTERVAL}s 轮询一次）"
        nohup env MODE=watch EXP_DIR="$exp" WATCH_INTERVAL="$UPLOAD_INTERVAL" \
            HF_TOKEN="$HF_TOKEN" HF_ENDPOINT="$HF_ENDPOINT" \
            bash upload_checkpoints.sh > /workspace/upload_checkpoints.log 2>&1 &
        echo $! > /workspace/upload.pid
        ok "守护上传 PID=$(cat /workspace/upload.pid)"
    elif [[ "$AUTO_UPLOAD" != "1" ]]; then
        warn "AUTO_UPLOAD=0，未启动守护上传"
    fi

    info "启动后 90 秒训练日志尾部："
    sleep 90
    tail -c 4000 /workspace/train.stdout.log || true

    cat <<EOF

============================================================
 训练在后台跑。start.sh 会【阻塞等待训练结束】，然后自动上传权重。
 日志：训练 stdout 会被上传到 HF（容器销毁后仍可诊断）
 exp 目录：${exp:-<未识别>}
============================================================
EOF

    # ------------------------------------------------------------ 等待训练结束
    # ⚠️ 用 `wait` 而不是轮询 `kill -0`：轮询只能知道"进程没了"，【拿不到退出码】，
    #    所以无法区分"训练正常结束"和"训练崩溃后静默退出"。实测踩过：训练只跑 5
    #    分钟就退出、什么都没保存，而我们看不到退出码，只能猜。
    #    wait 还会在信号中断时返回 >128（如 137=OOM kill），这是关键线索。
    step "等待训练结束（PID=$pid）"
    local rc=0
    set +e
    wait "$pid"
    rc=$?
    set -e
    echo "$rc" > /workspace/train.exit_code
    if [[ $rc -eq 0 ]]; then
        ok "训练进程正常结束（退出码 0）"
    else
        warn "训练进程【异常退出】退出码=$rc（已写入 /workspace/train.exit_code）"
        case "$rc" in
            137) warn "  137 = 128+9(SIGKILL) —— 常见于【被 OOM killer 杀掉】或平台强杀" ;;
            143) warn "  143 = 128+15(SIGTERM) —— 被要求终止（平台/超时）" ;;
            134) warn "  134 = 128+6(SIGABRT) —— C++ 层 abort（CUDA/NCCL 常见）" ;;
            139) warn "  139 = 128+11(SIGSEGV) —— 段错误" ;;
            *)   warn "  非零退出码，见训练 stdout 日志尾部" ;;
        esac
    fi

    # 训练日志尾部立刻打到平台日志里（万一上传失败，平台日志至少留一份）
    info "训练 stdout 尾部 120 行（完整日志会随收尾上传到 HF）："
    tail -n 120 /workspace/train.stdout.log 2>/dev/null || true

    # 等最终 checkpoint 落盘（finetune.py 保存后进程才退出，IO 可能仍在 flush）
    info "等 120s 让最终 checkpoint 落盘..."
    sleep 120

    if [[ -z "$exp" ]]; then
        warn "没有 exp 目录，跳过收尾上传。请手动："
        echo "     EXP_DIR=<exp目录> bash upload_output.sh"
        return 0
    fi

    # 停守护进程，避免与收尾上传抢同一个文件
    if [[ -f /workspace/upload.pid ]]; then
        kill "$(cat /workspace/upload.pid)" 2>/dev/null || true
        sleep 5
    fi

    step "收尾上传 checkpoint"
    MODE=all EXP_DIR="$exp" bash upload_checkpoints.sh || warn "checkpoint 收尾上传失败，可重跑"

    step "收尾上传最终权重"
    EXP_DIR="$exp" bash upload_output.sh || warn "权重收尾上传失败，可重跑：EXP_DIR=$exp bash upload_output.sh"

    step "全部完成"
    cat <<EOF

============================================================
 权重已上传：https://huggingface.co/Zenith6666/yam-backbone
   - model_state_dict.pt        最终权重
   - checkpoints/step_*.pt      各 step checkpoint
   - hydra_config.yaml / dataset_stats.json / action_tokenizer.pt
 容器内还留着的（销毁即失）：
   $exp/last.pt
============================================================
EOF

    if [[ "${KEEPALIVE:-0}" -gt 0 ]]; then
        info "KEEPALIVE=${KEEPALIVE}s，挂起以便进容器查看..."
        sleep "$KEEPALIVE"
    fi
}

# ============================================================================
# smoke：只做"便宜但高价值"的检查，不下载权重/数据、不跑训练
#   目的：把"配置级 / 库可用性"问题从 ~20 分钟压到 ~1 分钟暴露。
#   已覆盖的坑：max_epochs/max_steps 冲突、缺源码目录、torchcodec 缺 FFmpeg。
# ============================================================================
do_smoke() {
    # 优先用项目 venv；没有就用系统 python（镜像自带 torch）。
    # 这样 smoke 在【装环境之前】就能跑，最早暴露问题。
    local PY
    if [[ -x .venv/bin/python ]]; then
        PY=".venv/bin/python"
        info "使用项目 venv：$PY"
    else
        PY="python"
        warn "没有 .venv，用系统 python（镜像自带的 torch）"
    fi

    step "smoke 1/3 包完整性（g05.* import 是否都能解析）"
    python check_package.py . && ok "包完整性通过" || fail "包完整性失败（代码缺文件）"

    step "smoke 2/3 配置自检（hydra 组装 + validate + 互斥检查）"
    # 这一步需要 hydra/omegaconf/g05，只在 .venv 里 —— 没有时明确跳过并说明，
    # 而不是抛一个看不懂的 ModuleNotFoundError。
    if [[ ! -x .venv/bin/python ]]; then
        warn "没有 .venv —— 配置自检【无法运行】（需要 hydra/omegaconf/g05）"
        warn "它会在【装完环境后、下载权重前】自动跑（start.sh 的 1.5 阶段）"
        warn "想现在就验配置：加环境变量 SMOKE_AUTO_SETUP=1，或先跑 MODE=setup"
        if [[ "${SMOKE_AUTO_SETUP:-0}" == "1" ]]; then
            info "SMOKE_AUTO_SETUP=1 —— 先跑 setup 装环境（约 4-6 分钟），再验配置"
            do_setup
        fi
    fi
    if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import hydra, omegaconf" 2>/dev/null; then
        PY=".venv/bin/python"
        # 三条路径都验：默认、L2 的 --test、L4 的 --overfit
        "$PY" config_check.py task=yam \
            && ok "配置自检通过（task=yam 默认路径）" \
            || fail "配置自检失败（task=yam）"
        "$PY" config_check.py --test-mode \
            model.pretrained_ckpt=null model.max_steps=10 model.max_epochs=null \
            && ok "配置自检通过（L2 --test 路径）" \
            || fail "配置自检失败（L2 --test 路径）"
        "$PY" config_check.py --overfit \
            model.max_steps=500 model.max_epochs=null \
            && ok "配置自检通过（L4 --overfit 路径）" \
            || fail "配置自检失败（L4 --overfit 路径）"
    fi

    step "smoke 3/3 视频解码后端（真实解码一个样本；不需要数据集）"
    "$PY" - <<'PY'
import os, sys, tempfile, traceback
print(f"  G05_VIDEO_BACKEND = {os.environ.get('G05_VIDEO_BACKEND', '<未设，自动探测>')}")
try:
    import numpy as np, torch, torchvision
except Exception as e:
    print(f"  [WARN] torch/torchvision 不可用，跳过：{type(e).__name__}: {e}")
    sys.exit(0)

# 造一个小视频，用代码里同一套判据决定后端
rng = np.random.default_rng(0)
frames = rng.integers(0, 255, size=(16, 64, 64, 3), dtype=np.uint8)
tmp = tempfile.mkdtemp(prefix="smoke_vid_")
path = os.path.join(tmp, "probe.mp4")
torchvision.io.write_video(path, torch.from_numpy(frames), fps=8)

# 1) torchcodec 真实可用性（与 src/g05/.../video_utils.py 的探测一致）
tc_ok = False
try:
    from torchcodec.decoders import VideoDecoder
    md = VideoDecoder(path, device="cpu").metadata
    tc_ok = True
    print(f"  torchcodec: OK（metadata 读取成功，平均 fps={md.average_fps}）")
except Exception as e:
    print(f"  torchcodec: 不可用 -> {type(e).__name__}: {str(e).splitlines()[0][:140]}")
    print("             （缺系统 FFmpeg 共享库时就是这样；会回退 pyav）")

# 2) pyav 路径（torchvision 的 pyav 后端）—— 这是回退方案，必须能用
try:
    from g05.data.lerobot.datasets.video_utils import decode_video_frames
    for backend in (["pyav"] if not tc_ok else ["torchcodec", "pyav"]):
        out = decode_video_frames(path, [0.0, 0.5, 1.0], tolerance_s=1e-3, backend=backend)
        print(f"  decode_video_frames(backend={backend}): OK  shape={tuple(out.shape)}")
except Exception as e:
    print(f"  [FAIL] decode_video_frames 失败：{type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

import shutil; shutil.rmtree(tmp, ignore_errors=True)
if not tc_ok:
    print("  → 结论：torchcodec 不可用但 pyav 可用，训练会走 pyav（预期行为）")
PY
    ok "视频解码检查完成（若 torchcodec 不可用，pyav 可用即通过）"

    step "smoke 结束"
    cat <<'EOF'
 smoke 检查的都是"便宜但高价值"的项：
   1. 代码完整性（缺文件/被 .gitignore 吞）
   2. 配置可组装 + 校验通过 + max_epochs/max_steps 不冲突
   3. 视频解码后端真实可用（不用下载数据集）

 没验的（仍需真实运行，但可以放到正式任务里一次跑完）：
   - 11.44GB 权重能否完整加载（L3）
   - flash-attn / CuTe kernel 能否编译（已默认绕开，用 SDPA）
   - 显存是否够、DDP 是否正常
EOF
}

# ============================================================================
# smoke-ddp：多卡 DDP/NCCL 自检（极小模型，约 1-2 分钟）
#   用途：区分"4 卡通信环境本身有问题"还是"模型某处让 NCCL 越界"。
#   实测正式训练在 DDP 构造期间报 illegal memory access，1 卡却正常。
# ============================================================================
do_smoke_ddp() {
    # 优先用项目 venv；没有就用【镜像自带的 python/torch】——
    # 这样在多卡任务里 1-2 分钟就能出结果，不必先跑 6-8 分钟的 setup。
    local PY
    if [[ -x .venv/bin/python ]]; then
        activate_venv
        PY="python"
        info "使用项目 venv"
    else
        PY="python"
        warn "没有 .venv —— 用镜像自带的 python/torch（能验通信即可，不必装依赖）"
        if ! python -c "import torch, torch.distributed" 2>/dev/null; then
            fail "镜像自带的 python 没有 torch —— 请先跑 MODE=setup，或用带 torch 的镜像"
        fi
        python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'ndev', torch.cuda.device_count())"
    fi
    local n="${GPUS:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-${MLP_WORKER_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}}}"
    [[ "${n:-0}" -ge 1 ]] || n=1
    info "多卡自检，GPU 数=$n"

    if [[ ! -f tools/ddp_smoke_test.py ]]; then
        fail "缺少 tools/ddp_smoke_test.py"
    fi

    step "多卡 DDP/NCCL 自检（极小模型，不加载权重/数据）"
    # ⚠️ 参数与 scripts/run/finetune.sh:321-325 的官方单机路径【完全一致】，
    #    这样自检结果才能代表正式训练的启动方式。
    set +e
    "$PY" -m torch.distributed.run \
        --standalone --nnodes 1 --nproc-per-node "$n" \
        tools/ddp_smoke_test.py
    local rc=$?
    set -e
    echo
    if [[ $rc -eq 0 ]]; then
        ok "多卡自检【全部通过】—— 4 卡 DDP/NCCL 环境正常"
        cat <<'EOF'
 结论：环境没问题。正式训练崩溃的原因在【模型】侧。
EOF
        return 0
    fi

    # ---- 失败：自动尝试一组 NCCL 配置，找出哪种能跑通 --------------------
    # 实测第 1 步 dist.broadcast 就报 "Cuda failure 1 'invalid argument'"
    # （用镜像自带 torch、极小模型、无数据集），说明【多卡通信机制】不被支持。
    # 与其一轮一轮排队试，不如一次把候选配置全试一遍。
    warn "多卡自检失败 rc=$rc —— 自动尝试以下 NCCL 配置（每个约 20 秒）"
    cat <<'EOF'

 候选配置说明：
   A 保持现状                    —— 基线
   B NCCL_P2P_DISABLE=1          —— 关 GPU 间 P2P（最可能有效：单卡不用 P2P 所以正常）
   C NCCL_P2P_DISABLE=1 + IB 关  —— 退到共享内存/套接字
   D NCCL_P2P_LEVEL=LOC          —— 只用同 PCIe 交换机内的 P2P
   E NCCL_NET_PLUGIN=none        —— 不加载外部网络插件
EOF
    local cfg_names=(
        "A_基线"
        "B_关P2P"
        "C_关P2P+关IB"
        "D_P2P_LEVEL_LOC"
        "E_无net插件"
    )
    local cfg_envs=(
        ""
        "NCCL_P2P_DISABLE=1"
        "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1"
        "NCCL_P2P_LEVEL=LOC"
        "NCCL_NET_PLUGIN=none"
    )
    local worked=()
    local idx
    for idx in "${!cfg_names[@]}"; do
        local nm="${cfg_names[$idx]}"
        local ev="${cfg_envs[$idx]}"
        echo
        info "---- 尝试 [$nm]  env: ${ev:-<无追加>} ----"
        set +e
        # shellcheck disable=SC2086
        env $ev "$PY" -m torch.distributed.run \
            --standalone --nnodes 1 --nproc-per-node "$n" \
            tools/nccl_minimal_test.py 2>&1 | \
            grep -E 'nccl-min|Cuda failure|Unhandled|ncclUnhandled' | head -20
        local rc2=${PIPESTATUS[0]}
        set -e
        if [[ $rc2 -eq 0 ]]; then
            ok "[$nm] ✅ 通过"
            worked+=("$nm => $ev")
        else
            warn "[$nm] ❌ 失败 rc=$rc2"
        fi
    done

    echo
    step "NCCL 配置尝试结论"
    if [[ ${#worked[@]} -eq 0 ]]; then
        warn "所有配置都失败 —— 这个容器的多卡通信确实不可用"
        cat <<'EOF'

 建议（按优先级）：
   1) 换镜像：当前是 2.7.0a0+ecf3bae40a.nv25.02。改用平台【官方推荐】的
      PyTorch 镜像，或与 GPU 架构匹配的 CUDA 版本
   2) 查 GPU 真实型号：nvidia-smi 显示 "NVIDIA RPBZZZ6" 是【平台脱敏名】。
      若型号较特殊，通用镜像里的 NCCL/PyTorch 可能没有该架构的 kernel ——
      "Cuda failure 1 'invalid argument'" 正是这种典型症状
   3) 暂时用 1 卡：1 卡已证明完全可用（L4 过拟合 500 步通过），
      只是慢；先跑出可用权重，再解决多卡
EOF
    else
        ok "以下配置可以跑通 4 卡通信："
        local w
        for w in "${worked[@]}"; do
            echo "     $w"
        done
        cat <<'EOF'

 用法：把生效的那组变量加到平台执行命令里，例如：
     export NCCL_P2P_DISABLE=1
   （start.sh 也支持外部覆盖这些变量；下一轮我会把它们设为默认。）
EOF
    fi
    return 0
}

# ---------------------------------------------------------------- 分发
case "$MODE" in
    smoke)     do_smoke ;;
    smoke-ddp) do_smoke_ddp ;;
    setup)  do_setup ;;
    verify) do_verify ;;
    train)  do_train ;;
    all)
        do_setup
        do_verify          # L3 不通过会在这里非零退出，不会带着坏权重去训练
        do_train
        ;;
    *)      fail "未知 MODE=$MODE（可选 smoke | smoke-ddp | setup | verify | train | all）" ;;
esac

step "MODE=$MODE 结束"
