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
# ------------------------------------------------- NCCL：尊重平台注入，只做必要干预
#
# 【踩过的弯路，记录以免重犯】
# 曾在这里硬编码 NCCL_NET_PLUGIN=none / NCCL_IB_DISABLE=1 / NCCL_SHARP_DISABLE=1，
# 以为"单机多卡不需要 RDMA"。实测证明【无效且有害】：
#   - 无效：那几组配置全试过，4 卡 broadcast 依旧失败
#   - 有害：百度百舸的文档明确说「开启 RDMA 后平台会注入推荐的 NCCL 环境变量
#           （NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_IB_TIMEOUT / NCCL_TOPO_FILE …），
#           强烈建议使用平台预置的 NCCL 默认变量」。硬编码 IB_DISABLE=1 会
#           【直接废掉 RDMA】—— 一旦以后要跑多机就踩坑。
#   已确认平台确实注入了 NCCL_SOCKET_IFNAME、NCCL_IB_ADAPTIVE_ROUTING=0（日志可见）。
#
# 现在的原则：**不覆盖平台的注入**。只在平台【没设】时才补默认值，且只补真正安全的。
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"          # 同机 P2P 优先 NVLink；无 NVLink 时 NCCL 自行降级
export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"    # 关 SHARP（我们没开 collnet）
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"  # 关 collnet
# ⚠️ 不设 NCCL_IB_DISABLE / NCCL_NET_PLUGIN / NCCL_P2P_DISABLE ——
#    交回给平台与环境变量，避免与 RDMA 预设冲突。
#
# 若确实需要（无 P2P 的 Blackwell 卡：sm_120 且无 NVLink）：
#   平台执行命令里加  export NCCL_P2P_DISABLE=1
#   且 NCCL 版本必须 ≥ 2.27（2.25.1 does not know sm_120，连降级都做不到）

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

# ------------------------------------------------- 磁盘保护：限制 checkpoint 份数
# ★ 实测踩过：每份 checkpoint ≈ 34.3 GB（model 11.44 + optimizer 22.88），
#   官方代码【不清理】旧份数。1 纪元 68,570 步、每 2000 步一存 = 34 份
#   = 1,166 GB，而节点可用磁盘约 535 GB。
#   结果：写到第 15 份时磁盘满 -> Kubernetes 报 DiskPressure -> 直接 SIGTERM
#   驱逐容器（退出码 143，日志里【没有任何 Python 报错】，极难定位）。
# 默认只保留最近 2 份（约 69 GB），旧的在保存下一份后自动删除。
# 设 0 可关闭（恢复官方原始行为，但长训练会写满磁盘）。
export G05_KEEP_CHECKPOINTS="${G05_KEEP_CHECKPOINTS:-2}"
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
    prefer_venv_nccl
}

# ★ 让动态链接器优先加载【venv 里那份 NCCL】。
#
# 为什么必须做（实测踩坑）：
#   NGC 系镜像的全局 LD_LIBRARY_PATH 里已经含 /usr/local/lib/python3.12/dist-packages
#   （即容器自带 torch 的 nvidia/nccl/lib，版本 2.25.1/2.26.2，早于 sm_120 支持）。
#   而 venv 的 torch 对 libnccl.so.2 是【不带路径的 NEEDED】引用 ——
#   动态链接器按 LD_LIBRARY_PATH 顺序找，于是容器那份旧版【抢先被加载】。
#   现象：uv pip 明明报"已装 2.29.7"，但 torch.cuda.nccl.version() 仍是 (2,26,2)。
#   解法：把 venv 的 nvidia/nccl/lib 前置到 LD_LIBRARY_PATH 最前面。
prefer_venv_nccl() {
    local d=""
    for cand in \
        "$PROJECT_ROOT/.venv/lib/python3.10/site-packages/nvidia/nccl/lib" \
        "$PROJECT_ROOT"/.venv/lib/python*/site-packages/nvidia/nccl/lib ; do
        [[ -d "$cand" ]] && d="$cand" && break
    done
    if [[ -z "$d" ]]; then
        warn "找不到 venv 的 nvidia/nccl/lib（可能尚未安装 NCCL）—— 跳过 LD_LIBRARY_PATH 调整"
        return 0
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$d:"*) : ;;                                  # 已在最前，幂等
        *) export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" ;;
    esac
    # 确认生效：用 ldd 看 torch 的 C 扩展【实际链接】到哪份 libnccl.so.2。
    # ⚠️ 不要用 torch.cuda.nccl.version() —— 它返回 torch【编译期】写入的常量，
    #    与运行时实际加载的库无关，永远报同一个数字（实测踩坑：z 升级后它仍报
    #    2.26.2，而 ldd 显示已指向 venv 的新版）。用它做判据会得出错误结论。
    local got_ldd
    got_ldd="$(ldd "$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib","libtorch_cuda.so"))' 2>/dev/null)" 2>/dev/null \
        | awk '/libnccl/{print $3; exit}')"
    if [[ -n "$got_ldd" ]]; then
        if [[ "$got_ldd" == *"$d"* || "$got_ldd" == *"nvidia/nccl"* ]]; then
            ok "torch 实际链接的 NCCL = $got_ldd（venv 的 nvidia/nccl 版本）"
        else
            warn "torch 实际链接的 NCCL = $got_ldd（不是 venv 的那份）"
        fi
    else
        warn "ldd 读不到 libnccl 链接目标（继续，不影响后续）"
    fi
    # 附带报告一下 torch 编译期常量（仅供参考，不作为判据）
    info "（参考）torch 编译期记录的 NCCL 常量 = $(python -c 'import torch;print(".".join(map(str,torch.cuda.nccl.version())))' 2>/dev/null || echo '?')"
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
    # ⚠️ 有些镜像只有 python3（例如百舸 aihc-daft-gpu），写死 python 会误报失败
    HOST_PY="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
    if [[ -z "$HOST_PY" ]]; then
        warn "镜像里既没有 python 也没有 python3，跳过包完整性检查"
    else
    info "包完整性检查（g05.* import 是否都能解析）... [$HOST_PY]"
    if "$HOST_PY" check_package.py .; then
        ok "包完整性通过"
    else
        fail "包完整性检查失败 —— 见上方缺失模块清单（这是【代码缺文件】，不是环境问题）"
    fi
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

    # SKIP_DATASET=1：跳过数据集下载（MODE=serve-check 不需要数据 —— 它用合成观测）。
    # 评测（MODE=eval）和训练（MODE=train）必须要有数据集。
    if [[ "${SKIP_DATASET:-0}" == "1" ]]; then
        warn "SKIP_DATASET=1 —— 跳过数据集下载（只有 MODE=serve-check 能这么用）"
    else
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
    fi

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
    # 磁盘预算检查：每份 checkpoint 约 34.3GB（model 11.44 + optimizer 22.88）
    local _avail
    _avail="$(df -BG --output=avail "$G05_OUTPUT_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')" || true
    if [[ -n "${_avail:-}" ]]; then
        info "checkpoint 保留 ${G05_KEEP_CHECKPOINTS} 份 × 34.3GB ≈ $(( ${G05_KEEP_CHECKPOINTS} * 34 )) GB（可用 ${_avail} GB）"
        if [[ "${G05_KEEP_CHECKPOINTS}" != "0" && $(( G05_KEEP_CHECKPOINTS * 34 )) -gt "$_avail" ]]; then
            warn "保留份数会超出可用磁盘！请调小：G05_KEEP_CHECKPOINTS=1"
        fi
    fi

    step "启动正式训练（后台 -> /workspace/train.stdout.log）"
    # ★ 透传 start.sh 收到的额外参数给 finetune.sh（它会把非 -- 开头的当作
    #   hydra 覆盖，例如 model.max_epochs=1）。
    #   ⚠️ 之前这里没透传，导致 `bash start.sh model.max_epochs=1` 静默失效 ——
    #      用户以为只跑 1 epoch，实际跑了 10 个 epoch（约 11 天）。
    if [[ $# -gt 0 ]]; then
        info "透传 hydra 覆盖给 finetune.sh：$*"
    fi
    nohup bash scripts/run/finetune.sh "$n" yam logger.mode=offline "$@" \
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
    # ★ 关键：要测【正式训练真正使用的那个 torch】。
    #   容器自带 torch 来自镜像；正式训练用的是 .venv 里的 torch（uv 按 uv.lock 装的）。
    #   两者 NCCL 可能完全不同 —— 只测容器 torch 会得出"换镜像就行"的错误结论。
    #   规则：
    #     - 有 .venv 时【默认用 .venv 的 python】测（这才是正式训练的环境）
    #     - 没有 .venv 时用容器 python 测，并明确标注"这不是正式训练用的"
    #     - 可用 G05_SMOKE_IN_VENV=0/1 强制指定
    local PY
    local want_venv="${G05_SMOKE_IN_VENV:-auto}"
    if [[ -x .venv/bin/python && "$want_venv" != "0" ]]; then
        activate_venv
        PY="python"
        ok "使用【.venv 的 python】测试 —— 这是正式训练真正使用的环境"
        python -c "import torch; print('  venv torch', torch.__version__, 'cuda', torch.version.cuda, 'ndev', torch.cuda.device_count())"
    elif [[ -x .venv/bin/python ]]; then
        PY="python"
        warn "G05_SMOKE_IN_VENV=0 —— 强制用【容器自带】python 测（注意：正式训练用的是 .venv）"
    else
        # ⚠️ 有些镜像（例如百舸 aihc-daft-gpu）只提供 python3，没有 python 命令。
        #    直接写死 PY="python" 会在这种镜像里报 command not found。
        if command -v python >/dev/null 2>&1; then
            PY="python"
        elif command -v python3 >/dev/null 2>&1; then
            PY="python3"
            warn "镜像没有 python 命令，改用 python3"
        else
            fail "镜像里既没有 python 也没有 python3"
        fi
        warn "没有 .venv —— 只能用【容器自带】python 测"
        warn "⚠️ 正式训练用的是 .venv 的 torch，两者 NCCL 不同 —— 本次结果不能代表正式训练"
        warn "   想测正式训练的环境：先跑 MODE=setup 装出 .venv，再跑 MODE=smoke-ddp"
        if ! "$PY" -c "import torch, torch.distributed" 2>/dev/null; then
            fail "镜像自带的 $PY 没有 torch —— 请先跑 MODE=setup，或用带 torch 的镜像"
        fi
        "$PY" -c "import torch; print('  容器 torch', torch.__version__, 'cuda', torch.version.cuda, 'ndev', torch.cuda.device_count())"
    fi
    local n="${GPUS:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-${MLP_WORKER_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}}}"
    [[ "${n:-0}" -ge 1 ]] || n=1
    info "多卡自检，GPU 数=$n"

    # ★ 打印【平台注入的 NCCL 变量】—— 百舸文档说：开启 RDMA 后平台会自动注入
    #   NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_IB_TIMEOUT / NCCL_TOPO_FILE 等，
    #   并"强烈建议使用平台预置的默认变量"。看清平台给了什么，才知道该不该动。
    info "平台/环境注入的 NCCL 变量："
    env | grep -E '^NCCL_' | sort | sed 's/^/       /' || echo "       （无 —— 可能未开启 RDMA）"
    info "平台注入的 PyTorch 分布式变量："
    for v in MASTER_ADDR MASTER_PORT RANK WORLD_SIZE NPROC_PER_NODE LOCAL_RANK \
             MLP_WORKER_NUM MLP_WORKER_GPU MLP_ROLE_INDEX MLP_WORKER_0_HOST; do
        if [[ -n "${!v:-}" ]]; then printf '       %-18s = %s\n' "$v" "${!v}"; fi
    done
    echo

    # ★ 关键交叉验证：平台注入的 NCCL_IB_HCA / NCCL_SOCKET_IFNAME 指向的设备，
    #   在【容器内】真的存在吗？百舸按宿主机拓扑注入这些变量，但容器里看到的
    #   设备/IP 可能与宿主机不同 —— 对不上就会让 NCCL 的 IB 传输初始化失败。
    info "容器内【真实存在】的网络/RDMA 设备（与上面注入的值对比）:"
    if command -v ibv_devices >/dev/null 2>&1; then
        echo "       --- ibv_devices（RDMA 设备名）---"
        ibv_devices 2>&1 | sed 's/^/       /' || true
    else
        echo "       ibv_devices: 命令不存在（未装 OFED 用户态工具）"
    fi
    if [[ -d /sys/class/infiniband ]]; then
        echo "       --- /sys/class/infiniband（RDMA 设备）---"
        ls -1 /sys/class/infiniband 2>/dev/null | sed 's/^/       /' || true
    else
        echo "       /sys/class/infiniband: 不存在（容器内看不到 RDMA 设备）"
    fi
    echo "       --- IP 接口（NCCL_SOCKET_IFNAME 应指向其中之一）---"
    (ip -o -4 addr show 2>/dev/null || ifconfig -a 2>/dev/null) | sed 's/^/       /' || true
    echo "       --- /dev/infiniband（RDMA 字符设备）---"
    ls -1 /dev/infiniband 2>/dev/null | sed 's/^/       /' || echo "       /dev/infiniband: 不存在"
    # ★ RoCE 网卡的有效 GID 列表：平台注入 NCCL_IB_GID_INDEX=7，
    #   若索引 >= 有效个数，IB 传输初始化会失败。这里把真实的有效索引列出来。
    for dev in $(ls -1 /sys/class/infiniband 2>/dev/null || true); do
        for port in $(ls -1 "/sys/class/infiniband/$dev/ports" 2>/dev/null || true); do
            gdir="/sys/class/infiniband/$dev/ports/$port/gids"
            if [[ -d "$gdir" ]]; then
                n=$(ls -1 "$gdir" 2>/dev/null | wc -l | tr -d ' ')
                echo "       $dev port $port: 有 $n 个 GID 索引（有效范围 0..$((n-1))）"
                for g in $(ls -1 "$gdir" 2>/dev/null | head -3); do
                    printf '         gid %s = %s\n' "$g" "$(cat "$gdir/$g" 2>/dev/null)"
                done
            fi
        done
    done
    echo
    # ★ GPU 拓扑：确认是否存在 P2P。
    #   Blackwell 入门专业卡/消费卡（含 RTX PRO 6000 / RTX 50 系）在普通 PCIe 上
    #   通常【没有可用的 P2P、也没有 NVLink】。
    #   GPU 之间若显示 SYS（经 PCIe 主机桥）而非 NV#/PIX，即为无 P2P ——
    #   这时必须 NCCL_P2P_DISABLE=1，且 NCCL 版本要 ≥2.27 才认识 sm_120。
    if nvidia-smi topo -m >/dev/null 2>&1; then
        info "GPU 拓扑（NV#/PIX = P2P 可用；SYS = 无 P2P，需 NCCL_P2P_DISABLE=1）:"
        nvidia-smi topo -m 2>/dev/null | sed 's/^/       /' || true
        echo
    fi

    # ★ 两套 NCCL 必须区分清楚（这是最容易搞错的地方）：
    #     容器自带 torch 的 NCCL  <- 来自【镜像】     -> 换镜像才能改
    #     .venv 里 torch 的 NCCL  <- 来自【uv.lock】  -> 换镜像【改不了】，要单独升级
    #   正式训练用的是后者。所以若此刻用的是容器 python，就把 venv 的情况也打出来对照。
    if [[ -x .venv/bin/python && "$want_venv" == "0" ]]; then
        echo
        info "【对照组】.venv 的 torch/NCCL（正式训练真正使用的）:"
        .venv/bin/python - <<'PYV' || true
import glob, os, site
try:
    import torch
    print(f"    venv torch={torch.__version__} cuda={torch.version.cuda} "
          f"nccl={torch.cuda.nccl.version()}")
    arch = torch.cuda.get_arch_list()
    print(f"    venv arch_list={arch}")
    print(f"    venv arch_list 含 sm_120 ? {'sm_120' in arch}")
except Exception as e:
    print(f"    venv torch 读取失败: {type(e).__name__}: {e}")

# ★ 关键：libnccl.so 到底来自【torch 自带】还是【nvidia-nccl-cu12 pip 包】？
#   决定修法：前者 -> 必须换 torch/镜像；后者 -> 只升级 pip 包即可（不动 torch）。
try:
    import torch as _t
    torch_dir = os.path.dirname(_t.__file__)
    site_dirs = []
    try:
        site_dirs = list(site.getsitepackages())
    except Exception:
        pass
    # 1) torch 自带（torch/lib/）
    d1 = glob.glob(os.path.join(torch_dir, "lib", "libnccl.so*"))
    print(f"    [A] torch 自带 torch/lib/libnccl.so*: {d1 or '无'}")
    # 2) pip 包 nvidia/nccl/lib/
    d2 = []
    for r in [os.path.dirname(torch_dir)] + site_dirs:
        d2 += glob.glob(os.path.join(r, "nvidia", "nccl", "lib", "libnccl.so*"))
        d2 += glob.glob(os.path.join(r, "**", "nvidia", "nccl", "**", "libnccl.so*"),
                        recursive=True)
    d2 = sorted(dict.fromkeys(d2))
    print(f"    [B] pip 包 nvidia/nccl/lib/libnccl.so*: {d2 or '无'}")
    # 3) 版本目录名（pip 包路径里带版本号）
    for f in d2[:3]:
        print(f"        版本线索: {f}")
    if not d1 and not d2:
        print("    -> 两者都没找到：NCCL 可能静态链接进 torch（升 pip 包无效）")
    elif d2 and not d1:
        print("    -> ★ 用 pip 包：可通过升级 nvidia-nccl-cu12 修，不必换 torch")
    elif d1:
        print("    -> ★ 用 torch 自带：升 pip 包无效，必须换 torch/镜像")
except Exception as e:
    print(f"    venv libnccl 查找失败: {e}")
PYV
    else
        warn "只有容器自带的 torch 可测 —— 无 .venv 对照组"
        warn "⚠️ 本次结果【不能代表正式训练】：正式训练用的是 .venv 的 torch，其 NCCL"
        warn "   由 uv.lock 决定（与镜像无关，换镜像也改不了）。"
        warn "   要验证正式训练的环境：先 MODE=setup，再 MODE=smoke-ddp（会自动用 .venv）"
    fi
    echo

    if [[ ! -f tools/ddp_smoke_test.py ]]; then
        fail "缺少 tools/ddp_smoke_test.py"
    fi

    step "多卡 DDP/NCCL 自检（极小模型，不加载权重/数据）"
    # ⚠️ 这里【不再用 torchrun】。
    #   实测：百舸平台会把「节点级」的 RANK/WORLD_SIZE/MASTER_* 注入容器，
    #   而 torchrun/elastic 会继承它们，导致子进程拿到的 RANK/LOCAL_RANK 是乱值
    #   （实测见过 16 / 29 / 206 / 228），于是
    #       torch.cuda.set_device(local) -> RuntimeError: device id must be
    #       non-negative! -28
    #   进程在【进入 NCCL 之前】就死了 —— 之前所有 NCCL 环境变量实验全部失效，
    #   因为换什么 NCCL 版本都走不到通信那一步。
    #   改用自包含启动器：自己分配端口、清空 RANK/WORLD_SIZE、逐进程显式设
    #   RANK/LOCAL_RANK，device 号取模防御，永不越界。
    set +e
    "$PY" tools/nccl_multiproc_test.py --all --nproc "$n"
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
    # 实测：GPU 是 sm_120（Blackwell 入门专业卡），容器 NCCL 2.25.1 / venv 2.26.2
    # —— 都早于 sm_120 支持，连最基本的 broadcast 都报
    #     Cuda failure 1 'invalid argument'（enqueue.cc:1500，kernel 启动失败）
    # 而 1 卡正常（单卡不需要跨卡 kernel）。
    #
    # ★ 另外：百舸【按宿主机拓扑】注入了 RDMA 变量（NCCL_IB_HCA / NCCL_IB_GID_INDEX
    #   / NCCL_SOCKET_IFNAME / NCCL_IB_ADAPTIVE_ROUTING）。若这些值在【容器内】
    #   对不上真实设备，NCCL 的 IB 传输初始化就会失败 —— 所以专门有一组把它们清掉。
    warn "多卡自检失败 rc=$rc —— 自动尝试以下 NCCL 配置（每个约 15 秒）"
    cat <<'EOF'

 候选配置（按成功率排序）：
   A 基线                             —— 保留平台注入的一切
   B 清掉平台注入的 RDMA 变量
   C P2P 关闭 + 清 RDMA 变量
   D 纯套接字
   E ★ 清掉【全部】平台注入的 NCCL 变量 —— 平台是按 A100/H800（有 NVLink+RoCE）
     注入的：NCCL_P2P_LEVEL=NVL / NCCL_MIN_NCHANNELS=8 / NCCL_VERSION=2.25.1
     / NCCL_IB_TC=162 / NCCL_IB_HCA==mlx5_2（值带多余 '='）等。本机【无 NVLink】。
   F 清全部 + 关 P2P + 关 IB
   G 只清 P2P_LEVEL 和 NCHANNELS
   H ★ 清全部 + MAX_NCHANNELS=1 —— 平台强制 MIN_NCHANNELS=8；纯 PCIe 上
     8 个 channel 可能建不起来，反向强制只 1 个
   I ★ 清全部 + PROTO=Simple —— LL/LL128 对延迟敏感，走最保守协议
   J ★ 清全部 + ALGO=Ring —— NVLS/Tree 需要 NVLink，强制最保守算法
   K ★ 清全部 + P2P_LEVEL=SYS —— 允许跨 NUMA 的 P2P（本机 NIC↔GPU 就是 SYS）
EOF
    local cfg_names=(
        "A_基线"
        "B_清RDMA变量"
        "C_关P2P+清RDMA"
        "D_纯套接字"
        "E_清全部平台NCCL"
        "F_清平台+关P2P+关IB"
        "G_只清P2P_LEVEL和NCHANNELS"
        "H_清平台+MAX_NCHANNELS=1"
        "I_清平台+PROTO=Simple"
        "J_清平台+ALGO=Ring"
        "K_清平台+P2P_LEVEL=SYS"
    )
    # 百舸会按【宿主机拓扑】注入一整套 NCCL 变量。实测注入里有：
    #   NCCL_VERSION=2.25.1, NCCL_P2P_LEVEL=NVL, NCCL_MIN_NCHANNELS=8,
    #   NCCL_IB_TC=162, NCCL_IB_TIMEOUT=22, NCCL_IB_QPS_PER_CONNECTION=8,
    #   NCCL_IB_HCA==mlx5_2, NCCL_SOCKET_IFNAME==eth0, ...
    # 这些是给 A100/H800（有 NVLink + RoCE）准备的；本机【无 NVLink】，且部分
    # 值带多余的 '='（疑似注入 bug）。所以专门试“全部清掉”。
    local CLEAR_ALL_NCCL="env -u NCCL_VERSION -u NCCL_P2P_LEVEL -u NCCL_MIN_NCHANNELS \
-u NCCL_IB_HCA -u NCCL_IB_GID_INDEX -u NCCL_IB_ADAPTIVE_ROUTING -u NCCL_IB_TC \
-u NCCL_IB_TIMEOUT -u NCCL_IB_QPS_PER_CONNECTION -u NCCL_TOPO_FILE \
-u NCCL_SOCKET_IFNAME -u NCCL_SHARP_DISABLE -u NCCL_COLLNET_ENABLE"
    local CLEAR_RDMA="env -u NCCL_IB_HCA -u NCCL_IB_GID_INDEX -u NCCL_IB_ADAPTIVE_ROUTING -u NCCL_TOPO_FILE"
    # ★ H~K 是还没试过的方向：
    #   NCCL_MIN_NCHANNELS=8 被平台注入，等于【强制至少 8 个 channel】。
    #   纯 PCIe、无 NVLink 的机器上 8 个 channel 可能建不起来 —— 所以反向试
    #   MAX_NCHANNELS=1（强制只 1 个 channel）。
    #   PROTO/ALGO 方面：LL/LL128 对延迟敏感，NVLS/Tree 需要 NVLink，
    #   强制 Simple + Ring 走最保守的路径。
    #   P2P_LEVEL=SYS 允许跨 NUMA 的 P2P（本机 NIC 与 GPU 就是 SYS）。
    local cfg_cmds=(
        "env"
        "$CLEAR_RDMA"
        "$CLEAR_RDMA NCCL_P2P_DISABLE=1"
        "env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1"
        "$CLEAR_ALL_NCCL"
        "$CLEAR_ALL_NCCL NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1"
        "env -u NCCL_P2P_LEVEL -u NCCL_MIN_NCHANNELS"
        "$CLEAR_ALL_NCCL NCCL_MAX_NCHANNELS=1 NCCL_P2P_DISABLE=1"
        "$CLEAR_ALL_NCCL NCCL_PROTO=Simple NCCL_P2P_DISABLE=1"
        "$CLEAR_ALL_NCCL NCCL_ALGO=Ring NCCL_P2P_DISABLE=1"
        "$CLEAR_ALL_NCCL NCCL_P2P_LEVEL=SYS NCCL_P2P_DISABLE=0"
    )
    local worked=()
    local idx
    for idx in "${!cfg_names[@]}"; do
        local nm="${cfg_names[$idx]}"
        local cmd="${cfg_cmds[$idx]}"
        echo
        info "---- 尝试 [$nm]  命令前缀: $cmd ----"
        set +e
        # 同样不用 torchrun（原因见上文：平台注入的 RANK/WORLD_SIZE 会污染子进程）
        # shellcheck disable=SC2086
        $cmd "$PY" tools/nccl_multiproc_test.py --minimal --nproc "$n" 2>&1 | \
            grep -E '\[mp\]|Cuda failure|Unhandled|ncclUnhandled|non-negative' | head -20
        local rc2=${PIPESTATUS[0]}
        set -e
        if [[ $rc2 -eq 0 ]]; then
            ok "[$nm] ✅ 通过"
            worked+=("$nm => $cmd")
        else
            warn "[$nm] ❌ 失败 rc=$rc2"
        fi
    done

    echo
    step "NCCL 配置尝试结论"
    if [[ ${#worked[@]} -eq 0 ]]; then
        warn "所有配置都失败 —— 这个容器的多卡通信确实不可用"
        cat <<'EOF'

 已排除的原因（本自检实测，7 组配置全试）：
   - 不是 P2P：NCCL_P2P_DISABLE=1 无效
   - 不是 IB/RDMA：NCCL_IB_DISABLE=1、清掉平台注入的 IB_HCA/GID_INDEX 均无效
   - 不是共享内存：NCCL_SHM_DISABLE=1 无效
   - 不是 GID 索引：换成 0 / 3 也无效
   - 不是平台注入干扰：把平台注入的 RDMA 变量全部清掉仍无效

 实测环境（关键事实）：
   GPU compute_capability = sm_120（Blackwell），显存 97887 MiB
   GPU 拓扑：GPU 之间是 PHB/SYS（PCIe 主机桥 / NUMA），【没有 NVLink】
   容器 torch=2.7.0a0+ecf3bae40a.nv25.02  cuda=12.8  nccl=(2,25,1)
   libnccl.so 在容器内【找不到】-> torch 的 NCCL 是静态链接，无法替换
   （NCCL 2.25.1 发布于 2025-02，早于 sm_120 支持）

 结论：镜像里的 NCCL 太老且换不掉 —— 必须换一个 NCCL ≥ 2.27 的【镜像】。

 建议动作（按优先级）：
   1) ★ 换镜像。要求：NCCL ≥ 2.27（越新越好）。
      你列表里 26.0x-py3（NGC 新 tag）或 2.9.x-cu13 系可作候选。
      换完用  GPUS=4 MODE=smoke-ddp bash start.sh  复验：
      只要 [nccl-min] 那行显示 nccl=(2,2X,...) 且 ✅ 通过即可。
   2) 同时把  export NCCL_P2P_DISABLE=1  加到执行命令（无 NVLink 的卡建议常开）。
   3) ★ 注意两套 NCCL：正式训练用的是 .venv 的 torch，
      其 NCCL 由 uv.lock 决定（nvidia-nccl-cu12==2.26.2）。
      setup_env.sh 已改为自动把它升到 2.29.7，并打印 libnccl.so 来源：
        [A] torch 自带 -> 换不掉，只能靠换镜像
        [B] pip 包     -> 升级生效
   4) 若短期拿不到新镜像：用 1 卡先出权重
        GPUS=1 bash start.sh model.max_epochs=1      # 约 27 小时

 可提供给平台支持的实测证据：
   GPU0 name='NVIDIA RPBZZZ6'  compute_capability=sm_120  97887 MiB
   torch=2.7.0a0+ecf3bae40a.nv25.02  nccl=(2,25,1)  ndev=4
   4 卡 dist.broadcast: NCCL enqueue.cc Cuda failure 1 'invalid argument'
   平台注入 NCCL_IB_HCA=mlx5_2 / NCCL_IB_GID_INDEX=7 / NCCL_SOCKET_IFNAME=eth0
   7 种 NCCL 配置组合全部失败
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

# ============================================================================
# eval：离线 open-loop 评测（A=我们微调的 vs B=官方基座）
#
# 为什么需要：训练只给出 loss，不证明「在没见过的新任务上好不好」。
#   我们的验证集是完整留出的两个任务（TASK-YAM-0014/0015），
#   所以 open-loop L1 衡量的是零样本泛化。
#
# 为什么必须 A/B 两次：单看 ar_action_l1=0.05 无法判断好坏，
#   必须和官方基座在同一份数据、同一参数下比。
#
# 前置：MODE=setup 已跑过（.venv / 基座权重 / 数据集齐备）。
#
# 环境变量：
#   EVAL_EPISODES     评多少条 episode，默认 20（脚本默认只评 1 条，统计无意义）
#   EVAL_RUN_NAME     我们 run 目录名，默认 yam-ft-1ep
#   EVAL_SRC_REPO     微调产物的 HF 仓库，默认 Zenith6666/yam-backbone
#   EVAL_SKIP_BASE    1 = 只评 A，不跑基座
#   EVAL_UPLOAD       1 = 把 metrics.json 上传到 HF（默认 1）
#   EVAL_UPLOAD_VIZ   1 = 连 episodes/ 可视化一起上传（默认 0，可能几十 MB）
# ============================================================================
# ============================================================================
# prepare_yam_run_dir：把 HF 上我们微调的 4 个产物组装成一个 run 目录。
#
# 供 MODE=eval 和 MODE=serve-check 共用（两者都需要同一个 run 目录）。
# 结果写进全局变量：
#     PREPARED_RUN_DIR  /  PREPARED_CKPT
#
# find_run_dir 会从 ckpt 往上找 5 层内的 .hydra/config.yaml，
# 所以必须是 <run>/checkpoints/model_state_dict.pt 这个层级。
#
# ⚠️ 故意【不】把 action_tokenizer.pt 放进 run 目录：
#   load_config_from_run_dir 的 _apply_action_tokenizer_sidecar 一旦发现
#   run_dir/action_tokenizer.pt 存在，就会 OmegaConf.update 三处 ckpt_dir；
#   而我们的配置里 tokenizer 是【链式插值】(tokenizer: ${tokenizer},
#   model_arch.action_tokenizer: ${model.tokenizer._target_})，
#   实测在评测环境里会变成
#       InterpolationKeyError: 'model.tokenizer._target_' not found
#   不放这个文件 → sidecar 直接返回 False、零改动；
#   配置里的 ckpt_dir: checkpoints/action_tokenizer_yam.pt（相对项目根）仍能命中。
# ============================================================================
prepare_yam_run_dir() {
    local run_name="${EVAL_RUN_NAME:-yam-ft-1ep}"
    local src_repo="${EVAL_SRC_REPO:-Zenith6666/yam-backbone}"
    PREPARED_RUN_DIR="$PROJECT_ROOT/runs/$run_name"
    PREPARED_CKPT="$PREPARED_RUN_DIR/checkpoints/model_state_dict.pt"

    if [[ -f "$PREPARED_CKPT" && -f "$PREPARED_RUN_DIR/.hydra/config.yaml" \
          && -f "$PREPARED_RUN_DIR/dataset_stats.json" \
          && -f "$PREPARED_RUN_DIR/.hydra/config.yaml.patched" ]]; then
        ok "run 目录已存在且完整：$PREPARED_RUN_DIR"
        return 0
    fi

    step "组装我们微调的 run 目录（$src_repo）"
    mkdir -p "$PREPARED_RUN_DIR/checkpoints" "$PREPARED_RUN_DIR/.hydra"
    local tmp="" rc=0
    tmp="$(mktemp -d)"
    _hfget() {   # $1=repo 内路径  $2=目标
        info "下载 $src_repo :: $1"
        rm -rf "$tmp/dl"; mkdir -p "$tmp/dl"
        if ! hf download "$src_repo" "$1" --repo-type model --local-dir "$tmp/dl" >/dev/null 2>&1; then
            warn "hf download 失败：$1"; return 1
        fi
        local src="$tmp/dl/$1"
        [[ -f "$src" ]] || { warn "下载后找不到 $src"; return 1; }
        mkdir -p "$(dirname "$2")"
        mv -f "$src" "$2"
    }
    _hfget "hydra_config.yaml"     "$PREPARED_RUN_DIR/.hydra/config.yaml" || rc=1
    _hfget "dataset_stats.json"    "$PREPARED_RUN_DIR/dataset_stats.json" || rc=1
    _hfget "model_state_dict.pt"   "$PREPARED_CKPT" || rc=1
    rm -rf "$tmp"
    [[ $rc -eq 0 ]] || fail "组装 run 目录失败（见上方日志）"

    # 把链式插值就地展开（否则 to_container 会报 InterpolationKeyError）
    # ⚠️ 用 .venv/bin/python（注意 $PY 只是 do_smoke_ddp 的局部变量，这里不能用）
    info "展开配置里的链式插值（model.tokenizer / action_tokenizer / AT_CONFIG）..."
    .venv/bin/python tools/prepare_eval_config.py "$PREPARED_RUN_DIR/.hydra/config.yaml" \
        --codec-path "checkpoints/action_tokenizer_yam.pt" \
        || fail "配置展开失败"
    touch "$PREPARED_RUN_DIR/.hydra/config.yaml.patched"
    ok "run 目录组装完成：$PREPARED_RUN_DIR"
}

do_eval() {
    require_hf_token
    activate_venv

    local ne="${EVAL_EPISODES:-20}"
    local B_RUN="$PROJECT_ROOT/checkpoints/g05-base"
    local B_CKPT="$B_RUN/checkpoints/model_state_dict.pt"

    step "0. 前置检查"
    [[ -x .venv/bin/python ]] || fail "没有 .venv —— 先跑 MODE=setup"
    [[ -d "$B_RUN" ]] || fail "缺少官方基座 $B_RUN —— 先跑 MODE=setup"
    if ! ls -d data/yam/*_lerobot >/dev/null 2>&1; then
        fail "缺少数据集 data/yam/*_lerobot —— 先跑 MODE=setup"
    fi
    local n_ds
    n_ds="$(ls -d data/yam/*_lerobot 2>/dev/null | wc -l | tr -d ' ')"
    ok "基座/数据集就绪（$n_ds 个数据集）"
    # cv2：eval_open_loop.py:45 顶层就 import visualize → import cv2，
    # opencv-python 需要系统 libGL。训练链路不 import cv2，所以没暴露过。
    # setup_env.sh 4.7 已自动修；这里前置拦一下，避免又跑到一半才抛原始 traceback。
    if ! .venv/bin/python -c "import cv2" >/dev/null 2>&1; then
        fail "venv 里 import cv2 失败（opencv 需要 libGL，或 opencv-python-headless 未装）—— 重跑 MODE=setup（已加自动修复）"
    fi
    ok "cv2 可用（$(.venv/bin/python -c 'import cv2;print(cv2.__version__)' 2>/dev/null)）"
    if [[ ! -f checkpoints/action_tokenizer.pt ]]; then
        warn "缺少官方 codec checkpoints/action_tokenizer.pt —— 基座基线(B)会失败"
        warn "  修复：MODE=setup 重跑（download_weights.sh 已加入该文件下载）"
    fi

    # ---------------------------------------------------------------- 组装 A 的 run 目录
    prepare_yam_run_dir
    local A_RUN="$PREPARED_RUN_DIR"
    local A_CKPT="$PREPARED_CKPT"

    # ---------------------------------------------------------------- A：我们的
    step "2. 评测 A（微调 backbone + YAM codec + YAM stats），${ne} 条 episode"
    local logd="$PROJECT_ROOT/eval_logs"
    mkdir -p "$logd"
    local A_LOG="$logd/eval_A.log"
    info "命令：bash scripts/run/eval_open_loop.sh --ckpt_path $A_CKPT eval_episodes_num=$ne"
    set +e
    bash scripts/run/eval_open_loop.sh --ckpt_path "$A_CKPT" \
        "eval_episodes_num=$ne" 2>&1 | tee "$A_LOG"
    local arc=${PIPESTATUS[0]}
    set -e
    [[ $arc -eq 0 ]] || { warn "A 评测失败（rc=$arc），日志：$A_LOG"; tail -40 "$A_LOG" || true; fail "A 失败"; }
    # ⚠️ metrics.json 不在 eval_model_state_dict/ 根下 —— mixture 模式下按 embodiment 分
    #    子目录（实测 .../eval_model_state_dict/galaxea_r1lite/metrics.json）。
    #    所以用 find 找，不要写死路径。
    local A_MET
    A_MET="$(find "$A_RUN/eval_model_state_dict" -maxdepth 2 -name metrics.json -print -quit 2>/dev/null || true)"
    [[ -n "$A_MET" && -f "$A_MET" ]] || fail "A 没有产出 metrics.json（在 $A_RUN/eval_model_state_dict 下未找到）"
    ok "A 完成：$A_MET"

    # ---------------------------------------------------------------- B：官方基座
    local B_MET=""
    local B_LOG="$logd/eval_B.log"
    if [[ "${EVAL_SKIP_BASE:-0}" == "1" ]]; then
        warn "EVAL_SKIP_BASE=1 —— 跳过基座基线（无法做 A/B 对比）"
    elif [[ ! -f checkpoints/action_tokenizer.pt ]]; then
        # 没有官方 codec 就没法让基座按它自己的 token 语义解码，此时跑 B 只会得到乱码，
        # 不如明确跳过 —— 避免"跑出个数字但其实无意义"。
        warn "缺少官方 codec checkpoints/action_tokenizer.pt，跳过 B（否则结果无意义）"
        warn "修复：重跑 MODE=setup（download_weights.sh 已加入该文件）"
    else
        step "3. 评测 B（官方基座 backbone + 官方 codec + 官方 stats），${ne} 条 episode"
        # ★ 关键：用 --task_config configs/task/yam.yaml 而不是基座自带的 .hydra/config.yaml。
        #   原因：基座那分配置是【官方预训练配置】，它的 data 段指向官方预训练数据集，
        #   那些目录在我们容器里不存在 —— 要么直接报错，要么（更糟）静默评了错的数据，
        #   导致 A/B 不可比。用 yam.yaml 保证 A 和 B 走【完全相同的 YAM 数据管线】。
        #   模型侧仍保持"原版"：基座权重 + 官方 codec（+ 官方 stats，见下）。
        #
        # 注：stats 不由 CLI 决定 —— eval_open_loop.py:707 固定读
        #     cfg.run_dir/dataset_stats.json，而 run_dir 由 ckpt 路径上溯得到，
        #     所以 B 会自动用 checkpoints/g05-base/dataset_stats.json（官方那份）。
        #
        # ⚠️ codec 路径必须【三处一起】override：
        #   load_config_from_task_yaml L253-254 已 OmegaConf.resolve + to_container，
        #   于是 tokenizer / model.tokenizer.vq_config / AT_CONFIG 变成三份【独立副本】。
        #   只改 tokenizer.vq_config.ckpt_dir 无效 —— tokenizer 实际读的是
        #   model.tokenizer.vq_config.ckpt_dir（实测 B 因此错误地加载了 YAM codec，
        #   基座不认识 YAM token 语义，对比被夸大）。
        local OFFICIAL_CODEC="$PROJECT_ROOT/checkpoints/action_tokenizer.pt"
        info "命令：eval_open_loop.sh --task_config configs/task/yam.yaml \\"
        info "        --ckpt_path $B_CKPT eval_episodes_num=$ne \\"
        info "        tokenizer.vq_config.ckpt_dir=<官方 codec>\\"
        info "        model.tokenizer.vq_config.ckpt_dir=<官方 codec>\\"
        info "        model.model_arch.AT_CONFIG.ckpt_dir=<官方 codec>"
        set +e
        bash scripts/run/eval_open_loop.sh \
            --task_config configs/task/yam.yaml \
            --ckpt_path "$B_CKPT" \
            "eval_episodes_num=$ne" \
            "tokenizer.vq_config.ckpt_dir=$OFFICIAL_CODEC" \
            "model.tokenizer.vq_config.ckpt_dir=$OFFICIAL_CODEC" \
            "model.model_arch.AT_CONFIG.ckpt_dir=$OFFICIAL_CODEC" \
            2>&1 | tee "$B_LOG"
        local brc=${PIPESTATUS[0]}
        set -e
        B_MET="$(find "$B_RUN/eval_model_state_dict" -maxdepth 2 -name metrics.json -print -quit 2>/dev/null || true)"
        # ---- 自检：B 是否真的用了【官方】codec。用错就会得到"基座+YAM码本"的
        #      错配结果（token 语义对不上），会让 A/B 的差距被严重夸大。
        # ⚠️ 必须先把换行去掉再 grep：平台日志按 80 字符折行，路径会被切成
        #    ".../checkpoint" + "s/action_tokenizer.pt"，逐行 grep 永远匹配不到。
        local b_codec
        b_codec="$(tr -d '\n' < "$B_LOG" 2>/dev/null \
                   | grep -oE 'checkpoints/action_tokenizer[a-z_]*\.pt' | tail -1 || true)"
        if [[ "$b_codec" == *"action_tokenizer_yam.pt"* ]]; then
            warn "★★ B 错误地加载了【YAM codec】（$b_codec）—— 基座不认识 YAM 的 token 语义，"
            warn "   A/B 对比【不可信】。override 未生效，需修 start.sh 后重跑。"
        elif [[ "$b_codec" == *"action_tokenizer.pt"* ]]; then
            ok "B 自检通过：加载的是【官方 codec】（$b_codec）"
        else
            warn "B 自检：未能从日志判定使用了哪个 codec（$b_codec）"
        fi
        if [[ $brc -eq 0 && -n "$B_MET" && -f "$B_MET" ]]; then
            ok "B 完成：$B_MET"
        else
            warn "B 评测失败或没产出 metrics（rc=$brc），日志：$B_LOG"
            tail -40 "$B_LOG" 2>/dev/null || true
            warn "仍会打印 A 的绝对指标；A/B 对比需修好 B 后再跑"
        fi
    fi

    # ---------------------------------------------------------------- 数据规模诊断
    # 实测踩过：数据集元信息显示 991 条 episode，但评测只加载到 ~27k 帧（约 1 个数据集）
    # —— 说明混合数据集的加载并不完整。A/B 因为用同一份数据仍然可比，
    # 但要知道"实际评了多少"，否则数字的含义会被误解。
    if [[ -f "$A_LOG" ]]; then
        local a_split a_batches
        a_split="$(tr -d '\n' < "$A_LOG" | grep -oE 'Split frame range: \[[0-9]+, ?[0-9]+\)' | tail -1 || true)"
        a_batches="$(grep -oE 'total_dataloader_batches=[0-9]+' "$A_LOG" | tail -1 || true)"
        info "数据规模诊断（A）：${a_split:-未解析到}   ${a_batches:-}"
        if [[ -n "$a_batches" ]]; then
            local nb="${a_batches#*=}"
            info "  → dataloader 共 $nb 个 batch（×batch_size_val 16 ≈ $((nb * 16)) 帧）"
            if [[ "$nb" -lt 500 ]]; then
                warn "  ⚠️ 帧数明显偏少（<8000）：混合数据集可能未完整加载，"
                warn "     结果只代表【已加载的那部分】；A/B 仍可比，但别当成全量结论"
            fi
        fi
    fi

    # ---------------------------------------------------------------- 可比性自检
    # 断言 A 和 B 看到的是【同一份数据集】。否则对比无意义 —— 这一步由脚本自己判定，
    # 不依赖人工看日志。
    if [[ -f "$A_LOG" && -f "$B_LOG" && -f "$B_MET" ]]; then
        local a_tot b_tot
        a_tot="$(grep -oE 'out of [0-9]+ total' "$A_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
        b_tot="$(grep -oE 'out of [0-9]+ total' "$B_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
        info "数据集规模自检： A out-of-total=${a_tot:-?}   B out-of-total=${b_tot:-?}"
        if [[ -n "$a_tot" && -n "$b_tot" && "$a_tot" != "$b_tot" ]]; then
            # 降级为 warn：平台日志按 ~80 字符折行，"out of N total" 常被切断，
            # 解析不可靠。真正的可比性由"两边都用 configs/task/yam.yaml"保证。
            warn "A/B 解析到的数据集规模不一致（$a_tot vs $b_tot）—— 但可能是日志折行导致的误判，请人工核对"
        elif [[ -n "$a_tot" && "$a_tot" == "$b_tot" ]]; then
            ok "A/B 数据集一致（共 $a_tot 条 episode），对比有效"
        else
            warn "未能解析出 episode 总数，跳过一致性断言（请人工核对两条日志）"
        fi
    fi

    # ---------------------------------------------------------------- 对比
    step "4. 结果对比"
    if [[ -f "$A_MET" && -f "$B_MET" && "${EVAL_SKIP_BASE:-0}" != "1" ]]; then
        set +e
        python tools/eval_compare.py "$A_MET" "$B_MET" \
            --label-a "A(微调)" --label-b "B(基座)" || true
        set -e
    else
        python - "$A_MET" <<'PYA' || true
import json, sys
path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
print("  === A（微调）的绝对指标 ===")
for k in sorted(d):
    print(f"    {k:<38} {float(d[k]):.6f}")
print()
print("  注意：没有 B 基线时，这些数字无法判断好坏。")
print("  参考点：训练时验证集上 fm_loss≈1.75 / ce_loss 见 logs/train.log")
PYA
    fi

    # ---------------------------------------------------------------- 上传
    if [[ "${EVAL_UPLOAD:-1}" == "1" ]]; then
        step "5. 上传评测结果到 HF"
        local repo="${HF_REPO_ID:-Zenith6666/yam-backbone}"
        # ⚠️ 绝不能无条件写 eval/A/metrics.json：
        #    曾出现"只想补可视化、只跑 3 条"的一轮把 eval/A/metrics.json 覆盖成
        #    3 条的结果，随后拉下来做 A/B 对照时 A 侧口径与 B 侧（10 条）不一致，
        #    差异从 −89.1% 变成 −88.8%，看起来像模型变了，其实是口径变了。
        # 做法：按 episode 数分目录存一份（永不互相覆盖），
        #       再把"完整 A/B 同轮"的结果额外写一份到 eval/{A,B}/metrics.json。
        hf upload "$repo" "$A_MET" "eval/A/${ne}ep/metrics.json" --repo-type model \
            >/dev/null 2>&1 && ok "已上传 eval/A/${ne}ep/metrics.json" || warn "A metrics 上传失败"
        if [[ -f "$B_MET" ]]; then
            hf upload "$repo" "$B_MET" "eval/B/${ne}ep/metrics.json" --repo-type model \
                >/dev/null 2>&1 && ok "已上传 eval/B/${ne}ep/metrics.json" || warn "B metrics 上传失败"
        fi
        if [[ -f "$B_MET" && "${EVAL_SKIP_BASE:-0}" != "1" ]]; then
            # 只有 A/B 同轮、同 episode 数时，才更新"默认入口"这两个文件
            hf upload "$repo" "$A_MET" "eval/A/metrics.json" --repo-type model \
                >/dev/null 2>&1 && ok "已更新 eval/A/metrics.json（A/B 同轮）" || warn "A 默认 metrics 上传失败"
            hf upload "$repo" "$B_MET" "eval/B/metrics.json" --repo-type model \
                >/dev/null 2>&1 && ok "已更新 eval/B/metrics.json（A/B 同轮）" || warn "B 默认 metrics 上传失败"
        else
            info "本轮不是完整 A/B（EVAL_SKIP_BASE=1 或 B 未成功），不动 eval/{A,B}/metrics.json"
        fi
        # 附带一份运行信息，便于日后判断"这个 metrics 是哪次跑的"
        # 用 python 生成 JSON（bash 里手拼引号太容易出错）
        local info_run="$logd/run_info_${ne}ep.json"
        .venv/bin/python - "$ne" "$A_LOG" > "$info_run" 2>/dev/null <<'PYRUNINFO' || true
import datetime, json, os, re, subprocess, sys

ne, alog = sys.argv[1], sys.argv[2]
batches = None
try:
    txt = open(alog, encoding="utf-8", errors="replace").read().replace("\n", "")
    m = re.findall(r"total_dataloader_batches=(\d+)", txt)
    batches = int(m[-1]) if m else None
except Exception:
    pass
try:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                  text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    sha = None
json.dump(
    {
        "episodes": int(ne) if str(ne).isdigit() else ne,
        "git_sha": sha,
        "run_at": datetime.datetime.now(datetime.timezone.utc)
                          .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "skip_base": os.environ.get("EVAL_SKIP_BASE", "0"),
        "total_dataloader_batches": batches,
        "approx_frames": (batches * 16) if batches else None,
    },
    sys.stdout, indent=2, ensure_ascii=False,
)
print()
PYRUNINFO
        if [[ -s "$info_run" ]]; then
            hf upload "$repo" "$info_run" "eval/run_info_${ne}ep.json" --repo-type model \
                >/dev/null 2>&1 && ok "已上传 eval/run_info_${ne}ep.json" || true
        fi
        if [[ "${EVAL_UPLOAD_VIZ:-0}" == "1" ]]; then
            for side in A B; do
                local d="$PROJECT_ROOT/runs/$run_name/eval_model_state_dict"
                [[ "$side" == "B" ]] && d="$B_RUN/eval_model_state_dict"
                # ⚠️ episodes 不在 <root>/episodes —— mixture 模式下是
                #    <root>/<embodiment>/episodes（实测 .../galaxea_r1lite/episodes/）。
                #    用 find 找，别写死层级。
                local ep
                ep="$(find "$d" -maxdepth 3 -type d -name episodes -print -quit 2>/dev/null || true)"
                if [[ -n "$ep" && -d "$ep" ]]; then
                    local nf
                    nf="$(find "$ep" -type f 2>/dev/null | wc -l | tr -d ' ')"
                    info "上传 $side 的可视化（$nf 个文件，$ep）..."
                    hf upload "$repo" "$ep" "eval/$side/episodes" --repo-type model \
                        >/dev/null 2>&1 && ok "已上传 eval/$side/episodes/（$nf 个文件）" \
                        || warn "$side 可视化上传失败（本地仍在 $ep）"
                else
                    warn "$side 没找到 episodes 目录（在 $d 下），可视化未上传"
                fi
            done
        else
            info "EVAL_UPLOAD_VIZ=0 —— 可视化留在容器内（销毁即失）；需要就设 1 再跑"
        fi
    fi

    cat <<EOF

  产物位置：
     A（微调）: $A_MET
                $A_RUN/eval_model_state_dict/episodes/   （GT vs 预测图）
     B（基座）: $B_MET

  ★ 必做：打开 episodes/ 里的对比图，逐关节检查【量级 / 符号 / 是否某关节不动】。
     L1 是平均值，会掩盖"符号反了""量级差 10 倍"这类上真机直接崩的问题。
EOF
}

# ============================================================================
# serve-check：离线验证「异步推理服务端」——不接真机、不接 gello_software
#
# 回答两个问题（这是能否上真机的最后一道门槛）：
#   ① T_infer = ? 一次模型推理多久？
#      判据：必须 <= action_steps / rate = 16/30 = 0.533s（前瞻窗口）
#      超过 → 每换段会"饿"，机器人会顿一下
#   ② 服务端链路对不对？(WebSocket + 协议 + chunk 缓存 + 反归一化)
#
# 四个子步：
#   S1  serving/tests 单测（无 GPU，验证 chunked 状态机 / obs 构建 / 编解码）
#   S2  T_infer 基准（action_steps=1 = RTC 模式，每步都重算 → 每步耗时≈T_infer）
#   S3  Plan B 时序（action_steps=16 + 30Hz 节拍 → 看换段处是否阻塞、实际能跑多少 Hz）
#   S4  WebSocket 端到端（起 serve_policy.py + fake_policy_client.py，验证协议）
#
# 为什么不做"真实观测 vs GT"的对比：那件事第 4 步（MODE=eval）已经做了 ——
#   eval 走的是同一个 processor / PolicyInferencer / postprocess 代码路径，
#   所以归一化、相对转绝对、27→14 维 slot 对齐都已被验证。
#   这里补的是"网络传输 + chunk 缓存"这一层没有破坏任何东西。
#
# 环境变量：
#   SERVE_ACTION_STEPS    每多少步重算一次，默认 16
#   SERVE_RATE            控制频率，默认 30
#   SERVE_PORT            端口，默认 8765
#   SERVE_TINFER_STEPS    S2 步数，默认 6
#   SERVE_PLANB_STEPS     S3 步数，默认 48（= 3 个 chunk）
#   SERVE_WS_STEPS        S4 步数，默认 64
#   SERVE_SKIP_WS         1 = 跳过 S4
# ============================================================================
do_serve_check() {
    require_hf_token
    activate_venv

    local logd="$PROJECT_ROOT/serve_logs"
    mkdir -p "$logd"
    local asteps="${SERVE_ACTION_STEPS:-16}"
    local rate="${SERVE_RATE:-30}"
    local port="${SERVE_PORT:-8765}"

    step "0. 前置检查"
    [[ -d serving ]] || fail "缺少 serving/ 目录"
    [[ -x .venv/bin/python ]] || fail "没有 .venv —— 先跑 MODE=setup"
    .venv/bin/python -c "import websockets" 2>/dev/null || fail "websockets 缺失（应在 pyproject 里）"
    if ! .venv/bin/python -c "import msgpack" 2>/dev/null; then
        info "msgpack 不在 venv 里（是间接依赖）—— 补装"
        uv pip install --python .venv/bin/python msgpack >>"$logd/pip_msgpack.log" 2>&1 \
            || fail "msgpack 安装失败，见 $logd/pip_msgpack.log"
    fi
    ok "依赖就绪（websockets / msgpack）"

    prepare_yam_run_dir
    local CKPT="$PREPARED_CKPT"
    ok "ckpt: $CKPT"

    # ---------------------------------------------------------------- S1 单测
    step "1. serving 单测（无 GPU）"
    set +e
    .venv/bin/python -m pytest serving/tests -q 2>&1 | tee "$logd/serve_pytest.log" | tail -12
    local pytest_rc=${PIPESTATUS[0]}
    set -e
    if [[ $pytest_rc -eq 0 ]]; then ok "单测全部通过"; else warn "单测有失败（不阻塞后续，但需排查）"; fi

    # ---------------------------------------------------------------- S2 T_infer
    step "2. 量 T_infer（action_steps=1 RTC 模式：每步都重算）"
    local S2="$logd/serve_T_infer.log"
    local n2="${SERVE_TINFER_STEPS:-6}"
    info "命令：offline_chunk_harness.py --action_steps 1 --steps $n2 --rate 0"
    set +e
    .venv/bin/python serving/offline_chunk_harness.py \
        --ckpt_path "$CKPT" --device cuda \
        --action_steps 1 --steps "$n2" --rate 0 2>&1 | tee "$S2" | tail -20
    local s2rc=${PIPESTATUS[0]}
    set -e
    [[ $s2rc -eq 0 ]] || { warn "S2 失败（rc=$s2rc），日志 $S2"; tail -30 "$S2" || true; }

    # ---------------------------------------------------------------- S3 Plan B 时序
    step "3. Plan B 时序（action_steps=$asteps @ ${rate}Hz：看换段是否阻塞）"
    local S3="$logd/serve_planb.log"
    local n3="${SERVE_PLANB_STEPS:-48}"
    info "命令：offline_chunk_harness.py --action_steps $asteps --steps $n3 --rate $rate"
    set +e
    .venv/bin/python serving/offline_chunk_harness.py \
        --ckpt_path "$CKPT" --device cuda \
        --action_steps "$asteps" --steps "$n3" --rate "$rate" 2>&1 | tee "$S3" | tail -20
    local s3rc=${PIPESTATUS[0]}
    set -e
    [[ $s3rc -eq 0 ]] || { warn "S3 失败（rc=$s3rc），日志 $S3"; tail -30 "$S3" || true; }

    # ---------------------------------------------------------------- S4 WS 端到端
    if [[ "${SERVE_SKIP_WS:-0}" == "1" ]]; then
        warn "SERVE_SKIP_WS=1 —— 跳过 WebSocket 端到端"
    else
        step "4. WebSocket 端到端（起服务端 + 合成客户端）"
        local SRV_LOG="$logd/serve_server.log"
        local CLI_LOG="$logd/serve_client.log"
        info "启动 serve_policy.py（host=127.0.0.1 port=$port action_steps=$asteps）"
        PYTHONUNBUFFERED=1 .venv/bin/python serving/serve_policy.py \
            --ckpt_path "$CKPT" --host 127.0.0.1 --port "$port" --device cuda \
            --action_steps "$asteps" > "$SRV_LOG" 2>&1 &
        local srv=$!
        local ready=0
        for _ in $(seq 1 120); do   # 最多等 240 秒（模型加载约 1 分钟）
            if tr -d '\n' < "$SRV_LOG" 2>/dev/null | grep -q "Policy server listening"; then
                ready=1; break
            fi
            if ! kill -0 "$srv" 2>/dev/null; then break; fi
            sleep 2
        done
        if [[ $ready -eq 1 ]]; then
            ok "服务端已就绪（ws://127.0.0.1:$port）"
            set +e
            .venv/bin/python serving/fake_policy_client.py \
                --host 127.0.0.1 --port "$port" --rate "$rate" \
                --steps "${SERVE_WS_STEPS:-64}" --verbose 2>&1 | tee "$CLI_LOG" | tail -25
            local crc=${PIPESTATUS[0]}
            set -e
            if [[ $crc -eq 0 ]]; then ok "S4 通过：WebSocket 链路正常"; else warn "S4 客户端失败（rc=$crc）"; fi
        else
            warn "服务端未在 240s 内就绪 —— 见 $SRV_LOG"
            tail -40 "$SRV_LOG" 2>/dev/null || true
        fi
        kill "$srv" 2>/dev/null || true
        wait "$srv" 2>/dev/null || true
    fi

    # ---------------------------------------------------------------- 结论
    step "5. 结论"
    local t_infer_ms t_avg_ms rate_hz slow_steps
    t_infer_ms="$(tr -d '\n' < "$S2" 2>/dev/null | grep -oE 'first-step latency \(inference baseline\): *[0-9.]+' | tail -1 | grep -oE '[0-9.]+$' || true)"
    t_avg_ms="$(tr -d '\n' < "$S2" 2>/dev/null | grep -oE 'latency: avg=[0-9.]+ms' | tail -1 | grep -oE '[0-9.]+' || true)"
    rate_hz="$(tr -d '\n' < "$S3" 2>/dev/null | grep -oE 'effective_rate=[0-9.]+Hz' | tail -1 | grep -oE '[0-9.]+' || true)"
    slow_steps="$(tr -d '\n' < "$S3" 2>/dev/null | grep -oE 'slow steps \(>= 20\.0ms, blocked recompute/swap\): \[[^]]*\]' | tail -1 | sed 's/.*: //' || true)"

    local window_ms
    window_ms="$(awk -v a="$asteps" -v r="$rate" 'BEGIN{printf "%.0f", a*1000.0/r}')"

    echo
    echo "  ┌──────────────────────────────────────────────────────────┐"
    printf "  │ T_infer (单次推理)        : %-28s│\n" "${t_infer_ms:-未解析}${t_infer_ms:+ ms}"
    printf "  │ T_infer 均值(RTC 模式)    : %-28s│\n" "${t_avg_ms:-未解析}${t_avg_ms:+ ms}"
    printf "  │ 前瞻窗口 (action_steps/Hz): %-28s│\n" "${window_ms} ms  (${asteps}/${rate})"
    printf "  │ Plan B 实测有效频率        : %-28s│\n" "${rate_hz:-未解析}${rate_hz:+ Hz}"
    printf "  │ 换段阻塞的 step            : %-28s│\n" "${slow_steps:-未解析}"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo

    if [[ -n "$t_infer_ms" ]] && awk -v t="$t_infer_ms" -v w="$window_ms" 'BEGIN{exit !(t<=w)}'; then
        ok "判定：✅ T_infer (${t_infer_ms}ms) <= 窗口 (${window_ms}ms) —— 异步前瞻能完全藏住推理延迟，实时可行"
    elif [[ -n "$t_infer_ms" ]]; then
        local over
        over="$(awk -v t="$t_infer_ms" -v w="$window_ms" 'BEGIN{printf "%.2f", t/w}')"
        warn "判定：⚠️ T_infer (${t_infer_ms}ms) 超过窗口 (${window_ms}ms) 约 ${over}×"
        warn "  → 每 ${asteps} 步会「饿」一下。可选优化（按性价比排序）："
        warn "    ① 确认 bf16 已开：serve_policy.py:103 读 cfg.model.model_weights_to_bf16（代码默认 True）"
        warn "       显式开：加 overrides  model.model_weights_to_bf16=true"
        warn "    ② torch.compile：加 overrides  model.use_torch_compile=true  model.torch_compile_mode=reduce-overhead"
        warn "       （首次推理会慢几十秒，属正常）"
        warn "    ③ 降控制频率 SERVE_RATE（如 15Hz → 窗口翻倍）；但 30Hz 是数据原始频率，降频有分布漂移风险"
        warn "    ④ 调 action_steps 无效来救：lookahead 要求 chunk_len >= 2*action_steps，"
        warn "       而我们 chunk_len=32、action_steps=16 已恰好是上限"
    else
        warn "判定：无法判定 —— S2 没跑出 T_infer，请检查 $S2"
    fi

    cat <<EOF

  日志位置：
    S1 单测     : $logd/serve_pytest.log
    S2 T_infer  : $S2
    S3 Plan B   : $S3
    S4 服务端   : $logd/serve_server.log
    S4 客户端   : $logd/serve_client.log

  ★ 关键判据：T_infer <= ${window_ms}ms（= ${asteps} 步 ÷ ${rate}Hz）
     满足 → 异步前瞻能完全吸收推理延迟，可以进入真机阶段（还缺 gello_software 的硬件层）
     不满足 → 先做推理优化，再谈真机
EOF
}

# ---------------------------------------------------------------- 分发
# ⚠️ 必须把 start.sh 收到的额外参数透传给 do_train，否则
#    `bash start.sh model.max_epochs=1` 会静默失效（用户以为 1 epoch，实际 10 epoch）。
case "$MODE" in
    smoke)     do_smoke ;;
    smoke-ddp) do_smoke_ddp ;;
    setup)  do_setup ;;
    verify) do_verify ;;
    eval)   do_eval ;;
    serve-check) do_serve_check ;;
    train)  do_train "$@" ;;
    all)
        do_setup
        do_verify          # L3 不通过会在这里非零退出，不会带着坏权重去训练
        do_train "$@"
        ;;
    *)      fail "未知 MODE=$MODE（可选 smoke | smoke-ddp | setup | verify | train | all）" ;;
esac

step "MODE=$MODE 结束"
