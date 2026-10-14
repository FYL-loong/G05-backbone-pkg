#!/usr/bin/env bash
# ============================================================================
# start.sh — G0.5 backbone 微调一键启动（百度百舸 AIHC，无人值守）
#
# 一条命令跑完：
#   装环境 -> 拉权重/数据 -> 自检 -> L1/L2/L3(门禁)/L4 -> 训练 -> 等待结束
#   -> 自动收尾上传（权重 + 日志 + 环境快照）
# 任何阶段失败都会自动把诊断信息传到 HF，便于事后排查。
#
# 平台「执行命令」框：
#   set -e
#   export HF_TOKEN="hf_xxxx"
#   git clone https://github.com/FYL-loong/G05-backbone-pkg.git G05-backbone-pkg
#   cd G05-backbone-pkg
#   bash start.sh
#
# 分阶段（调试用）：
#   MODE=setup   bash start.sh    装环境 + 拉权重/数据 + 自检
#   MODE=verify  bash start.sh    只跑 L1/L2/L3/L4
#   MODE=train   bash start.sh    只启动训练（环境/数据须已就绪）
#   MODE=all     bash start.sh    默认，全自动
#
# 环境变量：
#   HF_TOKEN          必需。HF 读写 token（写在平台执行命令里，勿提交到 git）
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
# ============================================================================
set -euo pipefail

MODE="${MODE:-all}"
G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-/workspace/output}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ONLY_TASK="${ONLY_TASK:-}"
SKIP_L4="${SKIP_L4:-0}"
STRICT_L3="${STRICT_L3:-1}"
AUTO_UPLOAD="${AUTO_UPLOAD:-1}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-600}"
export G05_OUTPUT_DIR HF_ENDPOINT

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- 日志留存
# 整个 start.sh 的 stdout 同时写一份到 START_LOG，
# 失败时 upload_diagnostics.sh 能把它一起传走。
START_LOG="${START_LOG:-/workspace/start.log}"
mkdir -p "$(dirname "$START_LOG")" 2>/dev/null || START_LOG="$PROJECT_ROOT/start.log"
exec > >(tee -a "$START_LOG") 2>&1
export START_LOG

# ---------------------------------------------------------------- 工具函数
C_R='\033[31m'; C_G='\033[32m'; C_Y='\033[33m'; C_C='\033[36m'; C_0='\033[0m'
info() { printf "${C_C}[INFO]${C_0} %s\n" "$*"; }
ok()   { printf "${C_G}[ OK ]${C_0} %s\n" "$*"; }
warn() { printf "${C_Y}[WARN]${C_0} %s\n" "$*"; }
fail() { printf "${C_R}[FAIL]${C_0} %s\n" "$*" >&2; exit 1; }
step() { printf "\n${C_C}========================================================${C_0}\n${C_C} %s${C_0}\n${C_C}========================================================${C_0}\n" "$*"; }

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

# ---------------------------------------------------------------- 失败处理
# 任何命令失败（set -e）都触发 ERR，自动把诊断传到 HF —— 容器销毁后这是唯一的现场。
CURRENT_STAGE="preflight"
EXP_DIR_FOUND=""
on_error() {
    local rc=$? line=${1:-?}
    printf "\n${C_R}[FAIL]${C_0} start.sh 在第 %s 行失败（退出码 %s），阶段: %s\n" "$line" "$rc" "$CURRENT_STAGE"
    printf "${C_R}[FAIL]${C_0} 正在上传诊断信息到 HF ...\n"
    STAGE="$CURRENT_STAGE" \
    START_LOG="$START_LOG" \
    EXP_DIR="${EXP_DIR_FOUND:-}" \
    EXTRA_FILES="/workspace/train.stdout.log /workspace/upload_checkpoints.log" \
    bash "$PROJECT_ROOT/upload_diagnostics.sh" || true
    printf "${C_Y}[WARN]${C_0} 诊断上传结束。现场仍在本容器内：%s\n" "$START_LOG"
    exit "$rc"
}
trap 'on_error $LINENO' ERR

# 标记阶段并打印标题（阶段名会出现在 diagnostics/<名字>_<时间戳>/ 里）
run_stage() {
    CURRENT_STAGE="$1"
    step "$1. $2"
}

# ---------------------------------------------------------------- 前置检查
step "0. 前置检查"

[[ -n "${HF_TOKEN:-}" ]] || fail "未设置 HF_TOKEN。请把它写在平台任务的执行命令里：export HF_TOKEN=\"hf_...\""
export HF_TOKEN

info "MODE           = $MODE"
info "PROJECT_ROOT   = $PROJECT_ROOT"
info "G05_OUTPUT_DIR = $G05_OUTPUT_DIR"
info "HF_ENDPOINT    = $HF_ENDPOINT"
info "ONLY_TASK      = ${ONLY_TASK:-<全部 10 个任务>}"

if nvidia-smi >/dev/null 2>&1; then
    info "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/       /'
    GPU_TOTAL_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
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
AVAIL_GB="$(df -BG "$G05_OUTPUT_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')"
if [[ "${AVAIL_GB:-0}" -lt 60 ]]; then
    warn "可用磁盘 ${AVAIL_GB}G 偏小（11GB 权重 + 每个 checkpoint 11.44GB）"
fi

# ---- 网络探测：setup 阶段要访问这些域名，提前暴露问题 ----
# 实测（2026-03）：github.com 在国内常不可达，而 uv 默认从 GitHub Releases 下载
# Python 3.10（NGC 镜像自带 3.12，不满足 requires-python >=3.10.16,<3.11）。
probe() {
    local host="$1" what="$2"
    if timeout 12 bash -c "exec 3<>/dev/tcp/$host/443" 2>/dev/null; then
        ok "  可达   $host   ($what)"
        return 0
    fi
    warn "  不可达 $host   ($what)"
    return 1
}
info "网络探测（setup 阶段要访问的域名）:"
probe mirrors.aliyun.com   "uv.lock 里 1049 个包" || true
probe download.pytorch.org "torch cu128 轮子" || true
probe pypi.nvidia.com      "nvidia 依赖" || true
probe astral.sh            "安装 uv" || true
if ! probe gh-proxy.com "Python 3.10 下载镜像（默认启用）"; then
    warn "  gh-proxy 不可达 → 若本地没有 Python 3.10，uv sync 会失败。"
    warn "  备选：UV_PYTHON_INSTALL_MIRROR=<其它镜像> 或 UV_PYTHON_INSTALL_MIRROR= （走官方源）"
fi

# ============================================================================
# setup：装环境 + 拉权重 + 拉数据 + 自检
# ============================================================================
do_setup() {
    run_stage setup-install "安装环境（uv sync，首次约 10-30 分钟）"
    # NGC 镜像自带 Python 3.12，但项目要求 3.10.16（.python-version / requires-python），
    # 所以 uv 必须联网下载一个独立 Python 3.10 —— 源默认是 GitHub Releases，
    # 国内不可靠。这里默认启用国内镜像；若它不通，用
    #   UV_PYTHON_INSTALL_MIRROR= bash start.sh    回退官方源
    if [[ -z "${UV_PYTHON_INSTALL_MIRROR+x}" ]]; then
        export UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
        info "启用 Python 下载镜像: $UV_PYTHON_INSTALL_MIRROR"
    else
        info "UV_PYTHON_INSTALL_MIRROR = ${UV_PYTHON_INSTALL_MIRROR:-<空，走官方 GitHub>}"
    fi
    # torch cu128 wheel 数百 MB，默认超时可能不够
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

    if [[ -x .venv/bin/python ]]; then
        ok ".venv 已存在，跳过 setup_env.sh（要重装先 rm -rf .venv）"
    else
        bash setup_env.sh
    fi
    activate_venv
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

    run_stage download-weights "下载权重（约 11.5GB，实测约 3 分钟；失败自动重试 3 次）"
    retry 3 60 bash download_weights.sh

    run_stage download-dataset "下载数据集（失败自动重试 3 次）"
    if [[ -n "$ONLY_TASK" ]]; then
        retry 3 60 env ONLY_TASK="$ONLY_TASK" bash download_dataset.sh
    else
        retry 3 60 bash download_dataset.sh
    fi

    run_stage data-check "数据完整性自检"
    local n_ds
    n_ds="$(find data/yam -maxdepth 1 -type d -name 'TASK-YAM-*_lerobot' 2>/dev/null | wc -l | tr -d ' ')"
    info "发现 $n_ds 个 _lerobot 数据集"
    [[ "$n_ds" -gt 0 ]] || fail "data/yam 下没有 TASK-YAM-*_lerobot，数据集没拉下来"
    # 用 heredoc 而不是 python -c，避免 $ 与引号在 bash 里层层转义
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

    run_stage stats-check "stats 自检（维度 + 退化检测）"
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

    run_stage L1-config "L1 配置组装（不加载权重、不占显存）"
    bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
    ok "L1 通过"

    run_stage L2-pipeline "L2 数据管线 + forward/backward（权重随机初始化，不验证权重加载）"
    bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
        model.pretrained_ckpt=null model.max_steps=10 logger.mode=offline
    ok "L2 通过（数据管线没崩）"

    run_stage L3-weight-load "L3 ★ 权重加载验证（11GB 加载；脚本自动判定并返回门禁退出码）"
    STRICT="$STRICT_L3" bash verify_weight_load.sh
    ok "L3 通过（权重完整加载）"

    if [[ "$SKIP_L4" == "1" ]]; then
        warn "SKIP_L4=1，跳过 L4"
    else
        run_stage L4-overfit "L4 单批过拟合（loss 应明显下降）"
        bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500 \
            logger.mode=offline
        ok "L4 跑完（请确认上面 loss 明显下降）"
    fi
}

# ============================================================================
# train：正式训练（后台）+ 守护上传 + 等待结束 + 收尾上传
# ============================================================================
do_train() {
    activate_venv

    local n="${GPUS:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-${MLP_WORKER_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}}}"
    [[ "${n:-0}" -ge 1 ]] || n=1
    info "GPU 数: $n（来源: GPUS / NPROC_PER_NODE / MLP_WORKER_GPU / MLP_WORKER_NUM / nvidia-smi）"
    info "注意：batch_size 是每卡的，全局 batch = batch_size × $n"

    run_stage train-launch "启动正式训练（后台，日志 -> /workspace/train.stdout.log）"
    nohup bash scripts/run/finetune.sh "$n" yam logger.mode=offline \
        > /workspace/train.stdout.log 2>&1 &
    local pid=$!
    echo "$pid" > /workspace/train.pid
    ok "训练 PID=$pid（已写入 /workspace/train.pid）"

    # 等 exp 目录出现（最多 5 分钟）
    local exp="" i=0
    while [[ $i -lt 30 ]]; do
        sleep 10; i=$((i+1))
        exp="$(ls -1dt "$G05_OUTPUT_DIR"/yam/*/ 2>/dev/null | head -1 || true)"
        [[ -n "$exp" ]] && break
    done
    if [[ -n "$exp" ]]; then
        ok "本次 exp 目录: $exp"
        EXP_DIR_FOUND="$exp"
        echo "$exp" > /workspace/exp_dir.txt
    else
        warn "5 分钟内没识别出 exp 目录。稍后手动：ls $G05_OUTPUT_DIR/yam/"
    fi

    # 守护上传：训练中陆续把 checkpoint 推到 HF
    if [[ "$AUTO_UPLOAD" == "1" && -n "$exp" ]]; then
        run_stage upload-daemon "启动 checkpoint 守护上传（每 ${UPLOAD_INTERVAL}s 轮询）"
        nohup env MODE=watch EXP_DIR="$exp" WATCH_INTERVAL="$UPLOAD_INTERVAL" \
            HF_TOKEN="$HF_TOKEN" HF_ENDPOINT="$HF_ENDPOINT" \
            bash upload_checkpoints.sh > /workspace/upload_checkpoints.log 2>&1 &
        echo $! > /workspace/upload.pid
        ok "守护上传 PID=$(cat /workspace/upload.pid)  日志 /workspace/upload_checkpoints.log"
    elif [[ "$AUTO_UPLOAD" != "1" ]]; then
        warn "AUTO_UPLOAD=0，未启动守护上传"
    fi

    info "启动后 90 秒训练日志尾部："
    sleep 90
    tail -c 4000 /workspace/train.stdout.log || true

    cat <<EOF

============================================================
 训练在后台跑。start.sh 会【阻塞等待训练结束】，然后自动收尾上传。
============================================================
 训练日志 : tail -f /workspace/train.stdout.log
 上传日志 : tail -f /workspace/upload_checkpoints.log
 exp 目录 : ${exp:-<未识别>}
============================================================
EOF

    # ------------------------------------------------------------ 等待训练结束
    CURRENT_STAGE="train-running"
    step "train-running. 等待训练结束（PID=$pid）"
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        waited=$((waited+60))
        if (( waited % 600 == 0 )); then
            local last_step
            last_step="$(grep -aoE '"step":[[:space:]]*[0-9]+' /workspace/train.stdout.log 2>/dev/null | tail -1 || true)"
            info "已等待 $((waited/60)) 分钟，日志最后 step: ${last_step:-未知}"
        fi
    done
    ok "训练进程已退出（累计等待 $((waited/60)) 分钟）"

    # 等最终 checkpoint 落盘（finetune.py:1329 保存后进程才退出，IO 可能仍在 flush）
    info "等 120s 让最终 checkpoint 落盘..."
    sleep 120

    if [[ -z "$exp" ]]; then
        warn "没有 exp 目录，跳过收尾上传。请手动："
        echo "     EXP_DIR=<exp目录> bash upload_checkpoints.sh"
        echo "     EXP_DIR=<exp目录> bash upload_output.sh"
        return 0
    fi

    # 停守护进程，避免与收尾上传抢同一个文件
    if [[ -f /workspace/upload.pid ]]; then
        kill "$(cat /workspace/upload.pid)" 2>/dev/null || true
        sleep 5
    fi

    run_stage final-checkpoints "收尾上传所有 checkpoint"
    MODE=all EXP_DIR="$exp" bash upload_checkpoints.sh || warn "checkpoint 收尾上传失败，可重跑"

    run_stage final-artifacts "收尾上传最终产物（权重 / 日志 / 配置 / 环境快照）"
    EXP_DIR="$exp" bash upload_output.sh || warn "产物收尾上传失败，可重跑"

    step "本容器内全部完成"
    cat <<EOF

============================================================
 结果查看位置
============================================================
 1) HF 权重仓库（推荐；容器销毁后唯一可靠的地方）
      https://huggingface.co/Zenith6666/yam-backbone
      - model_state_dict.pt    最终权重
      - checkpoints/step_*.pt  各 step checkpoint
      - train.log / finetune.jsonl / logs/   日志与指标
      - hydra_config.yaml / env_report.txt   配置与环境快照
 2) 容器内日志（销毁即失）
      /workspace/train.stdout.log
      $exp/train.log
      $exp/finetune.jsonl
 3) 若失败：diagnostics/<阶段>_<时间戳>/ 下有启动日志与环境快照

 趁容器还在，可跑 open-loop 评测：
   bash scripts/run/eval_open_loop.sh \\
     --ckpt_path $exp/last.pt --task_config configs/task/yam.yaml
EOF
}

# ---------------------------------------------------------------- 分发
case "$MODE" in
    setup)  do_setup ;;
    verify) do_verify ;;
    train)  do_train ;;
    all)
        do_setup
        do_verify          # L3 不通过会在这里非零退出，不会带着坏权重去训练
        do_train
        ;;
    *)      fail "未知 MODE=$MODE（可选 setup | verify | train | all）" ;;
esac

step "MODE=$MODE 结束"
