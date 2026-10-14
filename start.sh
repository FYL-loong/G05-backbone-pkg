#!/usr/bin/env bash
# ============================================================================
# start.sh — 百度百舸（AIHC）backbone 微调一键启动脚本
#
# 用法（粘进平台任务的「执行命令」框）：
#
#   git clone <你的仓库地址> G05-backbone-pkg && cd G05-backbone-pkg && \
#     MODE=setup bash start.sh
#
# 分阶段跑（推荐，每步看清楚再往下）：
#   MODE=setup    装环境 + 拉权重/数据 + 校验（不训练）
#   MODE=verify   跑 L1/L2/L3 三级验证（环境/数据已就绪）
#   MODE=train    正式训练（后台）
#   MODE=all      setup + verify（一条走完，不训练）
#
# 环境变量：
#   HF_TOKEN          必需。HF 读写 token（平台密钥注入，勿硬编码）
#   G05_OUTPUT_DIR    可选，默认 /workspace/output（容器内路径，靠 HF 落地）
#   GPUS              可选，MODE=train 的卡数；默认取 NPROC_PER_NODE /
#                     MLP_WORKER_GPU / nvidia-smi 探测
#   HF_ENDPOINT       可选，默认 https://huggingface.co
#                     注意：实测百度百舸【连不上 hf-mirror】，不要设成它
#   ONLY_TASK         可选，设了就只拉单任务，如 TASK-YAM-0006
# ============================================================================
set -euo pipefail

MODE="${MODE:-setup}"
G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-/workspace/output}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ONLY_TASK="${ONLY_TASK:-}"
export G05_OUTPUT_DIR HF_ENDPOINT

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------- 工具函数
C_R='\033[31m'; C_G='\033[32m'; C_Y='\033[33m'; C_C='\033[36m'; C_0='\033[0m'
info() { printf "${C_C}[INFO]${C_0} %s\n" "$*"; }
ok()   { printf "${C_G}[ OK ]${C_0} %s\n" "$*"; }
warn() { printf "${C_Y}[WARN]${C_0} %s\n" "$*"; }
fail() { printf "${C_R}[FAIL]${C_0} %s\n" "$*" >&2; exit 1; }
step() { printf "\n${C_C}========================================================${C_0}\n${C_C} %s${C_0}\n${C_C}========================================================${C_0}\n" "$*"; }

# 激活 venv。venv 的 activate 脚本在 set -u 下可能引用未定义变量而退出，
# 所以这里临时关掉 -u。
activate_venv() {
    [[ -f .venv/bin/activate ]] || fail "找不到 .venv/bin/activate，请先跑 MODE=setup"
    set +u
    # shellcheck disable=SC1091
    source .venv/bin/activate
    set -u
    info "python = $(command -v python)  ($(python -V 2>&1))"
}

# ---------------------------------------------------------------- 前置检查
step "0. 前置检查"

[[ -n "${HF_TOKEN:-}" ]] || fail "未设置 HF_TOKEN。请在平台任务里加环境变量 HF_TOKEN（读写权限 token）"
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
        warn "单卡显存 ${GPU_TOTAL_MB} MiB < 70 GB，全参微调可能 OOM（官方建议 >70GB）"
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
    warn "可用磁盘 ${AVAIL_GB}G 偏小（11GB 权重 + 每个 checkpoint 11.44GB + 数据集）"
fi

# ============================================================================
# setup：装环境 + 拉权重 + 拉数据 + 校验
# ============================================================================
do_setup() {
    step "1. 安装环境（uv sync，首次约 10-30 分钟）"
    if [[ -x .venv/bin/python ]]; then
        ok ".venv 已存在，跳过 setup_env.sh（要重装先 rm -rf .venv）"
    else
        bash setup_env.sh
    fi
    activate_venv
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

    step "2. 下载权重（约 11.5GB，实测约 3 分钟）"
    bash download_weights.sh

    step "3. 下载数据集"
    if [[ -n "$ONLY_TASK" ]]; then
        ONLY_TASK="$ONLY_TASK" bash download_dataset.sh
    else
        bash download_dataset.sh
    fi

    step "4. 数据完整性自检"
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
# verify：L1/L2/L3 三级验证
# ============================================================================
do_verify() {
    activate_venv

    step "L1 配置组装（不加载权重、不占显存）"
    bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
    ok "L1 通过"

    step "L2 数据管线 + forward/backward（显式置空权重，不验证权重加载）"
    bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
        model.pretrained_ckpt=null model.max_steps=10 logger.mode=offline
    ok "L2 通过（数据管线没崩）"

    step "L3 ★ 权重加载验证（这一级才有 11GB 加载）"
    bash verify_weight_load.sh
    warn "L3 请人工确认上面日志里有 Loaded N/N 且 Missing (rand init): 0"

    step "verify 阶段完成"
    cat <<'EOF'
下一步（人工确认 L3 后再执行）：

  # L4 单批过拟合：loss 应明显下降
  bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500 logger.mode=offline

  # L5 少量数据真实训练：确认存盘与上传通路
  bash scripts/run/finetune.sh 1 yam --max_datasets 1 model.max_steps=20 logger.mode=offline
EOF
}

# ============================================================================
# train：正式训练（后台）
# ============================================================================
do_train() {
    activate_venv

    local n="${GPUS:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}}"
    [[ "${n:-0}" -ge 1 ]] || n=1
    info "GPU 数: $n"

    step "正式训练（后台，日志 -> /workspace/train.stdout.log）"
    nohup bash scripts/run/finetune.sh "$n" yam logger.mode=offline \
        > /workspace/train.stdout.log 2>&1 &
    local pid=$!
    echo "$pid" > /workspace/train.pid
    ok "训练 PID=$pid（已写入 /workspace/train.pid）"

    sleep 60
    step "启动后 60 秒日志尾部"
    tail -c 3000 /workspace/train.stdout.log || true

    local exp
    exp="$(ls -1dt "$G05_OUTPUT_DIR"/yam/*/ 2>/dev/null | head -1 || true)"
    if [[ -n "$exp" ]]; then
        ok "本次 exp 目录: $exp"
        echo "$exp" > /workspace/exp_dir.txt
        cat <<EOF

训练期间的必做操作（容器临时，产物必须落 HF）：
  EXP_DIR=$exp bash upload_checkpoints.sh
  先看会传什么：DRY_RUN=1 EXP_DIR=$exp bash upload_checkpoints.sh
  只传最后一个：MODE=last EXP_DIR=$exp bash upload_checkpoints.sh
EOF
    else
        warn "还没识别出 exp 目录，稍后手动看：ls $G05_OUTPUT_DIR/yam/"
    fi
}

# ---------------------------------------------------------------- 分发
case "$MODE" in
    setup)  do_setup ;;
    verify) do_verify ;;
    train)  do_train ;;
    all)    do_setup; do_verify ;;
    *)      fail "未知 MODE=$MODE（可选 setup | verify | train | all）" ;;
esac

step "MODE=$MODE 结束"
