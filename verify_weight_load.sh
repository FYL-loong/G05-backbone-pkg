#!/usr/bin/env bash
# ============================================================================
# verify_weight_load.sh — 验证 11GB backbone 权重能被正确加载
#
# 为什么需要这个脚本：
#   官方 README 里的 smoke test 是
#       bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 \
#            model.pretrained_ckpt=null model.max_steps=10
#   `pretrained_ckpt=null` 会让 scripts/finetune.py 走
#       else: model = instantiate(cfg.model.model_arch)      # 随机初始化
#   这条分支（finetune.py:572-573），【完全不加载】那 11GB 权重。
#   它只能验证「数据管线 + forward/backward 不崩」，验证不了权重加载。
#
#   而权重加载恰恰是最容易出问题的一步：
#   load_state_dict_safely 有「静默 partial-load」行为 —— 维度不匹配时
#   只打 warning 就继续，不会报错。所以必须显式验证它加载成功。
#
# 用法：
#   bash verify_weight_load.sh            # 默认 1 卡、2 步
#   GPUS=8 STEPS=2 bash verify_weight_load.sh
#   TASK=yam bash verify_weight_load.sh
#
# 判据（两条都要满足）：
#   1) 日志里出现 "Loaded N/N"  且  "Missing (rand init): 0"
#   2) 2 步跑完，loss 是有限值（非 nan/inf）
# ============================================================================
set -euo pipefail

GPUS="${GPUS:-1}"
STEPS="${STEPS:-2}"
TASK="${TASK:-yam}"
CKPT="${CKPT:-checkpoints/g05-base/checkpoints/model_state_dict.pt}"
LOG="${LOG:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---- 前置检查 ------------------------------------------------------------
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }

[[ -f "$CKPT" ]] || fail "找不到权重 $CKPT —— 先跑 bash download_weights.sh"

CKPT_BYTES="$(wc -c < "$CKPT" | tr -d ' ')"
info "backbone 权重: $CKPT  ($CKPT_BYTES 字节)"
# 官方 OpenGalaxea/G05 的 g05-base/checkpoints/model_state_dict.pt 是 11440519964 字节。
# 少于 8GB 说明没下完。这里只做粗判，精确比对用 sha256（可选）。
if [[ "$CKPT_BYTES" -lt 8000000000 ]]; then
    fail "权重只有 $CKPT_BYTES 字节（应约 11440519964）—— 下载不完整，重跑 download_weights.sh"
fi

CODEC="${CODEC:-checkpoints/action_tokenizer_yam.pt}"
if [[ -f "$CODEC" ]]; then
    info "codec: $CODEC  ($(wc -c < "$CODEC" | tr -d ' ') 字节)"
else
    info "⚠ 未找到 $CODEC —— 确认 configs/task/yam.yaml 的 ckpt_dir 指向何处"
fi

STATS="data/stats/yam_stats.json"
if [[ -f "$STATS" ]]; then
    info "stats: $STATS  ($(wc -c < "$STATS" | tr -d ' ') 字节)"
else
    info "⚠ 未找到 $STATS —— 训练启动时会遍历全量数据现算（会慢很多）"
fi

[[ -n "${G05_OUTPUT_DIR:-}" ]] || fail "必须先 export G05_OUTPUT_DIR=<输出目录>（configs/train.yaml 用它，无默认值）"
info "输出目录: $G05_OUTPUT_DIR"

# ---- 记录输出目录，便于 run 结束后新目录比对 ------------------------------
mkdir -p "$G05_OUTPUT_DIR"
BEFORE="$(mktemp)"; AFTER="$(mktemp)"
trap 'rm -f "$BEFORE" "$AFTER"' EXIT
find "$G05_OUTPUT_DIR" -maxdepth 3 -type d 2>/dev/null | sort > "$BEFORE"

# ---- 跑 2 步（显式加载权重）----------------------------------------------
info "开始：加载权重 + 跑 $STEPS 步"
echo "--------------------------------------------------------------"
set +e
if [[ -n "$LOG" ]]; then
    bash scripts/run/finetune.sh "$GPUS" "$TASK" \
        --test --max_datasets 1 \
        "model.pretrained_ckpt=$CKPT" \
        "model.max_steps=$STEPS" 2>&1 | tee "$LOG"
    rc="${PIPESTATUS[0]}"
else
    bash scripts/run/finetune.sh "$GPUS" "$TASK" \
        --test --max_datasets 1 \
        "model.pretrained_ckpt=$CKPT" \
        "model.max_steps=$STEPS"
    rc=$?
fi
set -e
echo "--------------------------------------------------------------"

# ---- 定位本次 run 目录 ----------------------------------------------------
find "$G05_OUTPUT_DIR" -maxdepth 3 -type d 2>/dev/null | sort > "$AFTER"
NEWDIRS="$(comm -13 "$BEFORE" "$AFTER" || true)"
RUN_DIR="$(printf '%s\n' "$NEWDIRS" | grep -v '^$' | tail -1 || true)"

echo
printf '=%.0s' {1..62}; echo
if [[ $rc -ne 0 ]]; then
    fail "训练进程退出码 $rc —— 权重加载或 forward 失败，见上方日志"
fi
info "进程正常退出（rc=0）"

# ---- 检查判据 ------------------------------------------------------------
if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    info "本次 run 目录: $RUN_DIR"
    LOGFILE="$RUN_DIR/train.log"
    if [[ -f "$LOGFILE" ]]; then
        echo
        echo "--- 权重加载关键行 ---"
        grep -nE "Loaded |Missing \(|rand init|Unexpected|size mismatch|shape mismatch" "$LOGFILE" | head -20 || echo "（没找到 'Loaded' 相关行，请人工检查 $LOGFILE）"
        echo
        echo "--- 前几步 loss ---"
        grep -nE "loss|\"step\"" "$LOGFILE" | head -5 || true
    else
        echo "未找到 $LOGFILE"
    fi
    echo
    echo "--- run 目录内容 ---"
    ls -la "$RUN_DIR" | head -20
else
    info "未识别出新 run 目录（可能复用已有目录）。请人工检查 $G05_OUTPUT_DIR 下的 train.log。"
fi

cat <<'EOF'

============================================================
 人工确认清单（两条都满足才算通过）
============================================================
 1) 日志里出现      Loaded <N>/<N>
    且              Missing (rand init): 0
    —— 若不是 0，说明有参数没加载上（随机初始化），必须查清是哪些
 2) 2 步跑完，loss 是有限值（不是 nan / inf）

 不通过时先查：
   - 权重字节数：11440519964（ls -l checkpoints/g05-base/checkpoints/model_state_dict.pt）
   - configs/model/g05.yaml 的 pretrained_ckpt 路径
   - action_dim / proprio_dim 是否为 27（configs/task/yam.yaml）
============================================================
EOF
