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
# STRICT=1（默认）：加载不完整就非零退出，供无人值守流程判断
# STRICT=0：只报告，退出码始终为 0
STRICT="${STRICT:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---- 前置检查 ------------------------------------------------------------
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }

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
# 输出同时进 tee 和一个固定路径的日志，便于之后自动解析
RUNLOG="${LOG:-$(pwd)/verify_weight_load.log}"
info "开始：加载权重 + 跑 $STEPS 步（日志 -> $RUNLOG）"
echo "--------------------------------------------------------------"
set +e
bash scripts/run/finetune.sh "$GPUS" "$TASK" \
    --test --max_datasets 1 \
    "model.pretrained_ckpt=$CKPT" \
    "model.max_steps=$STEPS" 2>&1 | tee "$RUNLOG"
rc="${PIPESTATUS[0]}"
set -e
echo "--------------------------------------------------------------"

# ---- 定位本次 run 目录 ----------------------------------------------------
find "$G05_OUTPUT_DIR" -maxdepth 3 -type d 2>/dev/null | sort > "$AFTER"
NEWDIRS="$(comm -13 "$BEFORE" "$AFTER" || true)"
RUN_DIR="$(printf '%s\n' "$NEWDIRS" | grep -v '^$' | tail -1 || true)"
if [[ -n "$RUN_DIR" && -d "$RUN_DIR" && -f "$RUN_DIR/train.log" ]]; then
    # train.log 更完整（tee 的 stdout 可能被 hydra 重定向过）
    cp -f "$RUN_DIR/train.log" "$RUNLOG" 2>/dev/null || true
fi

echo
printf '=%.0s' {1..62}; echo
if [[ $rc -ne 0 ]]; then
    fail "训练进程退出码 $rc —— 权重加载或 forward 失败，见日志 $RUNLOG"
fi
info "进程正常退出（rc=0）"
[[ -n "$RUN_DIR" ]] && info "本次 run 目录: $RUN_DIR"

# ---- 自动解析加载摘要（用 Python，避免 grep BRE 的量词语义坑）-----------
# checkpoint_utils.py:496-509 打印的格式：
#   Loaded                    <a> / <b>  ✅   (或 ⚠️ )
#   Partial (handled)         <n> params  (dim mismatch, auto-fixed) / 0
#   Missing (rand init)       <n> params  / 0
#   Unexpected (skip)         <n> params  (not in model) / 0
echo
echo "--- 权重加载关键行（原文） ---"
grep -aE "Loaded|Missing \(rand init\)|Partial \(handled\)|Unexpected \(skip\)|Shape mismatch" "$RUNLOG" \
    | head -30 || echo "（没找到 —— 请人工检查 $RUNLOG）"
echo

VERDICT_FILE="$(mktemp)"
LOG="$RUNLOG" VERDICT_FILE="$VERDICT_FILE" python - <<'PY'
import os, re, sys

log_path = os.environ["LOG"]
try:
    txt = open(log_path, encoding="utf-8", errors="replace").read()
except Exception as e:
    print(f"[WARN] 读不到日志 {log_path}: {e}")
    open(os.environ["VERDICT_FILE"], "w").write("unknown")
    sys.exit(0)

def last_int(pattern):
    m = re.findall(pattern, txt)
    return m[-1] if m else None

loaded = re.findall(r"Loaded\s+(\d+)\s*/\s*(\d+)", txt)
missing = last_int(r"Missing \(rand init\)\s+(\d+)")
partial = last_int(r"Partial \(handled\)\s+(\d+)")
unexpected = last_int(r"Unexpected \(skip\)\s+(\d+)")

print("=" * 62)
print(" 自动解析结果")
print("=" * 62)
if loaded:
    a, b = loaded[-1]
    print(f"   Loaded              = {a} / {b}")
else:
    print("   Loaded              = 未解析到")
print(f"   Missing (rand init) = {missing if missing is not None else '未解析到'}")
print(f"   Partial (handled)   = {partial if partial is not None else '未解析到'}")
print(f"   Unexpected (skip)   = {unexpected if unexpected is not None else '未解析到'}")
print()

# 判据
if not loaded:
    verdict, why = "unknown", "没解析到 'Loaded a / b'。可能日志格式变了，或训练在加载前就失败了。"
else:
    a, b = int(loaded[-1][0]), int(loaded[-1][1])
    if a != b:
        verdict, why = "fail", f"Loaded {a}/{b}，有 {b - a} 个 key 没加载上"
    elif missing is not None and int(missing) != 0:
        verdict, why = "fail", f"Missing (rand init) = {missing}，有参数随机初始化"
    elif partial is not None and int(partial) != 0:
        verdict, why = "fail", f"Partial (handled) = {partial}，有参数维度不匹配被自动修复"
    elif missing is None:
        verdict, why = "unknown", "Loaded 全中，但没解析到 'Missing (rand init)' 行"
    else:
        verdict, why = "pass", f"Loaded {a}/{b}，Missing=0，Partial=0"

print(f"  判定: {verdict.upper()}  —— {why}")
open(os.environ["VERDICT_FILE"], "w").write(verdict)
PY

VERDICT="$(cat "$VERDICT_FILE" 2>/dev/null || echo unknown)"
rm -f "$VERDICT_FILE"

printf '=%.0s' {1..62}; echo
case "$VERDICT" in
  pass)
      ok "L3 通过（权重完整加载）。另请确认上面的 loss 是有限值、非 nan/inf"
      exit 0
      ;;
  fail)
      if [[ "$STRICT" == "1" ]]; then
          fail "L3 未通过。这是硬门禁：权重没完整加载，正式训练会白跑。日志：$RUNLOG"
      else
          warn "L3 未通过，但 STRICT=0，继续。日志：$RUNLOG"
          exit 0
      fi
      ;;
  *)
      if [[ "$STRICT" == "1" ]]; then
          fail "L3 无法自动判定（见上方）。请人工检查 $RUNLOG；确认无误后可 STRICT=0 跳过。"
      else
          warn "L3 无法自动判定，STRICT=0，继续。日志：$RUNLOG"
          exit 0
      fi
      ;;
esac
