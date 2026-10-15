#!/usr/bin/env bash
# ============================================================================
# upload_diagnostics.sh — 把【任意阶段】的诊断信息传到 HF，用于失败排查。
#
# 与 upload_output.sh 的区别：
#   upload_output.sh       训练【结束后】传产物（需要 exp 目录存在）
#   upload_diagnostics.sh  任何时刻都能跑；setup/L1/L2/L3/L4 失败时也能留证据
#
# 为什么需要它：
#   容器销毁后日志全丢。若失败发生在训练之前（装环境失败、权重拉不下来、
#   L3 门禁不过），此时还没有 exp 目录，upload_output.sh 用不了 ——
#   而这恰恰是最需要看日志的场景。
#
# 上传到 HF 仓库的 diagnostics/<STAGE>_<时间戳>/ 下：
#   start.log           整个 start.sh 的 stdout（调用方 tee 的）
#   env_report.txt      git revision / pip freeze / nvidia-smi / config 摘要
#   以及通过 EXTRA_FILES（空格分隔的路径列表）指定的额外文件
#
# 用法：
#   STAGE=setup EXP_DIR=/workspace/output/yam/xxx \
#     EXTRA_FILES="/workspace/train.stdout.log" \
#     bash upload_diagnostics.sh
#
# 环境变量：
#   HF_TOKEN      必需，写权限
#   STAGE         必需，阶段名（setup/verify/l3/train/eval），用于目录名
#   HF_REPO_ID    可选，默认 Zenith6666/yam-backbone
#   DIAG_DIR      可选，诊断工作目录，默认 /workspace/diagnostics
#   EXTRA_FILES   可选，额外要传的文件（空格分隔；不存在则跳过）
#   START_LOG     可选，start.sh 的完整 stdout 日志路径
#   EXP_DIR       可选，若已存在则一并把它的日志纳入
#   UPLOAD_ROOT   可选，项目根（生成 env_report 用），默认脚本所在目录
# ============================================================================
set -uo pipefail
set +o pipefail    # 见 start.sh 顶部注释：避免 | head 触发 SIGPIPE(141)    # 注意：故意不用 -e，诊断上传不能因为小错中断

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" 2>/dev/null || true

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-backbone}"
HF_TOKEN="${HF_TOKEN:-}"
STAGE="${STAGE:-unknown}"
DIAG_DIR="${DIAG_DIR:-/workspace/diagnostics}"
EXTRA_FILES="${EXTRA_FILES:-}"
START_LOG="${START_LOG:-/workspace/start.log}"
EXP_DIR="${EXP_DIR:-}"
UPLOAD_ROOT="${UPLOAD_ROOT:-$SCRIPT_DIR}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

info() { printf '\033[36m[diag]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[diag]\033[0m %s\n' "$*"; }

if [[ -z "$HF_TOKEN" ]]; then
    warn "没有 HF_TOKEN，诊断信息无法上传（只留在容器内）"
    exit 0
fi

TS="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo unknown)"
DEST="diagnostics/${STAGE}_${TS}"
BUNDLE="$DIAG_DIR/${STAGE}_${TS}"
mkdir -p "$BUNDLE" 2>/dev/null || { warn "无法创建 $BUNDLE"; exit 0; }

info "收集诊断信息 -> $BUNDLE  (仓库内: $DEST/)"

# 1) 环境快照
REPORT="$BUNDLE/env_report.txt"
{
    echo "================================================================"
    echo " 诊断快照"
    echo " 阶段    : $STAGE"
    echo " 时间    : $(date -Iseconds 2>/dev/null || date)"
    echo " 主机    : $(hostname 2>/dev/null)"
    echo "================================================================"
    echo
    echo "--- git revision ---"
    if [[ -d "$UPLOAD_ROOT/.git" ]]; then
        git -C "$UPLOAD_ROOT" log -1 --format='commit: %H%ndate:   %ci%nsubject:%s' 2>/dev/null || echo "(git log 失败)"
        echo
        echo "工作区状态:"
        git -C "$UPLOAD_ROOT" status --short 2>/dev/null | head -30 || true
    else
        echo "(不是 git 仓库)"
    fi
    echo
    echo "--- GPU ---"
    nvidia-smi 2>/dev/null || echo "(nvidia-smi 不可用)"
    echo
    echo "--- 关键库版本 ---"
    python -c "
import sys
print('python', sys.version.replace(chr(10),' '))
for m in ('torch','flash_attn','flash_linear_attention','liger_kernel','transformers','accelerate','deepspeed','huggingface_hub'):
    try:
        mod = __import__(m); print(f'{m:24s}', getattr(mod,'__version__','(n/a)'))
    except Exception as e:
        print(f'{m:24s} [不可用] {type(e).__name__}')
" 2>&1 || echo "(python 检查失败)"
    echo
    echo "--- pip freeze ---"
    pip freeze 2>/dev/null || echo "(pip freeze 失败)"
    echo
    echo "--- 磁盘 ---"
    df -h 2>/dev/null | head -10 || true
    echo
    echo "--- 环境变量名（只列名字和是否设置，不打印值）---"
    env | grep -oE '^[A-Za-z_][A-Za-z0-9_]*' | sort -u | head -80 || true
    echo
    echo "--- 已发现的相关文件 ---"
    for d in "$DIAG_DIR" "$EXP_DIR" /workspace; do
        [[ -n "$d" && -d "$d" ]] || continue
        echo "[$d]"
        find "$d" -maxdepth 2 -type f \( -name '*.log' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.json' \) 2>/dev/null | head -30 || true
    done
    echo
    echo "--- 配置摘要（已脱敏）---"
    CFG=""
    for c in "$EXP_DIR/.hydra/config.yaml" "$UPLOAD_ROOT/configs/train.yaml"; do
        [[ -n "$c" && -f "$c" ]] && { CFG="$c"; break; }
    done
    if [[ -n "$CFG" ]]; then
        echo "(来自 $CFG)"
        grep -viE 'token|secret|password|api[_-]?key' "$CFG" 2>/dev/null || true
    else
        echo "(没找到配置)"
    fi
} > "$REPORT" 2>&1
info "  env_report.txt ($(wc -c < "$REPORT" 2>/dev/null | tr -d ' ') 字节)"

# 2) 收集具体日志文件
collect() {
    local src="$1" name="$2"
    [[ -n "$src" && -f "$src" ]] || return 0
    cp -f "$src" "$BUNDLE/$name" 2>/dev/null && info "  $name ($(wc -c < "$BUNDLE/$name" 2>/dev/null | tr -d ' ') 字节)" || warn "  复制失败: $src"
}

collect "$START_LOG" "start.log"
collect "/workspace/train.stdout.log" "train.stdout.log"
collect "/workspace/upload_checkpoints.log" "upload_checkpoints.log"

if [[ -n "$EXP_DIR" && -d "$EXP_DIR" ]]; then
    collect "$EXP_DIR/train.log" "train.log"
    collect "$EXP_DIR/finetune.jsonl" "finetune.jsonl"
    collect "$EXP_DIR/env_report.txt" "env_report.txt"
    if [[ -d "$EXP_DIR/logs" ]]; then
        for f in "$EXP_DIR"/logs/*.log; do
            [[ -f "$f" ]] && collect "$f" "logs_$(basename "$f")"
        done
    fi
fi

# 额外指定
for f in $EXTRA_FILES; do
    [[ -f "$f" ]] && collect "$f" "$(basename "$f")"
done

# 3) 上传
# ⚠️ 诊断可能在【任何阶段】被调用，包括 preflight（此时 .venv 还不存在，
#    系统 python 里也没有 huggingface_hub）。实测会报
#    ModuleNotFoundError: No module named 'huggingface_hub'，导致诊断丢失 ——
#    而早期失败恰恰最需要诊断。所以这里按优先级挑一个可用的 python，
#    都没有就现场装一个（仅几十秒，且失败也不影响退出码）。
info "上传到 $HF_REPO_ID/$DEST/ ..."

pick_python() {
    local cands=(
        "${PROJECT_ROOT:-$SCRIPT_DIR}/.venv/bin/python"
        "$SCRIPT_DIR/.venv/bin/python"
        "$(command -v python3 2>/dev/null || true)"
        "$(command -v python 2>/dev/null || true)"
    )
    local c
    for c in "${cands[@]}"; do
        [[ -n "$c" && -x "$c" ]] || continue
        if "$c" -c "import huggingface_hub" >/dev/null 2>&1; then
            printf '%s' "$c"; return 0
        fi
    done
    # 有 python 但没有 hub：就地装一个
    for c in "${cands[@]}"; do
        [[ -n "$c" && -x "$c" ]] || continue
        warn "在 $c 里安装 huggingface_hub ..."
        if "$c" -m pip install -q "huggingface_hub" >/dev/null 2>&1 \
           || "$c" -m pip install -q --break-system-packages "huggingface_hub" >/dev/null 2>&1; then
            printf '%s' "$c"; return 0
        fi
    done
    return 1
}

PYBIN="$(pick_python || true)"
if [[ -z "$PYBIN" ]]; then
    warn "找不到可用 python / 装不上 huggingface_hub，诊断无法上传"
    warn "文件仍在容器内：$BUNDLE"
    exit 0
fi
info "使用 python: $PYBIN"

HF_REPO_ID="$HF_REPO_ID" HF_TOKEN="$HF_TOKEN" BUNDLE="$BUNDLE" DEST="$DEST" "$PYBIN" - <<'PY' 2>&1 | sed 's/^/  /'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
bundle = os.environ["BUNDLE"]
dest = os.environ["DEST"]
try:
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=bundle, path_in_repo=dest, repo_id=repo_id, repo_type="model")
    n = sum(len(files) for _, _, files in os.walk(bundle))
    print(f"[diag] OK 已上传 {n} 个文件 -> https://huggingface.co/{repo_id}/tree/main/{dest}")
except Exception as e:
    print(f"[diag] !! 上传失败: {type(e).__name__}: {e}")
    raise SystemExit(1)
PY
rc=$?

if [[ $rc -eq 0 ]]; then
    info "诊断已上传：https://huggingface.co/$HF_REPO_ID/tree/main/$DEST"
else
    warn "诊断上传失败（rc=$rc）。文件仍在容器内：$BUNDLE"
fi
exit 0
