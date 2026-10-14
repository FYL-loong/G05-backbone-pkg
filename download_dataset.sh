#!/usr/bin/env bash
# ============================================================================
# 从 HF 私有 dataset 仓库下载已转换好的 LeRobot 数据集到 data/yam/
#
# 路径对齐 configs/data/yam.yaml 的 dataset_dirs（data/yam/TASK-YAM-XXXX_lerobot）。
#
# 必需环境变量：
#   HF_TOKEN       HF 读权限 token（勿硬编码/勿提交）
# 可选：
#   HF_REPO_ID     数据集仓库，默认 Zenith6666/yam-lerobot
#   ONLY_TASK      只下载某个任务（如 TASK-YAM-0001），默认下载全部
#   OUTPUT_DIR     本地目录，默认 data/yam
#   HF_ENDPOINT    默认官方 https://huggingface.co；国内下载可 export HF_ENDPOINT=https://hf-mirror.com
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-lerobot}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 读权限 token，勿硬编码）}"
OUTPUT_DIR="${OUTPUT_DIR:-data/yam}"
ONLY_TASK="${ONLY_TASK:-}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "================================================================"
echo " 从 HF 下载 LeRobot 数据集"
echo "   repo   : $HF_REPO_ID"
echo "   to     : $OUTPUT_DIR"
[ -n "$ONLY_TASK" ] && echo "   only   : ${ONLY_TASK}_lerobot"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

if [ -n "$ONLY_TASK" ]; then
  hf download "$HF_REPO_ID" --repo-type dataset --local-dir "$OUTPUT_DIR" \
    --include "${ONLY_TASK}_lerobot/**"
else
  hf download "$HF_REPO_ID" --repo-type dataset --local-dir "$OUTPUT_DIR"
fi

echo "[done] 已下载到 $OUTPUT_DIR"
