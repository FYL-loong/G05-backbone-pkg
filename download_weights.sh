#!/usr/bin/env bash
# ============================================================================
# 下载 backbone 微调所需权重到本地（不入库，checkpoints/ 已在 .gitignore）
#
# 1) 官方 backbone + processor —— 来自公开【门控】仓库 OpenGalaxea/G05
#      checkpoints/g05-base/checkpoints/model_state_dict.pt  (~11.44GB)
#      checkpoints/g05-base/.hydra/config.yaml                (~41KB)
#      checkpoints/g05-base/dataset_stats.json                (~5.26MB)
#      checkpoints/qwen3_5_2b_base_processor/                 (~22MB)
#    （用 --include 只拉这两个子目录，避免下载全仓 55GB）
#
# 2) 微调后的 ActionCodec 码本 + YAM stats —— 来自私有仓库（codec 微调产物）
#      checkpoints/action_tokenizer_yam.pt
#      data/stats/yam_stats.json
#
# 必需环境变量：
#   HF_TOKEN       HF 读写权限 token（勿硬编码/勿提交）
#
# 可选：
#   BASE_REPO      公开权重仓库，默认 OpenGalaxea/G05
#   CODEC_REPO     微调码本仓库，默认 Zenith6666/yam-actioncodec
#   HF_ENDPOINT    默认官方 https://huggingface.co
#
# ⚠️ 关于 HF_ENDPOINT=hf-mirror.com：
#    实测百度百舸平台【连不上 hf-mirror】，会报
#    LocalEntryNotFoundError / ConnectionError。
#    而 OpenGalaxea/G05 是【门控仓库】，hf-mirror 也走门控校验，并不能绕过审批。
#    所以本脚本不再建议 mirror；若确实要用，请先确认平台上 curl 得通。
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }

BASE_REPO="${BASE_REPO:-OpenGalaxea/G05}"
CODEC_REPO="${CODEC_REPO:-Zenith6666/yam-actioncodec}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 读写权限 token，勿硬编码）}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p checkpoints data/stats

echo "================================================================"
echo " 1/3 官方 backbone + processor（$BASE_REPO）"
echo "     注：这是门控仓库，账号需已在仓库页面点过 Agree"
echo "================================================================"
hf download "$BASE_REPO" --repo-type model --local-dir checkpoints \
  --include "g05-base/**" \
  --include "qwen3_5_2b_base_processor/**"

echo "================================================================"
echo " 2/3 微调后 ActionCodec 码本（$CODEC_REPO）"
echo "================================================================"
hf download "$CODEC_REPO" action_tokenizer_yam.pt --repo-type model --local-dir checkpoints

echo "================================================================"
echo " 3/3 YAM 归一化 stats"
echo "================================================================"
hf download "$CODEC_REPO" yam_stats.json --repo-type model --local-dir data/stats

# ---------------------------------------------------------------- 校验
echo
echo "================================================================"
echo " 下载后校验（防截断）"
echo "================================================================"

# 官方权威字节数（来自 HF API）
declare -A WANT=(
  ["checkpoints/g05-base/checkpoints/model_state_dict.pt"]=11440519964
  ["checkpoints/g05-base/.hydra/config.yaml"]=41437
  ["checkpoints/g05-base/dataset_stats.json"]=5258626
  ["checkpoints/action_tokenizer_yam.pt"]=506886775
)

bad=0
for f in "${!WANT[@]}"; do
  want="${WANT[$f]}"
  if [[ ! -f "$f" ]]; then
    printf '\033[31m[缺失]\033[0m %s\n' "$f"; bad=$((bad+1)); continue
  fi
  got="$(wc -c < "$f" | tr -d ' ')"
  if [[ "$got" == "$want" ]]; then
    ok "$f  ($got 字节)"
  else
    printf '\033[31m[大小不符]\033[0m %s  本地 %s  期望 %s  差 %s\n' \
      "$f" "$got" "$want" "$((want - got))"
    bad=$((bad+1))
  fi
done

# processor 目录只做存在性 + 文件数检查
PROC="checkpoints/qwen3_5_2b_base_processor"
if [[ -d "$PROC" ]] && [[ "$(find "$PROC" -type f | wc -l | tr -d ' ')" -gt 0 ]]; then
  ok "$PROC/  ($(find "$PROC" -type f | wc -l | tr -d ' ') 个文件)"
else
  printf '\033[31m[缺失]\033[0m %s/\n' "$PROC"; bad=$((bad+1))
fi

# stats 只做下限检查（官方无公开字节数）
if [[ -f data/stats/yam_stats.json ]]; then
  s="$(wc -c < data/stats/yam_stats.json | tr -d ' ')"
  if [[ "$s" -gt 1000 ]]; then ok "data/stats/yam_stats.json  ($s 字节)"
  else printf '\033[31m[过小]\033[0m data/stats/yam_stats.json 只有 %s 字节\n' "$s"; bad=$((bad+1)); fi
else
  printf '\033[31m[缺失]\033[0m data/stats/yam_stats.json\n'; bad=$((bad+1))
fi

echo
if [[ $bad -eq 0 ]]; then
  ok "全部权重校验通过。下一步：bash verify_weight_load.sh"
else
  fail "有 $bad 项不通过。大文件可用 curl 断点续传："
  cat <<EOF >&2

  curl -L -C - --retry 10 --retry-all-errors \\
    -H "Authorization: Bearer \$HF_TOKEN" \\
    -o checkpoints/g05-base/checkpoints/model_state_dict.pt \\
    "\$HF_ENDPOINT/$BASE_REPO/resolve/main/g05-base/checkpoints/model_state_dict.pt"
EOF
  exit 2
fi
