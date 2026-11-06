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
set +o pipefail    # 见 start.sh 顶部注释：避免 | head 触发 SIGPIPE(141) 中止

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }

BASE_REPO="${BASE_REPO:-OpenGalaxea/G05}"
CODEC_REPO="${CODEC_REPO:-Zenith6666/yam-actioncodec}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 读写权限 token，勿硬编码）}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p checkpoints data/stats

# ---------------------------------------------------------------- 下载辅助
# 为什么不用 `hf download --include "dir/**"`：
#   实测该 glob 在平台上只下到了 processor 的 8 个文件，而 `g05-base/**` 一个都没下，
#   且 hf download 在 include 匹配不到时【只警告不报错】（退出码仍是 0），
#   导致失败被推迟到最后的校验环节才发现。
#   所以改为【显式文件路径】逐个下载：路径明确、每个文件都报告大小、
#   缺了立刻失败，完全绕开 include/glob 的版本语义差异。
#
# 下载 repo 里的 rpath，并【确保】它最终位于 exact_path。
# 做法：先按 hf 的默认行为下到 <tmpdir>/<rpath>，再 mv 到 exact_path。
# 这样既不依赖 glob，也不依赖 hf 的落盘规则。
dl() {
  local repo="$1" rpath="$2" exact_path="$3"
  if [[ -f "$exact_path" ]]; then
    info "已存在，跳过：$exact_path ($(wc -c < "$exact_path" | tr -d ' ') 字节)"
    return 0
  fi
  local tmpdir
  tmpdir="$(mktemp -d)"
  info "下载 $repo :: $rpath"
  if ! hf download "$repo" "$rpath" --repo-type model --local-dir "$tmpdir"; then
    warn "hf download 失败（退出码非零）：$repo/$rpath"
    rm -rf "$tmpdir"
    return 1
  fi
  local landed="$tmpdir/$rpath"
  if [[ ! -f "$landed" ]]; then
    warn "hf download 退出码 0，但预期落盘文件不存在：$landed"
    warn "  实际拿到："
    find "$tmpdir" -type f 2>/dev/null | head -10 | sed 's/^/    /' || true
    rm -rf "$tmpdir"
    return 1
  fi
  mkdir -p "$(dirname "$exact_path")"
  mv -f "$landed" "$exact_path"
  rm -rf "$tmpdir"
  ok "$exact_path ($(wc -c < "$exact_path" | tr -d ' ') 字节)"
  return 0
}

echo "================================================================"
echo " 1/3 官方 backbone + processor（$BASE_REPO）"
echo "     注：这是门控仓库，账号需已在仓库页面点过 Agree"
echo "     预期：11.44GB 的 model_state_dict.pt + 41KB config + 5.3MB stats"
echo "================================================================"
# 先列仓库清单，确认目标路径真实存在（拼错会立刻暴露，而不是静默跳过）
# 注意：CLI 是 `hf repo-files`，没有 `hf ls`（0.35.0 实测）。
info "核对 HF 仓库文件清单 ..."
if hf repo-files "$BASE_REPO" > /tmp/_g05_ls.txt 2>&1; then
  info "仓库条目数: $(grep -c . /tmp/_g05_ls.txt || echo 0)，其中含 'g05-base' 的："
  grep 'g05-base' /tmp/_g05_ls.txt | head -10 | sed 's/^/    /' \
    || warn "  没有任何 g05-base 条目 —— 路径或门控权限可能有问题"
else
  warn "hf repo-files 失败（仅诊断用途，不阻塞）："
  head -10 /tmp/_g05_ls.txt | sed 's/^/    /' || true
fi

dl "$BASE_REPO" "g05-base/checkpoints/model_state_dict.pt" \
   "checkpoints/g05-base/checkpoints/model_state_dict.pt" \
  || fail "主干权重下载失败 —— 见上方输出；也可用 curl 断点续传（见脚本末尾提示）"
dl "$BASE_REPO" "g05-base/.hydra/config.yaml" \
   "checkpoints/g05-base/.hydra/config.yaml" || warn "hydra config 缺失（非致命）"
dl "$BASE_REPO" "g05-base/dataset_stats.json" \
   "checkpoints/g05-base/dataset_stats.json" || warn "dataset_stats 缺失（非致命）"

# 官方 ActionCodec（506MB）—— 只在【离线评测做 A/B 对比】时需要：
#   官方基座的 run 目录 config 里 tokenizer.ckpt_dir 指向 checkpoints/action_tokenizer.pt，
#   不下载这份，用基座权重跑 eval 会因找不到 codec 而失败。
#   非致命：训练/推理都用不到它（我们用微调后的 action_tokenizer_yam.pt）。
dl "$BASE_REPO" "action_tokenizer.pt" "checkpoints/action_tokenizer.pt" \
  || warn "官方 codec 下载失败（仅影响 eval 的 B 基线，训练/推理不需要）"

echo "--- processor（8 个小文件；浅层 glob 已验证可用）---"
hf download "$BASE_REPO" --repo-type model --local-dir checkpoints \
  --include "qwen3_5_2b_base_processor/*" || fail "processor 下载失败"

echo "================================================================"
echo " 2/3 微调后 ActionCodec 码本（$CODEC_REPO）"
echo "     注意：这是【微调后】的 codec（约 170MB），不是官方 506MB 那个"
echo "================================================================"
dl "$CODEC_REPO" "action_tokenizer_yam.pt" "checkpoints/action_tokenizer_yam.pt" \
  || fail "codec 下载失败"

echo "================================================================"
echo " 3/3 YAM 归一化 stats"
echo "================================================================"
dl "$CODEC_REPO" "yam_stats.json" "data/stats/yam_stats.json" || fail "stats 下载失败"

# ---------------------------------------------------------------- 校验
echo
echo "================================================================"
echo " 下载后校验（防截断/防漏下）"
echo "================================================================"

# 官方权威字节数（来自 HF API 实测）
declare -A WANT=(
  ["checkpoints/g05-base/checkpoints/model_state_dict.pt"]=11440519964
  ["checkpoints/g05-base/.hydra/config.yaml"]=41437
  ["checkpoints/g05-base/dataset_stats.json"]=5258626
  # ⚠️ 这里必须是【微调后】字节数。官方 action_tokenizer.pt 是 506,886,775，
  #    但 backbone 训练用的是微调版 codec（用户已决定弃用官方基线），
  #    所以脚本下载的就是下面这个大小，不要写成官方值。
  ["checkpoints/action_tokenizer_yam.pt"]=169709565
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
