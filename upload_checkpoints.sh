#!/usr/bin/env bash
# ============================================================================
# upload_checkpoints.sh — 把训练中的 checkpoint 传到 HF（【可反复重跑、支持续传】）
#
# 存在的理由：
#   1) 平台容器【临时】，任务结束即销毁；且没有持久盘可挂
#   2) 表单里「优先级=中」会触发【低利用率自动停止】，自动容错可能重启任务
#   3) 单个 checkpoint ≈ 11.44 GB —— 远超 GitHub 的 100MB/文件 限制，
#      git / git-lfs 都存不了，只能走 HF
#
#   官方 upload_output.sh 只在【训练结束后】传一次。本脚本是它的补充：
#   训练过程中任何时候都能跑，把当前已有的 checkpoint 传上去。
#
# 典型用法：
#   # 训练在另一个终端跑着，你每隔一段时间打一次：
#   EXP_DIR=$G05_OUTPUT_DIR/yam/<exp> bash upload_checkpoints.sh
#
#   # 只传最后一个（最省时间）：
#   MODE=last EXP_DIR=... bash upload_checkpoints.sh
#
#   # 只传某个 step：
#   MODE=step STEP=4000 EXP_DIR=... bash upload_checkpoints.sh
#
# 必需环境变量：
#   HF_TOKEN    写权限 token
#   EXP_DIR     训练输出目录（$G05_OUTPUT_DIR/yam/<exp>/）
# 可选：
#   HF_REPO_ID  默认 Zenith6666/yam-backbone
#   MODE        all(默认) | last | step | watch
#   STEP        MODE=step 时用
#   DRY_RUN     1 = 只列出会传什么，不实际传
#   WATCH_INTERVAL  MODE=watch 时的轮询间隔秒数，默认 600（10 分钟）
#
# MODE=watch（无人值守用）：
#   常驻轮询 EXP_DIR/checkpoints/，发现【新出现或大小已稳定】的 *.pt 就上传，
#   已传过的跳过。容器被销毁时进程随之结束，不需要手动停。
#   典型用法（由 start.sh 在后台拉起）：
#     nohup env MODE=watch EXP_DIR=$EXP_DIR bash upload_checkpoints.sh \
#       > /workspace/upload_checkpoints.log 2>&1 &
# ============================================================================
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-backbone}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token）}"
EXP_DIR="${EXP_DIR:?请设置 EXP_DIR（训练输出目录，如 \$G05_OUTPUT_DIR/yam/2026-.../）}"
MODE="${MODE:-all}"
STEP="${STEP:-}"
DRY_RUN="${DRY_RUN:-0}"
WATCH_INTERVAL="${WATCH_INTERVAL:-600}"
export HF_TOKEN
export HF_REPO_ID

[[ -d "$EXP_DIR" ]] || { echo "[FAIL] EXP_DIR 不是目录: $EXP_DIR" >&2; exit 1; }
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

# ---------------------------------------------------------------- watch 模式
# 单独实现：不落盘状态，靠"已上传清单"文件避免重复传
if [[ "$MODE" == "watch" ]]; then
    STATE="${EXP_DIR}/.uploaded_checkpoints.txt"
    touch "$STATE"
    echo "[watch] 轮询 $EXP_DIR/checkpoints/，间隔 ${WATCH_INTERVAL}s，状态 $STATE"
    while true; do
        shopt -s nullglob
        for f in "$EXP_DIR"/checkpoints/*.pt; do
            b="$(basename "$f")"
            [[ "$b" == "last.pt" ]] && continue
            sz="$(wc -c < "$f" | tr -d ' ')"
            key="${b}:${sz}"
            grep -qxF "$key" "$STATE" 2>/dev/null && continue
            # 等文件大小稳定（避免传到写一半的 checkpoint）
            sleep 20
            sz2="$(wc -c < "$f" | tr -d ' ')"
            if [[ "$sz" != "$sz2" ]]; then
                echo "[watch] $b 仍在写入（$sz -> $sz2），跳过本轮"
                continue
            fi
            echo "[watch] 上传 $b ($((sz2/1024/1024)) MiB)"
            if hf upload "$HF_REPO_ID" "$f" "checkpoints/$b" --repo-type model 2>&1 | tail -3; then
                echo "$key" >> "$STATE"
                echo "[watch] 已上传 $b"
            else
                echo "[watch] 上传失败 $b，下轮重试" >&2
            fi
        done
        # sidecar（小，直接覆盖）
        for sc in dataset_stats.json hydra_config.yaml action_tokenizer.pt train.log; do
            case "$sc" in
                hydra_config.yaml) src="$EXP_DIR/.hydra/config.yaml" ;;
                *)                 src="$EXP_DIR/$sc" ;;
            esac
            [[ -f "$src" ]] && hf upload "$HF_REPO_ID" "$src" "$sc" --repo-type model >/dev/null 2>&1 || true
        done
        sleep "$WATCH_INTERVAL"
    done
fi

echo "================================================================"
echo " 上传 checkpoint 到 HF"
echo "   repo : $HF_REPO_ID"
echo "   exp  : $EXP_DIR"
echo "   mode : $MODE"
echo "================================================================"

echo "--- 输出目录现状 ---"
ls -la "$EXP_DIR" 2>/dev/null | head -20
echo "--- checkpoints/ ---"
ls -lah "$EXP_DIR/checkpoints" 2>/dev/null | tail -15 || echo "（还没有 checkpoints/ 目录）"
echo
df -h "$EXP_DIR" | tail -1
echo

MODE="$MODE" STEP="$STEP" DRY_RUN="$DRY_RUN" HF_REPO_ID="$HF_REPO_ID" EXP_DIR="$EXP_DIR" python - <<'PY'
import glob, os, sys
from huggingface_hub import HfApi

exp = os.environ["EXP_DIR"]
mode = os.environ["MODE"]
step = os.environ.get("STEP") or ""
dry = os.environ.get("DRY_RUN") == "1"
repo = os.environ["HF_REPO_ID"]
api = HfApi(token=os.environ["HF_TOKEN"])

# 1) checkpoint 列表
ckpts = sorted(glob.glob(os.path.join(exp, "checkpoints", "*.pt")))
ckpts = [c for c in ckpts if not c.endswith("last.pt")]

if mode == "last":
    ckpts = ckpts[-1:] if ckpts else []
elif mode == "step":
    if not step:
        print("[FAIL] MODE=step 需要同时设 STEP=<数字>", file=sys.stderr); sys.exit(1)
    ckpts = [c for c in ckpts if f"step_{step}" in os.path.basename(c)]
    if not ckpts:
        print(f"[FAIL] 没找到 step_{step} 的 checkpoint。现有：", file=sys.stderr)
        for c in glob.glob(os.path.join(exp, "checkpoints", "*.pt")):
            print("   ", os.path.basename(c), file=sys.stderr)
        sys.exit(1)

if not ckpts:
    print("[WARN] 没有找到任何 checkpoint（训练可能还没跑到 checkpointing_steps，默认 2000 步）")
    sys.exit(0)

print(f"待传 {len(ckpts)} 个 checkpoint：")
total = 0
for c in ckpts:
    sz = os.path.getsize(c)
    total += sz
    print(f"   {os.path.basename(c):34s} {sz/2**30:6.2f} GiB")
print(f"   合计 {total/2**30:.2f} GiB")

# 2) sidecar（每次覆盖，很小）
sidecars = [
    (os.path.join(exp, "dataset_stats.json"), "dataset_stats.json"),
    (os.path.join(exp, ".hydra", "config.yaml"), "hydra_config.yaml"),
    (os.path.join(exp, "action_tokenizer.pt"), "action_tokenizer.pt"),
    (os.path.join(exp, "train.log"), "train.log"),
]
sidecars = [(l, r) for l, r in sidecars if os.path.isfile(l)]

print(f"\nsidecar {len(sidecars)} 个：")
for l, r in sidecars:
    print(f"   {r:28s} {os.path.getsize(l)/2**20:8.2f} MiB")

if dry:
    print("\n[DRY_RUN] 只列清单，未实际上传。去掉 DRY_RUN=1 即执行。")
    sys.exit(0)

# 3) 建仓
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
print(f"\n仓库就绪: {repo}")

# 4) 上传（大文件放 HF_HUB_ENABLE_HF_TRANSFER）
for local in ckpts:
    name = os.path.basename(local)
    remote = f"checkpoints/{name}"
    print(f"[上传] {name} ({os.path.getsize(local)/2**30:.2f} GiB) -> {remote} ...", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=repo, repo_type="model")
    print(f"       完成")

for local, remote in sidecars:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=repo, repo_type="model")
    print(f"[上传] {remote}")

print("\n[done] 全部上传完成。")
print("      HF 上路径：checkpoints/<step>.pt")
PY

echo
echo "================================================================"
echo " 提示"
echo "================================================================"
echo " 本脚本可【反复重跑】：已上传的文件若内容未变，HF 会跳过重复传输。"
echo " 训练结束后的最终收尾，仍建议跑官方 upload_output.sh（它会挑 last.pt 的目标并扁平命名）。"
