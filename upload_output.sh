#!/usr/bin/env bash
# ============================================================================
# 把微调后的 backbone 权重上传到 HF 私有仓库（防止随临时容器销毁而丢失）。
#
# 为什么要传：容器是临时的，任务结束即销毁；平台只保留日志，不保留权重文件。
#             11.44GB 的权重远超 GitHub 限制（单文件 100MB），只能放 HF。
#
# 上传内容（只看推理/评测需要什么 + 崩溃诊断必需）：
#   model_state_dict.pt      最终权重（last.pt 指向的目标，约 11.4GB）
#   dataset_stats.json       归一化 stats（eval/deploy 需要）
#   action_tokenizer.pt      训练时用的 codec 快照
#   hydra_config.yaml        完整 Hydra 配置（复现实验用）
#   hf_processor/**          推理需要的 processor 文件
#   logs/**                  ★ 训练日志 —— 容器销毁后唯一的崩溃现场，必须传
#
# 为什么连日志一起传（实测踩过的坑）：
#   训练 stdout 被 start.sh 重定向到 /workspace/train.stdout.log 这个【文件】，
#   平台日志采集的是容器 stdout，**采不到重定向进文件的内容**。
#   曾经因此把"训练跑 5 分钟就退出"的崩溃现场永久丢失，只能猜。所以现在必须传。
#
# 必需环境变量：
#   HF_TOKEN      HF 写权限 token
#   EXP_DIR       微调输出 exp 目录（如 $G05_OUTPUT_DIR/yam/2026-.../）
# 可选：
#   HF_REPO_ID    目标仓库，默认 Zenith6666/yam-backbone
#   HF_ENDPOINT   默认 https://huggingface.co
#   SKIP_CKPT     1 = 不传 model_state_dict.pt（只传 sidecar）
# ============================================================================
set -euo pipefail
set +o pipefail    # 避免 | head 之类触发 SIGPIPE(141)，见 start.sh 顶部注释

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-backbone}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token，勿硬编码）}"
EXP_DIR="${EXP_DIR:?请设置 EXP_DIR（微调输出 exp 目录，如 \$G05_OUTPUT_DIR/yam/2026-.../）}"
SKIP_CKPT="${SKIP_CKPT:-0}"
export HF_REPO_ID HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$EXP_DIR" ]] || fail "EXP_DIR 不是目录: $EXP_DIR"
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

info "exp 目录 : $EXP_DIR"
info "目标仓库 : $HF_REPO_ID"

REPORT="$EXP_DIR/.upload_result.txt"
: > "$REPORT"

EXP_DIR="$EXP_DIR" HF_REPO_ID="$HF_REPO_ID" SKIP_CKPT="$SKIP_CKPT" REPORT="$REPORT" python - <<'PY'
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("[FAIL] 当前 python 缺 huggingface_hub，无法上传", file=sys.stderr)
    sys.exit(1)

repo_id = os.environ["HF_REPO_ID"]
exp = os.environ["EXP_DIR"].rstrip("/")
skip_ckpt = os.environ.get("SKIP_CKPT") == "1"
report = os.environ["REPORT"]
api = HfApi(token=os.environ["HF_TOKEN"])

# 建仓（首次运行会创建；已存在则幂等）
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] 仓库就绪: {repo_id}")

# (本地路径, 仓库内路径, 说明)
jobs = []

# 最终权重：last.pt 是符号链接，上传它的真实目标
if not skip_ckpt:
    last = os.path.join(exp, "last.pt")
    ckpt = os.path.realpath(last) if os.path.islink(last) else last
    jobs.append((ckpt, "model_state_dict.pt", "最终权重"))

jobs += [
    (os.path.join(exp, "dataset_stats.json"),    "dataset_stats.json",  "归一化 stats"),
    (os.path.join(exp, "action_tokenizer.pt"),   "action_tokenizer.pt", "codec 快照"),
    (os.path.join(exp, ".hydra", "config.yaml"), "hydra_config.yaml",   "Hydra 配置"),
]

ok_n = skip_n = fail_n = 0
lines = []
for local, remote, desc in jobs:
    if not os.path.exists(local):
        msg = f"[upload] -  跳过（不存在） {remote}   [{desc}]"
        print(msg); lines.append(msg); skip_n += 1
        continue
    size = os.path.getsize(local)
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo_id, repo_type="model")
        msg = f"[upload] OK {size:>13,}  {remote}   [{desc}]"
        print(msg); lines.append(msg); ok_n += 1
    except Exception as e:
        msg = f"[upload] !! 失败 {remote}   [{desc}]  {type(e).__name__}: {e}"
        print(msg, file=sys.stderr); lines.append(msg); fail_n += 1

# hf_processor/ 整个目录（推理需要）
proc = os.path.join(exp, "hf_processor")
if os.path.isdir(proc) and os.listdir(proc):
    try:
        api.upload_folder(folder_path=proc, path_in_repo="hf_processor",
                          repo_id=repo_id, repo_type="model")
        msg = "[upload] OK             hf_processor/   [推理用 processor]"
        print(msg); lines.append(msg); ok_n += 1
    except Exception as e:
        msg = f"[upload] !! 失败 hf_processor/  {type(e).__name__}: {e}"
        print(msg, file=sys.stderr); lines.append(msg); fail_n += 1
else:
    msg = "[upload] -  跳过（不存在） hf_processor/"
    print(msg); lines.append(msg); skip_n += 1

# ---- 训练日志（★ 崩溃诊断必需）------------------------------------------
# train.stdout.log 在容器【外面】，必须单独传：
#   start.sh 把训练 stdout 重定向到了这个文件，平台日志采集容器 stdout，
#   采不到重定向进文件的内容 —— 不传就会在容器销毁后永久丢失崩溃现场。
log_jobs = [
    ("/workspace/train.stdout.log", "logs/train.stdout.log", "训练 stdout（完整）"),
    ("/workspace/train.exit_code",  "logs/train.exit_code",  "训练进程退出码"),
    (os.path.join(exp, "train.log"), "logs/train.log",        "训练主日志"),
]
for local, remote, desc in log_jobs:
    if not os.path.exists(local):
        msg = f"[upload] -  跳过（不存在） {remote}   [{desc}]"
        print(msg); lines.append(msg); skip_n += 1
        continue
    size = os.path.getsize(local)
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo_id, repo_type="model")
        msg = f"[upload] OK {size:>13,}  {remote}   [{desc}]"
        print(msg); lines.append(msg); ok_n += 1
    except Exception as e:
        msg = f"[upload] !! 失败 {remote}   [{desc}]  {type(e).__name__}: {e}"
        print(msg, file=sys.stderr); lines.append(msg); fail_n += 1

# exp/logs/ 整个目录（含 model_debug_rank*.log / pretrain_log.txt 等）
elogs = os.path.join(exp, "logs")
if os.path.isdir(elogs) and os.listdir(elogs):
    try:
        api.upload_folder(folder_path=elogs, path_in_repo="logs/exp",
                          repo_id=repo_id, repo_type="model")
        n = sum(len(f) for _, _, f in os.walk(elogs))
        msg = f"[upload] OK             logs/exp/   [{n} 个日志文件]"
        print(msg); lines.append(msg); ok_n += 1
    except Exception as e:
        msg = f"[upload] !! 失败 logs/exp/  {type(e).__name__}: {e}"
        print(msg, file=sys.stderr); lines.append(msg); fail_n += 1
else:
    msg = "[upload] -  跳过（不存在） logs/exp/"
    print(msg); lines.append(msg); skip_n += 1

summary = (f"成功 {ok_n}，跳过 {skip_n}，失败 {fail_n}\n"
           f"https://huggingface.co/{repo_id}\n")
print("\n" + "=" * 64)
print(summary)
print("=" * 64)
open(report, "w", encoding="utf-8").write(summary + "\n".join(lines) + "\n")

if fail_n:
    print("[WARN] 有上传失败项，可重跑本脚本（幂等）", file=sys.stderr)
    sys.exit(1)
PY

info "upload_output.sh 结束"
