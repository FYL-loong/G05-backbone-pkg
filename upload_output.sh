#!/usr/bin/env bash
# ============================================================================
# 把微调产物与【全部诊断信息】上传到 HF 私有仓库，避免随临时容器销毁。
#
# 设计目标（按重要性排序）：
#   1) 失败时也留下可诊断的证据 —— 日志比权重值钱，权重能重跑、原因不查清就卡死
#   2) 成功时保存完整结果 —— 权重 + 配置 + 指标 + 运行环境快照
#   3) 可反复重跑、幂等
#
# 上传内容：
#   model_state_dict.pt      最终权重（last.pt 指向的目标，约 11.4GB）
#   checkpoints/step_*.pt    各 step checkpoint（守护进程模式另传）
#   train.log                完整训练日志
#   finetune.jsonl           逐 step 指标
#   logs/*.log               pretrain_log.txt + 每卡 model_debug_rank*.log
#   dataset_stats.json       归一化 stats（eval/deploy 需要）
#   action_tokenizer.pt      训练时用的 codec 快照
#   hydra_config.yaml        完整 Hydra 配置
#   hf_processor/**          推理需要的 processor 文件
#   env_report.txt           ★ 环境快照（git revision / pip freeze / nvidia-smi /
#                              config 摘要，已脱敏去掉 *TOKEN*）
#
# 必需环境变量：
#   HF_TOKEN      HF 写权限 token（平台上用密钥注入，勿硬编码/勿提交）
#   EXP_DIR       微调输出 exp 目录（如 $G05_OUTPUT_DIR/yam/2026-.../）
# 可选：
#   HF_REPO_ID    目标仓库，默认 Zenith6666/yam-backbone
#   HF_ENDPOINT   默认官方 https://huggingface.co（hf-mirror 只读，不支持 push）
#   SKIP_CKPT     1 = 不传 model_state_dict.pt（只传诊断，省时间）
#   UPLOAD_ROOT   项目根目录（生成 env_report 用），默认脚本所在目录
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-backbone}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token，勿硬编码）}"
EXP_DIR="${EXP_DIR:?请设置 EXP_DIR（微调输出 exp 目录，如 \$G05_OUTPUT_DIR/yam/2026-.../）}"
SKIP_CKPT="${SKIP_CKPT:-0}"
UPLOAD_ROOT="${UPLOAD_ROOT:-$SCRIPT_DIR}"
export HF_REPO_ID HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$EXP_DIR" ]] || fail "EXP_DIR 不是目录: $EXP_DIR"
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

info "exp 目录: $EXP_DIR"
info "目标仓库: $HF_REPO_ID"
echo "--- 目录内容 ---"
ls -la "$EXP_DIR" 2>/dev/null | head -25 || true
[[ -d "$EXP_DIR/checkpoints" ]] && { echo "--- checkpoints/ ---"; ls -lah "$EXP_DIR/checkpoints" | tail -15; }
echo "--- 磁盘 ---"
df -h "$EXP_DIR" | tail -1

# ---------------------------------------------------------------- 环境快照
# 出问题时，环境信息比日志正文更难复原，所以主动抓一份。
REPORT="$EXP_DIR/env_report.txt"
{
    echo "================================================================"
    echo " 环境快照 env_report.txt"
    echo " 生成时间: $(date -Iseconds 2>/dev/null || date)"
    echo "================================================================"
    echo
    echo "--- hostname / 容器 ---"
    hostname 2>/dev/null || true
    echo
    echo "--- git revision（复现实验的关键）---"
    if [[ -d "$UPLOAD_ROOT/.git" ]]; then
        git -C "$UPLOAD_ROOT" log -1 --format='commit: %H%ndate:   %ci%nsubject:%s' 2>/dev/null || true
        echo
        echo "工作区状态（有改动说明代码与 commit 不一致）:"
        git -C "$UPLOAD_ROOT" status --short 2>/dev/null | head -30 || true
    else
        echo "(不是 git 仓库)"
    fi
    echo
    echo "--- GPU ---"
    nvidia-smi 2>/dev/null || echo "(nvidia-smi 不可用)"
    echo
    echo "--- python / torch / 关键库 ---"
    python -c "
import sys
print('python  ', sys.version.replace(chr(10),' '))
print('prefix  ', sys.prefix)
for m in ('torch','flash_attn','flash_linear_attention','liger_kernel','transformers','accelerate','deepspeed','huggingface_hub','bitsandbytes'):
    try:
        mod = __import__(m)
        print(f'{m:24s}', getattr(mod,'__version__','(no __version__)'))
    except Exception as e:
        print(f'{m:24s} [不可用] {type(e).__name__}')
try:
    import torch
    print('cuda available:', torch.cuda.is_available())
    print('cuda version  :', torch.version.cuda)
    print('device count  :', torch.cuda.device_count())
except Exception:
    pass
" 2>&1 || true
    echo
    echo "--- pip freeze（完整依赖，复现用）---"
    pip freeze 2>/dev/null || echo "(pip freeze 失败)"
    echo
    echo "--- 训练用配置（已脱敏，去掉 *TOKEN*/*SECRET*/*KEY*）---"
    CFG=""
    for c in "$EXP_DIR/.hydra/config.yaml" "$EXP_DIR/hydra_config.yaml"; do
        [[ -f "$c" ]] && { CFG="$c"; break; }
    done
    if [[ -n "$CFG" ]]; then
        grep -viE 'token|secret|password|api[_-]?key' "$CFG" || true
    else
        echo "(没找到 hydra config)"
    fi
    echo
    echo "--- 训练 stdout 尾部 200 行（若有）---"
    tail -200 /workspace/train.stdout.log 2>/dev/null || echo "(没有 /workspace/train.stdout.log)"
} > "$REPORT" 2>&1
info "环境快照已生成: $REPORT ($(wc -c < "$REPORT" | tr -d ' ') 字节)"

# ---------------------------------------------------------------- 上传
REPORT="$REPORT" EXP_DIR="$EXP_DIR" HF_REPO_ID="$HF_REPO_ID" SKIP_CKPT="$SKIP_CKPT" python - <<'PY'
import os, sys
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
exp = os.environ["EXP_DIR"].rstrip("/")
skip_ckpt = os.environ.get("SKIP_CKPT") == "1"
api = HfApi(token=os.environ["HF_TOKEN"])

api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] 仓库就绪: {repo_id}")

# (本地路径, 仓库内路径, 说明)
jobs = []

# 1) 最终权重：last.pt 是指向 checkpoints/step_<N>.pt 的符号链接
if not skip_ckpt:
    last = os.path.join(exp, "last.pt")
    ckpt = os.path.realpath(last) if os.path.islink(last) else last
    jobs.append((ckpt, "model_state_dict.pt", "最终权重"))

# 2) 诊断信息 —— 失败时最需要的部分
jobs += [
    (os.path.join(exp, "train.log"),                "train.log",           "完整训练日志"),
    (os.path.join(exp, "finetune.jsonl"),           "finetune.jsonl",      "逐 step 指标"),
    (os.path.join(exp, "env_report.txt"),           "env_report.txt",      "环境快照"),
    (os.path.join(exp, "dataset_stats.json"),       "dataset_stats.json",  "归一化 stats"),
    (os.path.join(exp, "action_tokenizer.pt"),      "action_tokenizer.pt", "codec 快照"),
    (os.path.join(exp, ".hydra", "config.yaml"),    "hydra_config.yaml",   "Hydra 配置"),
    (os.path.join(exp, "logs", "pretrain_log.txt"), "logs/pretrain_log.txt", "预训练日志"),
]
# 每卡 debug 日志（排查 NCCL / OOM 必看）
logs_dir = os.path.join(exp, "logs")
if os.path.isdir(logs_dir):
    for fn in sorted(os.listdir(logs_dir)):
        if fn.startswith("model_debug_rank") and fn.endswith(".log"):
            jobs.append((os.path.join(logs_dir, fn), f"logs/{fn}", "单卡 debug 日志"))

# 训练 stdout（容器内绝对路径，如果还能读到）
if os.path.isfile("/workspace/train.stdout.log"):
    jobs.append(("/workspace/train.stdout.log", "train.stdout.log", "训练 stdout"))

ok_n = skip_n = fail_n = 0
for local, remote, desc in jobs:
    if not os.path.exists(local):
        print(f"[upload] -  跳过（不存在）  {remote}   [{desc}]")
        skip_n += 1
        continue
    size = os.path.getsize(local)
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo_id, repo_type="model")
        print(f"[upload] OK {size:>13,}  {remote}   [{desc}]")
        ok_n += 1
    except Exception as e:
        print(f"[upload] !! 失败          {remote}   [{desc}]  {type(e).__name__}: {e}")
        fail_n += 1

# hf_processor/ 整个目录（推理需要）
proc = os.path.join(exp, "hf_processor")
if os.path.isdir(proc) and os.listdir(proc):
    try:
        api.upload_folder(folder_path=proc, path_in_repo="hf_processor",
                          repo_id=repo_id, repo_type="model")
        print(f"[upload] OK {'':>13}  hf_processor/   [推理用 processor 文件]")
        ok_n += 1
    except Exception as e:
        print(f"[upload] !! 失败          hf_processor/  {type(e).__name__}: {e}")
        fail_n += 1
else:
    print("[upload] -  跳过（不存在）  hf_processor/")
    skip_n += 1

print()
print("=" * 64)
print(f" 完成：成功 {ok_n}，跳过 {skip_n}，失败 {fail_n}")
print(f" 查看：https://huggingface.co/{repo_id}")
print("=" * 64)
if fail_n:
    print("[WARN] 有上传失败项，可重跑本脚本（幂等）")
    sys.exit(1)
PY

info "upload_output.sh 结束"
