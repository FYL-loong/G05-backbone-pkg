#!/usr/bin/env bash
# ============================================================================
# 把微调后的 backbone 产物上传到 HF（私有 model 仓库），避免随临时容器销毁。
#
# 微调输出 exp 目录（$G05_OUTPUT_DIR/yam/<exp>/）里关键文件：
#   checkpoints/step_<N>.pt   各 step 权重（大，默认不传）
#   last.pt                   指向最新 step 权重的符号链接（上传它的目标文件）
#   dataset_stats.json        归一化 stats（eval/deploy 需要的 sidecar）
#   .hydra/config.yaml        Hydra 配置 sidecar
#   action_tokenizer.pt       训练时用的 codec 快照
#
# 必需环境变量：
#   HF_TOKEN      HF 写权限 token（平台上用密钥注入，勿硬编码/勿提交）
#   EXP_DIR       微调输出 exp 目录（如 $G05_OUTPUT_DIR/yam/2026-.../）
# 可选：
#   HF_REPO_ID    目标仓库，默认 Zenith6666/yam-backbone
#   HF_ENDPOINT   默认官方 https://huggingface.co（hf-mirror 只读，不支持 push）
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-backbone}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token，勿硬编码）}"
EXP_DIR="${EXP_DIR:?请设置 EXP_DIR（微调输出 exp 目录，如 \$G05_OUTPUT_DIR/yam/2026-.../）}"
export HF_REPO_ID HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

HF_REPO_ID="$HF_REPO_ID" HF_TOKEN="$HF_TOKEN" EXP_DIR="$EXP_DIR" python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
exp = os.environ["EXP_DIR"].rstrip("/")
api = HfApi(token=os.environ["HF_TOKEN"])

api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] 仓库就绪: {repo_id}")

# 最终权重：last.pt 是指向 checkpoints/step_<N>.pt 的符号链接，上传它的目标文件
last = os.path.join(exp, "last.pt")
if os.path.islink(last):
    ckpt = os.path.realpath(last)
else:
    ckpt = last

upload_map = {
    ckpt: "model_state_dict.pt",
    os.path.join(exp, "dataset_stats.json"): "dataset_stats.json",
    os.path.join(exp, ".hydra", "config.yaml"): "hydra_config.yaml",
    os.path.join(exp, "action_tokenizer.pt"): "action_tokenizer.pt",
}

for local, remote in upload_map.items():
    if not os.path.exists(local):
        print(f"[upload] ⚠ 跳过（本地不存在）: {local}")
        continue
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=repo_id, repo_type="model")
    print(f"[upload] {local} -> {repo_id}/{remote}")

print("[upload] 完成")
PY
