#!/usr/bin/env bash
# ============================================================================
# 安装 backbone 微调环境：装 uv（如缺失）+ uv sync
#
# 依赖 pyproject.toml + uv.lock + .python-version（Python 3.10）。
# 上游用 uv 管理依赖（cu128 torch + flash-attn / flash-linear-attention /
# liger-kernel 等需要预编译轮子的原生扩展），uv.lock 已锁死版本，最稳。
#
# 国内网络可选：先 export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if command -v uv >/dev/null 2>&1; then
  echo "[env] uv 已存在：$(uv --version)"
else
  echo "[env] 安装 uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv 默认装到 ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[env] uv sync（pyproject.toml + uv.lock，Python 3.10 + cu128 torch）..."
uv sync

echo ""
echo "[env] 完成。运行方式二选一："
echo "   source .venv/bin/activate   # 激活后 python = 项目 venv"
echo "   uv run bash scripts/run/finetune.sh ...   # 或每次用 uv run 前缀"
