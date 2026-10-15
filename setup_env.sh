#!/usr/bin/env bash
# ============================================================================
# 安装 backbone 微调环境：确保 uv 存在 + 确保 Python 3.10 存在 + uv sync
#
# 依赖 pyproject.toml + uv.lock + .python-version。
#
# ⚠️ 关键约束（.python-version / pyproject）：
#     .python-version            = 3.10.16
#     pyproject requires-python  = ">=3.10.16,<3.11"
#   NGC pytorch 镜像自带的是 Python 3.12 —— 【不满足约束】，所以 uv 必须
#   自己下载一个独立的 Python 3.10.16。而它的下载源是 GitHub Releases：
#     https://github.com/astral-sh/python-build-standalone/releases/download/...
#   国内/百度百舸访问 GitHub 不可靠，这一步最容易卡住。所以下面默认启用
#   国内镜像加速，并做"先探测再决定"的前置检查。
#
# 可用环境变量覆盖：
#   UV_PYTHON_INSTALL_MIRROR    Python 下载镜像前缀（默认走 gh-proxy）
#   UV_PYTHON_DOWNLOADS         never = 禁止自动下载（强制用系统 python）
#   UV_DEFAULT_INDEX            覆盖默认包索引（uv.lock 里大多是阿里云镜像）
#   UV_HTTP_TIMEOUT             uv 网络超时秒数（大 wheel 建议调大）
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

info() { printf '\033[36m[env]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[env]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[env]\033[0m %s\n' "$*" >&2; exit 1; }

WANT_PY="$(cat .python-version 2>/dev/null | tr -d '[:space:]')"
[[ -n "$WANT_PY" ]] || fail "找不到 .python-version"
info "项目要求 Python = $WANT_PY（pyproject: requires-python >=3.10.16,<3.11）"

# ---------------------------------------------------------------- 1. 确保 uv
if command -v uv >/dev/null 2>&1; then
    info "uv 已存在：$(uv --version)"
else
    info "安装 uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || fail "uv 安装后仍不在 PATH（检查 ~/.local/bin）"
    info "uv 就绪：$(uv --version)"
fi

# ---------------------------------------------------------------- 2. Python 下载镜像
# 默认启用国内镜像。若该镜像不通，UV_PYTHON_INSTALL_MIRROR= 置空可回退官方 GitHub。
if [[ -z "${UV_PYTHON_INSTALL_MIRROR+x}" ]]; then
    UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
    export UV_PYTHON_INSTALL_MIRROR
    info "已默认启用 Python 下载镜像：$UV_PYTHON_INSTALL_MIRROR"
    info "（如该镜像不可用，可 UV_PYTHON_INSTALL_MIRROR= bash setup_env.sh 回退官方源）"
else
    info "UV_PYTHON_INSTALL_MIRROR = ${UV_PYTHON_INSTALL_MIRROR:-<空，走官方 GitHub>}"
fi

# 网络超时：torch cu128 wheel 有数百 MB，默认超时可能不够
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

# ---------------------------------------------------------------- 3. 前置检查：Python 是否已有
info "检查本地是否已有 Python $WANT_PY ..."
if uv python find "$WANT_PY" >/dev/null 2>&1; then
    info "✅ 已找到：$(uv python find "$WANT_PY" 2>/dev/null)"
    info "无需下载 Python，uv sync 可直接进行"
else
    warn "本地没有 Python $WANT_PY，uv 需要联网下载"
    warn "下载源：$UV_PYTHON_INSTALL_MIRROR"
    info "（首次约 30-60 秒；失败则下面 uv sync 会报错并终止）"
fi

# ---------------------------------------------------------------- 4. uv sync
info "uv sync（pyproject.toml + uv.lock；包主要来自 mirrors.aliyun.com，torch 来自 download.pytorch.org）..."
info "首次约 10-30 分钟：其中 deepspeed 等 12 个包需要本地编译（依赖镜像里的 nvcc）"

# 多级回退。历史上出现过两类失败：
#   A) 为 aarch64 平台求解失败（packaging==25.0 只在 aliyun 上，而 uv 先撞上
#      pytorch-cu128 索引里同名但版本不符的包，默认不回退）→ 已用
#      [tool.uv] environments 收窄平台 + index-strategy=unsafe-best-match 解决
#   B) 锁文件与 pyproject 不一致导致重新解析 → 用 --frozen 直接按锁文件安装
# 无人值守场景下不能只赌一种方式，所以依次尝试。
SYNC_OK=0
run_sync() {
    local desc="$1"; shift
    info "尝试：uv sync $*"
    if uv sync "$@"; then
        ok_sync "$desc"
        return 0
    fi
    warn "失败：$desc"
    return 1
}
ok_sync() { printf '\033[32m[env]\033[0m ✅ %s\n' "$1"; SYNC_OK=1; }

run_sync "方式1：常规 uv sync" \
    || run_sync "方式2：--index-strategy unsafe-best-match（允许跨索引选版本）" --index-strategy unsafe-best-match \
    || run_sync "方式3：--frozen（完全按 uv.lock 安装，不重新解析）" --frozen \
    || run_sync "方式4：--frozen + unsafe-best-match" --frozen --index-strategy unsafe-best-match \
    || {
        warn "uv sync 全部方式失败。排查方向："
        warn "  1) Python 3.10 下载失败 → UV_PYTHON_INSTALL_MIRROR= bash setup_env.sh 回退官方源"
        warn "  2) 包索引不可达        → export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple"
        warn "  3) aarch64 求解问题    → 确认 pyproject.toml 里有 [tool.uv] environments"
        warn "  4) 编译失败（deepspeed 等）→ 确认镜像里有 nvcc：nvcc --version"
        warn "  完整日志见上方输出；start.sh 会把诊断上传到 HF"
        exit 1
    }

[[ "$SYNC_OK" == "1" ]] || fail "uv sync 未成功"

# ---------------------------------------------------------------- 5. 验证
info "验证环境 ..."
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$VENV_PY" ]] || fail "uv sync 后找不到 $VENV_PY"
info "venv python: $("$VENV_PY" -V 2>&1)"
"$VENV_PY" - <<'PY'
import sys
print(f"[env] python {sys.version.split()[0]} @ {sys.prefix}")
try:
    import torch
    print(f"[env] torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[env] ⚠️ CUDA 不可用！可能装成了 CPU 版 torch，或容器没挂上 GPU")
    else:
        print(f"[env] device_count={torch.cuda.device_count()}")
except Exception as e:
    print(f"[env] ⚠️ 无法导入 torch: {type(e).__name__}: {e}")

for m in ("flash_attn", "fla", "liger_kernel", "deepspeed", "transformers", "accelerate"):
    try:
        mod = __import__(m)
        print(f"[env] {m:16s} {getattr(mod, '__version__', '(n/a)')}")
    except Exception as e:
        print(f"[env] {m:16s} [不可用] {type(e).__name__}")
PY

info "完成。激活方式：source .venv/bin/activate"
