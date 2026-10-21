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

# ---------------------------------------------------------------- 4.5 NCCL 升级（可选）
# 为什么需要：实测 GPU 是 sm_120（Blackwell 入门专业卡，96GB），而 uv.lock 锁的
#   nvidia-nccl-cu12 == 2.26.2（2025-02）【不认识 sm_120】→ 4 卡连最基本的
#   broadcast 都报 `NCCL enqueue.cc Cuda failure 1 'invalid argument'`。
#   （1 卡可用，因为单卡不需要跨卡 kernel。）
#
# ⚠️ 关键认识：.venv 是【独立用 uv 装的】，它的 torch/NCCL 不来自容器镜像 ——
#    所以要改的是 venv 里的 NCCL，而不是换镜像。
#
# 做法：uv sync 完成后，直接在 venv 里 pip 升级 nvidia-nccl-cu12。
#   ★ 不动 pyproject.toml、不动 uv.lock、不动 torch —— 零配置风险
#     （曾尝试给 pyproject 注入 [tool.uv] override-dependencies，但注入后
#       TOML 校验失败；而且 pyproject.toml 里原本就有一句注释警告过
#       "故意不写进 [tool.uv]，避免万一键名不被支持导致 manifest 直接无效"。）
#   装错了也不怕：uv sync 会按 lock 纠正回来。
#
# 开关：G05_NCCL_UPGRADE=1（默认）| 0 关闭；G05_NCCL_VER 指定版本。
NCCL_VER="${G05_NCCL_VER:-2.29.7}"
if [[ "${G05_NCCL_UPGRADE:-1}" == "1" ]]; then
    VENV_PY_PRE="$PROJECT_ROOT/.venv/bin/python"
    NCCL_LIB_DIR="$PROJECT_ROOT/.venv/lib/python3.10/site-packages/nvidia/nccl/lib"

    info "把 venv 里的 nvidia-nccl-cu12 升级到 $NCCL_VER（支持 sm_120；不动 torch）"
    # ⚠️ 踩过的坑（两次失败都出在这里）：
    #   1) `pip install nvidia-nccl-cu12==2.29.7` 被拒 —— torch 2.7.1 的元数据把它
    #      【精确锁定】为 ==2.26.2。
    #   2) **`.venv/bin/python -m pip` 直接报 `No module named pip`** ——
    #      uv 创建的 venv 默认【不安装 pip】。所以任何 `python -m pip` 都跑不了。
    #   正确做法：用 `uv pip`（uv 能管理自己创建的 venv，无需 pip），
    #             并加 --no-deps 绕开 torch 的精确版本锁。
    NCCL_OK=0
    NCCL_LOG=/tmp/nccl_upgrade.log
    : > "$NCCL_LOG"

    _try_nccl() {
        local desc="$1"; shift
        info "尝试：$desc"
        : > "$NCCL_LOG"          # 每次清空，避免打印出上一次的旧报错
        if "$@" >>"$NCCL_LOG" 2>&1; then
            NCCL_OK=1
            return 0
        fi
        warn "失败：$desc（输出末尾 8 行）"
        tail -8 "$NCCL_LOG" | sed 's/^/       /' || true
        return 1
    }

    # 方式1：uv pip —— 首选（uv 能管理自己创建的 venv，不需要 pip）
    _try_nccl "uv pip install --no-deps nvidia-nccl-cu12==$NCCL_VER（--python .venv）" \
        uv pip install --python "$VENV_PY_PRE" --no-deps \
        "nvidia-nccl-cu12==$NCCL_VER" \
      || _try_nccl "uv pip install --no-deps --reinstall ..." \
        uv pip install --python "$VENV_PY_PRE" --no-deps --reinstall \
        "nvidia-nccl-cu12==$NCCL_VER"

    if [[ "$NCCL_OK" == "1" ]]; then
        ok_sync "nvidia-nccl-cu12==$NCCL_VER（uv pip）"
    else
        # 方式2：--target 装到临时目录，只覆盖 .so（完全不碰 site-packages 元数据）
        info "尝试兜底方式：--target 到临时目录，只取 libnccl.so.* 覆盖"
        TMPNCCL="$(mktemp -d)"
        if uv pip install --python "$VENV_PY_PRE" --no-deps --target "$TMPNCCL" \
                "nvidia-nccl-cu12==$NCCL_VER" >>"$NCCL_LOG" 2>&1; then
            SRCDIR="$TMPNCCL/nvidia/nccl/lib"
            if [[ -d "$SRCDIR" ]]; then
                mkdir -p "$NCCL_LIB_DIR"
                # ⚠️ glob 必须是 libnccl.so.*（不能是 libnccl.so*），
                #    否则会把刚做的 .g05bak 备份也一起拷进去。
                if [[ -f "$NCCL_LIB_DIR/libnccl.so.2" && ! -f "$NCCL_LIB_DIR/libnccl.so.2.g05bak" ]]; then
                    cp -a "$NCCL_LIB_DIR/libnccl.so.2" "$NCCL_LIB_DIR/libnccl.so.2.g05bak" || true
                fi
                for _f in "$SRCDIR"/libnccl.so.*; do
                    [[ -e "$_f" ]] || continue
                    cp -a "$_f" "$NCCL_LIB_DIR/" || true
                    info "  覆盖 $(basename "$_f")"
                done
                NCCL_OK=1
                ok_sync "覆盖 $NCCL_LIB_DIR/libnccl.so.* 成功（兜底方式）"
            else
                warn "临时目录里没有 nvidia/nccl/lib：$(ls -R "$TMPNCCL" 2>/dev/null | head -10)"
            fi
        else
            warn "兜底方式也失败，输出末尾："
            tail -8 "$NCCL_LOG" | sed 's/^/       /' || true
        fi
        rm -rf "$TMPNCCL" 2>/dev/null || true
    fi

    if [[ "$NCCL_OK" != "1" ]]; then
        warn "全部方式都失败 —— NCCL 仍是旧版，4 卡会失败。完整日志：$NCCL_LOG"
    fi

    # ★ 验证装了哪个版本，以及磁盘上有哪几份 libnccl.so.2。
    #   ⚠️ 判据必须是 ldd（下面那段），不能用 torch.cuda.nccl.version() ——
    #      后者是 torch 编译期常量，与运行时实际加载的库无关。
    info "查看已安装的 NCCL 与磁盘上所有 libnccl.so.2："
    "$VENV_PY_PRE" "$PROJECT_ROOT/tools/which_nccl.py" 2>&1 | sed 's/^/  /' || true

    # 硬判定：用 ldd 看 torch 实际链接到哪份 libnccl.so.2。
    # ⚠️ 不要用 torch.cuda.nccl.version() 判定 —— 它是 torch 编译期常量，
    #    与运行时实际加载的库无关（实测：升级后它仍报旧版，而 ldd 已指向新版）。
    info "判定 torch 实际链接的 NCCL（用 ldd，不用 nccl.version()）："
    "$VENV_PY_PRE" - <<'PYLDD' || true
import os, re, shutil, subprocess
try:
    import torch
    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so")
    if not os.path.exists(lib):
        print("[ldd] 找不到 libtorch_cuda.so")
    elif not shutil.which("ldd"):
        print("[ldd] 环境里没有 ldd 命令")
    else:
        out = subprocess.run(["ldd", lib], capture_output=True, text=True).stdout or ""
        hits = [ln.strip() for ln in out.splitlines() if "nccl" in ln.lower()]
        for h in hits:
            print("[ldd] " + h)
        if not hits:
            print("[ldd] libtorch_cuda.so 没有 nccl 条目")
        else:
            tgt = hits[0].split("=>")[-1].strip().split(" ")[0]
            if "nvidia/nccl" in tgt and ".venv" in tgt:
                print(f"[ldd] ✅ 链接到 venv 的 NCCL：{tgt}")
            else:
                print(f"[ldd] ⚠️ 链接到其他位置：{tgt}")
    try:
        print(f"[ldd] （参考，不可作为判据）torch 编译期常量 nccl="
              f"{torch.cuda.nccl.version()}")
    except Exception as e:
        print(f"[ldd] （参考）读编译期常量失败: {type(e).__name__}")
except Exception as e:
    print(f"[ldd] 失败: {type(e).__name__}: {e}")
PYLDD
else
    info "G05_NCCL_UPGRADE=0 —— 不改 NCCL（sm_120 上 4 卡会失败）"
fi

# ---------------------------------------------------------------- 4.7 cv2 的 libGL 依赖
# 为什么需要：训练链路不 import cv2，所以这个问题一直没暴露；
#   但【离线评测】会走 scripts/eval_open_loop.py:45 → g05/utils/eval/visualize.py:18
#   → `import cv2`，而依赖里的 opencv-python==4.10.0.84 需要系统 libGL.so.1，
#   pytorch-ngc 镜像默认不带 → ImportError: libGL.so.1: cannot open shared object file。
#
# 两种修法，优先 apt（不动 Python 依赖，uv.lock 保持原样）：
#   ① apt install libgl1
#   ② 换 opencv-python-headless（不需要 libGL），版本与 pyproject 对齐为 4.10.0.84
# 判据用【真的 import 一次】，不信 ldconfig（ldconfig 常在 PATH 外）。
CV_PY="$PROJECT_ROOT/.venv/bin/python"
if [[ -x "$CV_PY" ]] && ! "$CV_PY" -c "import cv2" >/dev/null 2>&1; then
    warn "venv 里 import cv2 失败（opencv 需要系统 libGL，仅【评测】用到，训练不受影响）"
    if apt-get update -qq >/tmp/apt_libgl.log 2>&1 \
       && apt-get install -y -qq libgl1 >>/tmp/apt_libgl.log 2>&1; then
        : # 下面用 import 复验，不在这里断言
    else
        warn "apt 安装 libgl1 失败（见 /tmp/apt_libgl.log），改走 headless 方案"
    fi
    if "$CV_PY" -c "import cv2" >/dev/null 2>&1; then
        ok_sync "cv2 可用（apt libgl1，$( "$CV_PY" -c 'import cv2;print(cv2.__version__)' 2>/dev/null )）"
    else
        info "替换 opencv-python → opencv-python-headless==4.10.0.84（后者不依赖 libGL）"
        uv pip uninstall --python "$CV_PY" opencv-python >>/tmp/cv_headless.log 2>&1 || true
        if uv pip install --python "$CV_PY" "opencv-python-headless==4.10.0.84" \
                >>/tmp/cv_headless.log 2>&1; then
            if "$CV_PY" -c "import cv2" >/dev/null 2>&1; then
                ok_sync "cv2 可用（opencv-python-headless，$( "$CV_PY" -c 'import cv2;print(cv2.__version__)' 2>/dev/null )）"
            else
                warn "装了 headless 但 import cv2 仍失败 —— 评测会起不来，见 /tmp/cv_headless.log"
            fi
        else
            warn "opencv-python-headless 安装失败 —— 评测会起不来，见 /tmp/cv_headless.log"
        fi
    fi
else
    if [[ -x "$CV_PY" ]]; then
        ok_sync "cv2 可用（$("$CV_PY" -c 'import cv2;print(cv2.__version__)' 2>/dev/null)）"
    fi
fi

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
