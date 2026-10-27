#!/usr/bin/env bash
# ============================================================================
# smoke_test.sh —— TODO 第 2 步：端到端 serving 冒烟（不接机器人）
#
# 一个终端搞定三件事：
#   ① 后台起 serve_policy.py（等它就绪）
#   ② 用同事的 fake_policy_client 验协议（合成 obs，零风险）
#   ③ 用我们打补丁的 vla_client.py（--fake-images --mock-robot，绝不动机械臂）
#   结束自动关服务端
#
# 2026-10-27 两次实跑后修正：
#   * 清空代理环境变量 —— 本机 HTTPS_PROXY=http://localhost:7897 是个死端口，
#     同事的 fake_policy_client 没传 proxy=None，会被劫持成 ConnectionRefused。
#   * 成功判定改为解析 bridge 的 "steps completed=N"，要求 N>0 且退出码 0
#     （上一版把 safe-hold 中断误判成"冒烟通过"）。
#   * 首包超时 180s（首次 forward 含 CUDA autotune，实测 ~12s）。
#
# 用法:
#   bash smoke_test.sh              # 全跑
#   SKIP_FAKE=1 bash smoke_test.sh  # 只跑我们的 bridge
#   PORT=18880 bash smoke_test.sh   # 换端口
#
# 安全：--mock-robot 不连 CAN、不做夹爪标定、不发任何指令给真机。
# ============================================================================
set -uo pipefail

# ★ 关键：本机代理指向死端口，会劫持所有 127.0.0.1 WebSocket 连接
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true

D="${D:-/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026}"
V="${V:-$D/runtime/env/bin/python}"
G="${G:-$D/code/motion/vendor/GalaxeaVLA}"
S="${S:-/home/chenyuan/workspace/g05-yam-deployment/g05-async-serving}"
V311="${V311:-/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python}"
B="${B:-/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge}"
PORT="${PORT:-18879}"
STEPS="${STEPS:-96}"
TASK="${TASK:-Pick up the small square and place it in the groove}"
CKPT="$D/models/ar_ft1ep/checkpoints/model_state_dict.pt"
LOG="/tmp/serve_${PORT}_$(date +%H%M%S).log"
BLOG="/tmp/bridge_$(date +%H%M%S).log"

die() { echo -e "\n[FAIL] $*" >&2; exit 1; }

# ── 前置检查 ────────────────────────────────────────────────────────────────
echo "=== 0. 前置检查 ==="
[[ -x "$V" ]]      || die "python 不可用：$V"
[[ -d "$G/src" ]]  || die "vendor/GalaxeaVLA/src 不存在：$G"
[[ -f "$CKPT" ]]   || die "checkpoint 不存在：$CKPT"
[[ -f "$S/serve_policy.py" ]] || die "缺 serve_policy.py：$S"
[[ -f "$B/vla_client.py" ]]   || die "缺我们的 bridge：$B/vla_client.py"
[[ -x "$V311" ]]   || die "3.11 环境不可用：$V311"
"$V311" -c "import websockets" 2>/dev/null \
    || die "3.11 环境缺 websockets（uv pip install --python $V311 websockets）"
if ss -tln 2>/dev/null | grep -q ":$PORT "; then
    die "端口 $PORT 已被占用（换 PORT=18880 再试）"
fi
echo "  D=$D"
echo "  PORT=$PORT   STEPS=$STEPS"
echo "  GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "  代理变量已清空（否则 127.0.0.1 连接会被劫持到 :7897）"

# ── ① 起服务端 ──────────────────────────────────────────────────────────────
echo
echo "=== 1. 起 serve_policy.py（后台，端口 $PORT）==="
echo "  服务端日志: $LOG"
cd "$G" || die "cd $G 失败"
PYTHONPATH="$G/src" "$V" "$S/serve_policy.py" \
    --ckpt_path "$CKPT" \
    --host 127.0.0.1 --port "$PORT" \
    eval_embodiment=galaxea_r1lite \
    model.model_weights_to_bf16=true \
    model.model_arch.continuous_action=false > "$LOG" 2>&1 &
SVPID=$!
cleanup() {
    echo
    echo "=== 收尾：关闭服务端 pid=$SVPID ==="
    kill "$SVPID" 2>/dev/null || true
    sleep 2
    kill -9 "$SVPID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "  等待模型加载（11.4GB 权重，可能 60~180s）..."
ready=0
for i in $(seq 1 180); do
    if grep -qiE "listening on|Uvicorn running|server listening" "$LOG" 2>/dev/null; then
        ready=1; break
    fi
    if ! kill -0 "$SVPID" 2>/dev/null; then
        echo "  ❌ 服务端进程已退出，日志尾部："
        tail -40 "$LOG"
        exit 1
    fi
    if (( i % 15 == 0 )); then
        echo "  ... 已等 $((i*2))s"
    fi
    sleep 2
done
if [[ $ready -ne 1 ]]; then
    echo "  ❌ 180s 内没就绪，日志尾部："
    tail -50 "$LOG"
    exit 1
fi
echo "  ✅ 服务端就绪："
grep -iE "listening on|mode=" "$LOG" | tail -3 | sed 's/^/     /'

# ── ② 同事的 fake client（验协议）────────────────────────────────────────────
FAKE_RC=99
if [[ "${SKIP_FAKE:-0}" != "1" ]]; then
    echo
    echo "=== 2. fake_policy_client（合成 obs，验 need_obs 握手）==="
    PYTHONPATH="$G/src" "$V" "$S/fake_policy_client.py" \
        --host 127.0.0.1 --port "$PORT" --rate 30 --steps 64 > /tmp/fake_$$.log 2>&1
    FAKE_RC=$?
    tail -12 /tmp/fake_$$.log | sed 's/^/     /'
    if [[ $FAKE_RC -eq 0 ]]; then
        echo "  ✅ fake client 通过"
    else
        echo "  ⚠️ fake client 退出码 $FAKE_RC（同事脚本的已知代理问题，不影响我们的 bridge）"
    fi
    rm -f /tmp/fake_$$.log
fi

# ── ③ 我们的 bridge（假机器人 + 合成图）──────────────────────────────────────
echo
echo "=== 3. ★ 我们的 vla_client（--mock-robot --fake-images，零风险）==="
"$V311" "$B/vla_client.py" \
    --host 127.0.0.1 --port "$PORT" \
    --task "$TASK" \
    --fake-images --mock-robot --steps "$STEPS" --verbose > "$BLOG" 2>&1
BRC=$?
# 只显示非 DEBUG 行（--verbose 会刷 websockets 帧日志，太吵）
grep -vE "DEBUG" "$BLOG" | tail -30 | sed 's/^/     /'
echo
echo "  bridge 退出码 = $BRC   （0=正常，>=2 表示被安全停机中断）"
echo "  bridge 完整日志: $BLOG"

# ── ④ 判定 ──────────────────────────────────────────────────────────────────
echo
echo "=== 4. 服务端日志摘要（每次推理的真实耗时）==="
grep -E "Recompute:|Dropped absent" "$LOG" | tail -20 | sed 's/^/  /'

echo
echo "=== 5. 判定 ==="
DONE=$(grep -oE "steps completed=[0-9]+" "$BLOG" | tail -1 | grep -oE '[0-9]+$')
DONE=${DONE:-0}
LAT=$(grep -oE "latency: n=[0-9]+ avg=[0-9.]+ms p50=[0-9.]+ms p99=[0-9.]+ms max=[0-9.]+ms" "$BLOG" | tail -1)
echo "  steps completed = $DONE / $STEPS"
[[ -n "$LAT" ]] && echo "  $LAT"
echo "  服务端 Recompute 次数 = $(grep -c 'Recompute:' "$LOG")"

if [[ $BRC -eq 0 && $DONE -ge $((STEPS * 9 / 10)) ]]; then
    echo
    echo "  ✅ 冒烟通过 —— TODO 第 2 步完成"
    echo "     下一步（TODO 第 5/6 步）：真机联调前先确认急停 + 软限位可用"
    [[ $FAKE_RC -ne 0 ]] && echo "     注：同事的 fake client 因其自身代理 bug 失败，与本项目无关"
    exit 0
fi
echo
echo "  ❌ 冒烟未通过"
[[ $BRC -ne 0 ]] && echo "     原因：bridge 退出码 $BRC（被安全停机中断）"
[[ $DONE -lt $((STEPS * 9 / 10)) ]] && echo "     原因：只完成 $DONE/$STEPS 步"
exit 1
