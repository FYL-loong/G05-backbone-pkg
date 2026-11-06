#!/usr/bin/env bash
# =============================================================================
# run_episode.sh —— 一次完整真机试验：预检 → 跑策略 → 自动精确复位
#
# 设计原则（都是今晚真机上踩出来的）：
#   1. 复位是流程的一部分，不是可选项 —— 手工记"跑完要复位"一定会漏，
#      漏了下次起跑姿态就不一致，实验不可比。
#   2. 无论策略跑成功、超时还是被看门狗停掉，**都必须复位**（用 trap）。
#   3. 复位走确定性脚本 home_robot.py（8.6°/s、收敛到 <1.5°、有卡死检测），
#      不让模型负责回位 —— 模型的 home 和 runtime.yaml 的全零不一样（实测差 24°/14°）。
#   4. 每次跑都留 CSV，出问题可以事后精确分析。
#
# 用法：
#     ./run_episode.sh                 # 默认 15 秒
#     DURATION=20 ./run_episode.sh     # 20 秒
#     RESET=0  ./run_episode.sh        # 跑完不复位（调试用）
#     TASK_ID=TASK-YAM-0013 ./run_episode.sh
#     DRY=1 ./run_episode.sh           # 只打印将要执行的命令
# =============================================================================
set -uo pipefail

# ── 可覆盖的配置 ─────────────────────────────────────────────────────────────
TASK_ID=${TASK_ID:-TASK-YAM-0006}
DURATION=${DURATION:-40}           # 宽松上限；通常 --stop-when-still 会提前结束
ARM_SPEED=${ARM_SPEED:-1.0}       # rad/s，57°/s。>1.5 会因 0.53s 延迟而震荡
LP_TAU=${LP_TAU:-0.10}            # 命令低通时间常数，压高频抖动
STILL=${STILL:-60}                # 模型输出连续多少步不变就正常结束（兜底，60=2秒）
RELEASE_STOP=${RELEASE_STOP:-45}  # 夹爪释放后再跑多少步就停（45=1.5秒）——主结束条件
HOME_SPEED=${HOME_SPEED:-0.12}    # 复位速度 rad/s，6.9°/s
HOME_WAIT=${HOME_WAIT:-3}         # 策略结束后等几秒再申请会话（守护进程需~1s判心跳失效）
RESET=${RESET:-1}                 # 1=跑完复位；0=不复位
DRY=${DRY:-0}
SERVER_PORT=${SERVER_PORT:-18879}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-/tmp/g05_runs}
mkdir -p "$OUTDIR" 2>/dev/null

# ── 路径 ─────────────────────────────────────────────────────────────────────
D=${D:-/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026}
B=${B:-/home/chenyuan/workspace/g05-yam-deployment/G05-backbone-pkg/serving/bridge}
V311=${V311:-/home/chenyuan/workspace/YAM/gello_software/.venv/bin/python}

CSV="$OUTDIR/ep_${STAMP}.csv"
LOG="$OUTDIR/ep_${STAMP}.log"
HOMELOG="$OUTDIR/home_${STAMP}.log"

say()  { printf '%s\n' "$*"; }
head1(){ printf '\n\033[1m=== %s ===\033[0m\n' "$*" 2>/dev/null || printf '\n=== %s ===\n' "$*"; }
fail() { printf '\n\033[31m[失败] %s\033[0m\n' "$*" 2>/dev/null || printf '\n[失败] %s\n' "$*"; }

# ── 0. 环境清理：代理会劫持 websockets（§已实测）────────────────────────────
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true

say "G0.5 × YAM 真机试验   ${STAMP}"
say "  任务        : $TASK_ID"
say "  时长上限    : ${DURATION}s（夹爪释放后 ${RELEASE_STOP} 步即停；兜底：连续 ${STILL} 步不动）"
say "  臂限速      : ${ARM_SPEED} rad/s"
say "  低通 tau    : ${LP_TAU}s"
say "  跑完复位    : $([ "$RESET" = "1" ] && echo "是 (${HOME_SPEED} rad/s)" || echo "否")"
say "  日志        : $LOG"
say "  CSV         : $CSV"

# ── 1. 依赖检查 ──────────────────────────────────────────────────────────────
head1 "1/5 依赖检查"
rc=0
[ -x "$V311" ] || { fail "找不到 python: $V311"; rc=1; }
for f in vla_client.py home_robot.py; do
    if [ -f "$B/$f" ]; then say "  ✓ $f"; else fail "缺文件: $B/$f"; rc=1; fi
done
# 关键参数是否存在于当前 vla_client.py（防止拖着旧文件白跑一轮）
if ! "$V311" "$B/vla_client.py" --help 2>/dev/null | grep -q "cmd-lowpass-tau"; then
    fail "vla_client.py 是旧版（没有 --cmd-lowpass-tau），请先拖新文件（md5 25038294814fac6cdd6296ca3c257366）"
    rc=1
fi
if [ $rc -ne 0 ]; then say ""; say "依赖不全，退出。"; exit 1; fi
say "  ✓ vla_client.py 含 --cmd-lowpass-tau"

# ── 2. 服务端 ────────────────────────────────────────────────────────────────
head1 "2/5 服务端 (port $SERVER_PORT)"
if ss -tln 2>/dev/null | grep -q ":${SERVER_PORT} "; then
    say "  ✓ 已在监听"
else
    fail "服务端没在跑。先起 serve_policy.py（见 RUNBOOK），再重跑本脚本。"
    exit 1
fi

# ── 3. 机器人预检（只读）─────────────────────────────────────────────────────
head1 "3/5 机器人预检"
CHK=$("$V311" "$B/vla_client.py" --task x --check-robot 2>&1)
CHK_RC=$?
say "$CHK" | grep -E "safety.state|hardware_health|ok=|✅|❌|state" | head -8
if [ $CHK_RC -ne 0 ]; then
    fail "预检未通过（退出码 $CHK_RC）。常见原因：state 不是 safe_hold / 有其它客户端持有会话 / 硬件不健康。"
    say ""
    say "如果刚跑过一轮，等 2 秒再试（守护进程需约 1s 判定心跳失效）。"
    exit $CHK_RC
fi
if printf '%s' "$CHK" | grep -q "❌"; then
    fail "预检输出里有 ❌，退出。"
    exit 6
fi
say "  ✓ 预检通过"

# ── 4. 跑策略 ────────────────────────────────────────────────────────────────
POLICY_RC=0
if [ "$DRY" = "1" ]; then
    head1 "4/5 策略（DRY，不执行）"
    say "  \"$V311\" \"$B/vla_client.py\" --host 127.0.0.1 --port $SERVER_PORT \\"
    say "      --task-id $TASK_ID --robot-backend zmq --command-timing legacy_30hz \\"
    say "      --camera-config \"$D/live_configs/robot.yaml\" \\"
    say "      --motion-src \"$D/code/motion/src\" \\"
    say "      --arm-max-speed-rad-s $ARM_SPEED --cmd-lowpass-tau $LP_TAU \\"
    say "      --stop-after-release $RELEASE_STOP --stop-when-still $STILL \\"
    say "      --log-csv \"$CSV\" --duration $DURATION --verbose"
else
    head1 "4/5 跑策略（上限 ${DURATION}s，稳定即停）—— 急停在手边"
    "$V311" "$B/vla_client.py" \
        --host 127.0.0.1 --port "$SERVER_PORT" \
        --task-id "$TASK_ID" \
        --robot-backend zmq --command-timing legacy_30hz \
        --camera-config "$D/live_configs/robot.yaml" \
        --motion-src    "$D/code/motion/src" \
        --arm-max-speed-rad-s "$ARM_SPEED" \
        --cmd-lowpass-tau "$LP_TAU" \
        --stop-after-release "$RELEASE_STOP" \
        --stop-when-still "$STILL" \
        --log-csv "$CSV" \
        --duration "$DURATION" --verbose 2>&1 | tee "$LOG" | tail -25
    POLICY_RC=${PIPESTATUS[0]}
    say ""
    say "  策略退出码 = $POLICY_RC"
    case "$POLICY_RC" in
        0) say "  ✓ 正常跑完" ;;
        2) say "  ⚠ 被安全停机制中断（守护进程 unhealthy）" ;;
        3) say "  ⚠ 服务端错误" ;;
        4) say "  ⚠ 服务端超时" ;;
        5) say "  ⚠ 相机故障" ;;
        6) say "  ⚠ 预检失败" ;;
        7) say "  ⚠ 模型输出异常 —— 看门狗触发，已安全停止（这是好事）" ;;
        *) say "  ⚠ 未知退出码" ;;
    esac
fi

# ── 5. 复位（无论如何都做）───────────────────────────────────────────────────
if [ "$RESET" = "1" ]; then
    head1 "5/5 精确复位到全零 home"
    say "  等 ${HOME_WAIT}s 让守护进程释放上一个会话 ..."
    sleep "$HOME_WAIT"
    if [ "$DRY" = "1" ]; then
        say "  \"$V311\" \"$B/home_robot.py\" --speed $HOME_SPEED"
    else
        "$V311" "$B/home_robot.py" --speed "$HOME_SPEED" 2>&1 | tee "$HOMELOG" | tail -18
        HOME_RC=${PIPESTATUS[0]}
        say ""
        if [ "$HOME_RC" -eq 0 ]; then
            say "  ✓ 复位完成"
        else
            fail "复位未完全收敛（退出码 $HOME_RC）—— 看 $HOMELOG"
            say "  可重试: \"$V311\" \"$B/home_robot.py\" --speed $HOME_SPEED --max-seconds 180"
        fi
    fi
else
    head1 "5/5 跳过复位（RESET=0）"
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────────
head1 "汇总"
say "  策略退出码 : $POLICY_RC"
say "  日志       : $LOG"
say "  CSV        : $CSV"
say "  复位日志   : $([ "$RESET" = "1" ] && echo "$HOMELOG" || echo "（未复位）")"
say ""
say "  分析本次动作:"
say "    python3 \"$B/analyze_bridge_csv.py\" \"$CSV\""
say ""
if [ "$POLICY_RC" -eq 7 ]; then
    say "  ⚠ 本次是模型输出异常被看门狗停掉 —— 说明这次跑到了策略崩坏区。"
    say "    这不会损坏硬件（这正是看门狗的目的）。可缩短 DURATION 重试。"
fi
[ "$POLICY_RC" -eq 0 ] || exit "$POLICY_RC"
exit 0
