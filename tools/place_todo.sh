#!/usr/bin/env bash
# ============================================================================
# place_todo.sh —— 把更新后的 TODO 放到张恩同原 TODO 的位置
#
# 用法:  bash place_todo.sh [新TODO路径] [他那份 serving 仓库路径]
#        默认 ~/TODO.md  和  ~/workspace/g05-yam-deployment/g05-async-serving
#
# 安全规则（目标已存在时）：
#   ① md5 == 张恩同原稿(8de593ce…)  → 备份成 TODO_原始_张恩同_20261026.md 再写
#   ② md5 == 新 TODO               → 已是最新，什么都不做，退出 0（幂等）
#   ③ 内容含本台账标题              → 是我们之前放的旧版 → 备份成 .bak.<时间> 再写
#   ④ 其它情况                      → 拒绝覆盖，退出 1（交给人工判断）
# ============================================================================
set -uo pipefail
NEW="${1:-$HOME/TODO.md}"
SERV="${2:-$HOME/workspace/g05-yam-deployment/g05-async-serving}"
REPO="${REPO:-$HOME/workspace/g05-yam-deployment/G05-backbone-pkg}"

ORIG_MD5="8de593ce4f1e54208e4be35e549fbc45"          # 张恩同原稿（225 行）
MARKER="# TODO —— G0.5 VLA × I2RT YAM 真机部署"      # 我们台账的特征首行
BAK="$SERV/TODO_原始_张恩同_20261026.md"
DST="$SERV/TODO.md"

echo "  新 TODO : $NEW"
echo "  目标目录: $SERV"
echo

[ -f "$NEW" ]  || { echo "  [FAIL] 找不到新 TODO: $NEW"; exit 1; }
[ -d "$SERV" ] || { echo "  [FAIL] 目录不存在: $SERV"; exit 1; }

NEWMD5=$(md5sum "$NEW" | cut -d' ' -f1)

echo "=== ① 处理已存在的目标 ==="
if [ ! -f "$DST" ]; then
    echo "  [SKIP] 目标不存在，直接写入"
    action=write
else
    CUR=$(md5sum "$DST" | cut -d' ' -f1)
    if [ "$CUR" = "$NEWMD5" ]; then
        echo "  ✅ 目标已是这一版（md5=$CUR），无需改动"
        action=skip
    elif [ "$CUR" = "$ORIG_MD5" ]; then
        if [ -f "$BAK" ]; then
            echo "  [SKIP] 原稿备份已存在: $BAK"
        else
            cp -a "$DST" "$BAK"
            echo "  ✅ 已备份原稿: $BAK"
        fi
        action=write
    elif head -1 "$DST" | grep -qF "$MARKER"; then
        B2="$DST.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "$DST" "$B2"
        echo "  ✅ 是我们之前放的旧版，已备份: $B2"
        action=write
    else
        echo "  [FAIL] $DST 既不是张恩同原稿($ORIG_MD5)，也不是我们的台账。"
        echo "         当前 md5=$CUR"
        echo "         首行: $(head -1 "$DST")"
        echo "         拒绝覆盖 —— 请人工确认后手动处理。"
        exit 1
    fi
fi

echo
if [ "$action" = "skip" ]; then
    echo "=== ② 无需写入 ==="
else
    echo "=== ② 写入新 TODO ==="
    cp -f "$NEW" "$DST"
    GOT=$(md5sum "$DST" | cut -d' ' -f1)
    if [ "$GOT" != "$NEWMD5" ]; then
        echo "  [FAIL] 写入后 md5 不符（期望 $NEWMD5 实际 $GOT）"; exit 1
    fi
    echo "  ✅ $DST   md5=$GOT"
fi

echo
echo "=== ③ 结果 ==="
ls -la "$SERV"/TODO*.md 2>/dev/null | sed 's/^/    /'
echo
GOT=$(md5sum "$DST" | cut -d' ' -f1)
echo "  位置与版本："
echo "    $SERV/TODO.md          md5=$GOT"
[ -f "$BAK" ] && echo "    $BAK   md5=$(md5sum "$BAK" | cut -d' ' -f1)"
if [ -f "$REPO/TODO.md" ]; then
    r=$(md5sum "$REPO/TODO.md" | cut -d' ' -f1)
    echo "    $REPO/TODO.md   md5=$r"
    if [ "$r" = "$GOT" ]; then
        echo
        echo "  ✅ 与仓库版本一致，收工"
    else
        echo
        echo "  ⚠️ 与仓库版本不一致 —— 重新拖一次 TODO.md 到仓库根"
    fi
fi
exit 0
