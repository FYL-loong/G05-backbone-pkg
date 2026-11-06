#!/usr/bin/env bash
# ============================================================================
# build_deploy_pkg.sh —— 把【我们微调的 G0.5】打包成 yam_motion 能加载的部署包
#
# 对应 g05-async-serving/TODO.md 的「第 1 步」前置：需要一个 run dir 让
# load_config_from_run_dir() 能定位 .hydra/config.yaml + dataset_stats.json +
# action_tokenizer.pt + hf_processor。
#
# 只做文件操作：不加载模型 / 不占用 GPU / 不打开相机 / 不给机器人发指令。
#
# 用法:
#   export HF_TOKEN=hf_xxx
#   bash build_deploy_pkg.sh                     # 用默认路径
#   DEST=/mnt/sda/xxx/ours bash build_deploy_pkg.sh
#
# 可覆盖的环境变量:
#   SRC        取骨架的现成部署包（默认 ar_newcodec5000）
#   DEST       输出目录（默认 /mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026）
#   HF_HOME    HF 缓存目录（默认 /mnt/sda/chenyuan/hf_home，避免撑爆系统盘）
#   HF_CLI     指定 hf 可执行文件（默认自动探测）
#   REPO       我们的模型仓库（默认 Zenith6666/yam-backbone）
#   CODEC_REPO codec 仓库（默认 Zenith6666/yam-actioncodec）
#   PORT       服务端端口（默认 18879；8765 已被别人占用）
#   TASK       任务文本（必须与训练数据完全一致，含大小写）
# ============================================================================
set -euo pipefail

SRC="${SRC:-/mnt/sda/chenyuan/motion/g05_ar_newcodec5000_20261016}"
DEST="${DEST:-/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026}"
HFDIR="${HFDIR:-/mnt/sda/chenyuan/hf_dl_ft1ep}"
export HF_HOME="${HF_HOME:-/mnt/sda/chenyuan/hf_home}"
REPO="${REPO:-Zenith6666/yam-backbone}"
CODEC_REPO="${CODEC_REPO:-Zenith6666/yam-actioncodec}"
PORT="${PORT:-18879}"
# ★ 大写 P —— 与我们训练数据的 task 文本完全一致
TASK="${TASK:-Pick up the small square and place it in the groove}"

export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------- 网络通路
# 本机实测（2026-10-27）：
#   直连 huggingface.co        → 超时
#   走 clash 7890 / 7891 代理   → error:0A000126 SSL routines::unexpected eof
#   直连 hf-mirror.com         → ✅ 可用
# ⇒ 默认清空代理变量并指向 hf-mirror；如需回到官方站，用 HF_ENDPOINT=https://huggingface.co KEEP_PROXY=1
if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true
fi
export NO_PROXY="${NO_PROXY:-}${NO_PROXY:+,}huggingface.co,hf-mirror.com,cdn-lfs.huggingface.co,cdn-lfs.hf.co"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

fail() { echo -e "\n[FAIL] $*" >&2; exit 1; }
info() { echo -e "\n[INFO] $*"; }
ok()   { echo "[ OK ] $*"; }

info "0a 网络通路"
echo "  HF_ENDPOINT = $HF_ENDPOINT"
echo "  NO_PROXY    = $NO_PROXY"
echo "  HTTPS_PROXY = ${HTTPS_PROXY:-<已清空>}"
echo "  KEEP_PROXY  = ${KEEP_PROXY:-0}"

# ---------------------------------------------------------------- 0 前置检查
info "0/5 前置检查"
[[ -n "${HF_TOKEN:-}" ]] || fail "请先： export HF_TOKEN=hf_你的token"
[[ -d "$SRC" ]] || fail "找不到骨架包 SRC=$SRC"
[[ -f "$SRC/launch.sh" ]] || fail "$SRC/launch.sh 不存在"
# DEST 处理：允许"上次中断"的目录继续跑（网络抖动时不用删重来）
RESUME=0
if [[ -e "$DEST" ]]; then
    if [[ -f "$DEST/.build_partial" ]]; then
        RESUME=1
        echo "[WARN] 检测到上次中断的构建（存在 .build_partial），继续执行，骨架不重复复制"
    else
        fail "$DEST 已存在且不是中断的构建。确认无用后： rm -rf $DEST"
    fi
fi
python3 -c "import yaml" 2>/dev/null || fail "python3 缺 pyyaml"

# ---------------------------------------------------------------- HF 下载器
# ⚠️ 不用 `hf` CLI：这台机器上 runtime/env/bin/hf 的 shebang 指向 compute 机的
#    /mnt/pfs/.../bin/python，搬到 yam1 后是 bad interpreter。
#    直接用 python API（snapshot_download），版本无关且支持 HF_ENDPOINT。
PY="${PY:-$SRC/runtime/env/bin/python}"
[[ -x "$PY" ]] || fail "找不到 python：$PY（可用 PY=<path> 覆盖）"
"$PY" -c "import huggingface_hub" 2>/dev/null || fail "$PY 里没有 huggingface_hub"
ok "python: $PY"

hfd() {  # $1=repo  $2=local_dir  $3..=allow_patterns
    local repo="$1" dest="$2"; shift 2
    "$PY" - "$repo" "$dest" "$@" <<'PYDL' || return 1
import os, sys
from huggingface_hub import snapshot_download
repo, dest, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"  [hf] {repo} -> {dest}")
print(f"  [hf] patterns={pats}")
print(f"  [hf] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}", flush=True)
snapshot_download(repo_id=repo, repo_type="model", allow_patterns=pats,
                  local_dir=dest)
print("  [hf] done", flush=True)
PYDL
}

# 连通性预检：能拿到 dataset_stats.json 的元信息就算通
info "HF 连通性预检（$REPO）"
"$PY" - "$REPO" <<'PYCHK' || fail "HF 不可达 —— 换 HF_ENDPOINT 或改用本地下载+scp"
import sys
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info(sys.argv[1], files_metadata=False)
print(f"  ✅ 可达：{info.id}  (private={info.private})")
PYCHK
ok "HF 可达"

df -h /mnt/sda | tail -1
AVAIL_G=$(df -BG --output=avail /mnt/sda | tail -1 | tr -dc '0-9')
[[ "${AVAIL_G:-0}" -ge 40 ]] || fail "/mnt/sda 只剩 ${AVAIL_G}G，需要 >=40G"
mkdir -p "$HF_HOME" "$HFDIR"
ok "前置检查通过（/mnt/sda 可用 ${AVAIL_G}G）"

# ---------------------------------------------------------------- 1 复制骨架
if [[ $RESUME -eq 0 ]]; then
    info "1/5 复制部署骨架（code / runtime / live_configs / 元数据）"
    mkdir -p "$DEST"/models/ar_ft1ep/checkpoints "$DEST"/models/ar_ft1ep/.hydra \
             "$DEST"/logs "$DEST"/reports
    cp -a "$SRC/code"              "$DEST/code"
    cp -a "$SRC/runtime"           "$DEST/runtime"
    cp -a "$SRC/live_configs"      "$DEST/live_configs"
    cp -a "$SRC/launch.sh"         "$DEST/launch.sh"
    cp -a "$SRC/verify_offline.py" "$DEST/verify_offline.py"
    for f in motion_contract.json motion_package.json motion_experiment.json \
             motion_versions.json motion_split.json; do
        [[ -f "$SRC/models/ar_newcodec5000/$f" ]] && \
            cp -a "$SRC/models/ar_newcodec5000/$f" "$DEST/models/ar_ft1ep/$f"
    done
    # 官方 processor：我们的 HF 仓库没有 hf_processor/，用骨架包自带那份
    # （同为 qwen3_5_2b_base_processor，词表一致）
    cp -a "$SRC/models/ar_newcodec5000/hf_processor" "$DEST/models/ar_ft1ep/hf_processor"
    touch "$DEST/.build_partial"
    ok "骨架复制完成，当前大小：$(du -sh "$DEST" | cut -f1)"
else
    # 中断续跑：只补缺失的骨架文件
    info "1/5（续跑）检查骨架完整性"
    mkdir -p "$DEST"/models/ar_ft1ep/checkpoints "$DEST"/models/ar_ft1ep/.hydra \
             "$DEST"/logs "$DEST"/reports
    for d in code runtime live_configs; do
        [[ -d "$DEST/$d" ]] || cp -a "$SRC/$d" "$DEST/$d"
    done
    for f in launch.sh verify_offline.py; do
        [[ -f "$DEST/$f" ]] || cp -a "$SRC/$f" "$DEST/$f"
    done
    for f in motion_contract.json motion_package.json motion_experiment.json \
             motion_versions.json motion_split.json; do
        [[ -f "$DEST/models/ar_ft1ep/$f" ]] || \
            { [[ -f "$SRC/models/ar_newcodec5000/$f" ]] && \
              cp -a "$SRC/models/ar_newcodec5000/$f" "$DEST/models/ar_ft1ep/$f"; }
    done
    [[ -d "$DEST/models/ar_ft1ep/hf_processor" ]] || \
        cp -a "$SRC/models/ar_newcodec5000/hf_processor" "$DEST/models/ar_ft1ep/hf_processor"
    ok "骨架完整（续跑）"
fi

# ---------------------------------------------------------------- 2 下载产物
# 两种模式：
#   在线（默认）  —— 用 python snapshot_download 从 HF（hf-mirror）拉 11.6G
#   离线 LOCAL_HF —— 产物已在本地目录（先在自己电脑下好、再拖到机器上），
#                    完全不联网。目录布局（扁平）：
#                      $LOCAL_HF/model_state_dict.pt
#                      $LOCAL_HF/dataset_stats.json
#                      $LOCAL_HF/hydra_config.yaml
#                      $LOCAL_HF/action_tokenizer.pt
#                      $LOCAL_HF/yam_stats.json
#                    （不含 hf_processor：我们的 HF 仓库没有，用骨架包自带的）
if [[ -n "${LOCAL_HF:-}" ]]; then
    info "2/5 使用本地产物目录（LOCAL_HF=$LOCAL_HF），不走网络"
    [[ -d "$LOCAL_HF" ]] || fail "LOCAL_HF 不存在：$LOCAL_HF"
    # 注意：不含 hf_processor —— 我们的 HF 仓库没有它，官方 processor 用骨架包里的
    need=(model_state_dict.pt dataset_stats.json hydra_config.yaml \
          action_tokenizer.pt yam_stats.json)
    for x in "${need[@]}"; do
        [[ -e "$LOCAL_HF/$x" ]] || fail "缺少 $LOCAL_HF/$x"
    done
    ok "本地产物齐全（5 项；hf_processor 用骨架包自带那份）"
    M_TMP="$LOCAL_HF"
else
    info "2/5 从 HF 拉我们的训练产物（约 10.8 GiB，慢一点）"
    # 注意：不下载 hf_processor —— 我们的 HF 仓库里【没有】这个目录
    #      （根下只有 checkpoints/、eval/、logs/ 三个目录 + 6 个文件）。
    #      官方 processor 由第 1 步从骨架包复制。
    hfd "$REPO" "$HFDIR/$REPO" \
        "model_state_dict.pt" "dataset_stats.json" "hydra_config.yaml" \
        "action_tokenizer.pt" \
        || fail "我们的模型产物下载失败"
    hfd "$CODEC_REPO" "$HFDIR/$CODEC_REPO" "yam_stats.json" \
        || fail "codec stats（yam_stats.json）下载失败"
    M_TMP="$HFDIR/$REPO"
fi

M="$DEST/models/ar_ft1ep"
if [[ -n "${LOCAL_HF:-}" ]]; then
    # 离线：复制（保留源文件，便于重复构建）
    cp -f  "$M_TMP/model_state_dict.pt" "$M/checkpoints/model_state_dict.pt"
    cp -f  "$M_TMP/dataset_stats.json"  "$M/dataset_stats.json"
    cp -f  "$M_TMP/action_tokenizer.pt" "$M/action_tokenizer.pt"
    cp -f  "$M_TMP/hydra_config.yaml"   "$M/.hydra/config.yaml"
    cp -f  "$M_TMP/yam_stats.json"      "$M/ar_codec_stats.json"
    # hf_processor 已在第 1 步从骨架包复制，这里不动
else
    mv -f "$M_TMP/model_state_dict.pt"  "$M/checkpoints/model_state_dict.pt"
    mv -f "$M_TMP/dataset_stats.json"   "$M/dataset_stats.json"
    mv -f "$M_TMP/action_tokenizer.pt"  "$M/action_tokenizer.pt"
    mv -f "$M_TMP/hydra_config.yaml"    "$M/.hydra/config.yaml"
    # hf_processor 已在第 1 步从骨架包复制，这里不动
    mv -f "$HFDIR/$CODEC_REPO/yam_stats.json" "$M/ar_codec_stats.json"
    rm -rf "$HFDIR"
fi

check_size() {  # $1=path $2=expected bytes
    local got; got=$(stat -c%s "$1")
    if [[ "$got" == "$2" ]]; then ok "  $(basename "$1")  $got 字节 ✓"
    else fail "$1 大小 $got != 期望 $2"; fi
}
check_size "$M/checkpoints/model_state_dict.pt" 11440527634
check_size "$M/dataset_stats.json"                226311
check_size "$M/action_tokenizer.pt"             169709565
check_size "$M/.hydra/config.yaml"                 18337
check_size "$M/ar_codec_stats.json"               226311
NHP=$(find "$M/hf_processor" -type f | wc -l)
[[ "$NHP" -eq 8 ]] || fail "hf_processor 文件数=$NHP，期望 8"
ok "hf_processor 8 个文件 ✓"
ok "产物下载 + 校验通过：$(du -sh "$M" | cut -f1)"

# ---------------------------------------------------------------- 3 改 config
info "3/5 改 .hydra/config.yaml（展开链式插值 + 路径改成 \${run_dir}/...）"
python3 - "$M/.hydra/config.yaml" <<'PY'
import copy, pathlib, sys, yaml
p = pathlib.Path(sys.argv[1]); cfg = yaml.safe_load(p.read_text())
model = cfg.get("model") or {}; root_tok = cfg.get("tokenizer")
ma = model.get("model_arch") or {}
changed = []
# ① 展开链式插值（否则 OmegaConf.to_container 报 InterpolationKeyError）
if isinstance(root_tok, dict) and isinstance(model, dict):
    if isinstance(model.get("tokenizer"), str) and model["tokenizer"].strip() == "${tokenizer}":
        model["tokenizer"] = copy.deepcopy(root_tok); changed.append("model.tokenizer 展开")
if isinstance(ma, dict):
    if isinstance(ma.get("action_tokenizer"), str) and "model.tokenizer._target_" in ma["action_tokenizer"]:
        src = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else root_tok
        if isinstance(src, dict) and src.get("_target_"):
            ma["action_tokenizer"] = src["_target_"]; changed.append("action_tokenizer 展开")
    if isinstance(ma.get("AT_CONFIG"), str) and "model.tokenizer.vq_config" in ma["AT_CONFIG"]:
        src = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else root_tok
        if isinstance(src, dict) and isinstance(src.get("vq_config"), dict):
            ma["AT_CONFIG"] = copy.deepcopy(src["vq_config"]); changed.append("AT_CONFIG 展开")
# ② codec 路径 → ${run_dir}/...
codec = "${run_dir}/action_tokenizer.pt"
for owner, label in ((root_tok, "tokenizer"), (model.get("tokenizer"), "model.tokenizer"),
                     (ma.get("AT_CONFIG"), "model_arch.AT_CONFIG")):
    if isinstance(owner, dict) and isinstance(owner.get("vq_config"), dict):
        owner["vq_config"]["ckpt_dir"] = codec; changed.append(f"{label}.ckpt_dir")
    elif isinstance(owner, dict) and "ckpt_dir" in owner:
        owner["ckpt_dir"] = codec; changed.append(f"{label}.ckpt_dir")
# ③ 归一化统计
cfg["datastatics_path"] = "${run_dir}/dataset_stats.json"; changed.append("datastatics_path")
# ④ processor 路径
hfp = "${run_dir}/hf_processor"
if isinstance(ma, dict) and "hf_processor_path" in ma:
    ma["hf_processor_path"] = hfp; changed.append("model_arch.hf_processor_path")
proc = model.get("processor")
if isinstance(proc, dict):
    tp = proc.get("tokenizer_params")
    if isinstance(tp, dict) and "pretrained_model_name_or_path" in tp:
        tp["pretrained_model_name_or_path"] = hfp; changed.append("processor.tokenizer_params")
    if "hf_processor_path" in proc:
        proc["hf_processor_path"] = hfp; changed.append("processor.hf_processor_path")
p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("  已修改：" + ", ".join(changed))
PY
ok "config 改写完成"

# ---------------------------------------------------------------- 4 元数据 + ar.json + launch.sh
info "4/5 改元数据 / ar.json / launch.sh"
python3 - "$M" "$DEST" "$PORT" "$TASK" <<'PY'
import datetime, hashlib, json, pathlib, sys
M, DEST, PORT, TASK = (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),
                       int(sys.argv[3]), sys.argv[4])

def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()

ckpt = M / "checkpoints" / "model_state_dict.pt"

mp = json.loads((M / "motion_package.json").read_text())
mp.update({
    "checkpoint": "checkpoints/model_state_dict.pt", "source_checkpoint": str(ckpt),
    "step": 17143, "training_mode": "ar", "optimizer_states": 0,
    "scheduler_last_epoch": 17143, "model_tensor_count": 946,
    "initialization": "official_g05_base_with_finetuned_YAM_codec_full_1epoch_4gpu",
    "policy_rate_hz": 30, "action_steps": 32, "optimizer_in_package": False,
    "inference_heads": ["ar"],
    "codec_sha256": sha(M / "action_tokenizer.pt"),
    "ar_codec_stats_sha256": sha(M / "ar_codec_stats.json"),
    "dataset_stats_sha256": sha(M / "dataset_stats.json"),
    "ckpt_sha256": sha(ckpt), "ckpt_bytes": ckpt.stat().st_size,
})
(M / "motion_package.json").write_text(json.dumps(mp, indent=2, ensure_ascii=False) + "\n")

me = json.loads((M / "motion_experiment.json").read_text())
me["task"] = TASK
me["experiment"] = "yam10task_g05_ft1ep_30hz"
g = me.setdefault("g05", {})
for k in ("dataset_root", "output_dir", "asset_root", "pretrained_checkpoint",
          "action_tokenizer_checkpoint", "dataset_config", "python"):
    g.pop(k, None)
g.update({"checkpoint": str(M), "port": PORT, "uri": f"ws://127.0.0.1:{PORT}",
          "max_steps": 17143, "overrides": ["model.use_torch_compile=false"]})
me.setdefault("inference", {})["endpoint"] = f"ws://127.0.0.1:{PORT}"
(M / "motion_experiment.json").write_text(json.dumps(me, indent=2, ensure_ascii=False) + "\n")

(M / "motion_versions.json").write_text(json.dumps({
    "checked_date": datetime.date.today().isoformat(),
    "repositories": [{
        "name": "G05-backbone-pkg", "path": "(our fine-tune package)",
        "origin": "https://github.com/FYL-loong/G05-backbone-pkg",
        "note": "backbone fine-tuned on I2RT YAM, 10 tasks, 1 epoch = 17143 steps, 4xGPU"}],
    "note": "codec/vendor 沿用本包自带的 vendor/GalaxeaVLA（含其 AR 解码加速补丁）"},
    indent=2, ensure_ascii=False) + "\n")

(M / "motion_split.json").write_text(json.dumps({
    "schema_version": 1,
    "source": "Zenith6666/yam-lerobot (LeRobot, 10 tasks, 991 episodes, 30fps)",
    "unit": "episode", "val_set_proportion": 0.2,
    "note": "按 episode 顺序 80/20 切；验证集 = TASK-YAM-0014 + 0015（完整留出的两个任务）"},
    indent=2, ensure_ascii=False) + "\n")

ar = json.loads((DEST / "live_configs" / "ar.json").read_text())
ar["experiment"] = "yam10task_g05_ft1ep_30hz"
ar["task"] = TASK
ar["inference"] = {"endpoint": f"ws://127.0.0.1:{PORT}", "query_mode": "sync"}
g = ar.setdefault("g05", {})
for k in ("python", "dataset_root", "output_dir", "asset_root"):
    g.pop(k, None)
g.update({"checkpoint": str(M), "port": PORT, "uri": f"ws://127.0.0.1:{PORT}",
          "action_steps": 32, "frequency_hz": 30.0, "missing_action": "hold",
          "full_chunk": True, "training_mode": "ar", "inference_head": "ar",
          "layout": "r1lite27", "device": "cuda:0"})
(DEST / "live_configs" / "ar.json").write_text(json.dumps(ar, indent=2, ensure_ascii=False) + "\n")
print("  元数据 5 个 + ar.json 已更新")
print(f"  task = {TASK!r}   port = {PORT}")
PY

sed -i 's#models/ar_newcodec5000#models/ar_ft1ep#g' "$DEST/launch.sh"
sed -i "s#--port 18878#--port $PORT#g"              "$DEST/launch.sh"
grep -q "models/ar_ft1ep"  "$DEST/launch.sh" || fail "launch.sh 模型路径未替换"
grep -q -- "--port $PORT"  "$DEST/launch.sh" || fail "launch.sh 端口未替换"
ok "launch.sh 已改（ar_ft1ep / port $PORT）"

# ---------------------------------------------------------------- 5 汇总
info "5/5 产物核对"
printf "  %-44s %14s\n" "文件" "字节"
printf "  %-44s %14s\n" "--------------------------------------------" "--------------"
for f in checkpoints/model_state_dict.pt action_tokenizer.pt ar_codec_stats.json \
         dataset_stats.json .hydra/config.yaml motion_contract.json motion_package.json \
         motion_experiment.json motion_versions.json motion_split.json; do
    [[ -f "$M/$f" ]] || fail "缺文件：$M/$f"
    printf "  %-44s %14s\n" "$f" "$(stat -c%s "$M/$f")"
done
printf "  %-44s %14s\n" "hf_processor/ (文件数)" "$(find "$M/hf_processor" -type f | wc -l)"
echo
echo "  dataset_stats.json vs ar_codec_stats.json："
if [[ "$(sha256sum "$M/dataset_stats.json" | cut -d' ' -f1)" == \
      "$(sha256sum "$M/ar_codec_stats.json" | cut -d' ' -f1)" ]]; then
    echo "    → 完全相同（同源）"
else
    echo "    → 不同（正常）"
fi
echo
echo "  config.yaml 关键行："
grep -nE 'ckpt_dir|datastatics_path|hf_processor_path|pretrained_model_name_or_path' \
     "$M/.hydra/config.yaml" | head -10
echo
echo "  ar.json 关键字段："
python3 -c "
import json;d=json.load(open('$DEST/live_configs/ar.json'))
print('    task        =', repr(d['task']))
print('    experiment  =', d['experiment'])
print('    checkpoint  =', d['g05']['checkpoint'])
print('    port / uri  =', d['g05']['port'], d['g05']['uri'])
print('    action_steps=', d['g05']['action_steps'], ' full_chunk =', d['g05']['full_chunk'])
"
echo
rm -f "$DEST/.build_partial"
ok "全部完成 ✅  部署包：$DEST  ($(du -sh "$DEST" | cut -f1))"
cat <<EOF

  ⚠️ 本脚本没有加载模型、没占 GPU、没开相机、没给机器人发指令。

  下一步（TODO 第 1 步：测 T_infer）：
    V=$DEST/runtime/env/bin/python
    G=$DEST/code/motion/vendor/GalaxeaVLA
    cd "\$G"
    PYTHONPATH="\$G/src" "\$V" \\
      /home/chenyuan/workspace/g05-yam-deployment/g05-async-serving/offline_chunk_harness.py \\
      --ckpt_path $M/checkpoints/model_state_dict.pt --device cuda:0 \\
      --action_steps 1 --steps 8 --rate 0 \\
      model.model_weights_to_bf16=true model.model_arch.continuous_action=false

EOF
