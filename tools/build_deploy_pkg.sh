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

fail() { echo -e "\n[FAIL] $*" >&2; exit 1; }
info() { echo -e "\n[INFO] $*"; }
ok()   { echo "[ OK ] $*"; }

# ---------------------------------------------------------------- 0 前置检查
info "0/5 前置检查"
[[ -n "${HF_TOKEN:-}" ]] || fail "请先： export HF_TOKEN=hf_你的token"
[[ -d "$SRC" ]] || fail "找不到骨架包 SRC=$SRC"
[[ -f "$SRC/launch.sh" ]] || fail "$SRC/launch.sh 不存在"
[[ -e "$DEST" ]] && fail "$DEST 已存在。确认无用后： rm -rf $DEST"
python3 -c "import yaml" 2>/dev/null || fail "python3 缺 pyyaml"

# 自动探测 hf CLI（这台机器上 hf 不在 PATH）
HF="${HF_CLI:-}"
if [[ -z "$HF" ]]; then
    if command -v hf >/dev/null 2>&1; then
        HF="$(command -v hf)"
    elif [[ -x "$SRC/runtime/env/bin/hf" ]]; then
        HF="$SRC/runtime/env/bin/hf"
    else
        fail "找不到 hf CLI，请设置 HF_CLI=<path>"
    fi
fi
[[ -x "$HF" ]] || fail "hf 不可执行：$HF"
ok "hf: $HF"

df -h /mnt/sda | tail -1
AVAIL_G=$(df -BG --output=avail /mnt/sda | tail -1 | tr -dc '0-9')
[[ "${AVAIL_G:-0}" -ge 40 ]] || fail "/mnt/sda 只剩 ${AVAIL_G}G，需要 >=40G"
mkdir -p "$HF_HOME" "$HFDIR"
ok "前置检查通过（/mnt/sda 可用 ${AVAIL_G}G）"

# ---------------------------------------------------------------- 1 复制骨架
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
ok "骨架复制完成，当前大小：$(du -sh "$DEST" | cut -f1)"

# ---------------------------------------------------------------- 2 下载产物
info "2/5 从 HF 拉我们的训练产物（约 11.6G，慢一点）"
"$HF" download "$REPO" --repo-type model --local-dir "$HFDIR/$REPO" \
    --include "model_state_dict.pt" --include "dataset_stats.json" \
    --include "hydra_config.yaml"   --include "action_tokenizer.pt" \
    --include "hf_processor/*" || fail "我们的模型产物下载失败"
"$HF" download "$CODEC_REPO" --repo-type model --local-dir "$HFDIR/$CODEC_REPO" \
    --include "yam_stats.json" || fail "codec stats（yam_stats.json）下载失败"

M="$DEST/models/ar_ft1ep"
mv -f "$HFDIR/$REPO/model_state_dict.pt"  "$M/checkpoints/model_state_dict.pt"
mv -f "$HFDIR/$REPO/dataset_stats.json"   "$M/dataset_stats.json"
mv -f "$HFDIR/$REPO/action_tokenizer.pt"  "$M/action_tokenizer.pt"
mv -f "$HFDIR/$REPO/hydra_config.yaml"    "$M/.hydra/config.yaml"
rm -rf "$M/hf_processor"
mv -f "$HFDIR/$REPO/hf_processor"         "$M/hf_processor"
mv -f "$HFDIR/$CODEC_REPO/yam_stats.json" "$M/ar_codec_stats.json"
rm -rf "$HFDIR"

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
