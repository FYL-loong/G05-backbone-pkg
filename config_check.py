#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""快速配置自检 —— 几秒内发现"配置级"错误，不需要 GPU、不需要下载权重/数据。

为什么需要它：
    真实训练任务每次都要 12 分钟下载（11.44GB 权重 + ~20GB 数据集）才进入
    `finetune.py`。而踩过的坑里有好几个是**配置/参数级**的，本来几秒就能发现：

      - `Cannot set both max_epochs and max_steps`（max_epochs 来自 task=yam 的
        覆盖，和命令行 max_steps 冲突）—— 这个断言在 finetune.py:877，
        **发生在任何下载之前**，却被下载埋到第 17 分钟才暴露。
      - `.gitignore` 裸 `eval/` 吞掉源码目录 → `no module named g05.utils.eval`
        （由 check_package.py 静态扫描发现）。
      - Hydra override 写错 key（Hydra 默认缺 key 就报错，这里提前触发）。

本脚本做的事：**完全复刻 `scripts/finetune.py` 的配置组装路径**，然后停在那里 ——
不建模型、不建数据集、不碰 GPU。覆盖 finetune.py:440-460：

    _apply_dataset_override_if_needed(cfg)
    OmegaConf.resolve(cfg)
    validate_train_config(cfg)
    truncate_embodiments / truncate_datasets   （--test 模式）
    max_epochs / max_steps 互斥断言             （finetune.py:877）

用法（必须在装了依赖的 venv 里跑）：
    python config_check.py                      # 等价 task=yam
    python config_check.py task=yam model.max_steps=10 model.max_epochs=null
    python config_check.py --test-mode          # 模拟 start.sh 的 L2 调用
    python config_check.py --test-mode --overfit  # 模拟 L4

退出码：0 = 配置可组装且校验通过；1 = 有配置错误（附精确原因）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"

# ---- 在 import g05 之前设置，和 scripts/finetune.py:84 保持一致 ----------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def die(msg: str, code: int = 1):
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    ap = argparse.ArgumentParser(description="快速配置自检（不加载模型/数据）")
    ap.add_argument("overrides", nargs="*", help="Hydra 覆盖，如 model.max_steps=10")
    ap.add_argument("--test-mode", action="store_true",
                    help="模拟 scripts/run/finetune.sh --test（MAX_DATASETS=1 等）")
    ap.add_argument("--dry-run-mode", action="store_true",
                    help="模拟 --dry-run（DRY_RUN=1）")
    ap.add_argument("--overfit", action="store_true",
                    help="模拟 --overfit_batch（+overfit_mode=batch）")
    args = ap.parse_args()

    # ---- 复刻 run/finetune.sh 在 --test / --dry-run 下导出的环境变量 ----------
    if args.test_mode:
        os.environ.setdefault("MAX_EMBODIMENTS", "17")
        os.environ.setdefault("MAX_DATASETS", "1")
        os.environ.setdefault("EXP_NAME", "test")
    if args.dry_run_mode:
        os.environ["DRY_RUN"] = "1"
    if args.test_mode or args.dry_run_mode:
        if not any(o.startswith("logger.mode=") for o in args.overrides):
            args.overrides.append("logger.mode=offline")

    overrides = list(args.overrides)
    if not any(o.startswith("task=") for o in overrides):
        overrides.insert(0, "task=yam")
    if args.overfit and not any("overfit_mode" in o for o in overrides):
        overrides.append("+overfit_mode=batch")

    print("=" * 72)
    print(" 快速配置自检（不加载模型/数据/GPU）")
    print("=" * 72)
    print(f"  overrides = {overrides}")
    print(f"  env       = MAX_DATASETS={os.environ.get('MAX_DATASETS')} "
          f"MAX_EMBODIMENTS={os.environ.get('MAX_EMBODIMENTS')} "
          f"DRY_RUN={os.environ.get('DRY_RUN')}")
    print()

    # ---- 1) 拿到配置辅助函数 + resolver 注册（能导入 finetune 最好，导不了就降级）----
    # scripts/finetune.py 模块级会 register_default_resolvers()，所以优先复用它，
    # 保证和真实训练走同一套 resolver。但它会连带 import torch/accelerate —— 环境
    # 没装好时导入失败，此时降级成"只注册 resolver"，仍能抓配置级错误。
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    ft = None
    try:
        import finetune as ft  # noqa: F401
        if not hasattr(ft, "_apply_dataset_override_if_needed"):
            die("scripts/finetune.py 里找不到 _apply_dataset_override_if_needed —— 上游改过？")
        print("[ OK ] 已导入 scripts/finetune.py（与真实训练同一套 resolver）")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 导入 scripts/finetune.py 失败（{type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]}）")
        print("[WARN] 降级：只注册 resolver，跳过数据集 override —— 配置级检查仍然有效")
        try:
            from g05.utils.config.config_resolvers import register_default_resolvers
            register_default_resolvers()
        except Exception as e2:  # noqa: BLE001
            die(f"连 resolver 都注册不了：{type(e2).__name__}: {e2}\n"
                f"  必须在装了依赖的环境里跑：source .venv/bin/activate")

    # ---- 2) compose 配置（等价 hydra.main 对同一 config_path/name 的组装）----
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as e:
        die(f"缺 hydra/omegaconf（{e}）")

    try:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="train.yaml", overrides=overrides)
    except Exception as e:  # noqa: BLE001
        die(f"Hydra 组装失败：{type(e).__name__}: {e}\n"
            f"  常见原因：override key 拼错（Hydra 对不存在的 key 会报错）")
    print("[ OK ] Hydra 组装成功（override key 全部有效）")

    # ---- 3) 复刻 finetune.py:440-441 ----------------------------------------
    try:
        if ft is not None:
            ft._apply_dataset_override_if_needed(cfg)
            print("[ OK ] _apply_dataset_override_if_needed")
        else:
            print("[SKIP] 未导入 finetune，跳过 _apply_dataset_override_if_needed")
        OmegaConf.resolve(cfg)
        print("[ OK ] OmegaConf.resolve（自定义 resolver 可用）")
    except Exception as e:  # noqa: BLE001
        die(f"配置解析失败：{type(e).__name__}: {e}")

    # ---- 4) 复刻 finetune.py:443-445 ----------------------------------------
    try:
        from g05.utils.config.config_validator import validate_train_config
        validate_train_config(cfg)
        print("[ OK ] validate_train_config 通过")
    except Exception as e:  # noqa: BLE001
        die(f"配置校验失败：{type(e).__name__}: {e}")

    # ---- 5) 复刻 finetune.py:449-458（--test 的截断）-------------------------
    max_emb = os.environ.get("MAX_EMBODIMENTS")
    max_ds = os.environ.get("MAX_DATASETS")
    try:
        if max_emb is not None:
            from g05.utils.eval.eval_utils import truncate_embodiments
            truncate_embodiments(cfg, int(max_emb))
            print(f"[ OK ] truncate_embodiments({max_emb})")
        if max_ds is not None:
            from g05.utils.eval.eval_utils import truncate_datasets
            truncate_datasets(cfg, int(max_ds))
            print(f"[ OK ] truncate_datasets({max_ds})")
    except Exception as e:  # noqa: BLE001
        die(f"数据集截断失败：{type(e).__name__}: {e}")

    # ---- 6) 复刻 finetune.py:876-881 的互斥断言（最容易踩的那个）------------
    max_epochs = cfg.model.get("max_epochs")
    max_steps = cfg.model.get("max_steps")
    print()
    print(f"  model.max_epochs = {max_epochs}")
    print(f"  model.max_steps  = {max_steps}")
    if max_epochs and max_steps:
        die("Cannot set both `max_epochs` and `max_steps`!\n"
            "  finetune.py:877 会 assert。修法：用 max_steps 时同时传 "
            "model.max_epochs=null\n"
            "  （注意 configs/task/yam.yaml 把 max_epochs 覆盖成了 10）")
    if not max_epochs and not max_steps:
        die("max_epochs 和 max_steps 都是空 —— finetune.py 会走到 "
            "`max_steps = cfg.model.max_steps`（None），后续计算会炸。\n"
            "  必须指定其中之一。")
    print("[ OK ] max_epochs / max_steps 互斥检查通过")

    # ---- 7) 数据目录存在性（静态，不建数据集）--------------------------------
    missing = []
    n_dirs = 0
    data_cfg = cfg.get("data", {})
    for group in (data_cfg.get("dataset_groups") or []):
        for d in (group.get("dataset_dirs") or []):
            n_dirs += 1
            if not (ROOT / d).exists():
                missing.append(d)
    emb = data_cfg.get("embodiment_datasets") or []
    for e in emb:
        for d in (e.get("dataset_dirs") or []):
            n_dirs += 1
            if not (ROOT / d).exists():
                missing.append(d)
    if n_dirs == 0:
        print("[WARN] 没在配置里找到 dataset_dirs，跳过目录检查")
    elif missing:
        print(f"\n[WARN] {len(missing)} 个数据集目录不存在（本地跑时正常，"
              f"容器里由 setup 阶段下载）：")
        for d in missing[:4]:
            print(f"         {d}")
        if len(missing) > 4:
            print(f"         ...（共 {len(missing)} 个）")
    else:
        print(f"[ OK ] {n_dirs} 个数据集目录都存在")

    print()
    print("=" * 72)
    print(" [ OK ] 配置自检全部通过 —— 这个 bug 类别不会再等 20 分钟才暴露")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
