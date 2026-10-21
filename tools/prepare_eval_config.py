#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""把 Hydra 保存下来的配置里【链式插值】就地展开，避免 load_config_from_run_dir
在 OmegaConf.to_container(resolve=True) 时报

    InterpolationKeyError: Interpolation key 'model.tokenizer._target_' not found

背景（实测）：
    训练存档的 .hydra/config.yaml 是【组合后】的配置，存在链式引用：
        tokenizer:                                   # 顶层真实节点
          _target_: g05.tokenizer...VQActionTokenizer
          vq_config: {...}
        model:
          tokenizer: ${tokenizer}                    # ← 指向顶层
          model_arch:
            action_tokenizer: ${model.tokenizer._target_}    # ← 再穿一层
            AT_CONFIG: ${model.tokenizer.vq_config}
    `to_container(resolve=True)` 需要“穿过一个插值节点再取子键”，
    评测环境里会解析失败（本地 omegaconf 2.3.0 能过、容器里报 not found）。
    与其继续追根因，直接把这两处【展开成具体值】，让插值不再有歧义。

为什么用 pyyaml 而不是 OmegaConf：
    OmegaConf 的 `.get()` 会【解析】插值，拿到的是解析后的值而不是原始字符串，
    无法判断“这里原本是不是 ${tokenizer}”。pyyaml 读到的是原始文本，可精确匹配，
    写回后仍是合法 YAML（评测端从文件重新加载）。

用法：
    python3 tools/prepare_eval_config.py <config.yaml> [--codec-path <path>]
"""
from __future__ import annotations

import argparse
import copy
import sys

import yaml


def _is_interp(v, target: str) -> bool:
    return isinstance(v, str) and v.strip() == target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--codec-path", default=None,
                    help="把 tokenizer.vq_config.ckpt_dir 覆盖成这个路径（相对项目根）")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print(f"[cfg-patch] {a.config} 不是 YAML 映射", file=sys.stderr)
        return 2

    changed: list[str] = []
    root_tok = cfg.get("tokenizer")
    model = cfg.get("model")
    ma = (model or {}).get("model_arch") if isinstance(model, dict) else None

    # ---- ① model.tokenizer: ${tokenizer} -> 顶层 tokenizer 的深拷贝
    if isinstance(root_tok, dict) and isinstance(model, dict):
        if _is_interp(model.get("tokenizer"), "${tokenizer}"):
            model["tokenizer"] = copy.deepcopy(root_tok)
            changed.append("model.tokenizer: ${tokenizer} -> 展开为具体 dict")

    # ---- ② model_arch 里的两处链式引用展开
    if isinstance(ma, dict):
        if _is_interp(ma.get("action_tokenizer"), "${model.tokenizer._target_}"):
            src = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else root_tok
            tgt = src.get("_target_") if isinstance(src, dict) else None
            if tgt:
                ma["action_tokenizer"] = tgt
                changed.append("model.model_arch.action_tokenizer -> 具体字符串")
        if _is_interp(ma.get("AT_CONFIG"), "${model.tokenizer.vq_config}"):
            src = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else root_tok
            vq = src.get("vq_config") if isinstance(src, dict) else None
            if isinstance(vq, dict):
                ma["AT_CONFIG"] = copy.deepcopy(vq)
                changed.append("model.model_arch.AT_CONFIG -> 展开为具体 dict")

    # ---- ③ 可选：覆盖 codec 路径
    # 注意 AT_CONFIG 在 ② 里已被复制成独立副本，必须一并改，
    # 否则 AT_CONFIG 与 model.tokenizer.vq_config 会出现两个不同的 ckpt_dir。
    if a.codec_path:
        ma2 = (model or {}).get("model_arch") if isinstance(model, dict) else None
        for owner, label in (
            (root_tok, "tokenizer"),
            (model.get("tokenizer") if isinstance(model, dict) else None, "model.tokenizer"),
            (ma2.get("AT_CONFIG") if isinstance(ma2, dict) else None, "model.model_arch.AT_CONFIG"),
        ):
            if isinstance(owner, dict) and isinstance(owner.get("vq_config"), dict):
                owner["vq_config"]["ckpt_dir"] = a.codec_path
                changed.append(f"{label}.vq_config.ckpt_dir -> {a.codec_path}")
            elif isinstance(owner, dict) and label.endswith("AT_CONFIG"):
                owner["ckpt_dir"] = a.codec_path
                changed.append(f"{label}.ckpt_dir -> {a.codec_path}")

    if not changed:
        print("[cfg-patch] 无需改动（没有找到链式插值）")
        return 0

    with open(a.config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print("[cfg-patch] 已就地展开/覆盖：")
    for c in changed:
        print(f"           - {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
