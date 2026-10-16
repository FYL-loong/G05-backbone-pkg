#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""多卡 NCCL 连通性最小测试（只用一个 broadcast，参数可调）。

用法：
    torchrun --standalone --nnodes 1 --nproc_per_node=N tools/nccl_minimal_test.py

用途：MODE=smoke-ddp 会用它把几种 NCCL 配置各试一遍，找出哪种能在这个
      容器里跑通 4 卡通信。实测第 1 步 dist.broadcast 就报
      "Cuda failure 1 'invalid argument'"，说明 P2P 机制不被支持。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    # 打印环境信息（只在 rank0 打全）
    if rank == 0:
        print(f"[nccl-min] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={torch.cuda.nccl.version()} ndev={torch.cuda.device_count()}", flush=True)
        try:
            cap = torch.cuda.get_device_capability(0)
            print(f"[nccl-min] GPU0 name={torch.cuda.get_device_name(0)!r} "
                  f"compute_capability=sm_{cap[0]}{cap[1]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[nccl-min] 读 GPU 属性失败: {e}", flush=True)
        _nccl_env = ", ".join(f"{k}={os.environ[k]}" for k in sorted(os.environ)
                              if k.startswith("NCCL_"))
        print(f"[nccl-min] NCCL env: {_nccl_env or '(none)'}", flush=True)

    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:  # noqa: BLE001
        print(f"[nccl-min][rank {rank}] init_process_group 失败: "
              f"{type(e).__name__}: {e}", flush=True)
        return 1
    if rank == 0:
        print(f"[nccl-min] world={dist.get_world_size()} 初始化成功", flush=True)

    try:
        t = torch.tensor([float(rank)], device=dev)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize()
        assert t.item() == 0.0, f"broadcast 结果错: {t.item()}"
        if rank == 0:
            print("[nccl-min] ✅ broadcast 成功", flush=True)
        ok = True
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"[nccl-min] ❌ broadcast 失败: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:200]}", flush=True)
        ok = False

    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
