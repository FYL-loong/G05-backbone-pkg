#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""多卡 DDP / NCCL 自检 —— 用【极小模型】验证 4 卡通信本身是否正常。

为什么需要它：
    实测「4 卡正式训练」在 `DDP(...)` 构造期间报
        NCCL WARN enqueue.cc:1556 Cuda failure 700 'an illegal memory access'
    而 1 卡（L4 过拟合 500 步）完全正常。参数/buffer 已被证明设备一致、dtype 干净。
    所以必须先区分：

      A) 环境问题 —— 4 卡 NCCL/DDP 本身跑不起来（与模型无关）
      B) 模型问题 —— 某个张量让 NCCL 越界

    本脚本用【极小模型】跑一遍所有多卡通信路径。若它也崩 -> A；若它正常 -> B。

用法（由 start.sh 的 MODE=smoke-ddp 通过 torchrun 调起）：
    torchrun --nproc_per_node=N tools/ddp_smoke_test.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn


def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    print(f"[ddp-smoke][rank {rank}] {msg}", flush=True)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, 32)
        self.lin = nn.Linear(32, 32, bias=False)
        self.buf = nn.Parameter(torch.zeros(4), requires_grad=False)

    def forward(self, x):
        return self.lin(self.emb(x))


def step(title: str, fn):
    """跑一步，失败就打印并退出（让日志一眼看到卡在哪一步）。"""
    log(f"--- {title} ---")
    try:
        fn()
        log(f"    OK  {title}")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"    FAIL {title}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:  # noqa: BLE001
        log(f"init_process_group 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    log(f"world={world} local_rank={local} device={dev} "
        f"name={torch.cuda.get_device_name(local)}")

    ok = True

    # 1) 基础 broadcast（单张小张量）
    def t1():
        t = torch.tensor([float(rank)], device=dev)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize()
        assert t.item() == 0.0, f"broadcast 结果错: {t.item()}"
    ok &= step("1. dist.broadcast（单张小张量）", t1)

    # 2) all_reduce（大一点，走真实 ring/tree）
    def t2():
        t = torch.ones(1 << 20, device=dev)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        assert abs(t[0].item() - world) < 1e-6, f"all_reduce 结果错: {t[0].item()}"
    ok &= step("2. dist.all_reduce（1M float32）", t2)

    # 3) bf16 all_reduce（正式训练是 bf16）
    def t3():
        t = torch.ones(1 << 20, device=dev, dtype=torch.bfloat16)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        assert abs(t[0].item() - world) < 1e-6, f"bf16 all_reduce 结果错: {t[0].item()}"
    ok &= step("3. dist.all_reduce（1M bfloat16）", t3)

    # 4) DDP 构造（极小模型）—— 这一步对应正式训练崩溃的位置
    def t4():
        m = Tiny().to(dev)
        ddp = nn.parallel.DistributedDataParallel(
            m, device_ids=[local], gradient_as_bucket_view=True, bucket_cap_mb=128
        )
        out = ddp(torch.randint(0, 128, (4, 8), device=dev)).sum()
        out.backward()
        torch.cuda.synchronize()
        ddp.module.lin.weight.grad  # 触发梯度就绪检查
    ok &= step("4. DDP 构造 + forward + backward（极小模型）", t4)

    # 5) DDP + 混合 dtype（模仿真实模型：部分 fp32、部分 bf16）
    def t5():
        m = Tiny().to(dev).to(torch.bfloat16)
        # 故意留一个 fp32 子模块，模仿 apply_fp32_params 的效果
        m.buf = nn.Parameter(torch.zeros(4), requires_grad=False)
        ddp = nn.parallel.DistributedDataParallel(
            m, device_ids=[local], gradient_as_bucket_view=True, bucket_cap_mb=128
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(torch.randint(0, 128, (4, 8), device=dev)).sum()
        out.backward()
        torch.cuda.synchronize()
    ok &= step("5. DDP + bf16 权重 + autocast", t5)

    # 6) 一个大 embedding（模仿扩词表后的 embedding）
    def t6():
        big = nn.Embedding(252189, 64).to(dev)
        ddp = nn.parallel.DistributedDataParallel(big, device_ids=[local])
        out = ddp(torch.randint(0, 252189, (2, 4), device=dev)).sum()
        out.backward()
        torch.cuda.synchronize()
    ok &= step("6. DDP + 大 embedding（252189×64，模仿扩词表）", t6)

    log(f"===== 结果: {'全部通过' if ok else '有失败项'} =====")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
