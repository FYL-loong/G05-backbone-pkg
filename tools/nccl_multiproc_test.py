#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""多卡 NCCL/DDP 自检（自包含启动器，不依赖 torchrun）。

为什么不用 torchrun：
    实测在百舸平台上，torchrun 启动的子进程拿到的 RANK/LOCAL_RANK 是
    乱值（见过 16 / 29 / 206 / 228），于是
        torch.cuda.set_device(local)  ->  RuntimeError: device id must be
        non-negative! -28
    也就是说进程在【进入 NCCL 之前】就死了。这导致此前所有 NCCL 环境变量
    实验全部无效（换 NCCL 2.25 -> 2.30 结果一模一样，因为根本没走到通信）。

    根因是 torchrun/elastic 会继承平台注入的 RANK/WORLD_SIZE/MASTER_*，
    在 K8s 上这些值属于「节点」语义，与「进程组」语义冲突。

本脚本做法：
    1. 父进程分配一个空闲端口，显式设置 MASTER_ADDR/MASTER_PORT/WORLD_SIZE，
       并为每个 rank 单独设置 RANK/LOCAL_RANK（绝不继承外部的值）
    2. 用 env:// 方式 init_process_group
    3. device 号 = local_rank % device_count（防御式取模，永不越界）

用法：
    单次测试（全部 6 步）：
        python3 tools/nccl_multiproc_test.py --all
    只测一次 broadcast（很轻，用于批量试 NCCL 环境变量）：
        python3 tools/nccl_multiproc_test.py --minimal
    指定卡数：  --nproc 4
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import traceback
import time


# --------------------------------------------------------------------------- 子进程
def _child(minimal: bool) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    ndev = torch.cuda.device_count()
    # ★ 防御：即使外部传来怪值，也保证落在 [0, ndev)
    dev_id = local % ndev if ndev > 0 else 0

    print(f"[mp][rank {rank}] world={world} local={local} ndev={ndev} "
          f"-> device cuda:{dev_id} ({torch.cuda.get_device_name(dev_id)})",
          flush=True)

    torch.cuda.set_device(dev_id)
    dev = torch.device("cuda", dev_id)

    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:  # noqa: BLE001
        print(f"[mp][rank {rank}] init_process_group 失败: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 1

    ok = True

    def step(title, fn):
        nonlocal ok
        if rank == 0:
            print(f"[mp][rank 0] --- {title} ---", flush=True)
        try:
            fn()
            torch.cuda.synchronize()
            if rank == 0:
                print(f"[mp][rank 0]     OK  {title}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[mp][rank {rank}]     FAIL {title}: "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
            return False

    def t_bcast():
        t = torch.tensor([float(rank)], device=dev)
        dist.broadcast(t, src=0)
        assert t.item() == 0.0, f"broadcast 值错: {t.item()}"

    def t_allreduce_f32():
        t = torch.ones(1024 * 1024, device=dev, dtype=torch.float32)
        dist.all_reduce(t)
        assert abs(t[0].item() - world) < 1e-3, f"allreduce 值错: {t[0].item()}"

    def t_allreduce_bf16():
        t = torch.ones(1024 * 1024, device=dev, dtype=torch.bfloat16)
        dist.all_reduce(t)
        assert abs(float(t[0].item()) - world) < 1e-1, f"allreduce bf16 值错: {t[0].item()}"

    def t_ddp():
        import torch.nn as nn
        m = nn.Sequential(nn.Embedding(128, 32), nn.Linear(32, 32)).to(dev)
        ddp = torch.nn.parallel.DistributedDataParallel(
            m, device_ids=[dev_id], gradient_as_bucket_view=False)
        x = torch.randint(0, 128, (4, 8), device=dev)
        loss = ddp(x).float().sum()
        loss.backward()
        ddp.zero_grad(set_to_none=True)

    def t_amp():
        import torch.nn as nn
        m = nn.Linear(64, 64).to(dev).to(torch.bfloat16)
        ddp = torch.nn.parallel.DistributedDataParallel(
            m, device_ids=[dev_id], gradient_as_bucket_view=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(torch.randn(8, 64, device=dev, dtype=torch.bfloat16))
            loss = out.float().sum()
        loss.backward()
        ddp.zero_grad(set_to_none=True)

    def t_bigemb():
        import torch.nn as nn
        m = nn.Embedding(252189, 64).to(dev)  # 模仿扩词表后的 embedding
        ddp = torch.nn.parallel.DistributedDataParallel(
            m, device_ids=[dev_id], gradient_as_bucket_view=False)
        out = ddp(torch.randint(0, 252189, (4, 8), device=dev))
        out.float().sum().backward()
        ddp.zero_grad(set_to_none=True)

    steps = [("1. dist.broadcast（单张小张量）", t_bcast)]
    if not minimal:
        steps += [
            ("2. all_reduce float32（1M 元素）", t_allreduce_f32),
            ("3. all_reduce bfloat16（1M 元素）", t_allreduce_bf16),
            ("4. DDP 构造 + forward + backward", t_ddp),
            ("5. DDP + bf16 权重 + autocast", t_amp),
            ("6. DDP + 大 embedding（252189×64）", t_bigemb),
        ]
    for title, fn in steps:
        if not step(title, fn):
            break

    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass

    if rank == 0:
        print(f"[mp][rank 0] ===== 结果: {'全部通过' if ok else '有失败项'} =====",
              flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- 父进程
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _launch(nproc: int, minimal: bool) -> int:
    import torch  # 只为读 device_count

    ndev = torch.cuda.device_count()
    if ndev < 1:
        print("[mp] 没有可见的 GPU", flush=True)
        return 1
    n = min(nproc, ndev) if nproc > 0 else ndev
    print(f"[mp] 启动 {n} 个进程（可见 GPU {ndev} 张）", flush=True)

    port = _free_port()
    base = dict(os.environ)
    # ★ 干净地重建进程组环境，绝不让平台的节点级变量污染
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK",
              "LOCAL_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        base.pop(k, None)
    base.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": str(n),
        "OMP_NUM_THREADS": "1",
    })
    print(f"[mp] MASTER_ADDR=127.0.0.1 MASTER_PORT={port} WORLD_SIZE={n}", flush=True)

    me = os.path.abspath(__file__)
    procs = []
    for r in range(n):
        env = dict(base)
        env["RANK"] = str(r)
        env["LOCAL_RANK"] = str(r)
        cmd = [sys.executable, me, "--child"]
        if minimal:
            cmd.append("--minimal")
        procs.append((r, subprocess.Popen(cmd, env=env)))

    rc = 0
    for r, p in procs:
        code = p.wait()
        if code != 0:
            rc = 1
            print(f"[mp] rank {r} 退出码 {code}", flush=True)
    print(f"[mp] 全部进程结束，总退出码 {rc}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--nproc", type=int, default=0)
    a = ap.parse_args()
    if a.child:
        return _child(a.minimal)
    return _launch(a.nproc, minimal=not a.all)


if __name__ == "__main__":
    sys.exit(main())
