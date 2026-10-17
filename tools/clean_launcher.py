#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""干净的 torchrun 替代品：按 N 个进程启动任意命令，环境变量自己算。

为什么需要它：
    百舸(AIHC)平台会把「节点级」的 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
    注入容器。`python -m torch.distributed.run`(=torchrun) 会继承这些值，
    结果子进程拿到的 RANK/LOCAL_RANK 是乱值（实测见过 16 / 29 / 206 / 228），
    于是 `torch.cuda.set_device(local)` 直接报
        RuntimeError: device id must be non-negative! -28
    进程在【进入 NCCL 之前】就死了 —— 这会让 4 卡训练看起来像"NCCL 不可用"，
    实际和 NCCL 毫无关系（实测用本启动器后 4 卡 6 项通信测试全部通过）。

本脚本做的事：
    1. 分配一个空闲端口
    2. 从零构造进程组环境：删掉所有 RANK/LOCAL_RANK/WORLD_SIZE/GROUP_RANK/
       LOCAL_WORLD_SIZE/ROLE_* 以及节点级 MASTER_*，换成 127.0.0.1 + 新端口
    3. 为每个 rank 显式设置 RANK / LOCAL_RANK（绝不继承外部值）
    4. 并发启动命令，全部等待；任一非零退出则整体返回 1

用法：
    python3 tools/clean_launcher.py --nproc 4 -- python3 scripts/finetune.py task=yam
    python3 tools/clean_launcher.py --nproc 4 -- python3 -c "import torch,os;print(os.environ['RANK'])"
    （--nproc 0 表示用 torch.cuda.device_count()）
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

_CLEAR_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "LOCAL_WORLD_SIZE",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT",
    # 平台/MPI 可能注入的同类变量
    "MLP_ROLE_INDEX", "MLP_WORKER_NUM", "MLP_WORKER_GPU", "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK",
    "PMI_RANK", "PMI_SIZE", "PMIX_RANK",
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--nproc", type=int, default=0, help="每节点进程数；0=按 GPU 数")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- 之后是要执行的命令")
    a = ap.parse_args()

    cmd = list(a.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[clean-launch] 用法: clean_launcher.py --nproc N -- <cmd ...>",
              file=sys.stderr)
        return 2

    n = a.nproc
    if n <= 0:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            n = 1
    n = max(1, n)

    port = _free_port()
    base = dict(os.environ)
    for k in _CLEAR_KEYS:
        base.pop(k, None)
    base.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": str(n),
        # 这些 torchrun 会设，accelerate / transformers 可能读；显式给全，避免缺失
        "LOCAL_WORLD_SIZE": str(n),
        "GROUP_RANK": "0",
        "ROLE_NAME": "default",
        "ROLE_WORLD_SIZE": str(n),
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "OMP_NUM_THREADS": "1",
    })

    print(f"[clean-launch] nproc={n} MASTER_ADDR=127.0.0.1 MASTER_PORT={port}",
          flush=True)
    print(f"[clean-launch] cmd={' '.join(cmd)}", flush=True)

    procs = []
    for r in range(n):
        env = dict(base)
        env["RANK"] = str(r)
        env["LOCAL_RANK"] = str(r)
        env["NODE_RANK"] = "0"
        env["ROLE_RANK"] = str(r)
        procs.append((r, subprocess.Popen(cmd, env=env)))

    rc = 0
    for r, p in procs:
        code = p.wait()
        if code != 0:
            rc = 1
            print(f"[clean-launch] rank {r} 退出码 {code}", flush=True)
    print(f"[clean-launch] 全部进程结束，总退出码 {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
