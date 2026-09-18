#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简易多卡 DDP 测试脚本。

运行示例：
    torchrun --standalone --nproc_per_node=2 test_multi_gpu.py

若没有 torchrun：
    python -m torch.distributed.run --standalone --nproc_per_node=2 test_multi_gpu.py
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动、CUDA 和 PyTorch 安装。")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"只检测到 {gpu_count} 张 GPU，无法进行多卡测试。")

    # torchrun 会自动设置这些环境变量
    required_env = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"缺少环境变量 {missing}。\n"
            "请使用以下方式启动：\n"
            "torchrun --standalone --nproc_per_node=2 test_multi_gpu.py"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size > gpu_count:
        raise RuntimeError(
            f"启动了 {world_size} 个进程，但服务器只有 {gpu_count} 张 GPU。"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化 NCCL 多卡通信
    dist.init_process_group(backend="nccl", init_method="env://")

    gpu_name = torch.cuda.get_device_name(local_rank)
    print(
        f"[Rank {rank}] 初始化成功："
        f"local_rank={local_rank}, device={device}, GPU={gpu_name}",
        flush=True,
    )

    # 1. 测试 GPU 间通信
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2

    if abs(value.item() - expected) > 1e-6:
        raise RuntimeError(
            f"AllReduce 结果异常：得到 {value.item()}，期望 {expected}"
        )

    # 2. 测试一次真正的 DDP 前向、反向和参数更新
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10),
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 16
    inputs = torch.randn(batch_size, 1024, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # 等待所有进程完成
    dist.barrier()

    print(
        f"[Rank {rank}] DDP 前向、反向、梯度同步和参数更新均成功，"
        f"loss={loss.item():.6f}",
        flush=True,
    )

    if rank == 0:
        print("\n测试通过：该服务器当前环境可以进行 PyTorch 多卡 DDP 训练。")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n多卡测试失败：{exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        cleanup()
        sys.exit(1)
