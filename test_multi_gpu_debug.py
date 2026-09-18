#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
带分阶段输出和超时的单机多卡 NCCL/DDP 诊断脚本。

运行：
    torchrun --standalone --nproc_per_node=2 test_multi_gpu_debug.py

详细日志：
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,COLL \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    torchrun --standalone --nproc_per_node=2 test_multi_gpu_debug.py
"""

import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def log(rank, message):
    print(f"[Rank {rank}] {message}", flush=True)


def safe_cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if rank < 0 or local_rank < 0 or world_size < 1:
        raise RuntimeError(
            "请通过 torchrun 启动：\n"
            "torchrun --standalone --nproc_per_node=2 test_multi_gpu_debug.py"
        )

    log(rank, "步骤 1/8：检查 CUDA")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    gpu_count = torch.cuda.device_count()
    if world_size > gpu_count:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}，但当前只检测到 {gpu_count} 张 GPU。"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print("\n===== 环境信息 =====", flush=True)
        print(f"PyTorch            : {torch.__version__}", flush=True)
        print(f"PyTorch CUDA       : {torch.version.cuda}", flush=True)
        print(f"GPU 数量           : {gpu_count}", flush=True)

        try:
            print(f"NCCL 版本           : {torch.cuda.nccl.version()}", flush=True)
        except Exception as exc:
            print(f"NCCL 版本           : 无法读取（{exc}）", flush=True)

        try:
            print(f"编译支持架构       : {torch.cuda.get_arch_list()}", flush=True)
        except Exception as exc:
            print(f"编译支持架构       : 无法读取（{exc}）", flush=True)

        print("====================\n", flush=True)

    log(
        rank,
        f"GPU={torch.cuda.get_device_name(local_rank)}, "
        f"compute_capability={torch.cuda.get_device_capability(local_rank)}"
    )

    log(rank, "步骤 2/8：初始化 NCCL 进程组")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=45),
    )
    log(rank, "NCCL 进程组初始化成功")

    log(rank, "步骤 3/8：在本卡创建 CUDA 张量")
    value = torch.tensor([float(rank + 1)], device=device)
    torch.cuda.synchronize(device)
    log(rank, f"CUDA 张量创建成功，初值={value.item()}")

    log(rank, "步骤 4/8：开始 NCCL all_reduce")
    work = dist.all_reduce(value, op=dist.ReduceOp.SUM, async_op=True)
    work.wait()
    torch.cuda.synchronize(device)
    log(rank, f"NCCL all_reduce 完成，结果={value.item()}")

    expected = world_size * (world_size + 1) / 2
    if abs(value.item() - expected) > 1e-6:
        raise RuntimeError(
            f"all_reduce 结果错误：实际={value.item()}，期望={expected}"
        )

    log(rank, "步骤 5/8：创建普通 CUDA 模型")
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10),
    ).to(device)
    torch.cuda.synchronize(device)
    log(rank, "普通 CUDA 模型创建成功")

    log(rank, "步骤 6/8：封装 DDP（此处会同步模型参数）")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    log(rank, "DDP 封装成功")

    log(rank, "步骤 7/8：执行前向、反向和参数更新")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(16, 1024, device=device)
    targets = torch.randint(0, 10, (16,), device=device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    log(rank, f"训练步骤成功，loss={loss.item():.6f}")

    log(rank, "步骤 8/8：最终 barrier")
    dist.barrier()
    torch.cuda.synchronize(device)
    log(rank, "最终 barrier 成功")

    if rank == 0:
        print(
            "\n测试通过：NCCL 通信、DDP 参数同步、梯度同步和参数更新均正常。",
            flush=True,
        )

    safe_cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rank = int(os.environ.get("RANK", -1))
        print(
            f"\n[Rank {rank}] 测试失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        safe_cleanup()
        sys.exit(1)
