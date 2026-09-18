import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"[Rank {rank}] 开始初始化 Gloo", flush=True)

dist.init_process_group(
    backend="gloo",
    init_method="env://",
)

x = torch.tensor([float(rank + 1)])

print(f"[Rank {rank}] all_reduce 前：{x.item()}", flush=True)

dist.all_reduce(x)

print(f"[Rank {rank}] all_reduce 后：{x.item()}", flush=True)

expected = world_size * (world_size + 1) / 2

if x.item() == expected:
    print(f"[Rank {rank}] Gloo 测试通过", flush=True)
else:
    print(f"[Rank {rank}] Gloo 结果异常，期望 {expected}", flush=True)

dist.destroy_process_group()
