"""Two CPU ranks using file rendezvous (portable Windows Gloo smoke)."""
import argparse
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import _run_training


def worker(rank, config, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2",LOCAL_WORLD_SIZE="2",RANK=str(rank),LOCAL_RANK=str(rank))
    dist.init_process_group("gloo",init_method=Path(rendezvous).as_uri(),rank=rank,world_size=2)
    args=argparse.Namespace(config=config,resume=None)
    _run_training(args,"cpu","external_ddp",False)


if __name__=="__main__":
    mp.spawn(worker,args=(sys.argv[1],sys.argv[2]),nprocs=2,join=True)
