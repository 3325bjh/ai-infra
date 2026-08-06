import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ReduceOp

def setup(rank,world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="29500"
    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)



def distributed_demo(rank,world_size,data_size):
    setup(rank,world_size)
    warm_up=5
    iters=10
    data=torch.randn((data_size//4,),dtype=torch.float32).to("cuda")
    print(f"rank:{rank},device:{data.device}")
    for i in range(warm_up):
        dist.all_reduce(data,op=ReduceOp.SUM,async_op=False)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        dist.all_reduce(data,op=ReduceOp.SUM,async_op=False)
    end.record()
    torch.cuda.synchronize()
    total_time=start.elapsed_time(end)
    mean_time=total_time/iters
    print(f"rank:{rank},data_size:{data_size},world_size:{world_size},mean_time:{mean_time}")
    dist.destroy_process_group()



if __name__ == '__main__':
    #world_sizes=[2,4,6]
    world_sizes = [2,4]
    data_sizes = [size * 1024**2 for size in (1, 10, 100, 1024)]
    for world_size in world_sizes:
        for data_size in data_sizes:
            mp.spawn(fn=distributed_demo,args=(world_size,data_size),nprocs=world_size,join=True)