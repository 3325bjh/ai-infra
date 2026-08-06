import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard,DTensor,DeviceMesh
import torch.multiprocessing as mp

def dtensor_api(rank,world_size,master_port):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    mesh=dist.init_device_mesh("cuda",(2,))
    dtensor=dist.tensor.randn((4,8),device_mesh=mesh,placements=[Shard(0)])
    dtensor.__create_chunk_list__()

    dist.destroy_process_group()

mp.spawn(dtensor_api,(2,29500),nprocs=2,join=True)