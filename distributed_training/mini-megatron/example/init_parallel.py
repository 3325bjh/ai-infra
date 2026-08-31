import os,torch,torch.distributed as dist
def pp_rank(): return dist.get_rank(ParallelState.pp_group)
def pp_world(): return dist.get_world_size(ParallelState.pp_group)

class ParallelState:
    tp_group=pp_group=dp_group=None
    tp_size=pp_size=dp_size=1

def init_parallel(tp_size,pp_size):
    dist.init_process_group("nccl")
    world=dist.get_world_size()
    rank=dist.get_rank()
    torch.cuda.set_device(rank%torch.cuda.device_count())
    assert world%(tp_size*pp_size)==0
    dp_size=world//(tp_size*pp_size)
    ParallelState.tp_size,ParallelState.pp_size,ParallelState.dp_size=tp_size,pp_size,dp_size
    for pp in range(pp_size):
        for dp in range(dp_size):
            ranks=[tp+dp*tp_size+pp*tp_size*dp_size for tp in range(tp_size)]
            g=dist.new_group(ranks)
            if rank in ranks:ParallelState.tp_group=g
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks=[tp+dp*tp_size+pp*tp_size*dp_size for dp in range(dp_size)]
            g=dist.new_group(ranks)
            if rank in ranks:ParallelState.dp_group=g
    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = [tp + dp * tp_size + pp * tp_size * dp_size for pp in range(pp_size)]
            g = dist.new_group(ranks)
            if rank in ranks: ParallelState.pp_group = g