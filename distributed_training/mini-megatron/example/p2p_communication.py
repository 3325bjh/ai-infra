import torch
import torch.distributed as dist
from init_parallel import ParallelState,pp_rank,pp_world


def _global(pp_local):
    return dist.get_global_rank(ParallelState.pp_group,pp_local)

def send_next(t):
    if pp_rank()!=pp_world()-1:
        dist.send(t.contiguous(),_global(pp_rank()+1))

def recv_prev(shape,dtype=torch.float32):
    if pp_rank()==0:return None
    t=torch.empty(shape,dtype=dtype,device="cuda",requires_grad=True)
    dist.recv(t,_global(pp_rank()-1))
    return t

def send_prev(t):
    if pp_rank()!=0:
        dist.send(t.contiguous(),_global(pp_rank()-1))

def recv_next(shape,dtype=torch.float32):
    if pp_rank()==pp_world()-1:return None
    t=torch.empty(shape,dtype=dtype,device="cuda")
    dist.recv(t,_global(pp_rank()+1))
    return t
