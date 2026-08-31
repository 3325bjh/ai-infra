import torch
import torch.distributed as dist
import torch.nn as nn

from example.init_parallel import ParallelState


def all_reduce_grads(model:nn.Module):
    dp=ParallelState.dp_size
    if dp==1: return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad,group=ParallelState.dp_group)
            p.grad/=dp