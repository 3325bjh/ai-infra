from typing import Any

import torch
import torch.distributed as dist
from init_parallel import ParallelState
class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): return x
    @staticmethod
    def backward(ctx,g):
        dist.all_reduce(g,group=ParallelState.tp_group)
        return g
class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        dist.all_reduce(x,group=ParallelState.tp_group)
        return x
    @staticmethod
    def backward(ctx,g): return g


copy_to_tp=_CopyToTP.apply
reduce_from_tp=_ReduceFromTP.apply