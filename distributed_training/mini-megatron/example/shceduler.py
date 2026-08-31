import torch
import torch.distributed as dist

from example.init_parallel import pp_world, pp_rank
from example.p2p_communication import recv_prev, send_next,recv_next,send_prev


def forward_backward_1f1b(stage_model,data_iter,num_microbatches,seq_shape,loss_fn):
    p,r=pp_world(),pp_rank()
    num_warmup=min(p-r-1,num_microbatches)
    num_steady=num_microbatches-num_warmup
    inputs,outputs=[],[]
    def fwd():
        x=recv_prev(seq_shape) if r>0 else next(data_iter).cuda()
        x.requires_grad_(True)
        y=stage_model(x)
        send_next(y)
        inputs.append(x)
        outputs.append(y)
        return y
    def bwd():
        x,y=inputs.pop(),outputs.pop()
        grad_y=recv_next(seq_shape) if r<p-1 else None
        if r==p-1:
            loss=loss_fn(y)
            loss.backward()
        else:
            y.backward(grad_y)
        if x.grad is not None:send_prev(x.grad)

    for _ in range(num_warmup):fwd()
    for _ in range(num_steady):
        fwd()
        bwd()
    for _ in range(num_warmup):bwd()