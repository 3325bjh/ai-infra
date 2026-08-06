import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.distributed_c10d import Work
import torch
class Naive_DDP(nn.Module):
    def __init__(self,model:nn.Module):
        super().__init__()
        self.module = model
        for parameter in model.parameters():
            dist.broadcast(parameter.data, 0)

    def forward(self,x, *inputs, **kwargs):
        return self.module(x)

    def finish_gradient_synchronization(self):
        world_size=dist.get_world_size()
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                # dist.all_reduce(parameter.grad,op=ReduceOp.AVG)
                dist.all_reduce(parameter.grad, op=ReduceOp.SUM)
                parameter.grad.div_(world_size)


class DDP_with_flat_gradients(nn.Module):
    def __init__(self,model:nn.Module):
        super().__init__()
        self.module = model
        for parameter in model.parameters():
            dist.broadcast(parameter.data, 0)

    def forward(self,x, *inputs, **kwargs):
        return self.module(x)

    def finish_gradient_synchronization(self):
        world_size=dist.get_world_size()
        parameters=[p for p in self.module.parameters() if p.grad is not None]
        gradients=[p.grad for p in parameters]
        flat_gradients=torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat_gradients,op=ReduceOp.SUM)
        flat_gradients.div_(world_size)
        unflattened_gradients=torch._utils._unflatten_dense_tensors(
            flat_gradients,
            gradients,
        )
        for parameters,synced_grad in zip(parameters,unflattened_gradients):
            parameters.grad.copy_(synced_grad)


class DDP_overlap_individaul_parameters(nn.Module):
    def __init__(self,model:nn.Module):
        super().__init__()
        self.module = model
        self.handles:list[Work]=[]

        with torch.no_grad():
            for parameter in model.parameters():
                dist.broadcast(parameter.data, 0)

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self._make_grad_hook())



    def forward(self,x, *inputs, **kwargs):
        return self.module(x)

    def _make_grad_hook(self):
        def hook(parameter:torch.Tensor):
            if parameter.grad is None:
                return
            parameter.grad.div_(dist.get_world_size())
            handle=dist.all_reduce(parameter.grad,op=ReduceOp.SUM,async_op=True)
            self.handles.append(handle)

        return hook

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


