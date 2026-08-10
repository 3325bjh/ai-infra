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

class DDPBucket(nn.Module):
    def __init__(self,module:torch.nn.Module,bucket_size_mb:float):
        super().__init__()
        self.module=module
        self.world_size=dist.get_world_size()
        for param in self.module.parameters():
            dist.broadcast(param.data,src=0)

        self.buckets=[]
        self.bucket_param_counts=[]
        self.bucket_ready_counts=[]
        self.param_to_bucket_idx={}
        max_bytes_per_bucket=bucket_size_mb*1024*1024
        param_with_grad=[p for p in self.module.parameters() if p.requires_grad]

        param_with_grad.reverse()
        current_bucket=[]
        current_bucket_bytes=0
        for param in param_with_grad:
            param_bytes=param.element_size()*param.numel()
            if current_bucket_bytes+param_bytes>max_bytes_per_bucket and len(current_bucket)>0:
                self._save_bucket(current_bucket)
                current_bucket=[]
                current_bucket_bytes=0
            current_bucket.append(param)
            current_bucket_bytes+=param_bytes

        if len(current_bucket)>0:
            self._save_bucket(current_bucket)

        for param in param_with_grad:
            param.register_post_accumulate_grad_hook(self._make_hook(param))

        self.pending_communications=[]
    def _save_bucket(self,bucket_params):
        bucket_idx=len(self.buckets)
        self.buckets.append(bucket_params)
        self.bucket_param_counts.append(len(bucket_params))
        self.bucket_ready_counts.append(0)
        for p in bucket_params:
            self.param_to_bucket_idx[p]=bucket_idx

    def _make_hook(self,param):
        def hook(tensor):
            if not dist.is_initialized():
                return

            bucket_idx=self.param_to_bucket_idx[param]
            self.bucket_ready_counts[bucket_idx]+=1
            if self.bucket_ready_counts[bucket_idx]==self.bucket_param_counts:
                bucket_params=self.buckets[bucket_idx]
                grads=[p.grad.data for p in bucket_params]
                flat_grads=torch._utils._flatten_dense_tensors(grads)
                handle=dist.all_reduce(flat_grads,op=ReduceOp.SUM,async_op=True)
                self.pending_communications.append((handle,flat_grads,bucket_params))
        return hook
    def forward(self,*inputs,**kwargs):
        return self.module(*inputs,**kwargs)
    def finish_gradient_synchronization(self):
        if dist.is_initialized():
            for handle,flat_grads,bucket_params in self.pending_communications:
                handle.wait()
                flat_grads/=self.world_size
                grads_shape_ref=[p.grad.data for p in bucket_params]
                unflattened_grads=torch._utils._unflatten_dense_tensors(flat_grads,grads_shape_ref)
                for p,synced_grad in zip(bucket_params,unflattened_grads):
                    p.grad.data.copy_(synced_grad)
            self.pending_communications.clear()
            for i in range(len(self.bucket_ready_counts)):
                self.bucket_ready_counts[i]=0