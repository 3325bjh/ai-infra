import torch
import torch.nn as nn
from cs336_basics.model import Embedding, Linear
import torch.distributed as dist
from torch.distributed import ReduceOp

class FSDP(nn.Module):
    def __init__(self,module:torch.nn.Module,compute_dtype:torch.dtype|None=None):
        super().__init__()
        self.world_size=dist.get_world_size()
        self.rank=dist.get_rank()
        self.module=module
        self.shard_metadata={}
        self.compute_dtype=compute_dtype
        self.sharded_modules=[]
        self.module_indices={}
        with torch.no_grad():
            for mod in self.module.modules():
                if not isinstance(mod,(Linear,Embedding)):
                    continue
                for name,parameter in mod.named_parameters(recurse=False):
                    dist.broadcast(parameter.data,src=0)
                    full_shape=tuple(parameter.shape)
                    shard=torch.chunk(parameter.data,self.world_size)[self.rank].contiguous()
                    self.shard_metadata[id(parameter)]={
                        "module":mod,
                        "name":name,
                        "full_shape":full_shape,
                        "shard_dim":0,
                        "local_shard":None,
                        "gathered_shards":None,
                        "gather_handle":None
                    }
                    parameter.data=shard
                    parameter.register_post_accumulate_grad_hook(self.make_gradient_hook())

                mod.register_forward_pre_hook(hook=self.forward_pre_hook())
                mod.register_forward_hook(hook=self.forward_hook())
                mod.register_full_backward_pre_hook(hook=self.backward_pre_hook())
                self.module_indices[id(mod)] = len(self.sharded_modules)
                self.sharded_modules.append(mod)



    def start_all_gather(self, mod: nn.Module):
        """异步开始某层权重的 all-gather，但不等待。"""
        for parameter in mod.parameters(recurse=False):
            metadata = self.shard_metadata[id(parameter)]

            # 已在通信中或已经 materialize 时，不重复发起。
            if metadata["gather_handle"] is not None:
                continue

            master_shard = parameter.data
            compute_shard = master_shard

            if self.compute_dtype is not None:
                compute_shard = master_shard.to(self.compute_dtype)

            gathered_shards = [
                torch.empty_like(compute_shard)
                for _ in range(self.world_size)
            ]

            handle = dist.all_gather(
                gathered_shards,
                compute_shard,
                async_op=True,
            )

            # 必须保存，不能只是局部变量。
            metadata["local_shard"] = master_shard
            metadata["gathered_shards"] = gathered_shards
            metadata["gather_handle"] = handle

    def wait_and_materialize(self, mod: nn.Module):
        """等待通信完成，并把 shard 临时替换为 full weight。"""
        for parameter in mod.parameters(recurse=False):
            metadata = self.shard_metadata[id(parameter)]

            # 若此前没有预取，当前层自己发起通信。
            if metadata["gather_handle"] is None:
                self.start_all_gather(mod)

            metadata["gather_handle"].wait()

            full_weight = torch.cat(
                metadata["gathered_shards"],
                dim=metadata["shard_dim"],
            ).view(metadata["full_shape"])

            parameter.data = full_weight

            # full_weight 已独立保存，临时通信 buffer 可释放。
            metadata["gathered_shards"] = None
            metadata["gather_handle"] = None

    def forward_pre_hook(self):
        def hook(mod: nn.Module, inputs):
            # 当前层必须先有完整权重。
            self.wait_and_materialize(mod)

            # 当前层计算开始前，异步预取下一层。
            current_index = self.module_indices[id(mod)]
            next_index = current_index + 1

            if next_index < len(self.sharded_modules):
                next_mod = self.sharded_modules[next_index]
                self.start_all_gather(next_mod)

        return hook

    def backward_pre_hook(self):
        def hook(mod:nn.Module, grad_output):
            self.wait_and_materialize(mod)
            current_index=self.module_indices[id(mod)]
            pre_index=current_index-1
            if pre_index>=0:
                pre_mod=self.sharded_modules[pre_index]
                self.start_all_gather(pre_mod)

        return hook

    def make_gradient_hook(self):
        def hook(parameter: torch.Tensor):
            metadata = self.shard_metadata[id(parameter)]

            # 此时 parameter.grad 是完整参数的 full gradient。
            # The full gradient follows the compute dtype (e.g. FP16), but
            # master weights and optimizer updates remain FP32.  Gloo/NCCL
            # also require reduce-scatter input and output dtypes to match.
            full_grad = parameter.grad.to(torch.float32)

            # 恢复 FP32 master shard，释放 full weight。
            parameter.data = metadata["local_shard"]
            metadata["local_shard"] = None

            # 每个 rank 最终只保留本 rank 的 gradient shard。
            local_grad = torch.empty_like(parameter.data,dtype=torch.float32,)

            # 先除 world_size，后续 SUM reduce-scatter 得到平均梯度。
            full_grad.div_(self.world_size)

            handle = dist.reduce_scatter_tensor(
                local_grad,
                full_grad,
                op=ReduceOp.SUM,
                async_op=True,
            )

            # 异步通信期间必须保留这些 tensor。
            metadata["full_grad_buffer"] = full_grad
            metadata["local_grad"] = local_grad
            metadata["reduce_scatter_handle"] = handle

            # 不要让 parameter 保留完整 gradient。
            parameter.grad = None

        return hook

    def forward_hook(self):
        def hook(m:torch.nn.Module, args, output):
            for parameter in m.parameters(recurse=False):
                metadata=self.shard_metadata.get(id(parameter))
                parameter.data=metadata["local_shard"]
                metadata["local_shard"]=None

        return hook

    def forward(self,x):
        return self.module(x)


    def finish_gradient_synchronization(self):
        for parameter_id, metadata in self.shard_metadata.items():
            handle = metadata.get("reduce_scatter_handle")

            if handle is None:
                continue

            handle.wait()

            parameter = None
            module = metadata["module"]
            name = metadata["name"]

            # 找回该层直接拥有的 Parameter。
            for current_name, current_parameter in module.named_parameters(
                    recurse=False
            ):
                if current_name == name:
                    parameter = current_parameter
                    break

            parameter.grad = metadata["local_grad"]

            metadata["full_grad_buffer"] = None
            metadata["local_grad"] = None
            metadata["reduce_scatter_handle"] = None
            
        for parameter in self.module.parameters():
            if id(parameter) in self.shard_metadata:
                continue

            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=ReduceOp.SUM)
                parameter.grad.div_(self.world_size)

    def gather_full_params(self)-> dict[str, torch.Tensor]:
        full_tensor={}
        with torch.no_grad():
            for name,parameter in self.module.named_parameters():
                metadata=self.shard_metadata.get(id(parameter))

                if metadata is None:
                    full_tensor[name]=parameter.detach().clone()
                    continue

                gathered_shards = [
                    torch.empty_like(parameter.data)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_shards,parameter.data)
                full_tensor[name]=torch.cat(gathered_shards,dim=metadata["shard_dim"]).view(metadata["full_shape"])
        return full_tensor

