from typing import Any, Iterable, Type
from torch.nn import Parameter
import torch
from torch.optim import Optimizer
import torch.distributed as dist

class Optimizer_state_sharding(Optimizer):
    """Replicate model weights while sharding wrapped-optimizer state by rank."""

    def __init__(self, params: Iterable[Parameter] | Iterable[dict[str, Any]], optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls=optimizer_cls
        self.rank=dist.get_rank()
        self.world_size=dist.get_world_size()
        self.parameter_owners: dict[int, int] = {}
        self.local_param_groups=[]
        self.local_optimizer=None
        self.next_parameter_index=0
        super().__init__(params, defaults=kwargs)
        self.local_optimizer=optimizer_cls(
            self.local_param_groups,
            **kwargs
        )


    def add_param_group(self,param_group:dict[str,Any]):
        super().add_param_group(param_group)
        full_group=self.param_groups[-1]
        local_parameters=[]
        for parameter in full_group["params"]:
            owner=self.next_parameter_index%self.world_size
            self.parameter_owners[id(parameter)] = owner
            if owner==self.rank:
                local_parameters.append(parameter)
            self.next_parameter_index+=1
        local_group={
            key:value
            for key,value in full_group.items()
            if key!="params"
        }
        local_group["params"]=local_parameters
        self.local_param_groups.append(local_group)
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        loss = self.local_optimizer.step(closure=closure, **kwargs)

        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    owner = self.parameter_owners[id(parameter)]
                    dist.broadcast(parameter.data, src=owner)

        return loss
