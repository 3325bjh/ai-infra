import torch
import torch.distributed as dist
from init_parallel import ParallelState
from mappings import copy_to_tp,reduce_from_tp
def divide(a,b):
    assert a%b==0
    return a//b
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        tp=ParallelState.tp_size
        self.out_part=divide(out_features,tp)
        self.weight=torch.nn.Parameter(torch.empty(self.out_part,in_features).cuda())
        torch.nn.init.normal_(self.weight,std=0.02)
    def forward(self,x):
        x=copy_to_tp(x)
        return torch.matmul(x,self.weight.t())

class RowParallelLinear(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        tp=ParallelState.tp_size
        self.in_part=divide(in_features,tp)
        self.weight=torch.nn.Parameter(torch.empty(out_features,self.in_part.cuda()))
        torch.nn.init.normal_(self.weight, std=0.02)
    def forward(self,x):
        out=torch.matmul(x,self.weight.t())
        return reduce_from_tp(out)

class ParallelMLP(torch.nn.Module):
    def __init__(self,h):
        super().__init__()
        self.fc1=ColumnParallelLinear(h,4*h)
        self.fc2=RowParallelLinear(4*h,h)
    def forward(self,x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))