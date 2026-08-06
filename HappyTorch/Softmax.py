import torch
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max,_=torch.max(x,dim=dim,keepdim=True)
    x=x-x_max
    exp_x=torch.exp(x)
    sum_exp=torch.sum(exp_x,dim=dim,keepdim=True)
    return exp_x/sum_exp

# def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     x_max=torch.max(x,dim=dim,keepdim=True).values
#     e_x=torch.exp(x-x_max)
#     return e_x/e_x.sum(dim=dim,keepdim=True)