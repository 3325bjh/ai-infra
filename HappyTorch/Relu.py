import torch
def relu(x: torch.Tensor) -> torch.Tensor:
    # mask=x>0
    # return torch.where(mask,x,0)
    return (x*(x>0)).float()