import torch
def silu(x: torch.Tensor) -> torch.Tensor:
    return x/(1+torch.exp(-x))