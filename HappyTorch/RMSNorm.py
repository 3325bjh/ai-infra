import torch


def rms_norm(x:torch.Tensor, weight, eps=1e-6):
    rms=torch.sqrt((x**2).mean(dim=-1,keepdim=True)+eps)
    return x/rms*weight