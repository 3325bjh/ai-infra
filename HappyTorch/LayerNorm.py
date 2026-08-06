import torch
import torch.nn as nn
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean=torch.mean(x,dim=-1,keepdim=True)
    var=torch.var(x,dim=-1,keepdim=True,unbiased=False)
    return gamma*(x-mean)/torch.sqrt(var+eps)+beta