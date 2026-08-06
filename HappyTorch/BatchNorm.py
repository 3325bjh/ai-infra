import torch
def my_batch_norm(x, gamma, beta, eps=1e-5):
    mean=torch.mean(x,dim=0,keepdim=True)
    var=torch.var(x,dim=0,keepdim=True,unbiased=False)
    return gamma*(x-mean)/torch.sqrt(var+eps)+beta