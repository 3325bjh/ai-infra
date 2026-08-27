import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    m=torch.max(logits,dim=-1,keepdim=True).values
    target_logits=torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    shifted_logits=logits-m
    log_sum_exp=m.squeeze(-1)+torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    loss=log_sum_exp-target_logits
    return torch.mean(loss)

