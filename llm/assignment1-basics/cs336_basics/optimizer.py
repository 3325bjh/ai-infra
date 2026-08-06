import math

import torch
from torch.optim import Optimizer
from collections.abc import Iterable

class AdamW(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if not 0.0<=betas[0]<1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[0]}")
        if not 0.0<=betas[1]<1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[1]}")
        if eps<0.0:
            raise ValueError(f"Invalid epsilon value:{eps}")
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self):
        loss=None
        for group in self.param_groups:
            beta1,beta2=group["betas"]
            eps=group['eps']
            lr=group['lr']
            wd=group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad
                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['exp_avg']=torch.zeros_like(p,memory_format=torch.preserve_format)
                    state['exp_avg_sq']=torch.zeros_like(p,memory_format=torch.preserve_format)
                exp_avg,exp_avg_sq=state['exp_avg'],state['exp_avg_sq']
                state['step']+=1
                t=state['step']
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcuml_(grad,grad,value=1-beta2)
                bias_correction1=1-beta1**t
                bias_correction2=1-beta2**t
                step_size=lr*(math.sqrt(bias_correction2)/bias_correction1)
                denom=exp_avg.sqrt().add_(eps)
                p.addcdiv_(exp_avg,denom,value=-step_size)
                if wd!=0:
                    p.add_(p,alpha=-lr*wd)
        return loss


def clip_gradient_norm(parameters:Iterable[torch.nn.Parameter],max_norm:float):
    params_with_grad=[p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm=0.0
    for p in params_with_grad:
        param_norm=torch.norm(p.grad.detach(),p=2)
        total_norm+=param_norm**2
    total_norm=total_norm**0.5
    eps=1e-6
    if total_norm>max_norm:
        clip_conf=max_norm/(total_norm+eps)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_conf)
