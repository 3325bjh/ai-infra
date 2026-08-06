import math
from token import tok_name

import torch
import torch.nn as nn
from einops.array_api import rearrange
from numpy import dtype
from torch.nn import factory_kwargs
from torch.xpu import device


class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_size,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((vocab_size,embedding_size),**factory_kwargs))
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]



class Linear(nn.Module):
    def __init__(self,in_dim,out_dim,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((out_dim,in_dim),**factory_kwargs))
        std=(2.0/(in_dim+out_dim))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=-3*std)
    def forward(self,x):
        return torch.einsum("...i,oi -> ...o",x,self.weight)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.d_k=d_k
        powers=torch.arange(0,d_k,2,device=device)/d_k
        freq=1.0/(theta**powers)
        t=torch.arange(max_seq_len,device=device).float()
        freqs_matrix=torch.outer(t,freq)
        self.register_buffer("cos_cached",freqs_matrix.cos(),persistent=False)
        self.register_buffer("sin_cached",freqs_matrix.sin(),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]

        if x.ndim>cos.ndim and cos.ndim>=3:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)

        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)

        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        output=torch.empty_like(x)
        output[...,0::2]=x_even*cos-x_odd*sin
        output[...,1::2]=x_odd*cos+x_even*sin

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len=None,theta=None,device=None,dtype=None):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device)
        else:
            self.rope=None
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        b,s,d=x.shape
        q=rearrange(self.q_proj(x),"...s(hd)->...hsd",h=self.num_heads)
        k=rearrange(self.q_proj(x),"...s(hd)->...hsd",h=self.num_heads)
        v=rearrange(self.q_proj(x),"...s(hd)->...hsd",h=self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arrange(s,device=x.device).expand(b,s)
            q=self.rope(q,token_positions)
            v=self.rope(k,token_positions)
        mask=torch.tril(torch.ones((s,s),device=x.device,dtype=torch.bool))
        attn_out=scaled_dot_product_attention(q,k,v,mask=mask)
        attn_out=rearrange(attn_out,"...hsd->...s(hd)")
        return self.output_proj(attn_out)



def softmax(x,dim):
    max_x,_=torch.max(x,dim=dim,keepdim=True)
    x=x-max_x
    exp_x=torch.exp(x)
    sum_exp=torch.sum(exp_x,dim=dim,keepdim=True)
    return exp_x/sum_exp

def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask:torch.tensor=None
)->torch.Tensor:
    dk=Q.size(-1)
    scores=torch.einsum("...nk,...mk ->...nm",Q,K)/math.sqrt(dk)
    if mask is not None:
        scores=scores.masked_fill(mask==False,float('-inf'))
    probs=softmax(scores,dim=-1)
    return torch.einsum("...nm,...mk->...nk",probs,V)

def silu_fn(in_features):
    return in_features*torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.d_ff=d_ff
        self.d_model=d_model
        self.w1=Linear(d_model,d_ff,device,dtype)
        self.w3=Linear(d_model,d_ff,device,dtype)
        self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate=silu_fn(self.w1(x))
        signal=self.w3(x)
        return self.w2(gate*signal)


class LayerNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.ones(d_model),**factory_kwargs)
        self.bias=nn.Parameter(torch.zeros(d_model),**factory_kwargs)
        self.eps=eps
    def foward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x_float=x.to(torch.float32)
        mean=x_float.mean(dim=-1,keepdim=True)
        var=x_float.var(dim=-1,keepdim=True,unbiased=True)
        x_normed=(x_float-mean)/torch.sqrt(var+self.eps)
        result=x_normed*self.weight+self.bias
        return result.to(in_dtype)


class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.ones(d_model),**factory_kwargs)
        self.eps=eps
    def foward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x_float=x.to(torch.float32)
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
        result=(x_float/rms)*self.weight
        return result.to(in_dtype)

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float,device=None,dtype=None,use_rms_norm=True,norm_mode=None,ffn_type=None):
        super().__init__()
        self.attn=CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2=RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        x=x*self.attn(self.ln1(x),token_positions=token_positions)
        x=x*self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,max_seq_len:int,d_model:int,num_layers:int,num_heads:int,d_ff:int,rope_theta:float,device=None,dtype=None,use_rms_norm:bool=True,norm_mode:str="pre",ffn_type:str="swiglu"):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.token_embeddings=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([
            TransformerBlock(
                d_model,num_heads,d_ff,max_seq_len,rope_theta,
                device=device,dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            ) for _ in range(num_layers)
        ])

        if use_rms_norm:
            self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
        else:
            self.ln_final=nn.Identity()
        self.lm_head=Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        b,s=token_ids.shape
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)
        x=self.token_embeddings(token_ids)
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)
        x=self.ln_final(x)
        return self.lm_head(x)
    @torch.no_grad()
    def generate(
          self,
            prompt_ids:torch.Tensor,
            max_new_tokens:int,
            eos_token_id:int=None,
            temperature:float=1.0,
            top_p:float=1.0
    ):
        self.eval()
        generated=prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond=generated[:,-self.context_length:]
            logits=self.forward(idx_cond)
            logits=logits[:,-1,:]
            if temperature!=1.0:
                logits=logits/(temperature+1e-8)
            if top_p<1.0:
                logits=self._top_p_filter(logits,top_p)
            probs=softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,num_samples=1)
            generated=torch.cat((generated,next_token),dim=1)

            if eos_token_id is not None and (next_token==eos_token_id).all():
                break

            return generated

    def _top_p_filter(self,logits:torch.Tensor,p:float)->torch.Tensor:
        sorted_logits,sorted_indices=torch.sort(logits,descending=True,dim=-1)
        cumulative_probs=torch.cumsum(softmax(sorted_logits,dim=-1),dim=-1)
        sorted_indices_to_remove=cumulative_probs>p
        sorted_indices_to_remove[...,1:]=sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0]=False
        indices_to_remove=sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
        logits=logits.masked_fill(indices_to_remove,float('-inf'))
        return logits