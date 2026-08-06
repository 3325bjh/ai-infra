import math

import torch
import torch.nn as nn
from einops import rearrange
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)

    def forward(self, Q, K, V):
        B,S_q,_=Q.shape
        S_k=K.shape[1]
        Q=self.W_q(Q)
        K=self.W_k(K)
        V=self.W_v(V)
        Q=Q.view(B,S_q,self.num_heads,self.d_k).transpose(1,2)
        K=K.view(B,S_k,self.num_heads,self.d_k).transpose(1,2)
        V=V.view(B,S_k,self.num_heads,self.d_k).transpose(1,2)
        # Q=rearrange(Q,"...s(hd)->...hsd",h=self.num_heads)
        # K=rearrange(K,"...s(hd)->...hsd",h=self.num_heads)
        # #(B,H,S,d_k)
        # V=rearrange(V,"...s(hd)->...hsd",h=self.num_heads)
        #(B,H,S,S)
        qk=torch.matmul(Q,K.transpose(-1,-2))
        weight=torch.softmax(qk/math.sqrt(self.d_k),dim=-1)
        #(B,H,S,d_k)
        out=torch.matmul(weight,V)
        out=out.transpose(1,2).contiguous().view(B,S_q,-1)
        # out=rearrange(out,"...hsd->s(hd)")
        return self.W_o(out)
