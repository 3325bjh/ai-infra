import torch
import math
def scaled_dot_product_attention(Q, K, V):
    dk=Q.shape[-1]
    qk=torch.einsum("...mk,...nk->...mn",Q,K)
    score=torch.softmax(qk/math.sqrt(dk),dim=-1)
    return torch.einsum("...mn,...nk->...mk",score,V)