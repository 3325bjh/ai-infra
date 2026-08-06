import math

import torch
def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)