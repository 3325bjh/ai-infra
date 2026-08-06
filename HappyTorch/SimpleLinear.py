import torch
import torch.nn as nn
import math
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight=nn.Parameter(torch.randn((out_features,in_features))/math.sqrt(in_features))
        self.bias=nn.Parameter(torch.zeros((out_features,)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T+self.bias
if __name__ == '__main__':
    bias=torch.zeros((100,))
    print(bias.shape)