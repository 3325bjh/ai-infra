import torch
import torch.nn as nn
class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate_proj=nn.Linear(d_in,d_out)
        self.up_proj=nn.Linear(d_in,d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate=self.gate_proj(x)
        up=self.up_proj(x)
        swish=gate*torch.sigmoid(gate)
        return swish*up