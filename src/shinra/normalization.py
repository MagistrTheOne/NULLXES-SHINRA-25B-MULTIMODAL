import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Keep float64 for numerical gradient checks; FP16/BF16 accumulate in FP32.
        y = x if x.dtype == torch.float64 else x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.to(y.dtype)).to(
            x.dtype
        )
