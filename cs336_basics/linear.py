import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import Parameter
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.init_params()

    def init_params(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        trunc_normal_(self.weight, mean=0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x):
        return einsum(self.weight, x, "out_features in_features, ... in_features  -> ... out_features")
