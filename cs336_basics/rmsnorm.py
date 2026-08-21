import torch
from torch.nn import Parameter, Module
from einops import reduce


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def root_mean_square(self, x: torch.Tensor):
        return torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self.root_mean_square(x)
        res = x / rms * self.weight
        res = res.to(in_dtype)
        return res
