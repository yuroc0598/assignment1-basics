import torch
from torch.nn import Module


class Rope(Module):
    def __init__(self, theta: float, dk: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = dk
        self.max_seq_len = max_seq_len
        assert self.d_k % 2 == 0
        # theta(i, k) = i / base ^ (2k - 2)/d
        pos = torch.arange(self.max_seq_len, device=device)
        dim = torch.arange(self.d_k // 2, device=device)
        freq = self.theta ** ((-2 * dim) / dk)
        angles = pos[:, None] * freq[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert token_positions.max() < self.max_seq_len
        assert x.shape[-1] == self.d_k
        assert token_positions.ndim in [1, x.ndim - 1]
        even = x[..., ::2]
        odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = even * cos - odd * sin
        x_odd = even * sin + odd * cos
        x_stack = torch.stack([x_even, x_odd], dim=-1)
        return torch.flatten(x_stack, -2)
