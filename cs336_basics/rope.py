import torch
from torch.nn import Module


class Rope(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        assert self.d_k % 2 == 0
        # theta(i, k) = i / base ^ (2k - 2)/d
        pos = torch.arange(self.max_seq_len, device=device)
        sub = torch.arange(self.d_k // 2, device=device)
        freq = self.theta ** ((-2 * sub) / d_k)
        theta_matrix = pos[:, None] * freq[None, :]
        cos = torch.cos(theta_matrix)
        sin = torch.sin(theta_matrix)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        even = x[..., ::2]
        odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = even * cos - odd * sin
        x_odd = even * sin + odd * cos
        x_stack = torch.stack([x_even, x_odd], dim=-1)
        return torch.flatten(x_stack, -2)
