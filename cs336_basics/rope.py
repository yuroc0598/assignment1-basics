import torch
from torch.nn import Module


class Rope(Module):
    def __init__(self, theta: float, dk: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_k = dk
        assert self.d_k % 2 == 0
        # theta(i, k) = i / base ^ (2k - 2)/d
        pos = torch.arange(max_seq_len, device=device)
        dim = torch.arange(self.d_k // 2, device=device)
        freq = theta ** ((-2 * dim) / dk)
        angles = pos[:, None] * freq[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x could be of shape (b, seq, dk) or (b, h, seq, dk)
        assert x.shape[-1] == self.d_k

        # if x has shape (b, seq, dk), legit token_positions could be (seq,), (b, seq), (1, seq)
        if x.ndim == 3:
            assert token_positions.ndim in [1, 2]
        elif x.ndim == 4:
            # if x has shape (b, h, seq, dk), legit token_positions could be (seq,), (b, seq), (1, seq), (b, h, seq)
            assert token_positions.ndim in [1, 2, 3]
            if token_positions.ndim == 2:
                token_positions = token_positions.unsqueeze(dim=1)

        even = x[..., ::2]
        odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = even * cos - odd * sin
        x_odd = even * sin + odd * cos
        x_stack = torch.stack([x_even, x_odd], dim=-1)
        return torch.flatten(x_stack, -2).to(x.dtype)
