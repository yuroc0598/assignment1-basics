import torch
from torch.nn import Module
from torch import Tensor
from .utils import sdpa
from .linear import Linear
from einops import rearrange
from .rope import Rope


class Cmsa(Module):
    # causal multi head self attention
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        assert d_model % num_heads == 0
        self.dk = d_model // num_heads
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if max_seq_len is not None:
            if theta is not None:
                self.theta = theta
            else:
                self.theta = 10000
            self.max_seq_len = max_seq_len
            self.rope = Rope(self.theta, self.dk, self.max_seq_len, device=device, dtype=dtype)
        else:
            self.rope = None

    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        k = rearrange(self.k_proj(x), "b seq (h dk) -> b h seq dk", h=self.h)
        q = rearrange(self.q_proj(x), "b seq (h dk) -> b h seq dk", h=self.h)
        seq = k.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            k = self.rope(k, token_positions)
            q = self.rope(q, token_positions)
        v = rearrange(self.v_proj(x), "b seq (h dk) -> b h seq dk", h=self.h)
        mask = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))
        dp = sdpa(q=q, k=k, v=v, mask=mask)
        dp = rearrange(dp, "b h seq dk -> b seq (h dk)")
        return self.output_proj(dp)
