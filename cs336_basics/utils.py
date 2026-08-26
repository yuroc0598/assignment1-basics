import math
import torch
from einops import einsum
from torch import Tensor


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 1. Find the maximum value along the target dimension
    # keepdim=True preserves dimensions for correct broadcasting
    max_x = torch.max(x, dim=dim, keepdim=True).values

    # 2. Subtract the max and compute the exponentials (prevents overflow)
    exp_x = torch.exp(x - max_x)

    # 3. Sum the exponentials along the target dimension
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    # 4. Divide elements by the sum to get probabilities
    return exp_x / sum_exp_x


def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None):
    # k: [B, ..., S, dk]
    # q: [B, ..., S, dk]
    # v: [B, ..., S, dv]

    # attention: softmax((k qT) / sqr(dk)) * v
    # mask: [S, S]

    dk = k.shape[-1]
    pre_softmax = einsum(q, k, "... s dk, ... t dk -> ... s t")
    pre_softmax = pre_softmax / math.sqrt(dk)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
    return einsum(softmax(pre_softmax), v, "... s t, ... t dv -> ... s dv")
