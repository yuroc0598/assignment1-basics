import math
import torch
from einops import einsum, reduce
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
    # scaled dot product
    # k: [B, ..., S, dk]
    # q: [B, ..., S, dk]
    # v: [B, ..., S, dv]
    # FLOPs: 4bssd + 2bhss

    # attention: softmax((k qT) / sqr(dk)) * v
    # mask: [S, S]

    dk = k.shape[-1]
    # FLOPs 2bssd
    pre_softmax = einsum(q, k, "... s dk, ... t dk -> ... s t")
    # FLOPs bhss
    pre_softmax = pre_softmax / math.sqrt(dk)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
    # FLOPs  bhss + 2bssd
    return einsum(softmax(pre_softmax), v, "... s t, ... t dv -> ... s dv")


def cross_entropy(logits: Tensor, target: Tensor):
    """
    input shape B S V
    target shape B S
    cross entropy loss:
    L = -logP = -log softmax(x) = -log exp(xi)/sum(exp(xi))
    = logsum(exp(xi)) - xi
    = log sum(exp(xi-max))*exp(max) - xi
    = log sum(exp(xi-max)) + max - xi
    """
    max_x = torch.max(logits, dim=-1, keepdim=True).values
    exp_x = torch.exp(logits - max_x)
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)
    log = (torch.log(sum_exp_x) + max_x).squeeze(-1)
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(-1)
    xi = logits.gather(-1, target).squeeze(-1)
    loss = log - xi
    # loss has shape B S
    return loss.mean()
