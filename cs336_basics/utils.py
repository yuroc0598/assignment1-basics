import math
import torch
import numpy as np
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


def cos_learning_rate(t, alpha_max, alpha_min, Tw, Tc):
    """
    Warmup t < Tw: t * alpha_max / Tw
    cosine annealing Tw <= t <= Tc: alpha_min + 1/2 *(1 + cos(pie * (t-Tw)/(Tc-Tw)))(alpha_max - alpha_min)
    post-annealing t > Tc: alpha_min
    """
    if t < Tw:
        return t * alpha_max / Tw
    if Tw <= t <= Tc:
        return alpha_min + 1 / 2 * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min)
    return alpha_min


def gradient_clipping(params, M, eps=10**-6) -> None:
    """
    params: a list of Parameters
    M: max l2-norm
    update the gradients in-place
    """
    val = math.sqrt(sum(torch.sum(p.grad**2) for p in params if p.grad is not None))
    if val > M:
        fac = M / (eps + val)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(fac)


def data_loading(x, batch_size, context_len, device):
    starts = np.random.randint(0, len(x) - context_len, size=batch_size)
    inputs = np.stack([x[i : i + context_len] for i in starts])
    targets = np.stack([x[i + 1 : i + 1 + context_len] for i in starts])
    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)
    return inputs, targets
