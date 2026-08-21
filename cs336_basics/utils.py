import torch


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
