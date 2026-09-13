import math
import torch
from typing import Callable, Iterable, Optional


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the bias-corrected step size for this iteration
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # Apply weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Apply moment-adjusted parameter update
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

        return loss


"""
memory analaysis:

let P be the number of parameters, so
P=L(4D^2+3D*D_ff+2D)+2VD+D

adamw has m and v, each parameter is 4bytes if using fp32, so static memory is 8P
during step(), extra temp memory because of v.sqrt(), 4P
so in total 12P for peak memory
that is 12 * (L(4D^2+3D*D_ff+2D)+2VD+D)

in the case of
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1, 600)

Static memory: 12.22 GiB
Peak during step(): 18.33 GiB

so plus the memory of parameters themselves if using fp16/bf16 2bytes each

total static memory: 16.4GB
total peak memory:22.97GB

if model params use fp32 as well
total static: 18.33GB
total peak: 24.44GB


FLOPs per step(): 13 FLOPs

│ m.mul_(beta1).add_(grad, alpha=1-beta1)           │ m ← β1·m + (1-β1)·g  │ 2 mults + 1 add = 3                      │
  ├───────────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
  │ v.mul_(beta2).addcmul_(grad, grad, value=1-beta2) │ v ← β2·v + (1-β2)·g² │ 3 mults (g·g, ·(1-β2), β2·v) + 1 add = 4 │
  ├───────────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
  │ p.data.mul_(1 - lr*weight_decay)                  │ θ ← θ·(1-αλ)         │ 1 mult                                   │
  ├───────────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
  │ v.sqrt().add_(eps)                                │ √v + ε               │ 1 sqrt + 1 add = 2                       │
  ├───────────────────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
  │ p.data.addcdiv_(m, ..., value=-alpha_t)           │ θ ← θ - α_t·m/denom  │ 1 div + 1 mult + 1 add = 3               │
  └───────────────────────


Total FLOPs = 5.546e21 (400,000 steps x 1.3865e16 FLOPs/step, where per-step = 3 x forward = 3 x 4.62e15).

Achievable throughput = 495 TFLOP/s x 0.5 = 2.475e14 FLOP/s.

Time = 5.546e21 / 2.475e14 = 2.241e7 seconds = 259 days = 8.5 months = 0.71 years.

"""
