from torch.nn import Parameter
from torch.optim import Optimizer
import math
import torch
from typing import Callable


class SGD(Optimizer):
    def __init__(self, params, lr):
        if lr < 0:
            raise ValueError(f"default learning rate {lr} should not be negative")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for pg in self.param_groups:
            lr = pg["lr"]
            for p in pg["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= (lr / math.sqrt(t + 1)) * p.grad.data
                state["t"] = t + 1
        return loss


def main():
    for lr in [10, math.e, math.e**2, math.e**3]:
        weights = Parameter(5 * torch.randn((10, 10)))
        sgd = SGD([weights], lr)
        epoch = 10
        for _ in range(epoch):
            sgd.zero_grad()
            loss = (weights**2).mean()
            loss.backward()
            sgd.step()
            print(f"learning rate is {sgd.param_groups[0]["lr"]} and loss is {loss}")


if __name__ == "__main__":
    main()
