from torch.nn import Module
from .linear import Linear
from torch.nn import Sigmoid

sigmoid = Sigmoid()

# sigmoid(x)
# SiLu (x) = x * sigmoid(x)
# GLu (x) = sigmoid(w1 x) * w2 x
# swiglu (x) = w3 silu(w1 x) * w2 x


class SwiGlu(Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        if not d_ff:
            self.d_ff = round((8 / 3) * d_model / 64) * 64
        else:
            self.d_ff = d_ff
        self.linear1 = Linear(in_features=d_model, out_features=self.d_ff)
        self.linear2 = Linear(in_features=d_model, out_features=self.d_ff)
        self.linear3 = Linear(in_features=self.d_ff, out_features=d_model)

    def forward(self, x):
        # W3 swish(w1 x) * (w2 x)
        # x d_model
        out_l1 = self.linear1(x)
        silu_l1 = sigmoid(out_l1) * out_l1
        return self.linear3(silu_l1 * self.linear2(x))
