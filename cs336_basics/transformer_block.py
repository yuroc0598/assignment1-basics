from torch.nn import Module
from torch import Tensor
from .cmsa import Cmsa
from .rmsnorm import RMSNorm
from .swiglu import SwiGlu


class Transformer(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.cmsa = Cmsa(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.rmsnorm_1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm_2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.swiglu = SwiGlu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor = None):
        # y = cmsa(rmsnorm(x)) + x
        y = self.cmsa(self.rmsnorm_1(x), token_positions) + x
        # ffn(rmsnorm(y)) + x
        return self.swiglu(self.rmsnorm_2(y)) + y
