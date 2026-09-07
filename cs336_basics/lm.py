from torch.nn import Module
from .transformer_block import Transformer


class LM(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = None,
        vocab_size: int = None,
        context_length: int = None,
        num_layers: int = 1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.transformers = [
            Transformer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]


    def forward(self):
