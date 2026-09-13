from torch.nn import Module, ModuleList
from .bpe_opt import BPE
from .embedding import Embedding
from .transformer_block import Transformer
from .rmsnorm import RMSNorm
from .linear import Linear

TOKENIZER_TRAINING_INPUT = (
    "/Users/yuroc/workspace/learn/assignment1-basics/cs336_basics/inputs/TinyStoriesV2-GPT4-valid.txt"
)

"""
mem/computation analysis

1. number of trainable parameters
- embedding: vocab_size x d_model
- RMSNorm: 2 x d_model x num_layers + d_model
- Transformer block (per layer) 4 x d_model^2 + 3 d_model x d_ff:
    - K weight: d_model x d_model
    - V weight: d_model x d_model
    - Q weight: d_model x d_model
    - projection: d_model x d_model
    - Swiglu: 3 x d_model x d_ff
- output project: d_model x vocab_size

in the case of
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1, 600)
total 1.64 billion

if using single-precision floating point (fp32), each parameter is 4 bytes, 6.56GB


2. FLOPs analysis

one forward through one transformer layer

RMSNorm × 2: ~8 × B × S × D
Q, K, V projections: 6 × B × S × D²
RoPE on Q, K: ~6 × B × S × D
QKᵀ: 2 × B × S² × D
Scaling: B × H × S²
Softmax: 5 x B × H × S²
Attention × V: 2 × B × S² × D
Output projection: 2 × B × S × D²
SwiGLU: 6 × B × S × D × D_ff + O(B × S × D_ff)
Residual × 2: 2 × B × S × D

Dominant terms:

Linear projections: 8 × B × S × D²
SwiGLU: 6 × B × S × D × D_ff
Attention: 4 × B × S² × D

All layers (dominant part):
num_layers x (8BSD^2+6BSDD_ff+4BDS^2) + 2BSDx(Vocab_size)

"""


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = None,
        specials: list = None,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.context_len: int = context_len
        self.num_layers: int = num_layers
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_ff: int = d_ff
        self.max_seq_len: int = context_len
        self.theta: float = theta
        self.device = device
        self.dtype = dtype
        self.tokenizer = BPE(vocab_size, TOKENIZER_TRAINING_INPUT, specials or ["<|endoftext|>"], "<|endoftext|>")
        self.train_tokenizer()
        self.transformer_blocks = ModuleList(
            [Transformer(d_model, num_heads, d_ff, context_len, theta, device, dtype) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        self.embedding = Embedding(len(self.tokenizer.vocab), self.d_model, self.device, self.dtype)  # B, S, D

    def train_tokenizer(self, tokenizer):
        self.tokenizer.train()

    def forward(self, input_text):
        # raw input text -> tokenization -> embedding -> num_layers Transformer block
        # -> Norm -> Linear -> softmax -> output logits
        token_ids = self.tokenizer.encode(input_text)  # B, S
        # go through num_layers transformer blocks
        x = self.embedding(token_ids)  # B, S, D
        for xb in self.transformer_blocks:
            x = xb(x)  # B, S, D
        # output shape is B, S, Vocab
        return self.linear(self.norm(x))
