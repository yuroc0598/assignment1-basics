import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import Parameter
from einops import einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.init_params()

    def init_params(self):
        trunc_normal_(self.weight, mean=0, std=1, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
