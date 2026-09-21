import os

import torch

from cs336_basics.bpe_opt import BPE
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import softmax


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the low-probability tail outside the nucleus and renormalize.

    Keeps the smallest set of highest-probability tokens whose cumulative
    probability covers top_p, always keeping at least the top token.
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Drop a token once the mass strictly before it already reaches top_p,
    # so the kept set is the smallest one whose cumulative probability >= top_p.
    drop_mask = (cumulative_probs - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)


@torch.no_grad()
def decode(
    model: TransformerLM,
    tokenizer: BPE,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    end_of_text: str = "<|endoftext|>",
    device=None,
) -> str:
    """Sample a completion for `prompt` from `model` until end_of_text or max_new_tokens.

    temperature scales the logits before softmax (temperature <= 0 means greedy
    decoding). top_p, if given, restricts sampling to the smallest set of
    tokens whose cumulative probability is at least top_p (nucleus sampling).
    The returned text excludes the end_of_text token itself, if generated.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if end_of_text not in tokenizer.special_token_to_id:
        raise ValueError(f"end_of_text token {end_of_text!r} is not in the tokenizer's special tokens")

    if device is None:
        device = next(model.parameters()).device

    end_of_text_id = tokenizer.special_token_to_id[end_of_text]

    token_ids = list(tokenizer.encode(prompt))
    if not token_ids:
        # Nothing to condition on: seed with end_of_text as a BOS-like token,
        # matching the convention that <|endoftext|> also marks the start of a document.
        token_ids = [end_of_text_id]
    prompt_len = len(token_ids)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = tokens[:, -model.context_len :]
            logits = model(context)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = softmax(next_token_logits / temperature, dim=-1)
                if top_p is not None:
                    probs = _top_p_filter(probs, top_p)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=-1)
            if next_token.item() == end_of_text_id:
                break
    finally:
        model.train(was_training)

    generated_ids = tokens[0, prompt_len:].tolist()
    if generated_ids and generated_ids[-1] == end_of_text_id:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids)
