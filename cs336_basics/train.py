from pathlib import Path
from cs336_basics.bpe_opt import BPE
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.experiment_logging import ExperimentLogger
import typer
import torch
from cs336_basics.utils import cross_entropy, data_loading, cos_learning_rate, gradient_clipping, save_checkpoint
import os
import shutil
import urllib.request
import numpy as np
from datetime import datetime, UTC


DEFAULT_TOKEN_ID_FILE_PATH = "token_ids.bin"
DEFAULT_TOKENIZER_FILE_PATH = "tokenizer.json"
TOKEN_STORAGE_DTYPE = "uint32"
NUMPY_TOKEN_DTYPE = "<u4"
TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

app = typer.Typer()


def _download(url: str, dest: str) -> None:
    """Download url to dest atomically (via a .part file), creating parent dirs as needed."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(dest_path)


def _ensure_local_file(path: str, url: str) -> None:
    if not os.path.exists(path):
        _download(url, path)


def _ensure_tokenized(
    token_id_file_path: str,
    text_path: str,
    text_url: str,
    tokenizer_path: str,
    vocab_size: int,
    special_tokens: list[str] | None,
) -> None:
    """Make sure token_id_file_path exists, downloading raw text / (re)tokenizing only as needed."""
    if os.path.exists(token_id_file_path):
        return
    _ensure_local_file(text_path, text_url)
    if os.path.exists(tokenizer_path):
        bpe = BPE.load(tokenizer_path)
    else:
        bpe = BPE(special_tokens=special_tokens, vocab_size=vocab_size, input_path=text_path)
        bpe.train()
        bpe.save(tokenizer_path, overwrite=True)
    bpe.encode_file(text_path, token_id_file_path, TOKEN_STORAGE_DTYPE)


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    """Convert a CLI-supplied dtype name (e.g. "float32") into a torch.dtype."""
    if dtype is None:
        return None
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise typer.BadParameter(f"unknown torch dtype: {dtype!r}")
    return resolved


@torch.no_grad()
def _evaluate(lm, token_stream, batch_size, context_len, device, eval_iters) -> float:
    lm.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = data_loading(token_stream, batch_size, context_len, device)
        logits = lm(inputs)
        losses.append(cross_entropy(logits, targets).item())
    lm.train()
    return sum(losses) / len(losses)


@app.command()
def tokenize(
    vocab_size: int,
    tokenizer_training_path: str,
    training_dataset_file_path: str,
    tokenizer_output_path: str = DEFAULT_TOKENIZER_FILE_PATH,
    token_id_file_path: str = DEFAULT_TOKEN_ID_FILE_PATH,
    special_tokens: list[str] | None = None,
):
    bpe = BPE(special_tokens=special_tokens, vocab_size=vocab_size, input_path=tokenizer_training_path)
    bpe.train()
    bpe.save(tokenizer_output_path, overwrite=True)
    bpe.encode_file(training_dataset_file_path, token_id_file_path, TOKEN_STORAGE_DTYPE)
    return bpe


@app.command()
def encode(
    tokenizer_path: str,
    dataset_file_path: str,
    token_id_file_path: str,
):
    """Encode a text file with an already-trained tokenizer (e.g. a validation split)."""
    bpe = BPE.load(tokenizer_path)
    bpe.encode_file(dataset_file_path, token_id_file_path, TOKEN_STORAGE_DTYPE)
    return bpe


@app.command()
def train(
    vocab_size: int,
    context_len: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
    max_gradient: float,
    steps: int,
    checkpoint_step: int,
    checkpoint_folder: str,
    token_id_file_path: str = DEFAULT_TOKEN_ID_FILE_PATH,
    val_token_id_file_path: str | None = None,
    tokenizer_path: str | None = None,
    train_text_path: str | None = None,
    val_text_path: str | None = None,
    eval_interval: int = 100,
    eval_iters: int = 20,
    run_name: str | None = None,
    log_dir: str = "runs",
    use_wandb: bool = False,
    wandb_project: str = "cs336-basics",
    wandb_entity: str | None = None,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    theta: float = 10_000.0,
    device: str | None = None,
    dtype: str | None = None,
    special_tokens: list[str] | None = None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    if not 0 <= Tw < Tc:
        raise ValueError("expected 0 <= Tw < Tc")
    if max_gradient <= 0:
        raise ValueError("expected max_gradient > 0")
    if steps <= 0:
        raise ValueError("expect step > 0")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")

    if tokenizer_path is None:
        tokenizer_path = str(Path(token_id_file_path).parent / DEFAULT_TOKENIZER_FILE_PATH)
    if train_text_path is None:
        train_text_path = str(Path(token_id_file_path).parent / "TinyStoriesV2-GPT4-train.txt")
    _ensure_tokenized(token_id_file_path, train_text_path, TINYSTORIES_TRAIN_URL, tokenizer_path, vocab_size, special_tokens)

    if val_token_id_file_path is not None:
        if val_text_path is None:
            val_text_path = str(Path(val_token_id_file_path).parent / "TinyStoriesV2-GPT4-valid.txt")
        _ensure_tokenized(
            val_token_id_file_path, val_text_path, TINYSTORIES_VALID_URL, tokenizer_path, vocab_size, special_tokens
        )

    torch_dtype = _resolve_dtype(dtype)
    lm = TransformerLM(
        vocab_size,
        context_len,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        theta,
        special_tokens,
        device,
        torch_dtype,
    )
    model_params = list(lm.parameters())
    adamw = AdamW(
        model_params,
        lr=alpha_max,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    token_id_stream = np.memmap(token_id_file_path, dtype=NUMPY_TOKEN_DTYPE, mode="r")
    if len(token_id_stream) <= context_len:
        raise ValueError("token file is too short")

    val_token_id_stream = None
    if val_token_id_file_path is not None:
        val_token_id_stream = np.memmap(val_token_id_file_path, dtype=NUMPY_TOKEN_DTYPE, mode="r")
        if len(val_token_id_stream) <= context_len:
            raise ValueError("validation token file is too short")

    checkpoint_folder = Path(checkpoint_folder)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    config = {
        "vocab_size": vocab_size,
        "context_len": context_len,
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "batch_size": batch_size,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "Tw": Tw,
        "Tc": Tc,
        "max_gradient": max_gradient,
        "steps": steps,
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "theta": theta,
        "dtype": dtype,
    }
    logger = ExperimentLogger(
        run_name,
        config,
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )

    try:
        for step in range(steps):
            adamw.zero_grad(set_to_none=True)
            inputs, targets = data_loading(token_id_stream, batch_size, context_len, device)
            logits = lm(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            gradient_clipping(model_params, max_gradient)
            if alpha_max:
                current_lr = cos_learning_rate(step, alpha_max, alpha_min, Tw, Tc)
            else:
                current_lr = lr
            for group in adamw.param_groups:
                group["lr"] = current_lr
            adamw.step()

            metrics = {"train_loss": loss.item(), "lr": current_lr}
            if val_token_id_stream is not None and (step % eval_interval == 0 or step == steps - 1):
                metrics["val_loss"] = _evaluate(
                    lm, val_token_id_stream, batch_size, context_len, device, eval_iters
                )
            logger.log(step, **metrics)

            if step != 0 and step % checkpoint_step == 0:
                timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                checkpoint_file_path = checkpoint_folder / f"checkpoint_{step}_{timestamp}.pt"
                save_checkpoint(lm, adamw, step, checkpoint_file_path)

        if step % checkpoint_step != 0:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            checkpoint_file_path = checkpoint_folder / f"checkpoint_{step}_{timestamp}.pt"
            save_checkpoint(lm, adamw, step, checkpoint_file_path)
    finally:
        logger.close()


if __name__ == "__main__":
    app()
