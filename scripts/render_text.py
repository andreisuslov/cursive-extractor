#!/usr/bin/env python3
"""Render arbitrary text as cursive from a locally-trained checkpoint, to a PNG.

Loads a checkpoint saved by ``scripts/train_local.py``, rebuilds the model with a
matching config, cleans the text down to the tokenizer's supported characters, and
renders it with sample.py's paragraph generation (``generate_paragraph`` +
``plot_paragraph``). Uses only sample.py public functions -- no architecture or
sampling-math change. Device auto-selects (CUDA -> MPS -> CPU); W&B is disabled.

    python scripts/render_text.py --text "hello world" --out runs/hello.png
"""

import argparse
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "disabled-for-render")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402  (must follow the env setup above)

from data import create_datasets  # noqa: E402
from model import Transformer, get_all_args  # noqa: E402
from sample import GenerationParams, generate_paragraph, plot_paragraph  # noqa: E402

RUNS_DIR = os.path.join(_REPO_ROOT, "runs")


def pick_device() -> str:
    """cuda if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_text(text: str, alphabet: str) -> str:
    """Keep only letters + space (the most reliably rendered characters), collapsing
    runs of whitespace. Anything else -- digits, punctuation, characters outside the
    tokenizer's alphabet -- is dropped. Returns the text that is actually rendered."""
    keep = {c for c in alphabet if c.isalpha() or c == " "}
    cleaned = "".join(c if c in keep else " " for c in text)
    return " ".join(cleaned.split())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render text as cursive from a local checkpoint")
    p.add_argument("--text", required=True, help="Text to render (cleaned to letters + space)")
    p.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "bigbank_local.pt"))
    p.add_argument("--dataset", default="bigbank", help="Dataset whose tokenizer/warmups to use")
    p.add_argument("--out", default=os.path.join(RUNS_DIR, "rendered_text.png"))
    # These must match the checkpoint's training config (defaults match train_local.py).
    p.add_argument("--num_words", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_ctx_head", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=600, help="Per-word generation budget")
    p.add_argument(
        "--line_width", type=float, default=8.0, help="Words wrap to a new line past this"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    cli = parse_args(argv)
    device = pick_device()

    args = get_all_args(use_argparse=False)
    args.device = device
    args.dataset_name = cli.dataset
    args.num_words = cli.num_words
    args.n_layer = cli.n_layer
    args.n_embd = cli.n_embd
    args.n_embd_context = cli.n_embd  # must equal n_embd
    args.n_ctx_head = cli.n_ctx_head
    args.max_seq_length = cli.max_seq_length
    args.seed = cli.seed

    cleaned = clean_text(cli.text, args.alphabet)
    print(f"[render] device {device} | checkpoint {cli.checkpoint}")
    print(f"[render] cleaned text ({len(cleaned.split())} words):\n  {cleaned}")

    # Build the dataset (tokenizer + warmup examples) and the model, then load weights.
    train_ds, _ = create_datasets(args)
    args.vocab_size = train_ds.get_vocab_size()
    args.block_size = train_ds.get_stroke_seq_length()
    args.context_block_size = train_ds.get_text_seq_length()
    args.context_vocab_size = train_ds.get_char_vocab_size()

    model = Transformer(args).to(device)
    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = GenerationParams(
        n_at_a_time=1,  # generate one word at a time (matches 2-word training)
        n_words=cli.num_words,
        num_steps=cli.num_steps,
        sentence_line_width=cli.line_width,
        do_sample=False,  # greedy -> deterministic, more legible
        seed=cli.seed,
        verbose=False,
    )
    offsets = generate_paragraph(model, train_ds, cleaned, params)
    fig, _ax = plot_paragraph(offsets, cleaned, params=params, include_title=False)
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    fig.savefig(cli.out, bbox_inches="tight")
    print(f"[render] saved -> {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
