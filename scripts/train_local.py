#!/usr/bin/env python3
"""Real local training on MPS/CPU with Weights & Biases disabled.

Trains the actual Transformer (no architecture or training-math changes) on a local
dataset, then saves the best checkpoint and a generated-handwriting PNG into the
gitignored ``runs/`` directory. The device auto-selects (CUDA -> MPS -> CPU) and W&B is
fully disabled (offline, no login). This is the heavier sibling of ``smoke_train.py``:
the smoke proves the pipeline runs in seconds; this actually trains a small-but-real
model for a few thousand steps.

    python scripts/train_local.py --steps 3000 --dataset easybank
"""

import argparse
import math
import os
import sys
import time

# Disable W&B BEFORE importing anything that imports wandb (model/train/sample all do).
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "disabled-for-local")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402  (must follow the env setup above)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data.dataloader import DataLoader  # noqa: E402

from data import InfiniteDataLoader, create_datasets  # noqa: E402
from model import get_all_args, get_checkpoint, save_checkpoint  # noqa: E402
from sample import (  # noqa: E402
    GenerationParams,
    generate,
    plot_strokes,
    word_offsets_to_points,
)

RUNS_DIR = os.path.join(_REPO_ROOT, "runs")


def pick_device() -> str:
    """cuda if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real local training (MPS/CPU, W&B disabled)")
    p.add_argument("--steps", type=int, default=3000, help="Training steps")
    p.add_argument("--dataset", default="easybank", help="Dataset name under data/")
    p.add_argument("--num_words", type=int, default=2, help="Words per training example")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64, help="(n_embd_context is set equal to this)")
    p.add_argument("--n_ctx_head", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=320)
    p.add_argument("--train_size", type=int, default=2000, help="Number of combinatorial examples")
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    # Learning-rate decay (StepLR, already in the model): defaults match get_all_args, so
    # the default run keeps constant lr. Pass a small --step_lr_every to actually decay.
    p.add_argument("--step_lr_every", type=int, default=33000, help="StepLR decay interval (steps)")
    p.add_argument("--lr_decay", type=float, default=0.333, help="StepLR multiplicative decay")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=3, help="Sample images to save at the end")
    # Sample generation budget. save_samples caps it at block_size-1, which can truncate the
    # last word of a multi-word prompt; raise this so every prompt word has room to render.
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="Token budget per sample (0 = block_size-1, the dataset default)",
    )
    return p.parse_args(argv)


def build_args(cli: argparse.Namespace, device: str):
    """Real arg defaults (get_all_args) overridden for a small-but-real local run."""
    args = get_all_args(use_argparse=False)
    args.device = device
    args.dataset_name = cli.dataset
    args.num_words = cli.num_words
    args.batch_size = cli.batch_size
    args.n_layer = cli.n_layer
    args.n_embd = cli.n_embd
    args.n_embd_context = cli.n_embd  # must equal n_embd
    args.n_ctx_head = cli.n_ctx_head
    args.max_seq_length = cli.max_seq_length
    args.train_size = cli.train_size
    args.test_size = cli.test_size
    args.learning_rate = cli.lr
    args.step_lr_every = cli.step_lr_every  # StepLR decay interval (model uses this)
    args.lr_decay = cli.lr_decay  # StepLR gamma
    args.max_steps = cli.steps
    args.local_checkpoint_path = os.path.join(RUNS_DIR, f"{cli.dataset}_local.pt")
    return args


def save_word_samples(model, dataset, num, max_new_tokens, out_dir, device, warmup=50):
    """Greedily sample `num` examples and save a PNG each into ``out_dir``.

    Mirrors sample.save_samples but takes an explicit ``max_new_tokens`` budget so a
    multi-word prompt has room to render every word (save_samples caps the budget at
    ``block_size - 1``, which can truncate the last word). Uses only sample.py's public
    functions -- no change to the model or the sampling math. Returns (paths, word_counts).
    """
    params = GenerationParams()
    strokes, contexts = [], []
    for i in range(num):
        x, c, _y = dataset[i]
        strokes.append(x)
        contexts.append(c)
    x_init = torch.stack(strokes).to(device)[:, :warmup]
    context = torch.stack(contexts).long().to(device)
    budget = max_new_tokens if max_new_tokens > 0 else dataset.get_stroke_seq_length() - 1
    x_samp = generate(model, x_init, context, budget, do_sample=False).to("cpu")
    paths, word_counts = [], []
    for i in range(x_samp.size(0)):
        offsets = dataset.decode_stroke(x_samp[i].numpy())
        points = np.vstack(word_offsets_to_points(offsets, params))
        text = dataset.decode_text(context[i])
        fig, _ax = plot_strokes(points, f'Sample {i + 1}: "{text}"')
        path = os.path.join(out_dir, f"{dataset.name}_sample_{i + 1}.png")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
        word_counts.append(sum(1 for w in offsets if len(w) > 1))  # words actually drawn
    return paths, word_counts


def main(argv=None) -> int:
    cli = parse_args(argv)
    device = pick_device()
    os.makedirs(RUNS_DIR, exist_ok=True)
    print(f"[train_local] device = {device}  dataset = {cli.dataset}  steps = {cli.steps}")

    args = build_args(cli, device)
    train_ds, test_ds = create_datasets(args)
    args.vocab_size = train_ds.get_vocab_size()
    args.block_size = train_ds.get_stroke_seq_length()
    args.context_block_size = train_ds.get_text_seq_length()
    args.context_vocab_size = train_ds.get_char_vocab_size()

    model, optimizer, scheduler, step, best_loss = get_checkpoint(args, sample_only=False)
    loader = InfiniteDataLoader(train_ds, batch_size=args.batch_size, num_workers=0)

    @torch.inference_mode()
    def evaluate(dataset, max_batches=10):
        model.eval()
        dl = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=0)
        losses = []
        for i, batch in enumerate(dl):
            xb, cb, yb = (t.to(device) for t in batch)
            _logits, loss = model(xb, cb, yb)
            losses.append(loss.item())
            if i >= max_batches:
                break
        model.train()
        return sum(losses) / len(losses)

    # --- training loop (same per-step ops as train.py) ---
    t0 = time.time()
    first_loss = last_loss = None
    while step < args.max_steps:
        x, c, y = (t.to(device) for t in loader.next())
        _logits, loss = model(x, c, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss
        if step % cli.eval_every == 0:
            test_loss = evaluate(test_ds)
            print(
                f"step {step:5d} | train {last_loss:.4f} | test {test_loss:.4f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s"
            )
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(
                    model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss
                )
        step += 1
    train_secs = time.time() - t0

    final_test = evaluate(test_ds)
    if best_loss is None or final_test < best_loss:
        best_loss = final_test
    save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss)

    # --- generate sample images into runs/ (explicit token budget so every word renders) ---
    model.eval()
    sample_paths, word_counts = save_word_samples(
        model, test_ds, cli.num_samples, cli.max_new_tokens, RUNS_DIR, device
    )

    print("\n[train_local] ===== summary =====")
    print(f"  device:        {device}")
    print(f"  steps:         {step}")
    print(f"  loss finite:   {first_loss is not None and math.isfinite(last_loss)}")
    print(f"  train loss:    {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  best test:     {best_loss:.4f}  (final test {final_test:.4f})")
    print(f"  lr decay:      step_lr_every={args.step_lr_every} lr_decay={args.lr_decay}")
    print(f"  max_new_tokens:{cli.max_new_tokens or args.block_size - 1}")
    print(
        f"  wall-clock:    {train_secs:.1f}s ({step} steps, {1000 * train_secs / step:.0f} ms/step)"
    )
    print(f"  checkpoint:    {args.local_checkpoint_path}")
    print(f"  words drawn per sample: {word_counts}")
    print(f"  sample images: {', '.join(os.path.basename(p) for p in sample_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
