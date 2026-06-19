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

import torch  # noqa: E402  (must follow the env setup above)
from torch.utils.data.dataloader import DataLoader  # noqa: E402

from data import InfiniteDataLoader, create_datasets  # noqa: E402
from model import get_all_args, get_checkpoint, save_checkpoint  # noqa: E402
from sample import save_samples  # noqa: E402

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
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=3, help="Sample images to save at the end")
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
    args.max_steps = cli.steps
    args.local_checkpoint_path = os.path.join(RUNS_DIR, f"{cli.dataset}_local.pt")
    return args


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
                f"| {time.time() - t0:.0f}s"
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

    # --- generate sample images into runs/ (save_samples writes relative to cwd) ---
    model.eval()
    cwd = os.getcwd()
    os.chdir(RUNS_DIR)
    try:
        save_samples(model, test_ds, num=cli.num_samples, do_sample=False, log_wandb=False)
    finally:
        os.chdir(cwd)
    sample_png = os.path.join(RUNS_DIR, f"{test_ds.name}_topk_1.png")

    print("\n[train_local] ===== summary =====")
    print(f"  device:        {device}")
    print(f"  steps:         {step}")
    print(f"  loss finite:   {first_loss is not None and math.isfinite(last_loss)}")
    print(f"  train loss:    {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  best test:     {best_loss:.4f}  (final test {final_test:.4f})")
    print(
        f"  wall-clock:    {train_secs:.1f}s ({step} steps, {1000 * train_secs / step:.0f} ms/step)"
    )
    print(f"  checkpoint:    {args.local_checkpoint_path}")
    print(f"  sample image:  {sample_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
