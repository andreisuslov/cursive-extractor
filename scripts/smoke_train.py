#!/usr/bin/env python3
"""Local end-to-end smoke: train the Transformer a few steps + sample, no W&B, no GPU.

Proves the model trains and samples on THIS laptop (MPS if available, else CPU) using a
tiny subset of an existing local dataset. Weights & Biases is fully DISABLED (offline, no
login/getpass) and the device is auto-selected.

This does NOT modify train.py / sample.py / model.py / data.py -- it reuses their public
functions (create_datasets, get_checkpoint, save_samples, generate) with a tiny config,
running the same per-step ops as train.py's loop. The production path is untouched; the
only "enabling" needed for a local run is the env vars set below plus a small config.

    python scripts/smoke_train.py
"""

import math
import os
import sys
import tempfile
import time

# --- Disable Weights & Biases BEFORE importing anything that imports wandb (model.py /
# train.py / sample.py all do). WANDB_MODE=disabled makes wandb.init/log/watch no-ops
# with no network or login; the dummy key stops get_all_args() prompting via getpass. ---
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "disabled-for-smoke")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting for sample.py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import InfiniteDataLoader, create_datasets
from model import get_all_args, get_checkpoint
from sample import generate, save_samples


def pick_device() -> str:
    """cuda if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiny_args(device: str):
    """Real arg defaults (get_all_args), overridden for a fast local run."""
    args = get_all_args(use_argparse=False)
    args.device = device
    args.dataset_name = "easybank"
    args.train_size = 128
    args.test_size = 32
    args.num_words = 2
    args.batch_size = 8
    args.n_layer = 2
    args.n_embd = 32
    args.n_embd_context = 32  # must equal n_embd (context pos-emb add + cross-attn LN)
    args.n_ctx_head = 4
    args.max_seq_length = 200
    args.learning_rate = 1e-2
    args.max_steps = 12
    return args


def main() -> int:
    t_total = time.time()
    device = pick_device()
    print(f"[smoke] device = {device}")

    args = tiny_args(device)
    train_dataset, test_dataset = create_datasets(args)
    args.vocab_size = train_dataset.get_vocab_size()
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.context_vocab_size = train_dataset.get_char_vocab_size()

    model, optimizer, scheduler, _step, _best = get_checkpoint(args, sample_only=False)
    loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)

    # --- a handful of real training steps (same ops as train.py's loop body) ---
    t_train = time.time()
    losses = []
    for _ in range(args.max_steps):
        x, c, y = (t.to(device) for t in loader.next())
        _logits, loss = model(x, c, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    train_secs = time.time() - t_train
    print(f"[smoke] loss trajectory: {' '.join(f'{x:.3f}' for x in losses)}")

    # --- generate a sample via sample.py (the same entry train.py uses) ---
    # save_samples writes its PNG relative to cwd; run it in a temp dir so the smoke
    # never litters the repo root.
    t_sample = time.time()
    cwd = os.getcwd()
    sample_dir = tempfile.mkdtemp(prefix="smoke_samples_")
    os.chdir(sample_dir)
    try:
        save_samples(model, test_dataset, num=1, do_sample=False, log_wandb=False)
    finally:
        os.chdir(cwd)

    model.eval()
    x0, c0, _ = test_dataset[0]
    warmup = 50
    x_init = x0[:warmup].unsqueeze(0).to(device)
    context = c0.unsqueeze(0).long().to(device)
    x_samp = generate(model, x_init, context, warmup + 60, do_sample=False).to("cpu")
    decoded_words = test_dataset.decode_stroke(x_samp[0].numpy())  # list of (N,3) arrays
    sample_secs = time.time() - t_sample

    finite = all(math.isfinite(x) for x in losses)
    decreased = losses[-1] < losses[0]
    print("\n[smoke] ===== summary =====")
    print(f"  device:            {device}")
    print(f"  steps:             {len(losses)}")
    print(f"  loss finite:       {finite}")
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f}  (decreased: {decreased})")
    print(f"  sample tokens:     {tuple(x_samp.shape)}")
    print(f"  sample png:        {os.path.join(sample_dir, 'test_topk_1.png')}")
    print(f"  decoded words: {len(decoded_words)}  first stroke shape: {decoded_words[0].shape}")
    print(f"  train wall-clock:  {train_secs:.2f}s ({len(losses)} steps)")
    print(f"  sample wall-clock: {sample_secs:.2f}s")
    print(f"  total wall-clock:  {time.time() - t_total:.2f}s")

    ok = finite and decreased and len(decoded_words) >= 1
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: trained and sampled end-to-end on {device}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
