# Local training samples

Cursive generated entirely on a laptop (Apple Silicon, MPS) with Weights & Biases
disabled, via `scripts/train_local.py` on the `easybank` dataset. These demonstrate
that the pipeline trains and samples locally with no cloud/GPU cluster. Files are
`easybank_<run>_<words>.png`, where `<words>` are the two prompt words.

| run   | steps | context (max_seq_length) | lr schedule        | wall-clock (MPS) | trunc. % | best test loss |
|-------|-------|--------------------------|--------------------|------------------|----------|----------------|
| 3k    | 3000  | 320                      | constant 1e-2      | ~4 min           | ~82%     | 2.13           |
| 12k   | 12000 | 320                      | constant 1e-2      | ~16 min          | ~82%     | 1.82           |
| tuned | 7000  | 512                      | StepLR 1e-2 → 1.25e-3 | ~14 min       | ~20%     | **1.56**       |
| best  | 12000 | 512                      | StepLR 1e-2 → 1.25e-3 (step_lr_every 3000) | ~19 min | ~20% | 1.57 |

(`trunc. %` is the share of two-word training examples longer than the context, which
`create_datasets` now reports + warns about; it is the cause of dropped second words.)

## 3k → 12k (same 320-token config, constant lr)

12k steps improves quality on already-legible words ("laundry" goes jagged → smooth,
"zoom" becomes readable) and reaches real legibility on the easier words; hard cases
("onzn", "page") stay rough, and **the second word never renders**. Gains mostly land
by ~7–8k steps, then the loss plateaus at constant lr (~1.82).

## tuned run (lr decay + larger context)

```
python scripts/train_local.py --steps 7000 --dataset easybank \
  --max_seq_length 512 --batch_size 20 \
  --step_lr_every 2000 --lr_decay 0.5 --max_new_tokens 700
```

Two changes, both via CLI/config (no architecture or training-math change):

- **Larger context fixed the missing second word.** A diagnostic showed two-word
  easybank examples are ~400+ tokens *even after downsampling*, so a 320-token context
  **truncated 100% of them** — the model literally never saw the second word in
  training. That was the real cause of "loan"/word-2 never rendering (not the sample
  budget). Raising `max_seq_length` to 512 lets the model learn and draw both words:
  all three tuned samples now render **both** words.
- **Learning-rate decay broke the plateau.** StepLR (`--step_lr_every 2000 --lr_decay
  0.5`) drives lr 1e-2 → 5e-3 → 2.5e-3 → 1.25e-3; test loss keeps dropping at each
  decay (1.80 → 1.70 → 1.64 → 1.60 → 1.56) instead of flattening at 1.82.

Result — `easybank_tuned_laundry_loan.png` reads as a complete two-word cursive phrase
("laundry" smooth, "loan" legible), and `easybank_tuned_onzn_page.png` now renders both
words ("page" shows its p/g descenders; "onzn" stays the roughest word).

Honest caveats: the 1.56 vs 1.82 comparison is *indicative, not clean* — the tuned run
also changed `max_seq_length` (512 vs 320), which shifts the loss scale, so lr decay
isn't perfectly isolated. Legibility still varies per word, and the longer generation
budget occasionally emits a degenerate large offset that distorts a plot (the `ear
zoom` sample, omitted here). This is a ~15-minute laptop run of a small model; the
published model trains ~125k steps with a larger config/dataset.

## best (capstone) — the strongest samples this laptop produced

```
python scripts/train_local.py --steps 12000 --dataset easybank \
  --max_seq_length 512 --batch_size 16 \
  --step_lr_every 3000 --lr_decay 0.5 --num_samples 5 --max_new_tokens 650
```

- device MPS, 12000 steps, **~19 min** (96 ms/step), truncation **~20%** (the warning confirmed it)
- train loss 6.55 → 1.36, best test loss **1.57**; lr 1e-2 → 5e-3 → 2.5e-3 → 1.25e-3
- **all 5 sampled prompts rendered both words** ([2,2,2,2,2])

The three `easybank_best_*.png` are the cleanest cursive from any run here:
`laundry loan` (both words smooth and legible), `wood beqke` ("wood" especially clean),
and `ear zoom` (both legible — and, unlike the tuned run, **no degenerate-offset plot
artifact**, because the smaller `--max_new_tokens 650` leaves less trailing generation).
The other two prompts also rendered both words: `page` and `elder` are legible (nice
p/g and l/d ascenders/descenders); `onzn` and `urv` stay the roughest first words.

Honest read vs the tuned run: best test loss is **1.57 — essentially tied with the
tuned 1.56, not lower**. More steps (12k) + a smaller batch (16, noisier) bought no
loss improvement once lr had decayed; the win is *reliability and breadth*, not a lower
number — 5/5 prompts render both words with no plot artifacts, giving several complete,
legible two-word phrases. This is still a ~20-minute small-model laptop run, not the
published ~125k-step model; per-word legibility varies and the first word of a pair is
usually the roughest.

Regenerate (checkpoint + fresh PNGs land in the gitignored `runs/`):
`python3 scripts/train_local.py --dataset easybank --steps 12000 --max_seq_length 512 --batch_size 16 --step_lr_every 3000 --lr_decay 0.5 --num_samples 5 --max_new_tokens 650`
