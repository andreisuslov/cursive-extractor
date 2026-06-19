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

## wide-augmentation variety (`bigbank_varied_*.png`)

A full bigbank run **with the wide handwriting augmentation** (both-way slant, line
incline, 0.6-1.6x height, 0.8-1.3x width, per-word vertical jitter -- see
`data.augment_stroke`):

```
python scripts/train_local.py --dataset bigbank --num_words 2 --max_seq_length 512 \
  --batch_size 16 --step_lr_every 2500 --lr_decay 0.5 --steps 12000
```

- device MPS, 12000 steps, **~19 min** (97 ms/step), truncation ~23%
- train loss 6.46 -> 1.76, best test loss **2.00**

`bigbank_varied_01_grid.png` renders the same word six times each; `02`/`03` are
per-word strips for "writing" / "summer". Across the samples the **style visibly
varies** -- slant (left / upright / right), height (tall vs short), and width
(condensed vs spread) -- which the earlier fixed-style runs never showed.

Honest trade-off: best test loss **2.00 is higher than the no-augmentation bigbank run
(1.39)** -- a small model trained for ~20 min cannot fit the much wider variety as
tightly, so individual letters are rougher and some warmups generate degenerate output
(filtered out of these strips). The augmentation is clearly *learnable and does not break
training*; a rigorous with-vs-without legibility comparison at a larger step budget is
future work.

## controlled with-vs-without A/B (`aug_vs_noaug_01.png`)

The "future work" above, done as a **matched, single-variable A/B**. Two bigbank models,
**identical config and the same seed (42)**, differing only in augmentation:

```
# A (WITH, default):   python scripts/train_local.py --dataset bigbank --num_words 2 \
#   --max_seq_length 512 --batch_size 16 --step_lr_every 2000 --lr_decay 0.5 --steps 7000
# B (WITHOUT):         ...same flags... --no-augment
```

`--no-augment` (added to `train_local.py`) routes `StrokeDataset` through `augment_stroke`
with **identity** geometric ranges. Because `np.random.uniform(a, a)` still consumes one
RNG draw, the downsampling (hence sequence length and the 23% truncation) is **byte-identical**
between arms -- the only variable is the slant/incline/height/width/jitter geometry. The
fairness invariant is locked by `test_no_augment_flag_keeps_downsample_identical`.

| arm     | augmentation | steps | wall-clock (MPS) | train loss   | **best test loss** |
|---------|--------------|-------|------------------|--------------|--------------------|
| A       | WITH (wide)  | 7000  | ~11.4 min        | 6.45 → 1.87  | **2.08**           |
| B       | WITHOUT      | 7000  | ~11.5 min        | 6.50 → 0.96  | **1.18**           |

**Style variety** -- same 5 words rendered from 6 warmup seeds each (greedy, seeded for
reproducibility); spread is measured across the 6 renders of each word, averaged over words
(`runs/_ab_variety.py`, metrics in `runs/ab_variety_metrics.json`):

| spread metric        | WITH  | WITHOUT | WITH / WITHOUT |
|----------------------|-------|---------|----------------|
| slant (std of x-on-y slope) | 1.233 | 0.769 | **x1.60**  |
| width (coef. of var.)       | 0.472 | 0.379 | **x1.25**  |
| height (coef. of var.)      | 0.465 | 0.512 | x0.91 (≈equal) |

**Conclusion -- the two effects are both real and they trade off:**

- **Variety: augmentation wins on slant** (x1.60) and modestly on width (x1.25). In
  `aug_vs_noaug_01.png` the per-panel slant labels span a much wider range in the WITH-aug
  rows (e.g. "writing": −1.14 / +1.60 / +5.54) than the NO-aug rows (clustered near ±1).
  **Height variety is *not* increased** -- the layout/decode normalizes vertical scale, and
  bigbank's natural cross-writer variation already gives the NO-aug arm a height baseline.
- **Legibility: no-augmentation wins decisively** -- best test loss **1.18 vs 2.08**, and
  in the figure the NO-aug rows render *legible* "writing"/"summer" while the WITH-aug rows
  are rougher/jaggier (some of the WITH slant spread is genuine style, some is roughness --
  e.g. the +5.54 outlier is a jagged render, not clean italic).

So the wide augmentation does what it was meant to (more handwriting variety, slant above
all) but, at this small-model / ~11-min-per-arm scale, pays for it in legibility. A larger
model or longer schedule would be needed to get *both* variety and clean letters.

## legibility-first follow-up: the full trade-off curve (`legible_variety_01.png`, `aug_levels_01.png`)

Because **legibility is the hard requirement**, the wide band above is unacceptable. To find a
band that stays legible, two more matched runs (same config + seed) swept the augmentation
width down, and all four levels were rendered/measured together (`runs/_ab_variety.py`):

| augmentation level | shear / height band        | **best test loss** | slant variety (std) | renders |
|--------------------|----------------------------|--------------------|---------------------|---------|
| none (`--no-augment`) | identity                | **1.18**           | 0.91                | legible |
| **mild (new default)** | ±0.08 / 0.93–1.08      | **1.54**           | 0.99                | legible |
| moderate              | ±0.15 / 0.85–1.15       | 1.80               | 1.50                | jaggy   |
| wide (old default)    | ±0.30 / 0.60–1.60       | 2.08               | 1.10*               | illegible |

(*the wide slant number is noisy — the model is rough enough that the metric is unstable.)

**The decisive, honest finding: at laptop scale there is no band that delivers both variety
and legibility.** Variety stays near the no-aug baseline until the band is wide enough
(moderate+) to already break legibility — i.e. the only settings that *add* style variety are
the ones that *destroy* legibility. Mild augmentation keeps renders legible (in
`legible_variety_01.png` the mild rows read "writing"/"summer" just like the no-aug rows) but
its variety is barely above no-aug while still costing ~0.36 of loss.

**What changed:** `data.augment_stroke`'s defaults were narrowed from the wide band to the
**mild** band, so a default local run is legible. Real style variety with clean letters needs
the capacity/schedule of the **full cloud training (~125k steps)**, which absorbs a wider band
— not a ~7k-step laptop model. The wide band is still reachable by passing explicit ranges to
`augment_stroke`, and any future re-widening of the *default* is caught by
`test_augment_default_slant_is_symmetric_and_moderate` (asserts the default shear stays ≤ 0.2).
