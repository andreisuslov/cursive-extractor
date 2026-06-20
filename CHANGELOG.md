# Changelog

All notable changes to this project, grouped into development milestones.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

This document tracks the **`ocr-stroke-recovery`** branch: 53 commits off `main`
(2026-06-18 → 2026-06-19). The branch rewrites the OCR handwriting → pen-stroke
pipeline, hardens it into a tested, linted, locally-trainable project, and then
pursues a research arc into automatic per-letter segmentation. No formal versions
are tagged, so milestones are numbered in the order they were built; later
milestones build directly on earlier ones.

## [Unreleased] — `ocr-stroke-recovery`

**Current state.** The OCR vectorizer recovers clean pen strokes from scanned
diary crops (fabricated pen-ups down ~95 → 1–3 per word). About **91%** of real
page-4 words segment into per-letter pieces usably by geometry alone. Two attempts
to beat that geometric ceiling with a learned recognizer (recognition-guided cuts,
holistic HTR forced-alignment) were explored and — honestly — do **not** beat
geometry at the current data scale (~490 OCR'd words); the evidence points at *more
data*, not a better cut algorithm, as the next lever. The full suite is **147 tests**
passing, ruff-clean, with CI on every push/PR, and the whole train+sample loop runs
W&B-free on an Apple-Silicon laptop.

---

## M1 — OCR vectorizer stroke-recovery rewrite

The reason the branch exists. The old vectorizer ordered a scanned word's pixels by
greedy nearest-neighbor and called any >2px gap a pen-lift, fabricating ~95 fake
pen-ups per word. Replaced with a real skeletonize-and-trace pipeline that holds
visual overlap (IoU ~0.67) while cutting fake pen-ups to ~1–3 per word.

- `130c144` WIP baseline captured before the rewrite (OCR extraction + diary scraper + crude vectorizer).
- `2e787d3` Zhang-Suen 1px thinning + connected-component depth-first trace that retraces drawn edges at dead-ends; fake pen-ups ~95 → 1–3/word, IoU held (0.666/0.668).
- `f6243c3` Per-stroke overlay inspection helper for visual QA; confirmed the tracer is precise and surfaced crop over-capture (ruled lines, scan bands, neighbor words) as the next upstream problem.

## M2 — Crop-cleaning + never-regress safety gate

Scanned crops carried ruled lines, scan-edge bands, and ink from neighbouring
words/rows, which the precise tracer faithfully turned into junk strokes. Added a
cleaning pass, then *gated it so it can never make a word worse*.

- `f518645` `clean_word()` strips thin full-width ruled lines, clamps ink to a vertical band around the word box, keeps only components overlapping the target box; over-captured crops 43.5 → 15.4 pen-ups/word.
- `337fb48` Safety gate: trace each word both cleaned and uncleaned, keep the cleaned crop only when it doesn't raise the stroke count (recorded in `metadata["cleaned"]`); page-4 worse-than-uncleaned 7/40 → 0/40.

## M3 — Ruff lint/format baseline

Established an enforceable style baseline so later changes stay clean.

- `9150c05` Add `[tool.ruff]` (py312, line-length 100, select E/F/I/UP/B/SIM/C4/RUF); `ruff check` 377 → 0, experiment metrics unchanged.
- `96044bc` Fix the last two files with cosmetic format drift; `ruff format --check` reports all files already formatted (idempotent).

## M4 — Vectorizer & harness performance

Sped up the hot path and the experiment harness with output **proven byte-identical**
across all 80 easybank+bigbank words — no accuracy traded for speed.

- `c7eaf25` Vectorize Zhang-Suen connectivity + per-component thinning; end-to-end 3.68s → 3.36s.
- `34ed148` Vectorize the order-recovery harness metrics (path length, x-reversals, render coords); 3.39s → 2.77s.
- `c4599b1` Precompute unit vectors + reuse settled adjacency in the graph walk (no rebuild); 2.84s → 2.63s.
- `b801af8` Cache dataset JSON loads so each bank is parsed once, not twice; 2.65s → 2.49s.

## M5 — `ocr/` readability, dedup, type hints, dead-code removal

Paid down duplication and made the package legible and typed, with no behavior change.

- `bad853f` Factor duplicated env-flag parsing and 8 near-identical path builders into helpers (~80 fewer lines, all 216 generated paths byte-identical); doc fixes.
- `6c6570c` Module/function docstrings + dead-code removal in the data-pipeline scripts.
- `471bed0` Precise type hints across the `ocr/` package public APIs (annotation-only).
- `61c96d3` Docstrings + type hints across the torch core (model/data/sample/train); no architecture or signature change.
- `78cb253` Update `ocr/README.md` and add a Local-development section to the top-level README; every documented command run and verified.
- `5a43e59` Ponytail dead-code/binary cleanup (~908 lines, ~25 MB): drop the superseded Colab export, stray committed binaries, the never-False cross-attention branch in `model.py`, and unused builders.

## M6 — Test suite (0 → 147)

The branch started with no tests and ended with a 147-test suite covering the OCR
package, the safety-gate invariant, the torch core, and the generation/decoding path.
CPU-only and deterministic — no network, no `wandb.init`. The dedicated expansions:

- `4d45f23` 34 tests — config/paths/parsers/vectorize core + a metrics regression lock on the real banks.
- `ae90a8b` 49 — crop-cleaning never-regress invariant + helper units.
- `c684ba3` 61 — torch core: data-representation round-trips, augmentations, and a tiny-model forward/backward smoke.
- `1d986ab` 73 — `sample.py` generation/decoding path; Agg + W&B-disabled `conftest`.
- Feature commits added their own regression tests too (`1518831` truncation, `7287351` real-crop tracer, and each segmentation/recognizer/allograph/HTR commit), reaching **147** by branch end.

## M7 — Dev workflow: CI + requirements split

Made a fresh clone able to run every workflow, and added an automated quality gate.

- `cad237f` Split dependencies by workflow (`requirements-dev.txt`, `requirements-scraper.txt`) so every third-party import maps to a file (14/14 importable).
- `0297fa9` GitHub Actions CI: `ruff check` + `ruff format --check` + `pytest` on push/PR (ubuntu, Python 3.12, CPU torch; read-only, no deploy).

## M8 — Local MPS training capability + samples

Training and sampling previously assumed a GPU **and** Weights & Biases. Added a
W&B-free path so the whole loop runs on an Apple-Silicon laptop, with committed
sample images (not just claims) as evidence.

- `6e512b8` `scripts/smoke_train.py` — W&B-disabled end-to-end train+sample smoke (verified on MPS: 12 steps, loss 6.42 → 4.76).
- `1d6fdc6` `scripts/train_local.py` — real local training on MPS into a gitignored `runs/`; `.gitignore` covers run byproducts. A 3000-step easybank run (~4 min) produces recognizable cursive.
- `656b172` Expose StepLR + `--max_new_tokens` knobs; diagnosed that the old ctx 320 truncated 100% of two-word examples — training at ctx 512 fixes it (test loss 1.82 → 1.56).
- `a440bd4` `scripts/render_text.py` — render arbitrary text as cursive from a local checkpoint via `sample.py`'s public functions.
- `e8c82a3`, `922e459`, `72de990` Committed laptop sample images (3k / 12k / 30k steps) with honest legibility reads (longer training recovers legibility; minim-heavy words stay rough); checkpoints stay gitignored.
- **Fixed** `1518831` Surface silent dataset truncation (printed stats + `UserWarning`) — made visible the tail-dropping that lost the second word of two-word samples.
- **Fixed** `92ed7af` Blank renders from long warmup seeds in `generate_paragraph` (passed `num_steps - 2*warmup` to `generate()`); single words now render non-blank and multi-word no longer drops word 0. Regression test added.

## M9 — Handwriting-variety augmentation + legibility A/B

Widened the training-time augmentation to teach real style variation, then *honestly
measured the cost* instead of assuming it was free.

- `cb9a650` Widen `augment_stroke` (symmetric slant ±0.3, ±4° incline re-enabled, height 0.6–1.6, width 0.8–1.3, per-word vertical jitter); training-time only, signatures compatible.
- `d68b861` Variety samples: the same word now visibly varies in width/height/slant across draws (earlier fixed-style runs drew it near-identically).
- `c4f6a72` 4-level A/B (no-aug / mild / moderate / wide, matched seed and RNG draws): legibility **degrades monotonically** with augmentation strength (best test loss 1.18 → 1.54 → 1.80 → 2.00) while variety rises — even mild costs legibility at this small-model/short-run scale.
- `f0547f5` Docstring: mild defaults prioritize legibility on purpose; meaningful wider variety needs the full ~125k-step run, not a short laptop one.
- `642193a`, `e439b37` README write-up + comparison images.

## M10 — Automatic per-letter segmentation (the research arc)

The branch's largest experiment: cut a traced word into its known letters
automatically, guided only by the transcription. Built up from a 2/12 prototype to
**~91%** of real page-4 words segmenting usably by geometry, then probed whether a
recognizer could push past the geometric ceiling.

**Tracer robustness**
- `7287351` Stroke-width-aware noise floor in the tracer (provably a no-op on clean input); removes 32 spurious strokes across 147 real diary crops with no crop made worse. Locked with real-crop fixture tests.

**Geometric cuts — count-correct on 12, then scaled**
- `adb2827` Transcription-forced arc-length DP prototype; honest **2/12** plus two named real blockers (contaminated crops; backtracking DFS makes arc-length non-monotonic in x).
- `581a2db` Fix both blockers (clean per-word input + cut by horizontal **x** instead of arc-length): **2/12 → 8/12**.
- `0dcc235` Strip gutter blob / margin spiral / same-line neighbour ink: **8/12 → 11/12** (only fully-connected whole-line cursive left).
- `94dd60f` Gated skeleton-**topology** fallback for single-blob words — fires only on a wide uncut gap, so it can't regress the others (11 overlays byte-identical): translate reaches a **12/12** count.

**Slant- and angle-aware cuts**
- `b3d4365` Slant-aware cuts (de-shear → x-DP → re-shear) so cuts follow the writing lean instead of clipping slanted neighbours.
- `8df1d00` Per-boundary **local** cut angles + descender bending (replaces the single global slant); re-eval over all 183 page-4 crops: **91% usable** (conf ≥ 0.7), 96.7% at ≥ 0.6.

**Scale eval + bleed fixes**
- `ba32e56` Scale eval over ~140 crops + a gated ruled-line-bleed strip (fires only on a true straight rule); honest read at scale: **~85–90%** segment usably, the residual being gap-free connected cursive that needs a recognition-guided cut.

**Recognition-guided cuts — the first honest ceiling**
- `383b71e` Forced-alignment recognition-guided cuts with a dependency-free font-template recognizer (+18pp top-1 on the hard connected words geometry can't crack). Ablation-verified; the binding limit is the font→handwriting gap in the recognizer, not the cut search.
- `93da3b1` Tune the recognition weight (`ALIGN_ALPHA=1.5`, the peak); saturation confirms recognizer **quality**, not the weight, is now the constraint.

## M11 — Real-handwriting recognizer bootstrap (~2×) — honest negative on cut-guidance

Replaced the font-template recognizer with a small CNN trained on real segmented
diary letters. The recognizer roughly **doubles** font accuracy — but using it to
guide cuts is a dead end, and the commit says so plainly.

- `2ffaebe` CNN on real segmented letters (whole-word train/test split, no leak): held-out top-1 **40.8% vs 20.6% for fonts** (+20.2pp, 6/6 seeds). **Negative result:** plugging the CNN into the cut search scores 77.5% *by its own judge* (circular); under an **independent** font judge, CNN-guided cuts (19.4%) do **not** beat plain geometry (20.3%), and no alpha sweep rescues it. A better recognizer does not yield better cuts — cut-guidance is not the lever (the recognizer remains useful for noise rejection and allograph clustering).

## M12 — Per-letter allograph library — partial, and "more data didn't clean it"

Built a per-writer library of letter-form variants (allographs) from segmented
letters, to ask whether the data is clean enough to be useful. **Verdict: partial** —
it finds genuine allographs, but they sit next to recurring mis-cuts, and
**segmentation cut quality (not sample size) is the bottleneck**.

- `179c7b0` Build the library: extract 893 letters → CNN noise-reject (187 dropped, 706 kept) → numpy k-means into up to ~5 variant forms per letter. Finds real allographs (isolated vs ligatured e, four r shapes, open/narrow/ligatured o) but next to mis-cuts; silhouettes 0.14–0.37 (not cleanly separated).
- `cb59c01` Pool 4 OCR'd pages (~490 words) to test "more data averages out mis-cuts": **falsified** — mis-cut fraction rose and the cleanest letters regressed. Mis-cuts are structured/biased (slant over-fit, neighbour-row bleed, black-block over-ink), not zero-mean noise, so more words pile the *same* bias. More data bought coverage, not cleanliness.
- `de5610b` Attack the cuts instead: reject solid black-block over-ink slices + tighten neighbour-row/word stripping. The dominant **degenerate** variants (e solid-flag, a blobs, s rectangle, t flag) are **gone**, replaced by genuine letter forms — the real usability gain (the grid is the evidence; mean silhouette barely moves because a black-block cluster is itself tight/well-separated). Residual is cut-alignment error, geometrically inseparable from a thick hand without eating real letters.

## M13 — Holistic HTR forced-alignment experiment — clean data-starved negative

Tried the holistic alternative to geometric cutting: a CNN→BiLSTM→CTC handwriting
model that derives per-letter boundaries by CTC forced-alignment. Evaluated
**independently** (font judge, not the model's own score, avoiding the earlier
circularity). The result is a clean negative *at the current data scale*, and points
at the real lever.

- `5b2e3e7` HTR forced-alignment cut scores **4.5–5.6% vs geometry 12–13%** (about half), consistent across seeds; doubling epochs (60 → 120) barely moves it. ~490 words is far too little for HTR (these need thousands+) — the right architecture, the wrong data scale *here*. Motivates the parallel data-scaling effort (scraping more American Diary Project diaries).
