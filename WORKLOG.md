# Worklog

The **decision-making** and **results** behind the project — the *why* and the
*what-we-learned*, as opposed to `CHANGELOG.md`, which records the *what-we-shipped*
(implemented changes, by milestone). When a decision here turns into code, it graduates
to a CHANGELOG milestone.

Newest entries first. Dates are absolute.

---

## 2026-06-28 — InkSight probe: DECISIVE POSITIVE — it derenders diary ink cleanly

Ran InkSight Small-p (released Apache-2.0 weights, `tf.saved_model`) directly on 6 real,
faded diary word-crops (`box.jpg`), bypassing our entire tracer. **Result: it works, and
well.** The recovered ink traces the actual handwriting as smooth, well-ordered connected
cursive — real letterforms following sensible left-to-right pen motion — on historical ink
*with ruled lines and neighbor words present*. Night-and-day vs. our tracer's angular
zig-zag. This answers the survey's make-or-break domain-gap question: **a learned derenderer
generalizes to 19th-20th-c. diary ink off-the-shelf, zero fine-tuning.**

Details: 9–17 strokes/word (plausible); ~290 s/crop on this CPU Mac (fine for offline
dataset building, not interactive — batch on a GPU to scale). The model's *text recognition*
output was garbage ("John"→"khook-you") but irrelevant — we only use the ink. Setup that
worked: `uv venv --python 3.11`, `tensorflow==2.17.0` + `tensorflow-text==2.17.0` (arm64
wheel exists), `small-p-cpu.zip` from GCS, prompt "Recognize and derender." (fallback
"Derender the ink."). Reproducible recipe saved at `ocr/experiments/_inksight_probe.py`;
probe image at `ocr/experiments/inksight_probe_6.png`.

**Remaining issue (now the real one): crop tightness.** InkSight traces *whatever ink is in
the 224-crop*, so our over-capturing crops ("Uncle" → "Uncle Joh") make it trace neighbors
too. So the bottleneck moves from *stroke recovery* (solved by InkSight) to *clean
single-word crops* (segmentation) — or run InkSight at page level and split the resulting
ink by word.

**Decision / path forward:**
1. Replace `ocr/vectorize.py`'s tracer output with InkSight-derendered ink as the stroke
   source for the dataset.
2. Tighten crops to one word each (or page-level derender + split) so the ink is the target
   word only.
3. Rebuild a small `diarybank-v2` from InkSight ink, retrain the generator, and re-validate
   legibility directly. This is the first version with a real shot at legible output.

---

## 2026-06-28 — Data-quality autopsy: the strokes don't match the labels (plan revision)

Before building any trajectory recovery, inspected the actual `diarybank` training data
(plotted samples colored by pen-order). Findings:

- **Labels are fine.** `asciiSequence` is 100% single words, correct, and sequential (the
  first items reconstruct a real diary sentence). The text side is not the problem.
- **The strokes frequently are NOT the labeled word.** Point count is ~constant ~270/word
  regardless of length (`"I"` = 265 pts, `"to"` = 310) due to `downsample_points`' fixed
  target — but rendering the ink shows: some words trace cleanly (`"Church"` reads as
  Church), many carry a **garbage zig-zag blob** fused to the real word (`"Homick"` = word +
  scribble prefix), and several (`"to"`, `"Went"`, `"4th"`, `"Mr"`) are **fragmented,
  scattered pieces that don't read as the word at all.**
- **Two compounding causes**, not one: (a) the tracer emits dense zig-zag blobs on
  messy/junction/noisy ink; (b) imperfect crops leak neighbor / ruled-line / speck ink into
  each word's vectorization. The model was trained on text→stroke pairs where the strokes
  often don't match the text — which alone explains the scribble output.

**Plan revision.** "Step 0 (smoothness graph traversal)" **already exists** — `ocr/vectorize.py`'s
`trace_component` does a depth-first walk choosing the straightest continuation at branches,
plus component chaining (the naive nearest-neighbor was replaced back in M1). So Step 0 is
not the lever, and it would not fix the crop-contamination half anyway. The bottleneck is
**training-data stroke quality = {tracer order on messy ink} + {crop contamination}**.

**Decisive next experiment:** run **InkSight** (learned image→ink) directly on the diary
**crop images** (`box.jpg`), bypassing our entire tracer. If it derenders clean ink from the
same crops, the path is to replace the vectorizer with a learned recoverer (and the
crop-contamination still needs the segmentation side). If it also chokes on faded/contaminated
diary crops, the limiting factor is crop/segmentation quality, and the honest fallback is
collecting real pen strokes. Either way, validate stroke quality directly — never again only
through a downstream training run.

---

## 2026-06-28 — Trajectory recovery: how to seriously attack it

**Context.** The 2026-06-27 validation (below) proved the OCR pipeline's weak point is
**stroke-order recovery**: `ocr/vectorize.py`'s `order_points` walks skeleton pixels by
nearest-neighbor, which zig-zags across the page instead of following the pen. This is the
classic **offline → online handwriting trajectory recovery** problem. The ladder:

**Step 0 — cheap win, do first.** Replace nearest-neighbor with **smoothness-based graph
traversal**: skeletonize → build a graph (nodes = endpoints/junctions, edges = ink
segments) → traverse choosing, at each junction, the **minimal-direction-change**
continuation (the pen tends to go straight through crossings). Classic, ~a day of work,
and might alone turn the zig-zag scribble into roughly-correct strokes. Worth trying before
any ML — if it's enough, we're done.

**Step 1 — the real fix: a learned image → sequence recoverer.** CNN encoder over a word
crop → autoregressive decoder emitting `(dx, dy, pen)` steps (Sketch-RNN-style decoder;
cf. Bhunia et al. 2018, "Handwriting Trajectory Recovery using End-to-End Deep
Encoder-Decoder Network"). The blocker for any such model is paired `(image, trajectory)`
data — which brings the key insight:

**The unlock — manufacture unlimited perfect pairs for free.** We *have* real pen
trajectories: the hand-collected trackpad strokes (+ public **IAM-OnDB**, **IBM-UB**,
**Deepwriting**). **Render** each trajectory to a static image with diary-like augmentation
(paper texture, ink-width jitter, blur, noise) → perfectly-labeled `(rendered image →
known trajectory)` pairs, as many as wanted. Train the recoverer on those, then apply it to
real diary crops. Standard offline→online trick; sidesteps the "no labels for diary ink"
problem entirely.

**Critical: validate the recoverer *directly*, not through the generator.** Measure DTW
distance between recovered and true trajectories on held-out rendered pairs. We then know
the recoverer works *before* feeding the generator — avoiding the conflated failure we hit
on 2026-06-27 (where bad data looked like a model problem).

**Recommended sequence.** Step 0 → eyeball + re-run the diarybank validation → only build
Step 1 if Step 0 isn't enough.

### Survey results (2026-06-28 deep-research, 17 primary sources, 23/25 claims verified 3-0)

**The render-pairs "unlock" is confirmed standard practice** (Bhunia 2018, Nguyen/im2ink
2020, TRACE 2021, PEN-Net 2022, InkSight 2024): render online trajectories to raster +
degrade (stroke-width/contrast jitter, grid warp, Gaussian noise, blur) to bridge the
rendered-vs-real gap. **IAM-OnDB** is the one *substantiated* paired Latin source (221
writers, 10,426 lines); IBM-UB / Deepwriting / CASIA / RIMES / VNOnDB remain unconfirmed
for this use (open question).

**Reusable options, ranked for our purpose:**

1. **InkSight** — Google Research, TMLR 2024/25 (arXiv 2402.05804). ViT+mT5 VLM that
   "derenders" raster handwriting → digital ink. **Apache-2.0, released Small-p weights +
   Gradio + HF dataset** ([github.com/google-research/inksight](https://github.com/google-research/inksight)).
   Current generalization SOTA, runnable *today with zero training*. Caveats: TensorFlow;
   optimizes "valid tracing" not exact temporal order (only ~67% judged human-traced);
   checkpoint trainability unconfirmed.
2. **TRACE** — Archibald et al., BYU, ICDAR 2021 (arXiv 2105.11559). CNN→2-layer BiLSTM→
   1D-conv decoder, per-timestep relative (x,y)+SOS/EOS. **First trained end-to-end on whole
   lines of arbitrary width** (not isolated chars) → best architectural fit for connected
   Latin cursive. Trained on IAM-On rendered+degraded. **Adaptive-GT DTW loss directly
   resolves junction/loop pen-order ambiguity** (per step, at most one edit: swap adjacent
   strokes or invert a stroke's direction). Our render-pairs plan = literally TRACE's recipe.
3. **PEN-Net** — Chen et al., ACCV 2022 ([github.com/ChenZhounan/PEN-Net](https://github.com/ChenZhounan/PEN-Net)).
   Official code; introduces **AIoU** (glyph-fidelity metric computable from the binarized
   image mask with **no GT trajectory** — ideal for our unlabeled diary crops) and **LDTW**
   (length-normalized DTW, fixes DTW's bias toward fewer points). CJK/Indic focus.
4. **Bhunia 2018** — ConvLSTM-enc→LSTM-dec seq2seq, official code ([repo](https://github.com/AyanKumarBhunia/Handwriting-Trajectory-Recovery)). Isolated chars; simplest baseline.
5. **wor** — Diaz/Crispo et al., IJIMAI 2024 (arXiv 2406.03194; [github.com/gioelecrispo/wor](https://github.com/gioelecrispo/wor), MATLAB/MIT).
   Classical skeleton → good-continuity junction pairing → **Dijkstra** path tracing. This is
   the modern reference for **Step 0** — exactly the smoothness-traversal we planned.
   (`im2ink` is a demo-only repo with no source code — do NOT plan to fine-tune it.)

**Metrics:** evaluate the recoverer *directly* with **LDTW** + **AIoU** (AIoU needs no GT, so
it works on diary crops); DTW / nearest-neighbor distance / LPIPS as supporting. Plain RMSE
(needs exact point-count match) and raw DTW (prefers fewer points) are inadequate.

**Central risk (survey's top caveat):** only TRACE and InkSight were validated on connected
Latin at all, and **none** on faded, ruled, 19th-20th-century diary ink. Domain transfer to
our crops is the unverified make-or-break. Mitigate with heavy degradation augmentation and
possibly self-training on unlabeled diary crops.

### Revised, grounded plan

1. **Step 0 — smoothness graph traversal** to replace nearest-neighbor `order_points`,
   porting the good-continuity + Dijkstra logic from `wor`. ~A day; might suffice.
2. **Zero-training probe — run InkSight Small-p on ~10 diary crops** before building anything.
   It's free, SOTA, and released; if it derenders our ink sanely we may just fine-tune/use it
   and skip a from-scratch model. (Highest-ROI new finding.)
3. **Step 1 (only if 0+2 fall short) — fork a TRACE-style CNN+BiLSTM** word/line recoverer
   with the adaptive-GT DTW loss, trained on IAM-On rendered+degraded pairs, validated by
   LDTW+AIoU **before** feeding the generator.

Open questions to resolve while building: is metric-faithful pen-order even required for our
generator, or is InkSight "valid tracing" enough? Best degradation recipe for historical
scans? Are IBM-UB/Deepwriting usable to add connected-cursive variety beyond IAM-On?

---

## 2026-06-27 — First end-to-end training on diary strokes (diarybank) — VALIDATION

**Decision / why.** The entire OCR pipeline exists to feed the CursiveTransformer training
data, but the model had **never been trained on OCR-mined diary strokes**
(`datasets/diarybank.json.zip`, 16,846 word items, vocab 525) — only on hand-collected
trackpad strokes. Everything downstream (more segmentation work, letter cutting, more
diaries) was premature until we knew the diary stroke data *trains into legible cursive*.
So we ran that one validation as the gate. A CPU smoke-load confirmed the dataset parses,
tokenizes, and truncates 0/40k examples before spending any GPU time.

**What we did.**
- Added `train_colab.ipynb` (Colab harness). First Colab run was **killed** at step 2508
  (cutting the running cell stopped the kernel).
- Wanted < 1 h, so rented GPUs on RunPod via `runpodctl`. **Incident:** a multi-GPU
  availability probe left **6 pods running** (~$3/hr) because `runpodctl` mixes usage text
  into error output and the create-result parser misread successes as failures — caught via
  the spend rate, all torn down. Lesson: **reconcile against `pod list`, never trust create
  stdout.** Community 4090 host stalled; L4 host SSH died mid-run.
- Final clean run: single **L40S** (secure, $0.99/hr), unbuffered logs, monitored via the
  **W&B API** (SSH on these hosts was flaky; W&B logs over the network independently).
  Completed the full **20,132 steps in ~97 min**. Total spend across the whole adventure:
  **$2.41** (incl. ~$0.61 from the 6-pod goof).

**Results.**
- **Output is angular scribble, not cursive.** Across all held-out test samples the model
  reproduces baseline, left-to-right flow, word gaps, and ascender/descender *height* — but
  **no letter shapes**. (W&B project `diary_test`, run `runpod_l40s`.)
- **Loss plateaued, so it's not undertraining:** test_loss 1.96 (2.5k) → 1.87 (7.5k) →
  1.83 (12.5k) → **1.79 (17.6k)**, flat after ~7k.
- **Step time ~270 ms is GPU-bound**, not dataloader-bound: bumping dataloader workers
  8 → 24 changed nothing; GPU sat at ~100% util. The cost is the O(seq²) attention over
  1500-token sequences (fp32), not the 0.5M params. Levers would be mixed precision /
  shorter sequences — not more workers.

**Verdict.** Diary-OCR-derived strokes, as currently vectorized, do **not** train into
legible handwriting. Root cause: wrong **stroke order** from nearest-neighbor skeleton
traversal in `ocr/vectorize.py`. Transkribus (or any HTR) cannot fix this — it outputs
text, never pen trajectories. The two honest paths forward: (a) seriously attack trajectory
recovery (see 2026-06-28 entry), or (b) fall back to collecting real pen strokes, which
already train into legible cursive. **Do not** invest more in the layout/segmentation half
— it was never the blocker.
