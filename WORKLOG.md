# Worklog

The **decision-making** and **results** behind the project — the *why* and the
*what-we-learned*, as opposed to `CHANGELOG.md`, which records the *what-we-shipped*
(implemented changes, by milestone). When a decision here turns into code, it graduates
to a CHANGELOG milestone.

Newest entries first. Dates are absolute.

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

**Status.** A deep-research survey of current SOTA trajectory-recovery methods, datasets,
and open-source code was launched on 2026-06-28 to ground Step 1 before building it.

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
