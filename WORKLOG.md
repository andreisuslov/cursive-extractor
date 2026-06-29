# Worklog

The **decision-making** and **results** behind the project — the *why* and the
*what-we-learned*, as opposed to `CHANGELOG.md`, which records the *what-we-shipped*
(implemented changes, by milestone). When a decision here turns into code, it graduates
to a CHANGELOG milestone.

Newest entries first. Dates are absolute.

---

## 2026-06-28 — #2: trajectory-cue cutter + the reframe (cut quality is the hard problem)

Added a third, recognizer-free cutter: `trajectory_cut_word` proposes cuts from **pen-lifts +
baseline-valley minima** (the ligature dips between cursive letters) and snaps the L-1
width-prior boundaries to the nearest candidate. Three-way render (baseline | forced-align |
trajectory) + tests.

**Honest result: all three run, none reliably hits true letter boundaries.** Cursive letters
overlap in x and m/n/u/w have *internal* valleys, so cuts stay ambiguous even on a clean
trajectory — the same wall M10-M13 hit on ink. Knowing L + width-prior helps but doesn't
resolve it.

**Reframe (the useful takeaway):** the FONT goal doesn't need every word cut perfectly — it
needs **≥3 confident samples per letter**. So stop chasing a perfect every-word cutter and
build a **confidence-gated harvester**: keep only cuts we're sure of (clean pen-lift-separated
letters; words where valleys align crisply with the width prior), discard the ambiguous ones,
and accumulate per-letter variants until each letter has enough. That turns an unsolved
"segment everything" problem into a tractable "collect the easy wins" one — and is the natural
bridge to N3 (variant clustering) / N4 (font).

## 2026-06-28 — #2 progress: forced-alignment letter cutter on trajectories (marginal)

Built the real cutter for #2: `forced_align_word_strokes` in `_letter_segment_strokes.py`
rasterizes a clean InkSight word path and reuses `_recognizer.align_boundaries` (recognizer
score for the KNOWN letter sequence + per-letter width prior; geometry term off), then maps
the chosen column cuts back to split the trajectory points. Tests + baseline-vs-aligned render
added.

**Self-checked (honest): marginal over the equal-x baseline.** Forced alignment runs and moves
the boundaries to uneven, recognizer-driven positions, but the cuts still don't clearly land on
true letter boundaries — because the **font-template recognizer is weak** (the M11 finding,
reconfirmed on clean trajectories: a weak recognizer can't guide cuts well). So #2's cutter
exists and is correct, but the recognizer is the bottleneck, not the alignment.

**Next levers for #2:** (a) a stronger recognizer — the bootstrap CNN trained on real diary
letters (M11) roughly doubled font accuracy; plug it into `align_boundaries` (it already accepts
any object with `scores()`). (b) **trajectory-native cues** the image didn't have: cut at
pen-lifts and at the baseline-crossing minima between cursive letters (a strong handwriting
signal). Likely (a)+(b) combined.

## 2026-06-28 — First GPU run (InkSight at scale): pipeline works, data mostly noisy

Ran `inksight_vectorize --rich` on a full diary page (0-02 p1, 314 boxes) on a RunPod **L40S**.

**Mechanics: success.** ~**5 s/word** (110 s for 20, incl. load) — **~58× the Mac CPU** (~290 s/word).
All 314 boxes derendered; ~$1.60 for the session. (Process hung on TF exit after writing the
output, so the watcher never got a "done" signal — pulled the result manually before the pod's
auto-terminate; pod torn down. Note: scp is rejected on these hosts → upload/download via
`ssh "cat > f"` / `ssh "cat f"`; and **zsh doesn't word-split unquoted vars** so ssh `-o` flags
must be inline.)

**Quality: mostly noisy — corrected an earlier too-rosy read.** Top-of-page cherry-picks
(Uncle/Aunt/Isabel) are clean, but a representative spread is mostly fragmented/contaminated
(stacked neighbour rows, ruled lines, unreadable bits). Honest clean fraction is **low (~10-25%
by eye)**, not "mixed." Causes: (1) loose detection boxes + dense writing → neighbour bleed;
(2) **`inksight_vectorize` applies `mask_to_box` but NOT ruled-line/band removal (`clean_word`)
— a real omission**, so ruled lines pass through; (3) InkSight completes stray ink into the word.

**Conclusion:** `diarybank-v2` built from this would be as contaminated as v1. The GPU run's
value was validating speed/pipeline AND giving a scale-accurate picture: **crop-noise, not
stroke recovery, is the dominant blocker** — reconfirming the letter-level path. Page strokes
saved locally at `outputs/<slug>/page_001/<slug>_p001_strokes_inksight.json` (gitignored).

**Next (revised):** (a) quick — add `clean_word` ruled-line/band removal into
`inksight_vectorize` before derendering; (b) tighten detection boxes (Q2); (c) the real fix —
N2 letter-level segmentation + reassembly. Don't scale to the full corpus until a single page
comes out mostly clean.

## 2026-06-28 — Rendering/noise diagnosis + plan; CI rescued

**CI was red from the first push** (newly-created private repo). Three stacked causes, all
fixed (now green): (1) pre-existing — `test_recognizer` renders glyphs from macOS-only system
fonts, absent on the Linux runner → added a DejaVuSans fallback (`_recognizer.font_bank`, used
by both the recognizer and the test helper); (2) a copied recipe script wasn't lint-clean;
(3) I'd run `ruff check` but not `ruff format`. **Process fix: run `ruff check . && ruff
format --check . && pytest` (whole repo) before every push, not just changed files.**

**Rendering/noise problems (flat & rich both imperfect). Three distinct causes:**
- **P1 — neighbour-line ink in the crop:** loose detection boxes + dense cursive bleed bits of
  the line above/below into the crop.
- **P2 — wrong connections:** P1's stray ink + InkSight being a holistic tracer that "completes"
  paths it shouldn't.
- **P3 — fidelity:** InkSight makes a *plausible* tracing, not a pixel-faithful copy (survey:
  ~67% look human-traced), so it can miss/add strokes even on a clean crop (e.g. the "Uncle"
  leading loop was InkSight hallucinating, not crop bleed — the masked crop was clean).

**Obvious fix attempted + self-checked → NEGATIVE (not shipped):** a stricter "keep only
components mostly inside the box" mask to kill P1. Rendered and inspected it — **worse**: it
kept *more* neighbour text (the bbox-overlap÷pixel-area metric is ill-posed for sprawling
cursive components). The existing connected-component mask is at/near the crop-level ceiling.

**Conclusion:** crop-level masking has hit diminishing returns. P1/P2 are really solved by
**letter-level identification + reassembly** (cut clean strokes into letters → label from the
known transcript → keep clean per-letter glyphs → reassemble with *chosen* ligatures) — roadmap
N2→N3→N4 — which also sidesteps P3 (keep good letter *shapes*, not a perfect whole-word trace).
Progress now needs clean strokes at scale → the GPU run (N1). Note: **W&B is tracking, not a GPU
provider** (Launch only dispatches to compute you connect); use RunPod for the GPU work.

## 2026-06-28 — Planned: stroke thickness / faintness (richer than [x,y,pen])

**Gap (raised by the user):** the stroke format `[x, y, pen]` is trajectory-only — it
captures *where the pen went* but not line **thickness** or **faintness** ("weakness"), so
the rendered output can't convey pen weight. **Neither the skeleton tracer nor InkSight
emits this** — both are pure pen-*path* models — so it's a new capability, not just a format
change.

**Recoverable from a static scan:** *width* (per-point stroke thickness via the ink distance
transform — `vectorize._stroke_width` already computes a global version) and *intensity*
(per-point ink darkness = faintness). **Not** recoverable: true pen pressure/velocity (needs
a digitizing tablet); width+darkness are the visual proxies.

**Design (3 layers, increasing cost):**
1. Format → `[x, y, pen, width, intensity]` (forward-compatible; downstream ignores extra
   channels until used).
2. Capture → sample distance-transform width + pixel intensity at each derendered point;
   natural home is `inksight_vectorize` (it gives the path, we read the crop at each point).
3. Use → (a) render-time variable line-width/opacity = immediate visual win, no model change;
   (b) the transformer predicts width/intensity (new tokens/heads) so *generated* handwriting
   varies thickness — the real architecture lift.

**Sequencing:** capturing width+intensity in the data is cheap + forward-compatible (can do
alongside the vectorizer); teaching the model to *generate* thickness should wait until a
plain trajectory is confirmed legible (diarybank-v2) — thickness on garbage trajectories is
wasted, on good ones it's a clear upgrade.

---

## 2026-06-28 — Crop tightening (mask-to-box) so derendering sees ONE word

InkSight traces *all* ink in a crop, so our over-capturing crops made it derender neighbours
too ("Uncle" crop → "Uncle Joh"). Worked the fix:

- **Horizontal clamp** (symmetric to the existing vertical clamp in `fit_crop_to_ink`):
  helped (Uncle 327→255 px) but **insufficient** — in dense diary cursive any rectangle around
  a word still catches its neighbours above/below/beside.
- **Mask-to-box** (the fix): whiten all ink outside the detection box, padded vertically for
  ascenders/descenders and tightly horizontally. Re-ran InkSight on masked crops: **"Uncle"
  now derenders as just "Uncle"** (neighbour gone), "Aunt" clean. Residual minor **vertical**
  bleed ("John" still catches an "eet-" sliver from the line below) in tight line-spacing.

**Integrated (opt-in, no regression):** `config.CROP_MASK_TO_BOX` (+ `CROP_MASK_HPAD/VPAD_UP/
VPAD_DN`), `pdf_utils.box_mask_polygon` + `crop_to_box(mask_to_box=...)` reusing the existing
`whiten_outside_polygon` path, `crop.py` wiring + `--mask-to-box` CLI flag, and a unit test.
Off by default so existing crop/overlay outputs + tests are unchanged. **160 tests pass, ruff
clean.** Enable with `OCR_CROP_MASK_TO_BOX=1`. Evidence: `ocr/experiments/inksight_masked_clean.png`.

**Update — connected-component-aware masking (done).** Replaced the rectangle whiten with
`mask_crop_to_box`: keep ink that (a) belongs to a component overlapping the detection box
and (b) sits in a tight x-band / generous y-band, then whiten the rest. This drops *separate*
neighbour rows that overlap the crop rectangle, and the x-band still hard-cuts ligatured
horizontal neighbours. Result on the 4 probe words: **Uncle / Aunt / Isabel fully isolated.**

**"John" residual is a detection-box bug, not a masking one.** The "eet-" sliver persisted
because **John's own detection box dips down into the line below** — masking-to-box faithfully
keeps whatever the box contains. Tightening the downward band to `vpad_dn=0.15` clips the
near-miss overhang (the boxes here include descenders, so 0.15 doesn't cut them), but a box
that *genuinely spans two lines* can't be fixed by box-masking — that's upstream detection
(tighter/line-aware boxes). Defaults set to `vpad_up=0.30, vpad_dn=0.15`. 14 crop/helper tests
pass, ruff clean.

**Next:** wire InkSight in as the vectorizer (replace `vectorize.py` output) and rebuild
`diarybank-v2` from masked crops; separately, tighten detection boxes for the loose two-line
cases.

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
