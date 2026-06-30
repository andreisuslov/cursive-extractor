# Font pipeline (experiments) — handwriting → dynamic font

Toward the project's north star (ROADMAP.md): a document in someone's hand → a dynamic font of
their writing. These `_`-prefixed modules are the research pipeline. All are CPU/no-network and
covered by `tests/`.

## Stages

| Stage | Module | What it does | Status |
|---|---|---|---|
| Vectorize | `ocr/inksight_vectorize.py` | crop → clean pen strokes via **InkSight** derendering (GPU); `--rich` adds width/intensity | ✅ works |
| Clean crop | (in vectorize) `clean_word` + `mask_crop_to_box` | strip ruled lines/bands, isolate one word | ✅ partial |
| Cut → letters (geometry) | `_letter_segment_strokes.py` | 3 cutters: equal-x baseline, recognizer forced-align, trajectory | ⚠️ failed on cursive |
| **Cut → letters (CRAFT)** | `craft_segmenter.py` + `craft_train.py` | fine-tuned CRAFT char-region → L-1 cuts via known spelling | ✅ **works** (see below) |
| Harvest | `_letter_harvest.py` | confidence- + recognition-gated per-letter collection | ⚠️ gated by cuts |
| Cluster (N3) | `_variant_cluster.py` | k-means → ≤3 medoid variants per letter | ✅ works |
| Render (N4/N5) | `_font_render.py` | place variant glyphs, cycle variants per repeat, draw joins | ✅ works |
| Backend demo | `_font_pipeline_demo.py` | proves N3/N4 on clean font-traced letters → `font_backend_demo.png` | ✅ works |
| Font export | `_font_export.py` | variants → SVG glyph paths (`paths.json`) + `specimen.svg` | ✅ works |

## CRAFT letter cutter (the breakthrough on cutting)

Geometry/recognizer cutters all failed on connected cursive. A fine-tuned **CRAFT** character-region
detector solves it — trained only on **free synthetic cursive** (fonts → perfect per-character
labels), it transfers to real diary ink and localizes individual letters. Full story + before/after:
`outputs/.../page_001/craft_results.html`.

- `_craft_model/` — vendored CRAFT model (clovaai, MIT), patched for modern torchvision.
- `craft_train.py` — reproduces the weights: synthetic cursive + region/affinity GT, diary-style
  augmentation + CLAHE (closes the synthetic→real gap), OHEM loss (crisp peaks). CPU, ~20-40 min.
- `craft_weaksup.py` — real-domain weak-supervision: pseudo-label confident real words (model fired
  ~L separated peaks), fine-tune on synth+real → sharper, separated peaks (**v4**, the default).
- `craft_segmenter.py` — `CraftSegmenter().cut_word(crop_rgb, text)` → exactly `len(text)-1` cuts
  (model peaks where confident, width-prior backfill where faint). Extraction is pure numpy/cv2 and
  unit-tested; torch is lazy-loaded only for inference.

**Status (v4, after weak-supervision):** fires per-letter on every tested word with crisp, separated
peaks — peak counts close to the true letter counts (Isabel 6/6, John 4/4), cuts land roughly
per-letter on clear words; faint words still lean on the width prior. Next: more pages of
weak-supervision + sharper peak→box (affinity-aware) extraction.

**Setup (not committed — weights are large):**
```bash
pip install torch torchvision            # ~CPU build is fine
mkdir -p ocr/experiments/craft_weights   # base CRAFT weights (MIT, from EasyOCR's release):
curl -sSL https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip -o /tmp/c.zip
unzip -o /tmp/c.zip -d ocr/experiments/craft_weights/
python -m ocr.experiments.craft_train     # synthetic -> craft_weights/craft_finetuned_v3.pth
python -m ocr.experiments.craft_weaksup --page-dir outputs/<pdf>/page_001  # -> ..._v4.pth (default)
```

## Human-in-the-loop letter labelling

The lever that actually improves the cutter: human-verified labels (self-labels didn't help, see
WORKLOG). Workflow — fix cuts in the browser, labels auto-fill from the transcript:

```bash
python -m ocr.refine_boxes --boxes <boxes.json> --out <dir>/boxes_refined.json --page <page.png> --tighten
python -m ocr.experiments.label_export --page-dir <dir> --out <dir>/label_review.json   # CRAFT pre-cuts
# open ocr/experiments/letter_label.html, load label_review.json, drag cuts, Export -> labels_corrected.json
python -m ocr.experiments.label_ingest --corrected labels_corrected.json --page-dir <dir> --out-dir letters/
```

- `letter_label.html` — per word: the real crop + CRAFT's cuts as **draggable polylines** (top/bottom
  handles for slant, dbl-click to add midpoints for curves), an **eraser** brush to white-out
  non-letter ink, slices coloured + labelled from the spelling (you only fix cuts). `space` = next,
  `⏎` = done+next, `s` = skip; exports `{text, box, cuts(polylines), erase}`.
- `label_export.py` / `label_ingest.py` — feed the tool (CRAFT cuts) / turn corrections into labeled
  per-letter crops (clean supply for the font pipeline + real CRAFT weak-sup GT).

## The one blocker

Everything is wired end-to-end and the **backend is proven**: `font_backend_demo.png` shows a
readable, joined, variant-rotated "the quick brown fox" rendered from clean letters (traced from
cursive fonts) through the *real* N3/N4 stages.

What's missing is **clean per-letter glyph SUPPLY from diary cursive.** Automatic cutting of
connected cursive is unsolved here — 3 cutters + a geometric confidence gate + a recognition gate
all fail (cursive overlaps in x; m/n/u/w have internal valleys; the font-template recognizer
can't tell a good cut from a bad one). Even single-letter words don't yield textbook glyphs
(InkSight derender + ligature flourishes + residual crop noise). Full history: WORKLOG.md.

## Realistic paths to clean supply (pick one)

1. **Human-in-the-loop** cut review (prebuild auto-cuts, human confirms/nudges) — how
   Calligraphr-style tools ship a font; realistic for one writer. **Built:**
   `_letter_review.py` (export auto-cuts → `review.json`; ingest corrected → clean library) +
   `letter_review.html` (drag/add/delete cut lines per word, export). See "Run it" below.
2. **Letter template** collection (`ocr/handwriting_tool.html` / `datasets/collect.html`) — have
   the writer write each letter a few times → clean isolated glyphs straight into N3/N4. Sidesteps
   cutting entirely; changes the input from "arbitrary document" to "fill a template".
3. **Stronger recogniser/derenderer** — a CNN/HTR trained on more real letters to either guide
   cuts or judge harvested glyphs (data-starved today; needs thousands of labelled letters).

## Run it

```bash
# 1. strokes (GPU; needs InkSight model + a page's boxes.json) — see _inksight_probe.py for setup
OCR_INKSIGHT_MODEL=/path/to/small-p-cpu python -m ocr.inksight_vectorize --pdf <pdf> --page N --rich
# 2a. AUTOMATIC: harvest letters (recognition-gated) -- rough, see blocker above
python -m ocr.experiments._letter_harvest --strokes <strokes.json> --recognize --out lib.json
# 2b. HUMAN-IN-THE-LOOP (clean): export auto-cuts, fix them in the browser tool, ingest
python -m ocr.experiments._letter_review export --strokes <strokes.json> --out review.json
#    open ocr/experiments/letter_review.html, load review.json, fix cuts, save corrected.json
python -m ocr.experiments._letter_review ingest --review corrected.json --out lib.json
# 3. cluster into <=3 variants (pass several libs to pool a writer's pages -- N6)
python -m ocr.experiments._variant_cluster --library lib1.json lib2.json --k 3 --out variants.json
# 4. render text in the harvested hand
python -m ocr.experiments._font_render --variants variants.json --text "hello" --out hello.png
# 5. export reusable SVG glyph assets (paths.json + specimen.svg)
python -m ocr.experiments._font_export --variants variants.json --out-dir fontout/
# backend proof on clean font-traced letters:
python -m ocr.experiments._font_pipeline_demo --text "the quick brown fox" --out demo.png
```
