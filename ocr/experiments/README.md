# Font pipeline (experiments) — handwriting → dynamic font

Toward the project's north star (ROADMAP.md): a document in someone's hand → a dynamic font of
their writing. These `_`-prefixed modules are the research pipeline. All are CPU/no-network and
covered by `tests/`.

## Stages

| Stage | Module | What it does | Status |
|---|---|---|---|
| Vectorize | `ocr/inksight_vectorize.py` | crop → clean pen strokes via **InkSight** derendering (GPU); `--rich` adds width/intensity | ✅ works |
| Clean crop | (in vectorize) `clean_word` + `mask_crop_to_box` | strip ruled lines/bands, isolate one word | ✅ partial |
| Cut → letters | `_letter_segment_strokes.py` | 3 cutters: equal-x baseline, recognizer forced-align, trajectory (pen-lifts + baseline valleys) | ⚠️ **unsolved** |
| Harvest | `_letter_harvest.py` | confidence- + recognition-gated per-letter collection | ⚠️ gated by cuts |
| Cluster (N3) | `_variant_cluster.py` | k-means → ≤3 medoid variants per letter | ✅ works |
| Render (N4/N5) | `_font_render.py` | place variant glyphs, cycle variants per repeat, draw joins | ✅ works |
| Backend demo | `_font_pipeline_demo.py` | proves N3/N4 on clean font-traced letters → `font_backend_demo.png` | ✅ works |

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
   Calligraphr-style tools ship a font; realistic for one writer.
2. **Letter template** collection (`ocr/handwriting_tool.html` / `datasets/collect.html`) — have
   the writer write each letter a few times → clean isolated glyphs straight into N3/N4. Sidesteps
   cutting entirely; changes the input from "arbitrary document" to "fill a template".
3. **Stronger recogniser/derenderer** — a CNN/HTR trained on more real letters to either guide
   cuts or judge harvested glyphs (data-starved today; needs thousands of labelled letters).

## Run it

```bash
# 1. strokes (GPU; needs InkSight model + a page's boxes.json) — see _inksight_probe.py for setup
OCR_INKSIGHT_MODEL=/path/to/small-p-cpu python -m ocr.inksight_vectorize --pdf <pdf> --page N --rich
# 2. harvest letters (recognition-gated)
python -m ocr.experiments._letter_harvest --strokes <strokes.json> --recognize --out lib.json
# 3. cluster into <=3 variants
python -m ocr.experiments._variant_cluster --library lib.json --k 3 --out variants.json
# 4. render text in the harvested hand
python -m ocr.experiments._font_render --variants variants.json --text "hello" --out hello.png
# backend proof on clean font-traced letters:
python -m ocr.experiments._font_pipeline_demo --text "the quick brown fox" --out demo.png
```
