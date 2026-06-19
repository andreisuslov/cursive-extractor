# OCR dataset pipeline

Turns scanned handwriting PDFs into the `{points, metadata}` JSON format used to
train the cursive model. Refactored out of the Colab notebook
`cursive_ocr_extraction.ipynb` into importable, locally-runnable modules.

## Setup

```bash
pip install -r ocr/requirements.txt
brew install poppler          # macOS (Debian: apt-get install poppler-utils)
```

**`GOOGLE_API_KEY`** is required for the Gemini steps only (`extract_boxes`,
`trace_strokes`); `vectorize` and `verify_dataset` need no key. Get a key from
https://aistudio.google.com/apikey, then either:

```bash
# Option A (preferred, for Mac users) — store once in envchain, then prefix the command:
envchain --set gemini GOOGLE_API_KEY            # one-time; paste key at prompt
envchain gemini python -m ocr.extract_boxes --pdf "$PDF" --start-page 4 --end-page 4

# Option B — plain env var for the shell session:
export GOOGLE_API_KEY=...
python -m ocr.extract_boxes --pdf "$PDF" --start-page 4 --end-page 4
```

Run every command **from the repo root** so the default `data/content/...` paths
resolve, and so `python -m ocr.<module>` imports the package correctly.

## Output layout

Everything lands under one root (`outputs/` by default; override with
`--output-root` or `OCR_OUTPUT_ROOT`), namespaced by **PDF → page → box**, with
the origin baked into every name so a file is self-describing even if moved:

```
outputs/
  <pdf_slug>/                                     # from the PDF filename stem
    <pdf_slug>_tool.html                          # capture tool (spans a run)
    page_<NNN>[_V]/                               # zero-padded; _V = version (see below)
      <pdf_slug>_p<NNN>[_V]_transcript.txt        # full-page transcription (QA source)
      <pdf_slug>_p<NNN>[_V]_boxes.json            # detection: word+punct tokens + box_2d
      <pdf_slug>_p<NNN>[_V]_boxes_overlay.jpg     # page with boxes drawn
      <pdf_slug>_p<NNN>[_V]_strokes.json          # vectorized {points, metadata}
      <pdf_slug>_p<NNN>[_V]_qa.txt                # transcript-vs-boxes QA report
      <pdf_slug>_p<NNN>[_V]_box_<III>/            # one folder per word box
        text_recognized.txt                       # the OCR'd token (word + its punctuation)
        box.jpg                                   # cropped word image
        vectorized.jpg                            # box.jpg + teal strokes overlay
```

The whole pipeline is **`--pdf X --page N` driven**: each CLI reads its input
and writes its output at the canonical location above, so you never hand-wire
intermediate filenames. Box indices match the source `boxes.json` order (gaps =
entries with no `box_2d`). The layout is defined in one place, `ocr/paths.py`.
`outputs/` is git-ignored.

**Versioning.** Re-processing a page never overwrites. The first run is
`page_<NNN>` (no suffix); the next is `page_<NNN>_1`, then `_2`, …
`extract_boxes` auto-picks the next free version; the other CLIs default to the
**latest** version. Pass `--version N` anywhere to target a specific one
(`--version 0` = the base, no-suffix package).

**Limiting.** `--limit N` (on `extract_boxes`, `vectorize`, `package_boxes`)
processes only the first N words of the page — handy for sampling a page cheaply.

## Files

| File | Role |
|------|------|
| `config.py` | Defaults — PDF path, page range, Gemini model names, and crop padding / fit-ink / **cleaning** flags (all env-overridable, e.g. `OCR_CROP_CLEAN`). |
| `paths.py` | The output layout — per-PDF/per-page folders and origin-stamped filenames. |
| `pdf_utils.py` | Shared: load PDF pages; map 0-1000 `box_2d` to padded pixel crops. |
| `gemini_ocr.py` | Gemini word/box OCR (`extract_words_and_boxes`, `extract_with_fallback`), `draw_boxes_on_image`, and `trace_strokes`. |
| `reconcile.py` | Map indexed grounded detection back to transcript tokens **by index**, infilling any token the model missed (flagged `estimated`). |
| `tool.py` | `render_tool(words)` — inject a word bank into the HTML capture tool. |
| `handwriting_tool.html` | The browser tracing/capture tool (React, self-contained); `__WORD_BANK_JSON__` is the injection point. |
| `extract_boxes.py` | **CLI** — per page: transcribe (→ `transcript.txt` + counts) **and** detect word+punctuation boxes. |
| `vectorize.py` | **CLI** — recover `(x, y, pen)` strokes from cropped ink: ink connected-components → 1-px Zhang-Suen skeleton → continuous DFS trace, with optional per-word crop cleaning + a safety gate (batch over a page, or a single image). |
| `package_boxes.py` | **CLI** — one folder per box: `text_recognized.txt`, `box.jpg`, `vectorized.jpg` (teal overlay); runs QA. |
| `qa.py` | **CLI** — check the transcript equals the concatenation of all `text_recognized.txt`. |
| `crop.py` | **CLI** — crop a single OCR box to a high-DPI image. |
| `trace_strokes.py` | **CLI** — trace strokes for one box via Gemini (model-based alternative to `vectorize`). |
| `verify_dataset.py` | **CLI** — overlay a traced stroke dataset onto a whole PDF page (`--coords page\|box`). |
| `visualize_words.py` | **CLI** — draw the boxes of the first N words of a page. |

## Pipeline

`extract_boxes` (transcribe page + detect word/punctuation boxes) → `vectorize`
(Zhang-Suen skeleton + DFS stroke trace → `strokes.json`) → `package_boxes` (one folder per box,
then **QA**: the transcript must equal the concatenation of all
`text_recognized.txt`, token for token). `verify_dataset` gives an optional
whole-page overlay; the browser tool / `trace_strokes` are the manual / Gemini
alternatives to `vectorize` (their output verifies with `--coords page`).

**Transcript-grounded detection (default, `--grounded`).** Rather than re-reading
the page, detection is given the transcript's tokens as a **numbered** list and
returns each box tagged with its token number (`{index, box_2d}`). `ocr/reconcile.py`
maps boxes to tokens **by that index** — crucially *not* by position or text,
because the model drifts off the order on a dense page (skips a word and shifts),
which silently mislabels every box after it. Each box's text is set to its token,
and any token the model didn't return is **infilled** geometrically (flagged
`"estimated": true`; the content-aware crop then snaps it to ink). Result: every
transcript word gets exactly one correctly-located, correctly-labelled box. On
the sample page this raised QA from 96% to 100% and box completeness from 92% to
96% vs. free detection. `--no-grounded` reverts to free detection.

Each box folds in its trailing punctuation (e.g. `wealth.` is one box). To make
sure crops capture **whole letters, not parts**, cropping is content-aware
(`config.CROP_FIT_INK`, on by default): after padding, it finds the connected
ink components overlapping the word's box and returns the bounding box of those
*whole* components — so ascenders/descenders/dots are never cut, and separate
left/right neighbour words are excluded. A bounded vertical clamp stops a long
descender that merges into the next line from pulling in a whole adjacent row.
`vectorize` and `package_boxes` share these crop settings so the teal overlay
stays aligned; disable with `--no-fit-ink`.

**Stroke recovery.** `vectorize` turns a word crop into ordered `(x, y, pen)`
points: it thresholds the ink, splits it into **connected components** (each a
real pen-lift — separate letters, i-dots, t-crossbars), thins each to a 1-px
**Zhang-Suen skeleton**, and traces it as a single continuous **depth-first
path** (retracing back over already-drawn edges at dead-ends, so there are no
fabricated pen-ups and no spurious lines). This recovers exactly the lifts that
are *physically separable* from a static image — replacing the older greedy
nearest-neighbour skeleton tracer, which shattered one cursive word into ~95
fragments. The QA harness `ocr._order_recovery_experiment` measures the fidelity.

**Crop cleaning + safety gate** (`config.CROP_CLEAN`, on by default; disable with
`OCR_CROP_CLEAN=0`). Before tracing, `vectorize.clean_word` strips ruled lines and
scan-edge bands and keeps only the target word's ink (neighbouring words removed),
so a dense, ruled page still yields clean per-word strokes. A **safety gate** then
compares the cleaned trace against the uncleaned one and falls back to the
uncleaned strokes whenever cleaning would *increase* the stroke count (the
signature of a bad detection box) — so cleaning can never make a word worse. Each
box records which path was used in `metadata["cleaned"]`, and `package_boxes` crops
`box.jpg` to match so the overlay stays aligned.

## Usage

Pass `--pdf` and `--page`; everything routes through the canonical layout above.

```bash
PDF=data/content/test_document.pdf

# 1. Transcribe + detect boxes on page 4 (first run -> page_004; re-runs -> page_004_1, _2, ...)
python -m ocr.extract_boxes --pdf "$PDF" --start-page 4 --end-page 4 --no-tool

# 2. Vectorize the page -> ..._strokes.json   (Zhang-Suen skeleton + DFS trace)
python -m ocr.vectorize --pdf "$PDF" --page 4

# 3. One folder per box (text_recognized.txt, box.jpg, vectorized.jpg) + QA
python -m ocr.package_boxes --pdf "$PDF" --page 4

# QA on its own (transcript vs concatenated box texts) -> ..._qa.txt; exit 0 = PASS
python -m ocr.qa --pdf "$PDF" --page 4

# Only the first 15 words of the page (applies to steps 1-3):
python -m ocr.extract_boxes --pdf "$PDF" --start-page 4 --end-page 4 --limit 15 --no-tool
```

Override any default with `--boxes`, `--strokes`, `--output`, `--save`, or
`--output-root`; target a specific package with `--version N`.

## Development & QA helpers

Two `_`-prefixed modules support tuning the vectorizer; they are **not** part of the
dataset pipeline and need no Gemini key:

```bash
# Fidelity check: render easybank/bigbank ground-truth words to clean rasters, run them
# back through `vectorize`, and report visual IoU, fabricated pen-ups, path-length ratio,
# x-reversals, and i/j/t/x diacritic recovery. This is the regression behind stroke recovery.
python -m ocr._order_recovery_experiment

# Visual inspection: overlay the recovered strokes on the original ink (one colour per
# stroke, a dot at each stroke start) -> /tmp/overlay_*.png.
python -m ocr._overlay_inspect
```

## Where the old scripts went

The standalone `data/content/*.py` scripts were adapted into this package:

| Old script | Now |
|------------|-----|
| `process_all_words.py`, `vectorize_word.py` | `vectorize.py` (batch + single-image) |
| `generate_first_box.py` | `crop.py` |
| `generate_strokes.py` | `trace_strokes.py` + `gemini_ocr.trace_strokes` |
| `generate_overlay.py`, `visualize_all_vectors.py`, `visualize_first_entry.py` | `verify_dataset.py` (`--coords`, `--index`, `--crop`) |

## Notes on the refactor

- The Colab-only bits were dropped: `!pip`/`!apt-get` magics (see Setup above),
  `google.colab.userdata` (now plain `GOOGLE_API_KEY` env var), and the
  `drive.mount` workspace-backup cell (Colab housekeeping with no local meaning).
- The ~530-line HTML template is now the standalone asset `handwriting_tool.html`
  instead of a giant Python format-string.
- `visualize_words.py` was rewritten: the original notebook cell referenced
  undefined globals and a different box schema; it now reads the per-page JSON
  this pipeline produces.
- The default model is `gemini-pro-latest` (fallback `gemini-3-flash-preview` on
  a 404). Detection runs at **`temperature=0`** — at the default temperature the
  models intermittently return degenerate OCR (mostly punctuation) on a dense
  page; responses are parsed tolerantly (`gemini_ocr.parse_word_boxes`).
