# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A transformer that generates **cursive handwriting as pen strokes** (not images), conditioned on ASCII text via cross-attention. Core architecture is GPT-2/`makemore`-style self-attention plus a cross-attention layer that attends to the character context. By Sam Greydanus & Zachary Wimpee ([paper](http://arxiv.org/abs/2504.00051)). The README is a long, dated dev log — read its early sections ("How it works", "Preprocessing and Tokenization") for design rationale; the dated entries are historical and not authoritative about the current code.

## Commands

Training and sampling are designed to run on a **GPU** (`device` defaults to `cuda`) and **require Weights & Biases** — `train.py` calls `wandb.init`/`wandb.log` throughout and stores checkpoints as W&B artifacts. There is no test suite, linter config, or build step.

```bash
pip install -r requirements.txt   # covers training/sampling ONLY (torch, wandb, numpy, scipy, matplotlib)

# Train (dataset_name -> data/{name}.json.zip)
python train.py --wandb_entity <user> --wandb_project <proj> --wandb_api_key <key> \
  --dataset_name bigbank_3500 --batch_size 32 --max_seq_length 1050 --num_words 4 \
  --n_layer 5 --max_steps 125000 --learning_rate 1e-2 --downsample_mean 0.65

# Resume from a W&B run's best checkpoint (loads model + optimizer + scheduler + step)
python train.py ... --load_from_run_id <run_id>

# Sample (loads checkpoint, writes PNGs locally)
python sample.py ... --dataset_name bigbank --load_from_run_id <run_id>
```

Every hyperparameter lives in **`get_all_args()` in `model.py`** — this is the single source of truth for defaults, types, and help text. It works two ways: as an argparse CLI (scripts) and, with `get_all_args(use_argparse=False)`, as a `SimpleNamespace` for notebooks/Colab. `vocab_size`, `block_size`, and the context dims are NOT args — they are computed from the dataset at runtime and assigned onto `args` before the model is built.

## Architecture & data flow

Flat module layout, no package. Import graph: `train.py` → `model.py` (`get_all_args`, `get_checkpoint`, `save_checkpoint`) + `sample.py` (`save_samples`) + `data.py` (`create_datasets`, `InfiniteDataLoader`); `sample.py` → `model.py` + `data.py`.

- **`model.py`** — args, checkpoint I/O (local `best_checkpoint.pt` + W&B artifacts), and the `Transformer`. Each `Block` runs causal self-attention then `CrossAttention` over the encoded ASCII context (`forward(idx, context, targets)`). Note both stroke positions (`wpe`) and context positions (`wcpe`) get positional embeddings — a missing context positional embedding was historically a major bug.
- **`data.py`** — dataset construction, tokenization, augmentation. `create_datasets` loads `data/<name>.json.zip`, splits train/test, then **combinatorially expands** a few thousand single-word samples into hundreds of thousands of multi-word training examples via `generate_word_combos` (`num_words` per example). `StrokeDataset.__getitem__` augments and tokenizes on the fly.
- **`sample.py`** — generation + matplotlib plotting. `save_samples` is called from the train loop for W&B logging. `generate_paragraph` does sentence/paragraph generation (generate `n_at_a_time` words, split on WORD_TOKEN, line-wrap) and supports **regenerating specific word indices** to fix misspellings — pass the prior output back as `word_list_offsets` plus `regenerate_ixs`.
- **`train.py`** — the loop: `InfiniteDataLoader`, AdamW + StepLR, periodic eval, and checkpoint-on-best-test-loss to W&B.

### Stroke representation & tokenization (the core idea)

Raw data → absolute points `(x, y, pen_down)` → **offsets** (diffs) → **polar** `(r, theta, pen)` (`decompose_offsets`/`reconstruct_offsets`). Polar decouples magnitude from direction and is what's tokenized. Each stroke offset becomes **two tokens**: a `theta` bin (220 bins) and a combined `radius + pen-up/down` bin (`r_bins`, doubled for pen-up vs pen-down). Order matters — theta token precedes radius token ("point, then shoot"). Words within an example are separated by **pairs of `WORD_TOKEN`**; sequences also use `PAD_TOKEN` and `END_TOKEN`. `encode_stroke`/`decode_stroke` and `split_by_word_tokens`/`concat_with_word_tokens` are the round-trip.

Augmentation (`augment_stroke`) applies shear, per-axis scale, and **aggressive randomized downsampling** (`downsample_mean` ± `downsample_width`, ~60–70%). Randomizing the downsample rate per-sample is what decoupled letter position from token index and dramatically cut overfitting — don't disable it casually.

### Dataset format & files

Datasets are zipped JSON in `data/`, referenced by bare name via `--dataset_name`. Each item is `{'points': [[x, y, pen_down], ...], 'metadata': {author, asciiSequence, pointCount, strokeCount, aspectRatio}}`. `.gitignore` blocks `data/*.zip` by default with explicit `!data/<name>.json.zip` allowlist entries — **a new committed dataset must be added to that allowlist** or it won't be tracked. Helper scripts: `data/make_wordbank.py` (synthesize prompt word banks with realistic letter/punctuation distributions), `data/setup_easybank.py` (filter out words containing `i/j/t/x` — the "easybank" simplification), `data/collect.html` (self-contained browser tool for trackpad/mouse handwriting collection → JSON export).

## OCR dataset pipeline (`ocr/` package)

A separate, newer workflow that builds training data from **scanned handwriting PDFs** instead of hand collection. It is not wired into training and has its **own dependencies** (`ocr/requirements.txt`, not the root one): `google-generativeai`, `pdf2image` (needs system `poppler`), `opencv-python`, `Pillow`, etc. Set `GOOGLE_API_KEY` for the Gemini steps. Refactored out of the Colab notebook `cursive_ocr_extraction.ipynb` and the former `data/content/*.py` scripts; `ocr/README.md` has full usage. Stages:

1. **Transcribe + detect** (`python -m ocr.extract_boxes`) — per page it (a) transcribes the full text to `<prefix>_transcript.txt` at the page root (plain-text Gemini call, prints word/symbol counts) and (b) detects word+punctuation **tokens** (punctuation attaches to its word) as `box_2d` boxes (`[ymin, xmin, ymax, xmax]`, 0–1000 scale). The detection JSON is parsed by `gemini_ocr.parse_word_boxes`, which **accepts any `*_2d` key** — the model frequently emits `html_2d` instead of `box_2d`, and dropping those collapses a 183-box page to ~8. **Detection is transcript-grounded by default** (`--grounded`): the model gets the transcript tokens as a **numbered** list and returns each box tagged with its token number (`{index, box_2d}`); `ocr/reconcile.py` maps boxes to tokens **by that index** and infills (flagged `"estimated": true`) any the model omitted. Map by index, **not by position or text** — the model drifts off the order on a dense page (skips a word), so positional/text alignment silently mislabels every box after the skip (a real regression I hit: it dropped QA-true box-location to 80/183). Indexed grounding gives QA 100% *and* box-location 175/183 (3 major) — better than free detection's 92%/96.2%. `--no-grounded` = free detection.
2. **Vectorization** — turn cropped ink into ordered `(x, y, pen)` points, emitting the **same `{points, metadata}` format** as the collected datasets. Two ways: `python -m ocr.vectorize` (OpenCV threshold → `skeletonize` → nearest-neighbor `order_points`, **box-relative** coords) or `ocr.trace_strokes` / the browser tool (**page-relative** coords).
3. **Package + QA** (`python -m ocr.package_boxes`, then `ocr.qa`) — build the per-box folders, then check the transcript equals the concatenation of all `text_recognized.txt` token-for-token (report at `<prefix>_qa.txt`). `python -m ocr.verify_dataset --coords box|page` is an optional whole-page stroke overlay. Crops are **content-aware** (`config.CROP_FIT_INK`, default on; `pdf_utils.fit_crop_to_ink`): after padding, the crop is fitted to the connected ink components overlapping the word's box and takes their *whole* bounding box, so ascenders/descenders/dots are never cut and separate left/right neighbours are excluded (a bounded vertical clamp limits row-bleed from descenders that merge into the next line). **`vectorize` and `package_boxes` must share the same crop settings** or the teal overlay misaligns. Verified by a vision-agent workflow (target-word completeness): clipping dropped from 132/183 to 15/183 (only 6 "major", which are Gemini `box_2d` detection errors on short words, not croppable).

**Conventions:** run the CLIs from the **repo root** as `python -m ocr.<module>`; the whole pipeline is **`--pdf X --page N` driven** — each CLI reads its input and writes its output at a canonical location, so you rarely pass explicit paths. The output layout is centralized in **`ocr/paths.py`**: under `outputs/<pdf_slug>/page_<NNN>[_V]/` go the page-level `boxes.json`, `boxes_overlay.jpg`, `strokes.json`, plus **one folder per word box** `<pdf_slug>_p<NNN>[_V]_box_<III>/` containing `text_recognized.txt`, `box.jpg`, and `vectorized.jpg` (teal strokes over the crop). Origin is baked into every name. Root defaults to `outputs/` (`--output-root` / `OCR_OUTPUT_ROOT`); `outputs/` is git-ignored. **Re-processing a page is versioned, not overwritten**: first run is `page_<NNN>` (base), then `page_<NNN>_1`, `_2`, … (`extract_boxes` auto-increments; the other CLIs default to the latest, override with `--version N`, where `0` = base). `--limit N` processes only the first N words of a page. The per-box folders are built by `python -m ocr.package_boxes` (after `vectorize`). **Gemini detection must run at `temperature=0`** (`build_model` sets this) — at default temperature it intermittently returns degenerate OCR (mostly punctuation) on dense pages; the response is parsed tolerantly (`gemini_ocr.parse_word_boxes`). The Gemini key is read from `GOOGLE_API_KEY` — locally it's in the envchain `gemini` namespace, so prefix detection/tracing runs with `envchain gemini python -m ocr.extract_boxes ...`. `scrape_diary.py` (Selenium) is the upstream source-fetcher that downloads diary page images into a PDF under `outputs/`.
