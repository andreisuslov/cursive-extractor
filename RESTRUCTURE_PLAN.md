# Folder-Restructure Plan — cursivetransformer

Status legend: `[ ]` planned · `[x]` done

## Goal
Make the repo legible without fighting its intentional design. The flat,
bare-import **training core** (`data.py`, `model.py`, `sample.py`, `train.py`,
imported as `from data import …`) is load-bearing — pytest `pythonpath=["."]`,
the `scripts/` `sys.path` inserts, and `sample.py`/`train.py` self-appends all
rely on it. We work *around* it.

## Rejected: `src/` package rewrite
A `src/cursivetransformer/` layout was evaluated and rejected (adversarial audit
3/10): it breaks the documented flat layout on purpose, touches 6+ entry points
in lockstep, rewrites ~200 doc references, and leaves CI red if any single edit
is missed. The payoff (PyPI packaging / console entrypoints) is not a goal for a
research repo.

## Verification gate (run after every phase)
```
python3 -m ruff check . && python3 -m ruff format --check . && python3 -m pytest -q
```
Must stay **148 passed**, ruff clean.

---

## Phase 1 — Conservative cleanup (untracked/ignored only; no commit)
- [ ] `rm -rf diary_scraper/` — untracked, 1 file (39 B), vestigial
- [ ] `rm data/__init__.py` — untracked, vestigial half-package (`import data` → `data.py`)
- [ ] `rm -rf outputs/_fleet/` — gitignored orchestration scratch
- [ ] `data/content/` (20 MB, untracked) — **preserve**, not delete: it holds the
  only `test_document.pdf` (the OCR default slug anchor). It rides along to
  `datasets/content/` in Phase 2d.

## Phase 2 — Moderate grouping (4 independent commits, each gated)

### 2a — Isolate OCR experiments → `ocr/experiments/`
- [ ] `git mv` the 7 research modules (`_allograph_library`, `_bootstrap_recognizer`,
  `_htr_align`, `_order_recovery_experiment`, `_overlay_inspect`, `_recognizer`,
  `_segment_prototype`) into `ocr/experiments/`; add `ocr/experiments/__init__.py`.
- [ ] Update the 6 test imports: `from ocr import _x` → `from ocr.experiments import _x`
  (and the dotted `from ocr._order_recovery_experiment import run`).
- Safe: `ocr/__init__.py` doesn't re-export them; their `from ocr.vectorize import …`
  keeps working (vectorize stays put).

### 2b — Group the scraper pair → `scraper/`
- [ ] `git mv scrape_diary.py fetch_transcript.py scraper/`
- `fetch_transcript`'s `from scrape_diary import …` keeps working (co-located).

### 2c — Corral notebooks → `notebooks/`
- [ ] `git mv diary_extraction.ipynb train_sample_visualize.ipynb notebooks/`
- [ ] Fix the README Colab/GitHub link to `train_sample_visualize.ipynb`.

### 2d — Rename `data/` → `datasets/` (kills the name-clash; highest churn, last)
- [ ] `git mv data datasets` (carries untracked `content/` along)
- [ ] `data.py` — dataset-load path `…/data/{name}.json.zip` → `…/datasets/…`
- [ ] `data/build_diarybank.py` → `datasets/build_diarybank.py`: `REPO/"data"` → `"datasets"`
- [ ] `ocr/config.py` — `OCR_PDF_PATH` default `data/content/…` → `datasets/content/…`
- [ ] `.gitignore` — `/data/…` patterns + 9 `!data/*.json.zip` allowlist → `datasets/`
- [ ] `README.md`, `CLAUDE.md`, `ocr/README.md` — `data/` path refs → `datasets/`
- [ ] Full `pytest` (the regression test loads `bigbank`/`easybank` from the dir).

## Left intentionally alone
- Flat training core + bare imports + `pythonpath=["."]`.
- `ocr/` pipeline, `scripts/`, `tests/` layout (already clean).
- `static/` (~150 README dev-log assets — high churn, low payoff).
