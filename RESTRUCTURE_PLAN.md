# Folder-Restructure Plan — cursivetransformer

Status: **COMPLETE** (2026-06-22). Every phase landed on `ocr-stroke-recovery`,
each gated by `ruff check . && ruff format --check . && pytest -q` (148 passed).

## Goal
Make the repo legible without fighting its intentional design. The flat,
bare-import **training core** (`data.py`, `model.py`, `sample.py`, `train.py`,
imported as `from data import …`) is load-bearing — pytest `pythonpath=["."]`,
the `scripts/` `sys.path` inserts, and `sample.py`/`train.py` self-appends all
rely on it. We worked *around* it.

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

---

## Phase 1 — Conservative cleanup (untracked/ignored only; no commit)
- [x] `rm -rf diary_scraper/` — untracked vestigial (39 B)
- [x] `rm data/__init__.py` — untracked vestigial half-package
- [x] `rm -rf outputs/_fleet/` — gitignored orchestration scratch
- [x] `data/content/` — preserved, rode along to `datasets/content/` in 2d

## Phase 2 — Moderate grouping (4 commits, each gated)

### 2a — `ocr/experiments/` (commit `f4c12e9`)
- [x] `git mv` 7 research modules in; add `ocr/experiments/__init__.py`
- [x] rewrite parent imports `from . import paths, vectorize` / `from .config …`
  → absolute `from ocr import …` (audit missed these; siblings stay relative)
- [x] update 6 test modules to `from ocr.experiments import …`

### 2b — `scraper/` (commit `58c528f`)
- [x] `git mv scrape_diary.py fetch_transcript.py scraper/`; CLAUDE.md ref
- [x] ruff import-group fix for the now-relocated `from scrape_diary import …`

### 2c — `notebooks/` (commit `b52124a`), then removed (Python-only)
- [x] `git mv` both root notebooks in
- [x] README Colab links left intact (they point at the **upstream** repo)
- [x] **Follow-up:** all 4 tracked notebooks deleted (the 2 here + the 2 upstream
  ones in `static/`). The project moved entirely to Python — OCR lives in the
  `ocr/` package, training/sampling in `scripts/`. Notebooks remain recoverable
  from git history and the upstream repo. `notebooks/` dir removed.

### 2d — `data/` → `datasets/` (commit `<this branch>`)
- [x] `git mv` 13 tracked files; `datasets/` now holds the 9 zips + 3 scripts + html
- [x] `data.py` loader path, `build_diarybank.py` output path (`os.path.join … "datasets"`)
- [x] `ocr/experiments/*` bank loaders + `_segment_prototype` slug resolver
- [x] `ocr/config.py` `OCR_PDF_PATH` default + all `ocr/` CLI-example docstrings
- [x] `.gitignore` (`/datasets/…` + allowlist + `/datasets/content/`)
- [x] CLAUDE.md + ocr/README.md updated; README upstream links left intact
- [x] full `pytest` green

## Decisions / honest notes
- **README left mostly alone.** Its Colab/Dataset/`blob/main/…` links resolve
  against the **upstream `greydanus/cursivetransformer`** repo; CLAUDE.md flags
  the README as a historical, non-authoritative dev log. Rewriting those to the
  local layout would 404. Only the one local `datasets/easybank.json.zip`
  mention (smoke-train description) was updated.
- **`test_document.pdf` stub.** A 92-byte stub that upstream's "Ponytail
  cleanup" had deliberately removed got re-added by a `git add -A` in 2a. 2d
  untracks it and `.gitignore`s `datasets/content/` so it can't recur. It lives
  on, untracked, only as the OCR slug anchor (`config` treats it as usually-absent).
- **`git add -A` is a footgun** here — `datasets/content/` (and other scratch)
  is now explicitly ignored to make staging safe.

## Left intentionally alone
- Flat training core + bare imports + `pythonpath=["."]`.
- `ocr/` pipeline, `scripts/`, `tests/` layout (already clean).
- `static/` (~150 README dev-log assets — high churn, low payoff).
