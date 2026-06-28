# Roadmap — North Star & where we are

## The goal (north star)

A piece of software that takes **a document handwritten by one person** and produces a
**dynamic font of that person's handwriting**, then renders arbitrary text in their hand.

"Dynamic" = the font carries **at least 3 variants of every letter**, so generated text
rotates through them and never looks like a repeated stamp — it reads as real handwriting.

Concretely, end to end:

1. Feed in a scan/photo of someone's handwriting.
2. **Decode** it: find every letter and collect *all its samples* (many real instances of
   each `a`, `e`, `t`, …) — labelled by which letter they are.
3. Cluster those into **≥3 clean variants per letter** (allographs).
4. Assemble a **dynamic font** from the variants (with cursive joins between letters).
5. Type any text → render it in that person's handwriting.

## Pipeline stages & status (2026-06-28)

| # | Stage | What it produces | Status |
|---|-------|------------------|--------|
| 1 | Acquire | page images from a scan/PDF | ✅ working (`scraper/`, page render) |
| 2 | Detect + transcribe | word boxes + the text of each (so we know which letters to expect) | ✅ working (Gemini grounded, ~94% vs truth) |
| 3 | Recover clean pen strokes per word | ordered `(x,y,pen)` (now also width/intensity) | ✅ **solved in principle** via InkSight derendering (built; needs GPU to run at scale) |
| 4 | Segment words → **labelled letters** | each letter cut out + told which letter it is | ⚠️ **partial — the hard bottleneck.** ~85–91% per page by geometry; cut quality is the wall (see CHANGELOG M10–M13, several honest negatives). Not yet reliable. |
| 5 | Cluster → **≥3 variants per letter** | per-writer allograph library | ⚠️ partial — prototype exists (M12); gated entirely on stage 4 |
| 6 | Build the **dynamic font** | OpenType (or renderer) with randomized alternates + joins | ❌ not started |
| 7 | Render text in the person's hand | typed text → handwriting image/vector | ❌ not started |

## How far are we?

Front half (1–3, *get clean strokes from a scan*) is essentially working — stage 3, long
the blocker, was just cracked with InkSight. The **middle (4–5) is the hard, partly-solved
research core** and the current bottleneck: reliably cutting a word into *labelled* letters.
The **back half (6–7), the actual font, is unbuilt.** Rough honest read: **~40% of the
path**, with the two hardest pieces still ahead — reliable labelled-letter extraction, and
the font-assembly backend.

## Important strategic note (a fork)

There are **two different ways to "generate handwriting,"** and they're easy to conflate:

- **(A) The CursiveTransformer** (`train.py`/`sample.py`/`diarybank`) — a neural model that
  generates novel stroke trajectories from text. Infinite variety, natural cursive joins,
  but **data-hungry** (needs lots of strokes) and produces a *model*, not a *font*.
- **(B) A dynamic font from real samples** — extract the person's actual letters, keep ≥3
  variants each, assemble a font. **This is the stated goal.** Far less data per writer
  (just enough clean samples of each letter), faithful to the person, but needs the
  segmentation (stage 4) and font backend (stage 6) that don't fully exist yet.

The OCR/derender/segment/allograph work serves **(B) directly.** The neural transformer is
**not strictly required** for the stated goal — its natural role here is a **helper**:
synthesize variants for letters the person didn't write often enough (a single page rarely
contains every letter ≥3 times), and/or generate connecting strokes.

## Tracked next steps

The project backlog. Checked = done; ordered roughly by dependency. Keep this list current
as items land (the matching detail lives in `WORKLOG.md`).

**Critical path to the goal**
- [ ] **N1 — Validate stage 3 at scale.** Build `diarybank-v2` with the InkSight vectorizer
  on a GPU; confirm clean strokes for whole pages (not just probe crops).
- [ ] **N2 — Re-attack stage 4 on clean strokes (make-or-break).** Segment the *InkSight
  trajectory* into letters — cutting a clean pen path should beat cutting messy ink — and
  label each cut from the known transcription. The whole middle hinges on this.
- [ ] **N3 — Stage 5: variant clustering.** Cluster labelled letters into ≥3 variants per
  letter; extend the allograph library (M12) to the clean strokes.
- [ ] **N4 — Stage 6: font backend (new).** Variant glyphs → a real dynamic font
  (`fontTools`: outlines → OTF with `calt`/randomized alternates + cursive joins). Prototype
  on one well-covered writer.
- [ ] **N5 — Stage 7: render text** in the person's hand (type → font with variant rotation
  + joins). End-to-end demo.
- [ ] **N6 — Per-writer coverage.** A font needs ≥3 of *every* glyph; one page rarely has
  that. Decide gap-fill: more pages per writer, or neural synthesis (track A) for rare letters.

**Decisions to make**
- [ ] **D1 — Font tech:** real OpenType randomized alternates vs a custom SVG/stroke renderer
  (OpenType = portable; renderer = more flexible for joins). Blocks N4.
- [ ] **D2 — Keep the neural CursiveTransformer?** Drop it for the stated goal, or keep only
  as a gap-filler for rare letters (N6).

**Quality / deferred (not on the critical path)**
- [ ] **Q1 — Thickness in generation.** Data capture done (`--rich`); render-time variable
  width/opacity is free; model *generating* width/intensity is the deferred lift (new heads).
  Do after a plain trajectory is confirmed legible. (WORKLOG 2026-06-28.)
- [ ] **Q2 — Tighten loose two-line detection boxes** (e.g. "John" box dipping into the next
  row) — upstream fix the mask can't do.

## Open questions

- Does cutting the clean InkSight trajectory actually beat cutting raw ink at stage 4? (the
  whole middle hinges on this — untested.)
- Font tech: real OpenType with randomized alternates, or a custom SVG/stroke renderer that
  picks variants + draws joins? OpenType is portable; a renderer is more flexible for joins.
- Do we keep the neural transformer at all, or only as a gap-filler for rare letters?
