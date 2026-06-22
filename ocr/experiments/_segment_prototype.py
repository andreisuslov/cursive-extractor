"""EXPERIMENTAL transcription-forced per-letter segmentation of a traced word.

Given a word (its ``box_2d`` on the source page) and the KNOWN text
(``text_recognized.txt`` / ``metadata.asciiSequence``), cut the live-traced ink
into exactly ``L = len(word)`` per-letter stroke groups. The transcription
supplies the letter count (the hard constraint) and a per-character width prior;
a dynamic program then places the ``L-1`` internal boundaries at the
highest-scoring HORIZONTAL (x) cut columns near their expected x-positions.

Two blockers found on the first prototype (commit adb2827) drove this rewrite:

  (1) CONTAMINATED CROPS -- the stored ``box.jpg`` is padded and catches ink from
      the rows above/below. We now re-rasterize the source page and feed a CLEAN
      per-word image from ``vectorize.clean_word(page, box_2d)`` (ruled lines /
      scan bands / neighbour rows + same-line neighbours stripped). On this
      densely-ruled page the rows nearly touch and ``clean_word`` keeps whole ink
      components, so a residual band is masked to a tight horizontal band around
      the word's row (sized by the page row pitch -- see ``restrict_to_word_band``),
      which pixel-wise clips the rows above/below that clean_word still admits.

  (2) ARC-LENGTH != X -- ``vectorize.trace_component`` is a backtracking DFS, so
      cumulative arc-length along the traced path is NOT monotonic in x. Cutting
      by arc-length therefore cut at the wrong horizontal positions (the "leap"
      artifact). We now cut by HORIZONTAL X-POSITION: score vertical cut columns
      from the ink profile, run the DP over x, and assign every body point to a
      slot by its x relative to the ``L-1`` vertical boundaries -- trace order is
      irrelevant.

Why a NEW module (and why ``_``-prefixed = prototype): this reuses
``ocr.vectorize`` (live re-trace + ``clean_word``) and ``ocr.paths`` only, adds
numpy + cv2, and writes per-letter ``{points, metadata}`` files that round-trip
through ``data.decompose_offsets``/``strokes_to_offsets`` unchanged.

Algorithm:

  0. LOAD + LETTER SLOTS  -- ``L = len(word)``; ``L == 1`` -> emit whole word.
  1. CLEAN + LIVE TRACE   -- ``clean_word`` + central-band mask -> ``trace_ink``.
  2. CLASSIFY components  -- split BODY runs from DIACRITICS (dots / crosses).
  3. CUT PROFILE          -- per-column cut score over x. The normal MULTI-component
                             word uses the ink profile (thin column + compact vertical
                             extent + baseline band). A word that is ONE ligature-joined
                             connected stroke (e.g. *translate* -- a whole cursive line
                             with no pen-gaps) is x-density's weak case: the profile is
                             flat, so cuts pile onto the few incidental gaps and leave
                             whole letter-runs uncut. For that case ONLY we fall back to
                             a STROKE-TOPOLOGY score (``topology_cut_score``): the
                             Zhang-Suen skeleton's UPPER ENVELOPE dips to the baseline at
                             the low, thin connectors between letters and lifts into the
                             x-height/ascender zone inside a letter, so its height-minima
                             are the ligature cuts. The transcription still supplies the
                             ``L-1`` count and the DP still places the boundaries.
  4. EXPECTED X           -- per-char width prior -> expected boundary x-positions.
  5. TRANSCRIPTION DP      -- choose L-1 monotone x-boundaries maximising
                             sum(score) - lambda * (x-residual)^2 (banded over x).
  6. SLICE BY X + ATTACH  -- assign each body point to a slot by x; attach each
                             diacritic to a slot by its x-centre.
  7. EMIT                 -- format_strokes per slot -> letter_<kk>_<char>.json.
  8. CONFIDENCE + OVERLAY -- conf gate (low_confidence flag) + colour overlay PNG.

CLI::

    python -m ocr._segment_prototype --pdf test_document --page 4 \
        [--version N] [--box III] [--min-conf 0.4] [--dpi 600]

With no ``--box`` it segments a default sample of page-4 words and writes the
colour overlays to ``runs/seg_proto5_<word>.png`` for visual QA.
"""

import argparse
import json
import os

import cv2
import numpy as np
from pdf2image import convert_from_path

from ocr import paths, vectorize
from ocr.config import PDF_PATH, PDF_RENDER_FALLBACK

# --- tunables (Step 3 cut-score weights + Step 5 DP) ------------------------
W_THIN = 0.45  # thin ink column (a join between letters, not a stem)
W_COMPACT = 0.35  # small vertical ink extent (not a tall stem or a loop bowl)
W_LOWBAND = 0.20  # column ink centred near the baseline (a ligature rides it)
THIN_MEDMULT = 1.7  # col >= THIN_MEDMULT * median non-empty column -> thin score 0
COMPACT_HFRAC = 0.62  # vertical extent >= COMPACT_HFRAC * word-height -> compact 0
LOWBAND_HFRAC = 0.45  # |centroid - baseline| >= LOWBAND_HFRAC * word-height -> 0
DP_LAMBDA = 6.0  # x-residual weight in the DP objective
DP_BAND = 0.30  # search band +-30% of the word x-span around each expected x
GRID_MAX_CAND = 320  # cap the DP candidate columns (downsample wide words)

# Per-character width prior (relative). Wide letters get more x-span; thin stems
# (i, l, t, ...) get less. Anything not listed is 1.0.
WIDTH_PRIOR = {
    "m": 1.8,
    "w": 1.8,
    "M": 1.8,
    "W": 1.8,
    "i": 0.6,
    "l": 0.6,
    "t": 0.6,
    "I": 0.6,
    "j": 0.6,
    ".": 0.5,
    ",": 0.5,
    "'": 0.4,
}

# Distinct, saturated BGR colours cycled per letter slot (matches the teal-style
# overlay palette used in vectorize / _overlay_inspect).
PALETTE = [
    (0, 0, 230),
    (200, 130, 0),
    (0, 160, 0),
    (200, 0, 200),
    (0, 150, 255),
    (255, 160, 0),
    (130, 0, 200),
    (0, 110, 110),
    (110, 70, 0),
    (0, 0, 0),
]


# --- small helpers ----------------------------------------------------------


def _ipt(x: float, y: float) -> tuple[int, int]:
    """Round a (possibly numpy-float) coordinate to a Python int pixel tuple."""
    return (int(np.rint(x)), int(np.rint(y)))


def _stroke_bbox(stroke) -> tuple[int, int, int, int]:
    xs = [p[0] for p in stroke]
    ys = [p[1] for p in stroke]
    return min(xs), min(ys), max(xs), max(ys)


def render_page(pdf_path: str, page: int, dpi: int = 600):
    """Rasterize a SINGLE page (1-based) of ``pdf_path`` to a PIL image.

    The diary PDF has dozens of pages; ``pdf_utils.load_page`` renders them all,
    so here we render only the page we need via ``first_page``/``last_page``.

    ``pdf_path`` is the SLUG anchor and may be absent (the slug PDF is git-ignored);
    when it is, render from the real ``PDF_RENDER_FALLBACK`` diary PDF so the CLI/eval
    work out of the box with no manual symlink (see ``config.PDF_PATH``).
    """
    src = pdf_path if os.path.exists(pdf_path) else PDF_RENDER_FALLBACK
    return convert_from_path(src, dpi=dpi, first_page=page, last_page=page)[0]


# Central-band geometry (Fix 1, layered on clean_word). The diary's ruled rows
# are spaced barely more than a row height apart, and clean_word keeps WHOLE ink
# components -- on dense cursive a whole line is often one connected stroke, so
# its band + component-keeping still leaks the rows above/below. Masking ink to a
# tight band around the word's row CLIPS those rows pixel-wise (it cuts the
# connected stroke at the band edge), which keep-whole-component cannot do.
#
# The band half-height is sized from the page ROW PITCH, not the box height:
# individual boxes over-grow (heights here run 28..46 / median 39, while the row
# pitch is only ~34, so boxes are TALLER than the gap between rows and a box-height
# band would always re-admit the neighbour). The band is centred on the word's own
# box centre (reliable -- the box may be too tall but is centred on its row) and
# kept below the pitch so the neighbour rows fall outside it, at the cost of
# clipping the tallest of the target's own ascenders/descenders -- the legible
# compromise on these near-touching rows.
BAND_HALF_FRAC = 0.42  # band half-height as a fraction of the row pitch


def restrict_to_word_band(
    binary: np.ndarray,
    gray: np.ndarray,
    box_2d: list[int],
    crop_box: tuple[int, int, int, int],
    page_size: tuple[int, int],
    pitch_px: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Mask ``binary``/``gray`` to a tight horizontal band around the word's row,
    centred on the box centre and sized by ``pitch_px`` (the row pitch), then
    re-tighten. Removes residual adjacent-row ink that ``clean_word`` keeps on this
    densely-ruled page. Returns ``(band_binary, band_gray, crop_box)`` with
    ``crop_box`` updated to the tightened region (page px)."""
    _w, h = page_size
    sy = h / 1000.0
    left, top = crop_box[0], crop_box[1]
    y_c = 0.5 * (box_2d[0] + box_2d[2]) * sy - top
    half = BAND_HALF_FRAC * pitch_px
    y0 = max(0, int(y_c - half))
    y1 = min(binary.shape[0], int(y_c + half))
    masked = np.zeros_like(binary)
    masked[y0:y1, :] = binary[y0:y1, :]
    if not masked.any():  # band fell off the kept ink -> keep clean_word's output
        return binary, gray, crop_box
    # drop tiny specks the band-edge cut left behind (sub-stroke fragments).
    n, labels, stats, _ = cv2.connectedComponentsWithStats((masked > 0).astype(np.uint8), 8)
    keep = np.zeros_like(masked)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= 14:
            keep[labels == lbl] = 255
    masked = keep if keep.any() else masked
    ys, xs = np.nonzero(masked)
    tx0, ty0 = int(xs.min()), int(ys.min())
    tx1, ty1 = int(xs.max()) + 1, int(ys.max()) + 1
    band_binary = masked[ty0:ty1, tx0:tx1]
    band_gray = np.full((ty1 - ty0, tx1 - tx0), 255, dtype=gray.dtype)
    band_gray[band_binary > 0] = gray[ty0:ty1, tx0:tx1][band_binary > 0]
    new_box = (left + tx0, top + ty0, left + tx1, top + ty1)
    return band_binary, band_gray, new_box


# Fix 3 (layered after the band restriction): clean_word uses a deliberately wide
# crop (pad_frac >= 0.85) so its vertical-band cut can land INSIDE the crop. On
# words near the book's centre GUTTER that wide crop also pulls in the dark binding
# band, and clean_word keeps any component overlapping its generous +-0.12*bw rect,
# so a same-line neighbour word leaks into the first/last slot. Two pixel/component
# fixes here, both keyed off facts the cut algorithm cannot recover later:
#   - BINDING BAND: the gutter band is SOLID (column ink-fill ~0.6-1.0) where
#     cursive ink is sparse (~0.2); zero contiguous near-vertical dense columns
#     that reach a crop edge. Pixel-wise (not component-wise) so it also strips a
#     band traced into the SAME connected component as the word (e.g. *translate*).
#   - NEIGHBOUR WORD: keep only ink components whose x-range overlaps the TARGET
#     box (not just clean_word's padded rect), dropping a horizontally-adjacent
#     word (e.g. the `of` before *savings*) that shares the word's row band.
BIND_DENSE_FRAC = 0.45  # smoothed column ink-fill >= this -> binding band, not ink
BIND_MIN_W_FRAC = 0.04  # min binding-run width (and smoothing window) / crop width
X_KEEP_MARGIN_FRAC = 0.04  # keep components within this * box-width of the box x-span

# Horizontal ruled-line removal (Fix 4, found at scale on the page's BOTTOM row).
# A page rule (or scan-band edge) runs edge-to-edge through the crop; clean_word's
# content-aware fit then over-grows the crop to the line's full width (the bottom-row
# words *and* / *valuing* / *you.* grew to 2.7-2.8x their detection box) and, worse,
# the line BRIDGES the target word to its same-line neighbours into ONE component, so
# the neighbour-drop below cannot separate them. We isolate the line by a wide
# horizontal morphological OPEN (only a run >= RULE_KFRAC * box-width survives -- a
# real rule, never a short cursive horizontal) and subtract it, which also severs the
# bridge so the existing neighbour-drop then removes the neighbour on its own.
# GATE: a single image row reaches >= RULE_ROWFRAC ink fill. A straight edge-to-edge
# rule fills one whole row (the 3 bottom-row crops hit 1.00); the densest non-rule on
# this page, *translate*'s wavy cursive baseline, only reaches 0.94 -- so this is a
# strict no-op on translate and every other word, gating on the rule's defining shape
# (one full straight row) rather than on crop width (which is large pre-tighten for
# any word clean_word over-grew).
RULE_ROWFRAC = 0.97  # a row this full of ink == an edge-to-edge rule (translate: 0.94)
RULE_KFRAC = 0.5  # morph-open kernel width / box-width (min horizontal run = a rule)
RULE_KMIN = 120  # ...but never shorter than this many px (a rule is long)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D bool array as ``(start, end_exclusive)``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2], strict=True))


def strip_ruled_line(
    binary: np.ndarray,
    box_2d: list[int],
    crop_box: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> np.ndarray:
    """Zero a horizontal page rule from ``binary`` when the crop is over-wide.

    Returns ``binary`` unchanged unless a single image row is >= ``RULE_ROWFRAC``
    full of ink -- the signature of a straight edge-to-edge rule -- in which case a
    wide horizontal morphological OPEN isolates the long run (the rule), which is
    zeroed (plus a 1px vertical fringe for anti-aliasing). No re-tighten here: the
    caller's ``strip_binding_and_neighbors`` re-tightens and drops the now-severed
    neighbour."""
    h, w = binary.shape
    if h == 0 or w == 0 or not binary.any():
        return binary
    if (binary > 0).sum(1).max() < RULE_ROWFRAC * w:  # no edge-to-edge rule -> leave as-is
        return binary
    wpx, _hpx = page_size
    box_w_px = (box_2d[3] - box_2d[1]) / 1000.0 * wpx
    k = max(RULE_KMIN, int(RULE_KFRAC * box_w_px))
    if w < k:
        return binary
    mask = (binary > 0).astype(np.uint8)
    line = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    if line.sum() == 0:
        return binary
    line = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    out = binary.copy()
    out[line > 0] = 0
    return out if out.any() else binary


def strip_binding_and_neighbors(
    binary: np.ndarray,
    gray: np.ndarray,
    box_2d: list[int],
    crop_box: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Strip the page-binding gutter band and same-line neighbour words, then
    re-tighten. Returns ``(binary, gray, crop_box)`` (page px crop_box). See the
    Fix 3 note above for why each is keyed off density / box-x rather than the cut."""
    H, W = binary.shape
    if H == 0 or W == 0 or not binary.any():
        return binary, gray, crop_box
    work = binary.copy()
    wpx, _hpx = page_size
    sx = wpx / 1000.0
    bx0 = box_2d[1] * sx - crop_box[0]  # target box x-span in crop px
    bx1 = box_2d[3] * sx - crop_box[0]

    # (1) binding-band strip: smooth the per-column ink-fill so the band's internal
    # density dips don't break the run, then zero each dense run that is binding --
    # i.e. it reaches a crop edge OR sits OUTSIDE the word's box x-span (the gutter
    # band lives beyond the word). The density gate (cursive ink is sparse) keeps a
    # real ascender just past the box from being stripped; a split blob whose larger
    # half reaches neither edge (*pleasure*) is still caught by the outside-box test.
    col_fill = (work > 0).sum(axis=0).astype(float) / H
    k = max(9, int(BIND_MIN_W_FRAC * W)) | 1  # odd smoothing window ~ band width
    kernel = np.ones(k) / k
    smooth = np.convolve(np.pad(col_fill, k // 2, mode="edge"), kernel, mode="valid")
    min_w = max(20, int(BIND_MIN_W_FRAC * W))
    edge_tol = max(2, int(0.01 * W))
    for s, e in _runs(smooth >= BIND_DENSE_FRAC):
        if e - s < min_w:
            continue
        center = 0.5 * (s + e)
        if s <= edge_tol or e >= W - edge_tol or center < bx0 or center > bx1:
            work[:, s:e] = 0
    if not work.any():  # band-strip ate everything -> keep pre-strip ink
        work = binary.copy()

    # (2) neighbour-word drop: keep only components x-overlapping the target box.
    m = X_KEEP_MARGIN_FRAC * max(1.0, bx1 - bx0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), 8)
    kept = np.zeros_like(work)
    for lbl in range(1, n):
        x, _y, w, _h, area = stats[lbl]
        if area >= 14 and min(x + w, bx1 + m) > max(x, bx0 - m):
            kept[labels == lbl] = 255
    if kept.any():
        work = kept
    if not work.any():
        return binary, gray, crop_box

    ys, xs = np.nonzero(work)
    tx0, ty0 = int(xs.min()), int(ys.min())
    tx1, ty1 = int(xs.max()) + 1, int(ys.max()) + 1
    out_bin = work[ty0:ty1, tx0:tx1]
    out_gray = np.full((ty1 - ty0, tx1 - tx0), 255, dtype=gray.dtype)
    out_gray[out_bin > 0] = gray[ty0:ty1, tx0:tx1][out_bin > 0]
    new_box = (crop_box[0] + tx0, crop_box[1] + ty0, crop_box[0] + tx1, crop_box[1] + ty1)
    return out_bin, out_gray, new_box


# --- Step 1c: reject speck noise --------------------------------------------
# After clean_word + band + ruled-line + binding/neighbour strips, a few crops still
# carry sub-letter SPECKS -- a clipped neighbour-row tip, a scan fleck, a ligature
# crumb -- that are not part of the word. Left in the mask they shift the slant
# estimate and the per-column cut profile (a false cut landmark), and trace them as a
# spurious letter. We drop only components below a stroke-width-aware SPECK floor,
# which sits safely BELOW a real diacritic: an i-dot / t-cross / comma here is
# ~5*stroke_width^2 (~150-280 px on these scans) while a speck is sub-stroke
# (<~60 px), so the floor at ``NOISE_SPECK_K * stroke_width^2`` keeps every real mark.
# We do NOT try to drop letter-SIZED strays (a next-word edge sliver, a row-bleed
# fragment): on this dense page a real first/last letter is the same size and at the
# same crop edge, so size/position cannot tell them apart without recognition.
# ponytail: speck-floor only; letter-sized neighbour bleed needs a recognition-guided
# cut, not a geometry rule (which here would eat real edge letters -- *into*, *refer*).
NOISE_SPECK_K = 2.0  # speck floor = this * stroke_width^2 (below a diacritic's ~5*sw^2)
NOISE_SPECK_MIN = 14  # ...but never below this absolute area (clean low-DPI inputs)


def reject_noise(binary: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Drop sub-letter speck components, then re-tighten. Returns
    ``(binary, gray, changed)``. The floor is stroke-width-aware (see module note) so
    it removes specks/artifacts while keeping every real letter and diacritic."""
    H, W = binary.shape
    if H == 0 or W == 0 or not binary.any():
        return binary, gray, False
    dt = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)[binary > 0]
    sw = 2.0 * float(np.median(dt)) if dt.size else 0.0
    floor = max(NOISE_SPECK_MIN, round(NOISE_SPECK_K * sw * sw))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    keep = np.zeros_like(binary)
    dropped = False
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] < floor:
            dropped = True
            continue
        keep[labels == lbl] = 255
    if not dropped or not keep.any():
        return binary, gray, False

    ys, xs = np.nonzero(keep)
    tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    out_bin = keep[ty0:ty1, tx0:tx1]
    out_gray = np.full((ty1 - ty0, tx1 - tx0), 255, dtype=gray.dtype)
    out_gray[out_bin > 0] = gray[ty0:ty1, tx0:tx1][out_bin > 0]
    return out_bin, out_gray, True


# --- Step 1d: BLACK-BLOCK slice detector (solid over-inked blobs) -------------
# A mis-cut emits a SOLID over-inked blob as a "letter": a merged-neighbour smudge, a
# descender-bleed clump, or a degenerate sub-letter fragment the cut carved between two
# too-close boundaries. None are letters (a real pen stroke is THIN), but they recur and
# pollute a per-letter library as a dominant fake "variant" -- the page-1..4 allograph
# library's most common 'a'/'e'/'s' forms were exactly these (a 9x8 solid blob, a 3x6
# solid block, a filled triangle). A slice is a black-block when it is (a) far too small
# to be a letter at this DPI, OR (b) a fat solid mass -- very high ink-fill with almost no
# skeleton structure, OR (c) deeply over-inked: a real pen stroke is ~2 stroke-widths
# thick, so almost no pixel sits many px from an edge; a solid blob has a large DEEP-ink
# core. Tuned on the page-1..4 slices so NO real letter trips it (the deep gate clears the
# thickest real hands -- 'h','m','n' merged ligatures -- by a margin). Letter-SIZED
# moderate-fill blobs are deliberately LEFT IN: geometry cannot separate them from a
# genuinely thick hand, so removing them would eat real letters -- reported honestly.
# ponytail: thresholds assume the pipeline's fixed 600-DPI render (stroke ~6px); scale
# BLOCK_DEEP_PX / BLOCK_SMALL_PX with DPI if that default ever moves.
BLOCK_SMALL_PX = 20  # max(h, w) below this -> a sub-letter fragment (real letters are >=~40)
BLOCK_FILL = 0.62  # ink/bbox >= this AND...
BLOCK_SKEL = 0.060  # ...skeleton-length/area <= this -> a fat solid blob (no stroke core)
BLOCK_DEEP_PX = 7  # a pixel deeper than this many px from any edge is "core" (over-)ink
BLOCK_DEEP_FRAC = 0.24  # core-ink fraction >= this -> over-inked block (real letters < 0.20)
BLOCK_FLAT_AR = 2.2  # width/height >= this AND...
BLOCK_FLAT_FILL = 0.72  # ...fill >= this -> a wide-flat solid bar (real wide m/w fill ~0.4)


def black_block_features(mask: np.ndarray) -> tuple[int, int, float, float, float] | None:
    """``(h, w, fill, skel_len/area, deep_ink_frac)`` for an ink ``mask``, or ``None`` if
    empty. ``deep_ink_frac`` is the share of ink pixels more than ``BLOCK_DEEP_PX`` from any
    edge (high for a solid blob, ~0 for a thin pen stroke)."""
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area = int(m.sum())
    fill = area / float(h * w)
    skel_area = int((vectorize.zhang_suen(m * 255) > 0).sum()) / max(1, area)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)[m > 0]
    deep = float((dt > BLOCK_DEEP_PX).mean()) if dt.size else 0.0
    return h, w, fill, skel_area, deep


def is_black_block(mask: np.ndarray) -> bool:
    """True when ``mask`` is a solid over-inked blob / degenerate fragment, not a letter
    (see the module note). Used to reject such slices before they enter a per-letter
    library; conservative so it never drops a real (even thick) letter."""
    f = black_block_features(mask)
    if f is None:
        return False
    h, w, fill, skel_area, deep = f
    return (
        max(h, w) < BLOCK_SMALL_PX
        or (fill >= BLOCK_FILL and skel_area <= BLOCK_SKEL)
        or deep >= BLOCK_DEEP_FRAC
        or (w >= BLOCK_FLAT_AR * h and fill >= BLOCK_FLAT_FILL)
    )


# --- Step 1e: drop adjacent-ROW/WORD bleed (disconnected off-row ink) ---------
# On the densely-ruled pages 2-3 the row pitch is barely a row-height, so even after the
# central-band clip the tips of the rows above/below (and same-line neighbours the box-x
# drop missed) leak in as SMALL DISCONNECTED components riding the top/bottom of the band.
# Left in, they fall inside an edge letter's x-slot and contaminate it. We drop a non-main
# component that sits OUTSIDE the word's body rows AND has no main (word) ink in its
# x-column -- i.e. it is not perched over a stem. This SPARES diacritics (an i-dot / t-cross
# sits over its own stem, so body ink shares its x -> kept) and ascenders/descenders
# (connected to the body, so they extend INTO the body band, never fully off it). The
# "body" is the dense x-height band (the p20..p80 rows of ALL ink, not one component, so a
# diacritic over a BROKEN cursive body is still spared); only ink fully off that band with
# no body column under it is dropped. Conservative: a narrow off-row fragment that happens
# to align with a body column is kept, not risked -- so it never eats a real diacritic.
NEIGHBOR_BODY_LO = 0.20  # word body rows = ALL-ink y-percentiles between these bounds
NEIGHBOR_BODY_HI = 0.80


def drop_offrow_components(
    binary: np.ndarray, gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Drop free-floating adjacent-row/word ink (disconnected components that sit off the
    word's body band with no body column under them), then re-tighten. Returns
    ``(binary, gray, changed)``. Spares diacritics and connected ascenders/descenders (see
    the module note)."""
    H, W = binary.shape
    if H == 0 or W == 0 or not binary.any():
        return binary, gray, False
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if n <= 2:  # 0 or 1 real component -> nothing to strip
        return binary, gray, False
    ink = binary > 0
    ink_rows = np.nonzero(ink)[0]
    lo = float(np.percentile(ink_rows, 100 * NEIGHBOR_BODY_LO))
    hi = float(np.percentile(ink_rows, 100 * NEIGHBOR_BODY_HI))
    ilo, ihi = int(np.floor(lo)), int(np.ceil(hi)) + 1
    body_cols = ink[ilo:ihi, :].any(axis=0)  # columns carrying dense x-height ink
    dt = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)[binary > 0]
    tol = max(2, round(float(np.median(dt)))) if dt.size else 2  # ~1 stroke radius
    keep = np.zeros_like(ink)
    dropped = False
    for lbl in range(1, n):
        x, y, w, h, a = stats[lbl]
        overlaps_body = not (y + h <= lo or y >= hi)
        over_body = bool(body_cols[max(0, x - tol) : min(W, x + w + tol)].any())
        if a < 14 or overlaps_body or over_body:  # crumb / body-height / over a stem -> keep
            keep |= labels == lbl
        else:  # free-floating off-row ink -> drop
            dropped = True
    if not dropped:
        return binary, gray, False
    out_bin = np.where(keep, binary, 0)
    if not out_bin.any():
        return binary, gray, False
    ys, xs = np.nonzero(out_bin)
    tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    out_bin = out_bin[ty0:ty1, tx0:tx1]
    out_gray = np.full((ty1 - ty0, tx1 - tx0), 255, dtype=gray.dtype)
    out_gray[out_bin > 0] = gray[ty0:ty1, tx0:tx1][out_bin > 0]
    return out_bin, out_gray, True


# --- Step 2: classify components into BODY vs DIACRITIC ---------------------


def classify_components(
    strokes: list[list[tuple[int, int]]], shape: tuple[int, int], n_slots: int
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Split traced strokes into BODY runs (kept, in reading order) and DIACRITICS.

    A stroke is a DIACRITIC when it is tiny (``< max(12, 0.04 * total_pts)``) OR
    sits entirely in the top ~30% ascender band while being narrow
    (``x-width < 0.5 * W / L``). Everything else is body ink.
    """
    h, _w = shape
    total_pts = sum(len(s) for s in strokes)
    tiny = max(12, 0.04 * total_pts)
    median_letter_width = shape[1] / max(1, n_slots)
    band_y = 0.30 * h

    body: list[list[tuple[int, int]]] = []
    diacritics: list[list[tuple[int, int]]] = []
    for stroke in strokes:
        x0, _y0, x1, y1 = _stroke_bbox(stroke)
        width = x1 - x0
        is_tiny = len(stroke) < tiny
        in_band = y1 <= band_y and width < 0.5 * median_letter_width
        if is_tiny or in_band:
            diacritics.append(stroke)
        else:
            body.append(stroke)
    # Guard: never route ALL ink to diacritics (e.g. a short word of dots).
    if not body and strokes:
        body = [max(strokes, key=len)]
        diacritics = [s for s in strokes if s is not body]
    return body, diacritics


# --- Step 3: horizontal (x) cut-score profile -------------------------------


def column_ink_counts(binary: np.ndarray) -> np.ndarray:
    """Per-column ink-pixel count."""
    return (binary > 0).sum(axis=0).astype(float)


def cut_score_profile(binary: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Per-column cut score ``score[x]`` in [0, 1] over the crop width.

    A good between-letter cut is a vertical column that is (a) THIN (few ink
    pixels -- a ligature, not a stem), (b) COMPACT (small vertical ink extent --
    not a tall stem and, crucially, not the hollow centre of an ``o``/``a`` bowl
    where the two arcs make the extent large), and (c) LOW (its ink sits near the
    baseline, where ligatures ride). An empty interior column is a real pen gap
    and scores 1. The non-monotonic trace never enters here -- the score is a pure
    function of x.
    """
    h, w = binary.shape
    ink = binary > 0
    col = ink.sum(axis=0).astype(float)

    rows = np.arange(h)[:, None].astype(float)
    ymax_col = np.where(ink, rows, -1.0).max(axis=0)
    ymin_col = np.where(ink, rows, float(h)).min(axis=0)
    vext = np.where(col > 0, ymax_col - ymin_col, 0.0)
    centroid = np.where(col > 0, np.where(ink, rows, 0.0).sum(axis=0) / np.maximum(col, 1.0), 0.0)

    ys = np.nonzero(ink.any(axis=1))[0]
    hword = float(ys.max() - ys.min()) if ys.size else float(h)
    hword = max(hword, 1.0)
    baseline = float(np.percentile(np.nonzero(ink)[0], 82)) if ink.any() else h / 2.0

    nz = col[col > 0]
    med_col = float(np.median(nz)) if nz.size else 1.0

    thin = np.clip(1.0 - col / (THIN_MEDMULT * med_col + 1e-6), 0.0, 1.0)
    compact = np.clip(1.0 - vext / (COMPACT_HFRAC * hword + 1e-6), 0.0, 1.0)
    lowband = np.clip(1.0 - np.abs(centroid - baseline) / (LOWBAND_HFRAC * hword + 1e-6), 0.0, 1.0)

    score = np.clip(W_THIN * thin + W_COMPACT * compact + W_LOWBAND * lowband, 0.0, 1.0)
    # empty interior columns are real pen gaps -> strongest cut.
    xi0, xi1 = max(0, int(np.floor(x_min))), min(w - 1, int(np.ceil(x_max)))
    gap = (col == 0) & (np.arange(w) >= xi0) & (np.arange(w) <= xi1)
    score[gap] = 1.0
    # light 3-tap smooth so a 1-px valley reads as a small high plateau.
    kernel = np.array([0.25, 0.5, 0.25])
    score = np.convolve(np.pad(score, 1, mode="edge"), kernel, mode="valid")
    return np.clip(score, 0.0, 1.0)


# --- Step 3b: stroke-topology cut score (single connected-stroke fallback) ---
# Tunables for the connected-blob gate + the envelope smoothing windows.
BLOB_AREA_FRAC = 0.9  # one component holding >= this share of ink -> blob
BLOB_SPAN_FRAC = 0.9  # ...and spanning >= this share of the word width
BLOB_XDENS_MAXGAP = 1.8  # ...and x-density leaves a cut-gap > this * one letter-width
ENV_SMOOTH = 11  # upper-envelope smoothing window (px), odd
PEAK_SPREAD = 7  # widen each envelope-valley peak by this many px, odd


def is_connected_blob(binary: np.ndarray, x_min: float, x_max: float) -> bool:
    """True when ONE connected component holds ~all the ink AND spans ~the whole
    word width -- a single ligature-joined cursive stroke (the case x-density cuts
    cannot crack). Diacritics are tiny separate components and don't trip this."""
    n, _labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if n <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    k = 1 + int(areas.argmax())
    span = max(1.0, x_max - x_min)
    return bool(
        areas.max() / areas.sum() >= BLOB_AREA_FRAC
        and stats[k, cv2.CC_STAT_WIDTH] / span >= BLOB_SPAN_FRAC
    )


def max_cut_gap(boundaries: list[float], x_min: float, x_max: float) -> float:
    """Widest span between consecutive cuts (word ends included). A blob whose
    x-density cuts leave a gap far wider than one letter has a multi-letter run
    left uncut -- x-density's failure signature, the trigger for the topology cut."""
    cuts = np.asarray([x_min, *sorted(boundaries), x_max], dtype=float)
    return float(np.max(np.diff(cuts))) if cuts.size > 1 else (x_max - x_min)


def topology_cut_score(binary: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Per-column cut score in [0, 1] for ONE ligature-joined connected stroke.

    Zhang-Suen skeleton -> UPPER ENVELOPE ``top_y[x]`` (the topmost skeleton pixel
    per column). Inside a letter the envelope is lifted into the x-height/ascender
    zone (small y); at a between-letter connector the whole stroke drops to the
    baseline (large y). So the envelope's height-VALLEYS (local maxima of ``top_y``)
    are the ligatures -- we score those columns by how LOW the envelope sits there,
    and (as in the x-density profile) an empty interior column is a real pen gap and
    scores 1. The non-monotonic trace never enters -- this is a pure function of x.
    """
    h, w = binary.shape
    ink = binary > 0
    skel = vectorize.zhang_suen(binary) > 0
    if not skel.any():  # nothing to skeletonise -> let the caller use x-density
        return cut_score_profile(binary, x_min, x_max)
    col = skel.sum(axis=0).astype(float)
    rows = np.arange(h)[:, None].astype(float)
    top_y = np.where(skel, rows, float(h)).min(axis=0)  # upper envelope

    ys = np.nonzero(ink.any(axis=1))[0]
    y_top = float(ys.min()) if ys.size else 0.0
    baseline = float(np.percentile(np.nonzero(ink)[0], 82)) if ink.any() else h / 2.0
    hword = max(baseline - y_top, 1.0)

    # interpolate the envelope across skeleton-free columns, then smooth it.
    valid = col > 0
    top_y[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), top_y[valid])
    pad = ENV_SMOOTH // 2
    env = np.convolve(np.pad(top_y, pad, mode="edge"), np.ones(ENV_SMOOTH) / ENV_SMOOTH, "valid")

    lowness = np.clip((env - y_top) / hword, 0.0, 1.0)
    valley = np.zeros(w, dtype=bool)  # local maxima of envelope height = ligatures
    valley[1:-1] = (env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:])
    score = np.where(valley, lowness, 0.0)
    spad = PEAK_SPREAD // 2
    kern = np.ones(PEAK_SPREAD) / PEAK_SPREAD
    score = np.convolve(np.pad(score, spad, mode="edge"), kern, "valid")

    xi0, xi1 = max(0, int(np.floor(x_min))), min(w - 1, int(np.ceil(x_max)))
    gap = (ink.sum(axis=0) == 0) & (np.arange(w) >= xi0) & (np.arange(w) <= xi1)
    score[gap] = 1.0
    return np.clip(score, 0.0, 1.0)


# --- Step 3c: slant-aware de-shear (slanted cuts via the existing x DP) -------
# Cursive LEANS, so a strictly vertical cut at a boundary x clips into the slanted
# body of the neighbour letter. We estimate the word's slant, shear the ink upright
# (x_shear = x - tan(theta)*y, done as an integer per-row pixel SHIFT so ink counts
# and stroke topology are preserved exactly), run the EXISTING x-position profile +
# DP in that upright space -- a vertical cut there is a SLANTED cut in the original
# -- then map the sliced points + boundaries back. Gated on confidence: a flat
# de-shear curve (a connected blob like *translate*) keeps s = 0, i.e. identical
# vertical cuts, so the topology fallback path is left exactly as it was.
SLANT_TAN_MIN = -0.85  # de-shear search window for tan(slant); cursive leans right
SLANT_TAN_MAX = 0.30  # ...allow a little back-slant
SLANT_STEP = 0.03  # search step in tan units
SLANT_MIN_PROM = 0.06  # de-shear peak prominence below this -> unreliable, use s=0


def _shear_shifts(h: int, s: float) -> np.ndarray:
    """Integer per-row x-shift array realising ``x_shear = x - s*y``, offset so every
    shifted column is >= 0. ``shift[y]`` is ADDED to a point's x (and to row ``y``'s
    pixels) to de-slant; SUBTRACT it to map back. ``s == 0`` -> all zeros (identity)."""
    shift = np.rint(-s * np.arange(h)).astype(int)
    return shift - int(shift.min())


def _shear_binary(binary: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """De-slant ``binary`` by the integer per-row ``shift`` (pure pixel translation:
    no resampling, so ink counts/connectivity are preserved exactly)."""
    h, w = binary.shape
    out = np.zeros((h, w + int(shift.max())), binary.dtype)
    for y in range(h):
        out[y, shift[y] : shift[y] + w] = binary[y]
    return out


def _shear_points(
    strokes: list[list[tuple[int, int]]], shift: np.ndarray
) -> list[list[tuple[int, int]]]:
    """Add the per-row ``shift`` to each point's x (de-slant the traced ink)."""
    n = len(shift)
    return [[(x + int(shift[int(np.clip(round(y), 0, n - 1))]), y) for x, y in s] for s in strokes]


def _unshear_groups(groups: list[list[np.ndarray]], shift: np.ndarray) -> list[list[np.ndarray]]:
    """Subtract the per-row ``shift`` from each sliced point -> original geometry."""
    n = len(shift)
    out: list[list[np.ndarray]] = []
    for g in groups:
        ng: list[np.ndarray] = []
        for sub in g:
            arr = np.asarray(sub, dtype=float).copy()
            yy = np.clip(np.rint(arr[:, 1]).astype(int), 0, n - 1)
            arr[:, 0] = arr[:, 0] - shift[yy]
            ng.append(arr)
        out.append(ng)
    return out


def estimate_slant(binary: np.ndarray) -> tuple[float, float]:
    """Estimate the word slant as ``s = tan(theta)`` via the classic projection
    deslant (run-length variant): the shear that best aligns ink into long vertical
    runs. The run-length form rewards genuine near-vertical pen strokes and avoids
    the runaway of a plain sum-of-squared-columns profile (which an extreme shear
    inflates by stacking the diagonal connector strokes).

    Returns ``(s, prominence)`` where ``prominence`` (peak height above the median
    objective, in [0, 1]) is the confidence used to GATE application. A peak at the
    search-window edge is treated as unreliable and returns ``(0.0, 0.0)``.
    """
    ys, xs = np.nonzero(binary > 0)
    if xs.size < 50:
        return 0.0, 0.0
    h = binary.shape[0]
    grid = np.arange(SLANT_TAN_MIN, SLANT_TAN_MAX + 1e-9, SLANT_STEP)
    obj = np.empty(len(grid))
    for i, s in enumerate(grid):
        shift = _shear_shifts(h, float(s))
        sx = xs + shift[ys]
        sb = np.zeros((h, int(sx.max()) + 1), bool)
        sb[ys, sx] = True
        run = sb[0].astype(np.int32)  # longest continuous vertical run per column
        best = run.copy()
        for y in range(1, h):
            run = (run + 1) * sb[y]
            best = np.maximum(best, run)
        obj[i] = float((best.astype(float) ** 2).sum())
    k = int(obj.argmax())
    prom = float((obj[k] - np.median(obj)) / (obj[k] + 1e-12))
    if k == 0 or k == len(grid) - 1:  # peak pinned at the window edge -> unreliable
        return 0.0, 0.0
    y0, y1, y2 = obj[k - 1], obj[k], obj[k + 1]  # parabolic sub-step refinement
    den = y0 - 2 * y1 + y2
    s_best = float(grid[k]) - (0.5 * (y2 - y0) / den * SLANT_STEP if den != 0 else 0.0)
    return s_best, prom


# --- Step 3d: per-boundary cut seams (local angle + optional bending) --------
# The global de-shear (3c) puts ONE slant on the whole word; on words whose own
# descenders/ascenders inflate the slant objective it over-fits (e.g. *pleasure*
# -> tan -0.72, clipping the p). So we use the global slant only to PLACE the
# boundary x-columns (the DP, which works), then cut each boundary along a LOCAL
# angle measured from the near-vertical ink beside it, and let that cut BEND through
# the lowest-ink path so it routes AROUND a descender loop instead of slicing it.
# Two fallbacks keep words the global slant already handled untouched: an unreliable
# local angle (a between-letter connector / clean gap) reuses the global slant, and a
# clean pen-gap has no ink to route around so the seam stays on its straight slant line.
SEAM_SLOPE_FRAC = 0.90  # local-slant window half-width / one letter-width (spans adjacent stems)
SEAM_BAND_FRAC = 0.42  # seam may bend this * one letter-width either side of its slant line
SEAM_LOCAL_PROM = 0.10  # min local-window deslant prominence to trust (else use the global slant)
SEAM_W_INK = 1.0  # seam cost per ink pixel crossed (the dominant term)
SEAM_W_DEV = 0.004  # seam cost per px of straying from its slant line (anchors it near bx)
SEAM_W_CURV = 0.03  # seam cost per px of row-to-row bend (keeps it a smooth path)


def _local_slope(binary: np.ndarray, ox: float, half_w: float, fallback_s: float) -> float:
    """Local cut angle at original-x ``ox``: the deslant of a window spanning the
    adjacent letter STEMS (``+-half_w`` px), via the same run-length objective as the
    global ``estimate_slant``. The window must lean past the boundary's connector to
    the stems on either side -- a between-letter strip alone is horizontal and gives no
    angle. Returns ``fallback_s`` (the global slant) when the window deslant is
    unreliable, so a word the global slant already fits is not disturbed; returns the
    LOCAL angle where it is reliable, so a word whose slant varies along it (steep
    ascenders left, upright letters right -- e.g. *pleasure*) is cut per-region rather
    than at one over-fit global angle."""
    w = binary.shape[1]
    x0, x1 = max(0, int(ox - half_w)), min(w, int(ox + half_w))
    if x1 - x0 < 4:
        return fallback_s
    s, prom = estimate_slant(binary[:, x0:x1])
    if prom < SEAM_LOCAL_PROM:
        return fallback_s
    return s if SLANT_TAN_MIN <= s <= SLANT_TAN_MAX else fallback_s


def _l1_envelope(prev: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower envelope of ``prev`` under an L1 (``c`` per step) transition: returns
    ``m[k] = min_j prev[j] + c*|k-j|`` and the argmin index ``bp[k]`` (the seam's
    smoothness step). Two linear passes."""
    n = prev.size
    m = prev.astype(float).copy()
    bp = np.arange(n)
    for i in range(1, n):
        if m[i - 1] + c < m[i]:
            m[i], bp[i] = m[i - 1] + c, bp[i - 1]
    for i in range(n - 2, -1, -1):
        if m[i + 1] + c < m[i]:
            m[i], bp[i] = m[i + 1] + c, bp[i + 1]
    return m, bp


def _carve_seam(ink: np.ndarray, pref_x: np.ndarray, band: int) -> np.ndarray:
    """Min-cost vertical seam ``x(y)`` within +-``band`` px of the slant line
    ``pref_x(y)``: cost = ink crossed + deviation from the line + row-to-row bend.
    Returns the integer cut x per row. A clean gap -> the seam rides ``pref_x``; ink in
    the way -> it bends through the nearest low-ink path (around a descender)."""
    h, w = ink.shape
    D = np.arange(-band, band + 1)
    base = np.clip(np.rint(pref_x).astype(int), 0, w - 1)
    cols = np.clip(base[:, None] + D[None, :], 0, w - 1)  # (h, nd)
    emis = SEAM_W_INK * ink[np.arange(h)[:, None], cols] + SEAM_W_DEV * np.abs(D)[None, :]
    dp = np.empty_like(emis)
    back = np.zeros(emis.shape, dtype=np.int32)
    dp[0] = emis[0]
    for y in range(1, h):
        m, bp = _l1_envelope(dp[y - 1], SEAM_W_CURV)
        dp[y], back[y] = emis[y] + m, bp
    cut = np.empty(h, dtype=int)
    k = int(dp[-1].argmin())
    for y in range(h - 1, -1, -1):
        cut[y] = int(np.clip(base[y] + D[k], 0, w - 1))
        if y > 0:
            k = int(back[y, k])
    return cut


def build_cut_seams(
    binary: np.ndarray,
    bxs: list[float],
    shift: np.ndarray,
    s_slant: float,
    x_min: float,
    x_max: float,
    n_slots: int,
) -> np.ndarray:
    """One cut seam per boundary in ORIGINAL image coords, sorted left-to-right and
    forced non-crossing. ``bxs`` are de-sheared boundary x; the global slant line for
    boundary ``b`` is ``b - shift[y]`` (de-shear inverse). Each seam follows the LOCAL
    angle when reliable (else the global slant) and bends through low ink."""
    h, _w = binary.shape
    ink = (binary > 0).astype(float)
    rows = np.arange(h)
    yr = np.nonzero(ink.any(axis=1))[0]
    y_ref = int((yr.min() + yr.max()) // 2) if yr.size else h // 2
    letter_w = max(8.0, (x_max - x_min) / max(1, n_slots))
    slope_half = SEAM_SLOPE_FRAC * letter_w
    band = max(3, round(SEAM_BAND_FRAC * letter_w))

    seams: list[np.ndarray] = []
    prev_cut: np.ndarray | None = None
    for b in sorted(bxs):
        g = b - shift[rows].astype(float)  # global-slant line (original space)
        ox = b - float(shift[y_ref])  # boundary x in the original image at the mid row
        s_loc = _local_slope(binary, ox, slope_half, s_slant)
        # reliable local angle -> re-anchor the cut line at y_ref with it; else the
        # global-slant line g (unchanged, so global-good words don't move).
        pref = g if s_loc == s_slant else ox + s_loc * (rows - y_ref).astype(float)
        cut = _carve_seam(ink, pref, band)
        if prev_cut is not None:
            cut = np.maximum(cut, prev_cut + 1)  # keep seams ordered, non-crossing
        prev_cut = cut
        seams.append(cut)
    return np.asarray(seams) if seams else np.empty((0, h), dtype=int)


def grid_candidates(
    score: np.ndarray, x_min: float, x_max: float, max_cand: int = GRID_MAX_CAND
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate cut columns = a downsampled x-grid over ``[x_min, x_max]`` with
    their scores. Wide words are stepped so the DP stays small; the step is <=4 px
    for a typical word so boundaries still land on the right valley."""
    xi0, xi1 = max(0, int(np.ceil(x_min))), min(len(score) - 1, int(np.floor(x_max)))
    span = max(1, xi1 - xi0)
    step = max(1, span // max_cand + 1)
    cand_x = np.arange(xi0, xi1 + 1, step)
    return cand_x.astype(float), score[cand_x]


# --- Step 4/5: width prior + transcription-forced DP over x -----------------


def char_width_priors(word: str) -> list[float]:
    return [WIDTH_PRIOR.get(ch, 1.0) for ch in word]


def expected_boundary_x(word: str, x_min: float, x_max: float) -> np.ndarray:
    """Expected x at each of the ``L-1`` internal boundaries, from the per-char
    width prior scaled to the word's x-span."""
    w = np.asarray(char_width_priors(word), dtype=float)
    cum = np.cumsum(w)[:-1]  # boundaries are between chars
    return x_min + (x_max - x_min) * cum / w.sum()


def dp_boundaries_x(
    cand_x: np.ndarray,
    cand_score: np.ndarray,
    x_expected: np.ndarray,
    x_min: float,
    x_max: float,
    min_gap: float = 0.0,
    lam: float = DP_LAMBDA,
    band: float = DP_BAND,
) -> list[float]:
    """Choose exactly ``L-1`` monotone boundary x-positions maximising
    ``sum_j [ score(b_j) - lam * ((b_j - x_expected_j) / span)^2 ]``.

    Banded DP over (boundary j, candidate column): each boundary may only land
    within ``+-band*span`` of its expected x, and consecutive boundaries must be at
    least ``min_gap`` px apart so they cannot pile onto the same wide pen-gap (which
    would leave a whole connected run -- e.g. ``c+o`` of *could* -- uncut). Widens
    the band if no path is found.
    """
    m = len(x_expected)  # number of internal boundaries = L-1
    if m == 0:
        return []
    order = np.argsort(cand_x)
    cand_x = np.asarray(cand_x, dtype=float)[order]
    cand_score = np.asarray(cand_score, dtype=float)[order]
    span = max(1.0, x_max - x_min)
    n = len(cand_x)
    if n < m:
        return list(np.linspace(x_min, x_max, m + 2)[1:-1])

    def local(k: int, j: int) -> float:
        resid = (cand_x[k] - x_expected[j]) / span
        return cand_score[k] - lam * resid * resid

    NEG = -1e18
    dp = np.full((m, n), NEG)
    back = np.full((m, n), -1, dtype=int)

    for k in range(n):
        if abs(cand_x[k] - x_expected[0]) <= band * span:
            dp[0, k] = local(k, 0)
    for j in range(1, m):
        for k in range(n):
            if abs(cand_x[k] - x_expected[j]) > band * span:
                continue
            base = local(k, j)
            best_prev, best_pk = NEG, -1
            for pk in range(k):
                if cand_x[pk] <= cand_x[k] - min_gap and dp[j - 1, pk] > best_prev:
                    best_prev, best_pk = dp[j - 1, pk], pk
            if best_prev > NEG / 2:
                dp[j, k] = base + best_prev
                back[j, k] = best_pk

    if dp[m - 1].max() <= NEG / 2:
        if band < 1.0:
            wider = min(1.0, band * 2)
            return dp_boundaries_x(
                cand_x, cand_score, x_expected, x_min, x_max, min_gap, lam, wider
            )
        return list(np.linspace(x_min, x_max, m + 2)[1:-1])

    k = int(np.argmax(dp[m - 1]))
    chosen: list[float] = []
    j = m - 1
    while j >= 0 and k >= 0:
        chosen.append(float(cand_x[k]))
        k = back[j, k]
        j -= 1
    chosen.reverse()
    return chosen


def _ensure_count_x(boundaries: list[float], want: int, x_min: float, x_max: float) -> list[float]:
    """Force exactly ``want`` distinct interior boundary x-positions, sorted."""
    b = sorted({round(float(v), 3) for v in boundaries if x_min < v < x_max})
    if len(b) == want:
        return b
    if len(b) > want:
        return b[:want]
    for g in np.linspace(x_min, x_max, want + 2)[1:-1]:
        if all(abs(float(g) - e) > 1.0 for e in b):
            b.append(float(g))
        if len(b) == want:
            break
    return sorted(b)[:want]


# --- Step 6: slice body by x + re-attach diacritics -------------------------


def slice_by_x(
    body: list[list[tuple[int, int]]], boundaries_x: list[float], n_slots: int
) -> list[list[np.ndarray]]:
    """Assign every body point to a slot by its x relative to the ``L-1`` vertical
    boundaries; group consecutive same-slot points of each traced stroke into a
    contiguous sub-stroke. Trace order does not matter -- only x does."""
    bnds = np.asarray(sorted(boundaries_x), dtype=float)
    groups: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
    for stroke in body:
        cur_slot: int | None = None
        cur: list[tuple[int, int]] = []
        for px, py in stroke:
            slot = min(int(np.searchsorted(bnds, px, side="right")), n_slots - 1)
            if cur_slot is None:
                cur_slot = slot
            if slot != cur_slot:
                if cur:
                    groups[cur_slot].append(np.asarray(cur, dtype=float))
                cur, cur_slot = [], slot
            cur.append((px, py))
        if cur and cur_slot is not None:
            groups[cur_slot].append(np.asarray(cur, dtype=float))
    return groups


def attach_diacritics_x(
    groups: list[list[np.ndarray]],
    boundaries_x: list[float],
    diacritics: list[list[tuple[int, int]]],
    n_slots: int,
) -> list[list[np.ndarray]]:
    """Attach each diacritic (i-dot, t-cross, ...) to the slot its x-centre falls in."""
    bnds = np.asarray(sorted(boundaries_x), dtype=float)
    for d in diacritics:
        dx0, _dy0, dx1, _dy1 = _stroke_bbox(d)
        xc = 0.5 * (dx0 + dx1)
        slot = min(int(np.searchsorted(bnds, xc, side="right")), n_slots - 1)
        groups[slot].append(np.asarray(d, dtype=float))
    return groups


def _seam_slot(seams: np.ndarray, x: float, y: int, n_slots: int) -> int:
    """Slot of point ``(x, y)`` = the number of cut seams it lies to the right of at
    row ``y`` (seams are sorted ascending per row), clipped to the last slot."""
    yy = int(np.clip(y, 0, seams.shape[1] - 1))
    return min(int(np.searchsorted(seams[:, yy], x, side="right")), n_slots - 1)


def slice_by_seams(
    body: list[list[tuple[int, int]]], seams: np.ndarray, n_slots: int
) -> list[list[np.ndarray]]:
    """Assign every body point to a slot by which side of each per-boundary cut SEAM
    it falls on (a local-angle, possibly-bent cut), grouping consecutive same-slot
    points of each traced stroke into a contiguous sub-stroke. Generalises
    ``slice_by_x`` from straight global-slant lines to per-boundary seams."""
    groups: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
    for stroke in body:
        cur_slot: int | None = None
        cur: list[tuple[int, int]] = []
        for px, py in stroke:
            slot = _seam_slot(seams, px, round(py), n_slots)
            if cur_slot is None:
                cur_slot = slot
            if slot != cur_slot:
                if cur:
                    groups[cur_slot].append(np.asarray(cur, dtype=float))
                cur, cur_slot = [], slot
            cur.append((px, py))
        if cur and cur_slot is not None:
            groups[cur_slot].append(np.asarray(cur, dtype=float))
    return groups


def attach_diacritics_seams(
    groups: list[list[np.ndarray]],
    seams: np.ndarray,
    diacritics: list[list[tuple[int, int]]],
    n_slots: int,
) -> list[list[np.ndarray]]:
    """Attach each diacritic to the slot its centre falls in, by the cut seams."""
    for d in diacritics:
        dx0, _dy0, dx1, _dy1 = _stroke_bbox(d)
        yc = round(sum(p[1] for p in d) / len(d))
        slot = _seam_slot(seams, 0.5 * (dx0 + dx1), yc, n_slots)
        groups[slot].append(np.asarray(d, dtype=float))
    return groups


# --- Step 7/8: emit per-letter files + confidence + overlay -----------------


def compute_confidence(
    score: np.ndarray, boundaries_x: list[float], word: str, x_min: float, x_max: float
) -> float:
    """conf = mean(cut score at the boundary columns) * (1 - width-prior residual)."""
    if boundaries_x:
        idx = np.clip(np.rint(boundaries_x).astype(int), 0, len(score) - 1)
        cut_quality = float(np.mean(score[idx]))
    else:
        cut_quality = 1.0
    priors = np.asarray(char_width_priors(word), dtype=float)
    target = priors / priors.sum()
    cuts = np.asarray([x_min, *sorted(boundaries_x), x_max], dtype=float)
    widths = np.clip(np.diff(cuts), 0.0, None)
    actual = widths / max(1e-9, widths.sum())
    width_resid = float(np.mean(np.abs(actual - target)))
    return float(np.clip(cut_quality * (1.0 - width_resid), 0.0, 1.0))


def emit_letter_files(
    groups: list[list[np.ndarray]],
    word: str,
    shape: tuple[int, int],
    box_id: str,
    boundary_fracs: list[float],
    conf: float,
    low_conf: bool,
    out_dir: str,
    author: str = "robot",
) -> list[str]:
    """Write ``letters/letter_<kk>_<char>.json`` for every slot, in the native
    ``{points, metadata}`` schema via ``vectorize.format_strokes`` so it
    round-trips through ``data`` unchanged."""
    h, w = shape
    letters_dir = paths.ensure_dir(os.path.join(out_dir, "letters"))
    written: list[str] = []
    for k, (ch, substrokes) in enumerate(zip(word, groups, strict=False)):
        sub_px = [[_ipt(x, y) for x, y in sub] for sub in substrokes]
        points = vectorize.format_strokes(sub_px, w, h)
        safe_ch = _safe_char(ch)
        path = os.path.join(letters_dir, f"letter_{k:02d}_{safe_ch}.json")
        payload = {
            "points": points,
            "metadata": {
                "asciiSequence": ch,
                "author": author,
                "parent_box": box_id,
                "boundary_fracs": boundary_fracs,
                "confidence": round(conf, 4),
                "low_confidence": low_conf,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        written.append(path)
    return written


def _safe_char(ch: str) -> str:
    """Filesystem-safe single-char tag (digits/letters pass; others -> name)."""
    if ch.isalnum():
        return ch
    names = {
        ".": "dot",
        ",": "comma",
        "'": "apos",
        "-": "dash",
        "/": "slash",
        ";": "semi",
        ":": "colon",
        "!": "bang",
        "?": "qmark",
        '"': "quote",
        "(": "lparen",
        ")": "rparen",
    }
    return names.get(ch, f"u{ord(ch)}")


def _fade(gray: np.ndarray, keep: float = 0.32) -> np.ndarray:
    g = (255 - (255 - gray.astype(float)) * keep).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def draw_overlay(
    gray: np.ndarray,
    groups: list[list[np.ndarray]],
    word: str,
    boundaries_x: list[float],
    lw: int = 3,
    shift: np.ndarray | None = None,
    seams: np.ndarray | None = None,
) -> np.ndarray:
    """Colour overlay: faded ink background, one colour per letter slot, a dashed
    boundary at each cut, and the slot char labelled above it. With ``seams`` given,
    the dashed boundaries follow the actual per-row cut PATHS (local-angle / bent
    seams); otherwise ``boundaries_x`` are drawn as the global-slant lines implied by
    ``shift`` (``x = bx - shift[y]``), or plain vertical lines when ``shift`` is None."""
    img = _fade(gray)
    h = img.shape[0]
    for k, substrokes in enumerate(groups):
        col = PALETTE[k % len(PALETTE)]
        leftmost = None
        for sub in substrokes:
            pts = [_ipt(x, y) for x, y in sub]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], col, lw, cv2.LINE_AA)
            if pts:
                cv2.circle(img, pts[0], 4, col, -1)
                cv2.circle(img, pts[0], 4, (255, 255, 255), 1)
                lx = min(p[0] for p in pts)
                leftmost = lx if leftmost is None else min(leftmost, lx)
        ch = word[k] if k < len(word) else "?"
        if leftmost is not None:
            cv2.putText(img, ch, (leftmost, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    if seams is not None and len(seams):
        for cut in seams:
            for y0 in range(0, h, 12):
                y1 = min(h - 1, y0 + 6)
                p0 = (int(cut[min(y0, len(cut) - 1)]), y0)
                p1 = (int(cut[min(y1, len(cut) - 1)]), y1)
                cv2.line(img, p0, p1, (40, 40, 40), 1, cv2.LINE_AA)
        return img

    n = len(shift) if shift is not None else 0

    def _orig_x(bx: float, y: int) -> int:
        sh = int(shift[min(y, n - 1)]) if n else 0
        return int(np.rint(bx - sh))

    for bx in boundaries_x:
        for y0 in range(0, h, 12):
            y1 = min(h - 1, y0 + 6)
            cv2.line(
                img, (_orig_x(bx, y0), y0), (_orig_x(bx, y1), y1), (40, 40, 40), 1, cv2.LINE_AA
            )
    return img


# --- Orchestration ----------------------------------------------------------


def segment_word(
    page_image,
    box_2d: list[int],
    word: str,
    box_id: str,
    out_dir: str | None,
    min_conf: float,
    pitch_px: float,
    overlay_path: str | None = None,
    write_files: bool = True,
) -> dict:
    """Segment one word into ``L = len(word)`` per-letter stroke groups.

    Cleans the word crop from ``page_image`` (Fix 1: no adjacent-row ink), traces
    it, and cuts by horizontal x-position (Fix 2: arc-length-free). ``pitch_px`` is
    the page text-row pitch (page px) and sizes the central band. Returns a QA dict
    and (optionally) writes per-letter files + a colour overlay.
    """
    word = word.strip()
    L = len(word)
    clean_pil, binary, crop_box = vectorize.clean_word(page_image, box_2d)  # Fix 1a
    gray = np.asarray(clean_pil)
    # Fix 1b: clip residual adjacent-row ink to a tight band around the word's row.
    binary, gray, crop_box = restrict_to_word_band(
        binary, gray, box_2d, crop_box, page_image.size, pitch_px
    )
    # Fix 4: zero a full-width horizontal page rule (over-wide crops only) BEFORE the
    # neighbour-drop, so the rule no longer bridges the word to a same-line neighbour.
    binary = strip_ruled_line(binary, box_2d, crop_box, page_image.size)
    # Fix 3: strip the centre-gutter binding band + same-line neighbour words.
    binary, gray, crop_box = strip_binding_and_neighbors(
        binary, gray, box_2d, crop_box, page_image.size
    )
    # Step 1c: reject stray noise (clipped neighbour-row tips / next-word slivers) so
    # it is never a letter, a slant landmark, or a false cut column.
    binary, gray, _ = reject_noise(binary, gray)
    # Step 1e: drop free-floating adjacent-row/word ink the band clip still let through on
    # dense pages (spares diacritics + connected ascenders/descenders).
    binary, gray, _ = drop_offrow_components(binary, gray)
    h, w = binary.shape

    strokes = vectorize.trace_ink(binary)
    if not strokes:
        if overlay_path:
            paths.ensure_parent(overlay_path)
            cv2.imwrite(overlay_path, _fade(gray))
        return _qa(word, max(1, L), 0, 0, 0.0, True, 0, error="empty trace")

    if L <= 1:  # single char -> the whole word is one letter
        groups = [[np.asarray(s, dtype=float) for s in strokes]]
        if overlay_path:
            paths.ensure_parent(overlay_path)
            cv2.imwrite(overlay_path, draw_overlay(gray, groups, word or "?", []))
        if write_files and out_dir:
            emit_letter_files(groups, word or "?", (h, w), box_id, [], 1.0, False, out_dir)
        return _qa(word, max(1, L), 0, 0, 1.0, False, len(groups))

    body, diacritics = classify_components(strokes, (h, w), L)  # Step 2

    # Step 3c: estimate the word slant and de-slant everything into upright space, so
    # the vertical-cut DP below places SLANTED cuts in the original. Gated: a flat
    # de-shear curve (e.g. the connected blob *translate*) keeps s = 0 -> identity
    # shift -> the original vertical cuts + topology fallback, untouched.
    s_slant, slant_prom = estimate_slant(binary)
    if slant_prom < SLANT_MIN_PROM:
        s_slant = 0.0
    shift = _shear_shifts(h, s_slant)
    sbin = _shear_binary(binary, shift) if s_slant else binary
    body_s = _shear_points(body, shift) if s_slant else body

    body_pts = np.concatenate([np.asarray(p, dtype=float) for p in body_s])
    x_min, x_max = float(body_pts[:, 0].min()), float(body_pts[:, 0].max())
    if x_max - x_min < 2:
        return _qa(word, L, 0, 0, 0.0, True, 0, error="degenerate x-span")

    sw = sbin.shape[1]
    col = column_ink_counts(sbin)
    x_expected = expected_boundary_x(word, x_min, x_max)  # Step 4
    min_gap = 0.40 * (x_max - x_min) / L  # keep boundaries off the same pen-gap

    def cuts_for(profile: np.ndarray) -> list[float]:
        cand_x, cand_score = grid_candidates(profile, x_min, x_max)
        b = dp_boundaries_x(cand_x, cand_score, x_expected, x_min, x_max, min_gap)  # Step 5
        return _ensure_count_x(b, L - 1, x_min, x_max)

    # Step 3: x-density profile (the normal path), now over the de-sheared image. On
    # ONE ligature-joined connected stroke x-density can pile its cuts onto a few
    # incidental pen-gaps and leave a whole multi-letter run uncut; detect that (a
    # blob whose widest cut-gap exceeds ~one letter * BLOB_XDENS_MAXGAP) and ONLY then
    # fall back to the skeleton upper-envelope topology score. So a connected word
    # x-density already cuts well (e.g. *the*) is untouched -- the fallback fires for
    # the genuine failure (e.g. *translate*, a whole cursive line with no real gaps).
    score = cut_score_profile(sbin, x_min, x_max)
    bxs = cuts_for(score)
    if (
        is_connected_blob(sbin, x_min, x_max)
        and max_cut_gap(bxs, x_min, x_max) > BLOB_XDENS_MAXGAP * (x_max - x_min) / L
    ):
        score = topology_cut_score(sbin, x_min, x_max)
        bxs = cuts_for(score)

    # Step 6: cut each boundary along its LOCAL angle, bending through low ink, in the
    # ORIGINAL image (not the single global shear) -- then slice the original points by
    # those seams. The DP's de-sheared bxs still set WHERE each boundary sits.
    seams = build_cut_seams(binary, bxs, shift, s_slant, x_min, x_max, L)
    groups = slice_by_seams(body, seams, L)
    groups = attach_diacritics_seams(groups, seams, diacritics, L)

    n_free = sum(1 for bx in bxs if col[int(np.clip(round(bx), 0, sw - 1))] == 0)
    n_internal = len(bxs) - n_free
    span = max(1.0, x_max - x_min)
    boundary_fracs = [round((bx - x_min) / span, 4) for bx in bxs]
    conf = compute_confidence(score, bxs, word, x_min, x_max)  # Step 8
    low_conf = conf < min_conf

    if overlay_path:
        paths.ensure_parent(overlay_path)
        cv2.imwrite(overlay_path, draw_overlay(gray, groups, word, bxs, shift=shift, seams=seams))
    if write_files and out_dir:
        emit_letter_files(groups, word, (h, w), box_id, boundary_fracs, conf, low_conf, out_dir)

    # Count-reconciliation gate (mirrors ocr.qa): emitted == L.
    assert len(groups) == L, f"segment count {len(groups)} != L {L} for {word!r}"
    return _qa(word, L, n_free, n_internal, conf, low_conf, len(groups))


def _qa(
    word: str,
    L: int,
    n_free: int,
    n_internal: int,
    conf: float,
    low_conf: bool,
    n_segments: int,
    error: str | None = None,
) -> dict:
    return {
        "word": word,
        "L": L,
        "free_penlift_boundaries": n_free,
        "dp_internal_cuts": n_internal,
        "confidence": round(conf, 4),
        "low_confidence": low_conf,
        "n_segments": n_segments,
        "count_gate_ok": n_segments == L,
        "error": error,
    }


# Default sample of page-4 words to segment + eye-check when no --box is given.
DEFAULT_SAMPLE = {
    1: "Realized",
    2: "that",
    4: "could",
    5: "translate",
    10: "savings",
    30: "This",
    22: "the",
    8: "wealth",
    16: "pleasure",
    24: "without",
    50: "through",
    37: "period",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXPERIMENTAL transcription-forced per-letter segmentation (x-cut)"
    )
    p.add_argument("--pdf", default=PDF_PATH, help="Source PDF (slug or path)")
    p.add_argument("--page", type=int, default=4, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--box", default=None, help="Box index (e.g. 004). Omit for a sample.")
    p.add_argument("--min-conf", type=float, default=0.4, help="Confidence gate")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for the source page")
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Where to write seg_proto5_<word>.png overlays (sample mode)",
    )
    return p.parse_args(argv)


def _read_word(text_path: str, fallback: str) -> str:
    if os.path.exists(text_path):
        with open(text_path) as f:
            return f.read().strip()
    return fallback


def _resolve_pdf(pdf: str) -> str:
    """Accept a PDF path or a bare slug; map a slug to datasets/content/<slug>.pdf."""
    if os.path.exists(pdf):
        return pdf
    cand = os.path.join("datasets", "content", f"{pdf}.pdf")
    return cand if os.path.exists(cand) else pdf


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pdf_path = _resolve_pdf(args.pdf)
    version = (
        args.version if args.version is not None else paths.latest_version(pdf_path, args.page)
    )
    page_image = render_page(pdf_path, args.page, dpi=args.dpi)
    boxes_path = paths.boxes_json(pdf_path, args.page, version)
    with open(boxes_path) as f:
        boxes = json.load(f)
    pitch_px = _row_pitch(boxes, page_image.size[1])

    def box_info(index: int) -> tuple[list[int], str, str, str]:
        entry = boxes[index]
        txt = paths.box_text(pdf_path, args.page, index, version)
        out_dir = paths.box_dir(pdf_path, args.page, index, version)
        box_id = os.path.basename(out_dir)
        word = _read_word(txt, entry.get("text", ""))
        return entry["box_2d"], out_dir, box_id, word

    results: list[dict] = []
    overlays: list[str] = []

    if args.box is not None:
        index = int(args.box)
        box_2d, out_dir, box_id, word = box_info(index)
        overlay = os.path.join(out_dir, "letters_overlay.png")
        qa = segment_word(
            page_image,
            box_2d,
            word,
            box_id,
            out_dir,
            args.min_conf,
            pitch_px,
            overlay_path=overlay,
        )
        results.append(qa)
        overlays.append(overlay)
    else:
        paths.ensure_dir(args.runs_dir)
        for index, expected in DEFAULT_SAMPLE.items():
            try:
                box_2d, out_dir, box_id, word = box_info(index)
            except (IndexError, KeyError):
                continue
            word = word or expected
            run_overlay = os.path.join(args.runs_dir, f"seg_proto5_{paths.slugify(word)}.png")
            qa = segment_word(
                page_image,
                box_2d,
                word,
                box_id,
                None,
                args.min_conf,
                pitch_px,
                overlay_path=run_overlay,
                write_files=False,
            )
            results.append(qa)
            overlays.append(run_overlay)

    _print_summary(results, overlays)


def _row_pitch(boxes: list[dict], page_h_px: int) -> float:
    """Estimate the page text-row pitch (page px) by clustering the detected boxes
    into rows by their y-centres and taking the median centre-to-centre spacing.
    Box heights over-grow on this page, but the row pitch is stable."""
    sy = page_h_px / 1000.0
    valid = [b["box_2d"] for b in boxes if isinstance(b.get("box_2d"), list)]
    if not valid:
        return 0.04 * page_h_px
    med_h = float(np.median([b[2] - b[0] for b in valid]))
    centers = sorted(0.5 * (b[0] + b[2]) for b in valid)
    rows: list[list[float]] = []
    for c in centers:
        if not rows or c - rows[-1][-1] > 0.5 * med_h:  # gap -> new row
            rows.append([c])
        else:
            rows[-1].append(c)
    row_centers = np.array([float(np.mean(r)) for r in rows])
    diffs = np.diff(row_centers)
    pitch = float(np.median(diffs)) if diffs.size else med_h
    return pitch * sy


def _print_summary(results: list[dict], overlays: list[str]) -> None:
    print(f"\n{'word':<12} {'L':>2} {'free':>4} {'cuts':>4} {'conf':>6} {'low?':>5} {'gate':>5}")
    print("-" * 46)
    n_ok = 0
    for r in results:
        gate = "OK" if r["count_gate_ok"] else "FAIL"
        n_ok += r["count_gate_ok"]
        low = "yes" if r["low_confidence"] else ""
        print(
            f"{r['word']:<12} {r['L']:>2} {r['free_penlift_boundaries']:>4} "
            f"{r['dp_internal_cuts']:>4} {r['confidence']:>6.3f} {low:>5} {gate:>5}"
            + (f"  ERR={r['error']}" if r.get("error") else "")
        )
    print("-" * 46)
    print(f"count-gate passed: {n_ok}/{len(results)}")
    if overlays:
        print("overlays:")
        for o in overlays:
            print(f"  {os.path.abspath(o)}")


if __name__ == "__main__":
    main()
