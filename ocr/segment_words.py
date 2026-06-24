"""Deterministic word segmentation for handwriting pages (no model / no LLM).

Produces one tight, slant-following BOUNDING SHAPE per word, written as boxes.json
so the box editor + an OCR-labelling step can take over. It is a single step that
internally goes coarse -> fine:

  1. ink      -- COLOUR-AGNOSTIC extraction: ink is wherever a pixel is darker than
                 the local paper colour in ANY channel, so black/blue/green/red
                 writing are all captured (estimate paper per channel by closing,
                 take the max per-channel deficit, threshold).
  2. group    -- COARSE: horizontal run-length smoothing (dilate horizontally by ~an
                 intra-word gap, then connected components) merges each word's letters
                 into one blob. Purely horizontal, so it follows the handwriting slant
                 locally and never merges across lines (no global deskew needed).
  3. shapes   -- FINE: per blob, a polygon (~8 vertices) that follows the word's
                 top/bottom ink profile, hugging the slant + ascenders/descenders.

Printed margins/headers are segmented too (they're ink); delete that junk in the
box editor -- easier than detecting it, and trivial once it's an identified shape.

    python -m ocr.segment_words --pdf <pdf> --page N                 # -> boxes.json
    python -m ocr.segment_words --pdf <pdf> --page N --out /tmp/x.json --overlay /tmp/x.png

box_2d is the shape's bounding box ([ymin,xmin,ymax,xmax], 0-1000); polygon is the
tight outline ([[x,y],...], same 0-1000 scale). text is left empty for OCR to fill.
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import config, paths
from .pdf_utils import load_page


def extract_ink(rgb: np.ndarray, paper_k: int = 25) -> np.ndarray:
    """Colour-agnostic ink mask (uint8, 255=ink).

    Paper is estimated per channel by a morphological CLOSE (fills the thin ink),
    and ink is where a pixel falls below its local paper in ANY channel -- so the
    deficit fires for blue ink (dark in red), red ink (dark in green/blue), green,
    or plain black, regardless of colour.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (paper_k, paper_k))
    deficit = np.zeros(rgb.shape[:2], np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        paper = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, k)
        deficit = np.maximum(deficit, paper.astype(np.float32) - ch)
    d = cv2.normalize(deficit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d = cv2.GaussianBlur(d, (3, 3), 0)
    binv = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, close)
    binv[~page_mask(rgb)] = 0  # drop "ink" off the paper (desk / binding background)
    return binv


def page_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the bright cream PAGE area, excluding the dark desk/binding
    background a scan often includes (otherwise its texture reads as ink)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bright = ((hsv[:, :, 2] > 140) & (hsv[:, :, 1] < 90)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    h, w = rgb.shape[:2]
    keep = np.zeros((h, w), bool)
    for i in range(1, n):
        if st[i, 4] > 0.05 * h * w:  # the page(s): large bright regions
            keep |= lab == i
    return keep if keep.any() else np.ones((h, w), bool)


def remove_rules(binv: np.ndarray, frac: float = 0.40) -> np.ndarray:
    """Zero rows that are near-full-width ink -- printed ruled lines / header bands.
    Cursive text rows are sparse (<frac), so only the long horizontal rules go."""
    w = binv.shape[1]
    out = binv.copy()
    out[(binv > 0).sum(1) / w >= frac] = 0
    return out


def _q(v: float, n: int) -> int:
    """Pixel coord -> 0-1000 page scale, clamped."""
    return max(0, min(1000, round(v / n * 1000)))


def median_xheight(binv: np.ndarray) -> float:
    """Typical letter height -- the INK-AREA-WEIGHTED median component height, so the
    many tiny stroke fragments don't drag the estimate down to noise; real letters
    (more ink) dominate. Page-tall blobs (signatures) are excluded."""
    h = binv.shape[0]
    _, _, st, _ = cv2.connectedComponentsWithStats(binv, 8)
    hs = st[1:, 3].astype(np.float64)
    ar = st[1:, 4].astype(np.float64)
    keep = (hs >= 4) & (hs <= 0.15 * h)
    hs, ar = hs[keep], ar[keep]
    if len(hs) == 0:
        return 20.0
    order = np.argsort(hs)
    hs, cum = hs[order], np.cumsum(ar[order])
    return float(hs[np.searchsorted(cum, cum[-1] / 2)])  # area-weighted median height


def word_blobs(
    binv: np.ndarray, xh: float, gap_frac: float = 0.6
) -> list[tuple[int, int, int, int]]:
    """Group ink into word blobs by horizontal run-length smoothing: dilate
    horizontally (kernel ~ an intra-word gap) so a word's letters merge, then take
    connected components. Purely horizontal, so it follows slant locally and never
    merges across lines. Returns (x, y, w, h) blob boxes."""
    kx = max(5, round(gap_frac * xh))
    ky = max(1, round(0.22 * xh))  # tiny vertical close so an i-dot joins its stem
    dil = cv2.dilate(binv, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, _, st, _ = cv2.connectedComponentsWithStats(dil, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.4 * xh * xh or (w < 0.3 * xh and h < 0.3 * xh):
            continue  # drop specks
        out.append((int(x), int(y), int(w), int(h)))
    return out


def word_polygon(
    binv: np.ndarray, x: int, y: int, w: int, h: int, nbins: int = 4, pad: int = 2
) -> list[tuple[float, float]] | None:
    """Tight outline following the word's top/bottom ink profile in page coords, so
    it hugs the slant + ascenders/descenders. ~``2*nbins`` vertices."""
    ys, xs = np.nonzero(binv[y : y + h, x : x + w])
    if len(xs) < 3:
        return None
    edges = np.linspace(0, w, nbins + 1)
    tops, bots = [], []
    for b in range(nbins):
        a, bnd = edges[b], edges[b + 1]
        col = (xs >= a) & (xs < bnd)
        if not col.any():
            continue
        cx = x + (a + bnd) / 2.0
        tops.append((cx, y + float(ys[col].min()) - pad))
        bots.append((cx, y + float(ys[col].max()) + pad))
    if len(tops) < 2:
        return None
    return [*tops, *bots[::-1]]  # top left->right, then bottom right->left


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D boolean array as (start, end)."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def split_words_in_blob(
    binv: np.ndarray, x: int, y: int, w: int, h: int, xh: float
) -> list[tuple[int, int, int, int]]:
    """Split a line/phrase blob into words at VALLEYS of the ink column-projection --
    the thin dips between words -- so connected cursive (near-zero word gaps) still
    splits where ink density drops, not only at blank columns."""
    col = (binv[y : y + h, x : x + w] > 0).sum(0).astype(np.float64)
    k = max(3, round(0.5 * xh) | 1)  # smooth out letter-level dips so only word gaps remain
    col = np.convolve(col, np.ones(k) / k, mode="same")
    on = col > max(1.0, 0.18 * col.max())
    spans = _runs(on)
    if not spans:
        return [(x, y, w, h)]
    # merge spans separated by a NARROW gap (within-word); split only at WIDE gaps
    min_gap = max(4, round(0.45 * xh))
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(x + a, y, b - a, h) for a, b in merged if b - a >= 0.4 * xh]


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    """Box-filter smooth a 1-D projection with an odd window >= 1."""
    k = max(1, k | 1)
    return np.convolve(v.astype(np.float64), np.ones(k) / k, mode="same")


def _column_blocks(binv: np.ndarray, xh: float) -> list[tuple[int, int]]:
    """Page COLUMN blocks (x-ranges) for a two-up scan: left page, right page (or a
    single block). Vertical ink projection -> on-columns -> contiguous runs, with an
    explicit gutter cut so a faint binding valley still separates the two pages."""
    w = binv.shape[1]
    colsum = (binv > 0).sum(0).astype(np.float64)
    sm = _smooth(colsum, max(9, round(2 * xh)))
    if sm.max() <= 0:
        return [(0, w)]
    absfloor = 0.5 * xh  # ink rows per column below this is faint-edge noise
    thr = max(0.10 * sm.max(), absfloor)
    blocks = [(a, b) for a, b in _runs(sm > thr) if (b - a) >= 3 * xh]
    if not blocks:
        blocks = [(0, w)]

    def gutter_cut(x0: int, x1: int) -> int | None:
        """Center-third colsum minimum, but only if it is a PROMINENT valley versus
        the text on each side -- i.e. a real two-up binding gutter. A single page
        (text spans the full width) has no such dip, so it is never split."""
        bw = x1 - x0
        c0, c1 = x0 + bw // 3, x0 + 2 * bw // 3
        if c1 - c0 < 1:
            return None
        cut = c0 + int(np.argmin(sm[c0:c1]))
        sides = np.concatenate([sm[x0:c0], sm[c1:x1]])
        side = float(np.median(sides)) if sides.size else 0.0
        return cut if side > 0 and sm[cut] < 0.35 * side else None

    # two-up split: a full-width block with a real central gutter valley -> two pages.
    out = []
    for x0, x1 in blocks:
        cut = gutter_cut(x0, x1) if (x1 - x0) > 0.55 * w else None
        if cut is not None:
            out.extend([(x0, cut), (cut, x1)])
        else:
            out.append((x0, x1))
    out.sort(key=lambda b: b[0])
    return out


def _deepest_valley(rowsum: np.ndarray, y0: int, y1: int, xh: float) -> int | None:
    """Index of the deepest interior valley in rowsum[y0:y1] if it's a real inter-line
    gap, else None. A band must be tall enough to plausibly hold two lines, and the
    valley must dip below the band peak (measured inter-line valleys here sit at ~0.2-0.4
    of the peak; descender/x-height dips within one line stay higher). A band that is
    clearly too tall to be a single line (> 2.2*xh) is forced to split at its deepest
    interior dip even when shallow -- it is a merged line by construction."""
    if (y1 - y0) < 1.5 * xh:  # too thin to contain two text lines
        return None
    seg = _smooth(rowsum[y0:y1], max(3, round(0.3 * xh)))
    margin = max(1, round(0.5 * xh))  # never cut inside a glyph at the band edges
    if 2 * margin >= len(seg) or seg.max() <= 0:
        return None
    interior = seg[margin : len(seg) - margin]
    cut = margin + int(np.argmin(interior))
    too_tall = (y1 - y0) > 2.2 * xh  # cannot be one line -> must contain a merge
    thr = 0.85 if too_tall else 0.5
    if seg[cut] >= thr * seg.max():  # not a deep enough valley
        return None
    return y0 + cut


def _split_band(rowsum: np.ndarray, y0: int, y1: int, xh: float, depth: int = 0):
    """Split a row band into single-line sub-bands at deep interior valleys. A too-tall
    band is several merged lines; recurse on each half (each accepted cut strictly
    shrinks the band, so the no-valley guard guarantees termination -- the depth cap is
    only a paranoia backstop). Probe for a valley regardless of band height so that even
    a short two-line band (both lines short) is split when it has a clear interior gap."""
    yc = _deepest_valley(rowsum, y0, y1, xh) if depth < 40 else None
    if yc is None or yc <= y0 or yc >= y1:
        return [(y0, y1)]
    return _split_band(rowsum, y0, yc, xh, depth + 1) + _split_band(rowsum, yc, y1, xh, depth + 1)


def _line_bands(binv: np.ndarray, x0: int, x1: int, xh: float) -> list[tuple[int, int, int, int]]:
    """LINE boxes (x, y, w, h) inside one column block, top-to-bottom. Horizontal
    row-projection -> on-rows -> bands; split tall bands; tighten x to actual ink."""
    blk = binv[:, x0:x1]
    rowsum = (blk > 0).sum(1).astype(np.float64)
    sm = _smooth(rowsum, max(3, round(0.3 * xh)))
    if sm.max() <= 0:
        return []
    absfloor = 0.5 * xh  # ink pixels per row below this is noise
    on = sm > max(0.06 * sm.max(), absfloor)
    bands = [(a, b) for a, b in _runs(on) if (b - a) >= 0.4 * xh]
    sub = []
    for a, b in bands:
        sub.extend(_split_band(rowsum, a, b, xh))
    out = []
    for a, b in sub:
        if (b - a) < 0.4 * xh:
            continue
        col = (blk[a:b, :] > 0).sum(0)  # tighten x to this band's ink columns
        ink = np.nonzero(col > max(1, round(0.05 * xh)))[0]
        if len(ink) == 0:
            continue
        bx0, bx1 = x0 + int(ink[0]), x0 + int(ink[-1]) + 1
        out.append((bx0, a, bx1 - bx0, b - a))
    return out


def detect_lines(binv: np.ndarray, xh: float) -> list[tuple[int, int, int, int]]:
    """Text-line boxes (x, y, w, h) in reading order, for a two-up (or single) scan
    with horizontal baselines (no deskew). Find page COLUMN blocks by vertical ink
    projection (with an explicit binding-gutter cut), then LINE bands inside each
    block by horizontal row projection (splitting any merged-line band). Reading
    order = left page top-to-bottom, then right page."""
    blocks = _column_blocks(binv, xh)
    lines = []
    for bi, (x0, x1) in enumerate(blocks):
        for box in _line_bands(binv, x0, x1, xh):
            lines.append((bi, box))
    lines.sort(key=lambda t: (t[0], t[1][1]))  # block index, then band top y
    return [box for _, box in lines]


def split_line_n(
    binv: np.ndarray, x: int, y: int, w: int, h: int, n: int, xh: float
) -> list[tuple[int, int, int, int]]:
    """Cut a line blob into exactly ``n`` words at the ``n-1`` lowest-ink columns
    (word gaps), keeping cuts spaced apart. ``n`` comes from Claude reading the line,
    so the geometry is deterministic but the WORD COUNT is correct."""
    if n <= 1:
        return [(x, y, w, h)]
    col = (binv[y : y + h, x : x + w] > 0).sum(0).astype(np.float64)
    k = max(3, round(0.5 * xh) | 1)
    col = np.convolve(col, np.ones(k) / k, mode="same")
    margin, spacing, cuts = max(2, round(0.3 * xh)), max(3, round(0.6 * xh)), []
    for c in np.argsort(col):  # lowest ink first = best gaps
        c = int(c)
        if c < margin or c > w - margin or any(abs(c - q) < spacing for q in cuts):
            continue
        cuts.append(c)
        if len(cuts) == n - 1:
            break
    bounds = [0, *sorted(cuts), w]
    return [(x + bounds[i], y, bounds[i + 1] - bounds[i], h) for i in range(len(bounds) - 1)]


def _otsu_threshold(vals: list[float]) -> float:
    """1-D Otsu: the value that best splits ``vals`` into two clusters (maximizes
    between-class variance). Used per line to separate inter-letter from inter-word
    gap widths. Returns a threshold; widths >= it are the wider (word-gap) cluster."""
    v = sorted(vals)
    n = len(v)
    if n < 2:
        return v[0] if v else 0.0
    arr = np.array(v, dtype=np.float64)
    best_t, best_var = v[0], -1.0
    for i in range(1, n):
        if v[i] == v[i - 1]:
            continue
        t = (v[i - 1] + v[i]) / 2.0
        lo, hi = arr[arr < t], arr[arr >= t]
        var = (len(lo) * len(hi)) * (lo.mean() - hi.mean()) ** 2  # n^2 * between-class var
        if var > best_var:
            best_var, best_t = var, t
    return float(best_t)


def split_line_words(
    binv: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    xh: float,
    drop_frac: float = 0.80,
    run_frac: float = 0.70,
    empty_run_frac: float = 0.22,
    min_word_frac: float = 0.55,
    col_min_ink: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Split a line into WORDS by UPPER-CONTOUR DROP.

    Between two words the top ink contour falls toward the baseline (the pen lifts /
    dips through the gap), while within a word the contour stays high on the letter
    bodies. So a column is "low" when it is empty OR its (smoothed) top contour sits
    below ``drop_frac`` of the band height; a sustained interior run of low columns is
    a word boundary. This finds gaps that ink-DENSITY projection can't, because in
    cursive adjacent words overlap in x-extent but not in their upper profile.
    Returns tight per-word ``(x, y, w, h)`` boxes.
    """
    band = (binv[y : y + h, x : x + w] > 0).copy()
    bh, bw = band.shape
    if bw < 2 or bh < 2:
        return [(x, y, w, h)]
    band[(band.sum(1) / bw) > 0.55, :] = False  # drop residual full-width rule rows
    col_ink = band.sum(0)
    occupied = col_ink >= col_min_ink
    top = band.argmax(0).astype(np.float64)  # first ink row per column (0 if none...)
    top[~band.any(0)] = bh  # ...so mark empty columns as "low" (contour at baseline)
    top = _smooth(top, max(3, round(0.05 * xh) | 1))
    # boundary columns: the top contour drops toward baseline (no tall letter) OR the
    # column is a THIN connecting stroke (few ink px) -- catches words that connect at
    # the same height (no contour drop) but through a single thin ligature.
    occ_med = float(np.median(col_ink[occupied])) if occupied.any() else 0.0
    thin = (col_ink > 0) & (col_ink <= 0.30 * occ_med)
    low = (~occupied) | (top >= drop_frac * bh) | thin
    # candidate boundaries = interior low-contour runs with ink on both sides
    cands = []  # (center, width)
    for a, b in _runs(low):
        if a == 0 or b >= bw or not (occupied[:a].any() and occupied[b:].any()):
            continue
        cands.append(((a + b) // 2, b - a))
    floor = 0.35 * xh  # absolute min: never cut at a dip narrower than this (intra-letter)
    if not cands:
        cuts = []
    else:
        widths = [wd for _, wd in cands]
        # ADAPTIVE per-line threshold: Otsu splits this line's narrow inter-letter dips
        # from its wider inter-word gaps, so it self-calibrates to the line's own
        # spacing instead of a hand-tuned constant. run_frac is only the single-gap
        # fallback (no distribution to split).
        thr = max(floor, _otsu_threshold(widths)) if len(set(widths)) > 1 else run_frac * xh
        cuts = [c for c, wd in cands if wd >= thr]
    ink = np.where(occupied)[0]
    if ink.size == 0:
        return []
    left, right = int(ink[0]), int(ink[-1])
    bounds = [left, *[c for c in cuts if left < c < right], right + 1]
    words = []
    for i in range(len(bounds) - 1):
        idx = np.where(occupied[bounds[i] : bounds[i + 1]])[0]
        if idx.size:
            words.append([bounds[i] + int(idx[0]), bounds[i] + int(idx[-1]) + 1])
    # merge a too-thin sliver into whichever neighbour it sits closest to
    minw = min_word_frac * xh
    changed = True
    while changed and len(words) > 1:
        changed = False
        for i, (a, b) in enumerate(words):
            if b - a >= minw:
                continue
            if i == 0:
                words[1][0] = a
            elif i == len(words) - 1:
                words[-2][1] = b
            elif a - words[i - 1][1] <= words[i + 1][0] - b:
                words[i - 1][1] = b
            else:
                words[i + 1][0] = a
            words.pop(i)
            changed = True
            break
    out = []
    for wa, wb in words:
        rows = np.where(band[:, wa:wb].any(1))[0]
        if rows.size:
            out.append((x + wa, y + int(rows[0]), wb - wa, int(rows[-1]) - int(rows[0]) + 1))
    return out


def _components(binv: np.ndarray, xh: float) -> list[tuple[int, int, int, int]]:
    """Ink connected-component boxes (x, y, w, h), dropping specks and page-huge blobs."""
    n, _, st, _ = cv2.connectedComponentsWithStats((binv > 0).astype(np.uint8), 8)
    return [
        (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]))
        for i in range(1, n)
        if 0.04 * xh * xh < st[i, 4] < 60 * xh * xh
    ]


def _word_hull(
    binv: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int]] | None:
    """Convex hull (simplified) of the ink in a word box -- a tight outline that hugs
    the word like a hand-drawn shape, rather than an axis-aligned rectangle."""
    ys, xs = np.nonzero(binv[y0:y1, x0:x1] > 0)
    if len(xs) < 3:
        return None
    pts = np.column_stack([xs + x0, ys + y0]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    return [(int(px), int(py)) for px, py in hull]


def group_line_words(
    binv: np.ndarray,
    comps: list,
    lx: int,
    ly: int,
    lw: int,
    lh: int,
    xh: float,
    gap_mult: float = 0.7,
) -> list[tuple[int, int, int, int]]:
    """Group a line's ink COMPONENTS into words: sort left-to-right, merge a component
    into the current word when its gap to the previous is below this line's adaptive
    (Otsu) word-gap threshold OR it overlaps in x (an i-dot / t-bar / stacked stroke).
    Returns word boxes (x0, y0, x1, y1). Components are real ink, so boxes sit on the
    words; the adaptive per-line gap separates inter-letter from inter-word spacing."""
    cs = sorted(
        (
            c
            for c in comps
            if ly - 0.3 * xh <= c[1] + c[3] / 2 <= ly + lh + 0.3 * xh
            and lx - 2 <= c[0] + c[2] / 2 <= lx + lw + 2
        ),
        key=lambda c: c[0],
    )
    if not cs:
        return []
    gaps = [g for i in range(len(cs) - 1) if (g := cs[i + 1][0] - (cs[i][0] + cs[i][2])) > 0]
    thr = (_otsu_threshold(gaps) if len(gaps) > 1 else 0.5 * xh) * gap_mult
    thr = max(0.2 * xh, thr)
    groups = [[cs[0]]]
    for i in range(1, len(cs)):
        prev = groups[-1][-1]
        gap = cs[i][0] - (prev[0] + prev[2])
        if gap < thr or cs[i][0] < prev[0] + prev[2]:  # close, or x-overlapping (i-dot)
            groups[-1].append(cs[i])
        else:
            groups.append([cs[i]])
    out = []
    for g in groups:
        out.append(
            (
                min(c[0] for c in g),
                min(c[1] for c in g),
                max(c[0] + c[2] for c in g),
                max(c[1] + c[3] for c in g),
            )
        )
    return out


def _neck_split(
    binv: np.ndarray, x0: int, y0: int, x1: int, y1: int, xh: float, medw: float, depth: int = 0
) -> list[tuple[int, int, int, int]]:
    """Split a too-wide word box (two words joined by a thin ligature) at the thinnest
    interior ink column, recursively. ``medw`` is the page's median word width; only
    boxes wider than 1.7x it are considered, and only a genuinely thin neck (< 0.45x
    the box's median column ink) actually cuts -- so normal words are left intact."""
    w = x1 - x0
    if depth > 4 or w <= 1.7 * medw:
        return [(x0, y0, x1, y1)]
    col = _smooth((binv[y0:y1, x0:x1] > 0).sum(0).astype(np.float64), max(3, round(0.3 * xh) | 1))
    m = max(2, round(0.45 * medw))  # don't cut near the box edges (inside a glyph)
    if w - 2 * m < 1:
        return [(x0, y0, x1, y1)]
    cut = m + int(np.argmin(col[m : w - m]))
    occ = col[col > 0]
    if occ.size and col[cut] < 0.45 * float(np.median(occ)):
        cx = x0 + cut
        left = _neck_split(binv, x0, y0, cx, y1, xh, medw, depth + 1)
        right = _neck_split(binv, cx, y0, x1, y1, xh, medw, depth + 1)
        return left + right
    return [(x0, y0, x1, y1)]


def segment_page(rgb: np.ndarray) -> list[dict]:
    """Return word shapes [{text:'', box_2d, polygon}, ...] in reading order, 0-1000.

    Deterministic: detect lines, group each line's ink COMPONENTS into words by an
    adaptive per-line gap, neck-split any box that merged two words, and hug each word
    with a convex hull. Component-based, so boxes sit on the actual ink."""
    H, W = rgb.shape[:2]
    binv = remove_rules(extract_ink(rgb))
    xh = median_xheight(binv)
    comps = _components(binv, xh)
    raw = []
    for line in detect_lines(binv, xh):
        raw.extend(group_line_words(binv, comps, *line, xh))
    medw = float(np.median([b[2] - b[0] for b in raw])) if raw else xh
    shapes = []
    for b in raw:
        for x0, y0, x1, y1 in _neck_split(binv, *b, xh, medw):
            w, h = x1 - x0, y1 - y0
            if (w > 5.0 * xh and h < 0.45 * xh) or w < 0.22 * xh or h < 0.22 * xh:
                continue  # drop rule-like and sliver/speck boxes
            rows = np.where((binv[y0:y1, x0:x1] > 0).any(1))[0]  # retighten y after a split
            if rows.size:
                y0, y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
            hull = _word_hull(binv, x0, y0, x1, y1)
            if not hull:
                continue
            box = [_q(y0, H), _q(x0, W), _q(y1, H), _q(x1, W)]
            shapes.append(
                {"text": "", "box_2d": box, "polygon": [[_q(px, W), _q(py, H)] for px, py in hull]}
            )
    return shapes


def draw_line_boxes(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """Draw (x, y, w, h) line boxes (page pixels) as rectangles, for debugging."""
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w, y + h], outline=(230, 20, 20), width=3)
    return im


def draw_overlay(rgb: np.ndarray, shapes: list[dict]) -> Image.Image:
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    H, W = rgb.shape[:2]
    for sh in shapes:
        p = [(x / 1000 * W, y / 1000 * H) for x, y in sh["polygon"]]
        d.line([*p, p[0]], fill=(230, 20, 20), width=2)
    return im


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Deterministic word segmentation -> boxes.json")
    ap.add_argument("--pdf", default=config.PDF_PATH)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--output-root", default=paths.OUTPUT_ROOT)
    ap.add_argument("--out", default=None, help="output path (default: canonical boxes.json)")
    ap.add_argument("--overlay", default=None, help="also write an overlay image here")
    a = ap.parse_args(argv)
    rgb = np.array(load_page(a.pdf, a.page - 1, dpi=a.dpi).convert("RGB"))
    shapes = segment_page(rgb)
    out = a.out or paths.boxes_json(a.pdf, a.page, None, a.output_root)
    with open(paths.ensure_parent(out), "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"segmented {len(shapes)} words -> {out}")
    if a.overlay:
        draw_overlay(rgb, shapes).save(a.overlay)
        print(f"overlay -> {a.overlay}")


if __name__ == "__main__":
    main()
