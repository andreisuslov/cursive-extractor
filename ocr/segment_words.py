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


def segment_page(rgb: np.ndarray) -> list[dict]:
    """Return word shapes [{text:'', box_2d, polygon}, ...] in reading order, 0-1000."""
    H, W = rgb.shape[:2]
    binv = remove_rules(extract_ink(rgb))
    xh = median_xheight(binv)
    # stage 1: RLSA merges each line into a blob; stage 2: split that blob into words
    words = []
    for x, y, w, h in word_blobs(binv, xh, gap_frac=1.5):
        words.extend(split_words_in_blob(binv, x, y, w, h, xh))
    words.sort(key=lambda b: (round((b[1] + b[3] / 2) / (1.3 * xh)), b[0]))  # line, L->R
    shapes = []
    for x, y, w, h in words:
        poly = word_polygon(binv, x, y, w, h)
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        box = [_q(min(ys), H), _q(min(xs), W), _q(max(ys), H), _q(max(xs), W)]
        shapes.append(
            {"text": "", "box_2d": box, "polygon": [[_q(px, W), _q(py, H)] for px, py in poly]}
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
