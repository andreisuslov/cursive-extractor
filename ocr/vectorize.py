"""Vectorize cropped handwriting ink into ordered (x, y, pen) stroke points.

Pipeline per crop: threshold/close -> ink CONNECTED COMPONENTS (one real pen-lift
per component: separate letters, i-dots, t-crosses) -> per component, a 1-px
Zhang-Suen skeleton traced as a single continuous depth-first path (retracing
along drawn edges at dead-ends, so no fabricated pen-ups and no spurious lines)
-> normalized points with one pen-up (state 0) marker per stroke.

This recovers the strokes that are *physically separable* from a static image
(a lift can only be recovered where the ink is actually disconnected). On the
easybank/bigbank ground-truth check it cut fabricated pen-ups from ~95/word to
1/word on connected words, held visual IoU (~0.67), and recovered the true ink
connected-component count on 37/40 diacritic words. See _order_recovery_experiment.py.

Two CLI modes:
  * batch  -- read a per-page boxes JSON, vectorize every word, write a dataset
              with {points, metadata} added to each entry:
                python -m ocr.vectorize --boxes datasets/content/bbox_data/page_2_data.json --page 2
  * single -- vectorize one cropped image file and (optionally) save an overlay:
                python -m ocr.vectorize --image datasets/content/temp_crop.jpg --save overlay.jpg
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from . import config, paths
from .pdf_utils import box_to_crop_box, crop_to_box, ensure_page_png, load_page


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Grayscale crop -> closed binary image (white ink on black)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# --- Clean skeleton + continuous path tracing -------------------------------
# The old path (morphological skeleton + greedy nearest-neighbour order_points +
# format_points' ">2px gap = pen-up") shattered one connected cursive word into
# ~95 fragments. Replaced with: ink connected-components for pen-ups, a real 1-px
# Zhang-Suen skeleton per component, and a depth-first trace that retraces over
# drawn edges at dead-ends -> one lift-free stroke per component.


def zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning -> a clean 1-px skeleton (uint8 0/1). ``binary``: ink>0.

    Runs two alternating sub-passes until no pixel can be deleted. A contour pixel
    is removed when its 8-neighbourhood meets the Zhang-Suen conditions: exactly one
    0->1 transition around the ring (A == 1), 2..6 non-zero neighbours (B), and the
    two pass-specific corner tests. The whole neighbourhood is evaluated at once with
    numpy (the 8 neighbours stacked along a leading axis), so there is no per-pixel
    Python loop.
    """
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            # P2..P9: the 8 neighbours, clockwise starting north (Zhang-Suen order).
            nb = np.stack(
                [
                    P[:-2, 1:-1],
                    P[:-2, 2:],
                    P[1:-1, 2:],
                    P[2:, 2:],
                    P[2:, 1:-1],
                    P[2:, :-2],
                    P[1:-1, :-2],
                    P[:-2, :-2],
                ]
            )
            B = nb.sum(0)  # number of non-zero neighbours
            ring = np.concatenate([nb, nb[:1]])  # P2..P9 then wrap back to P2
            A = ((ring[:-1] == 0) & (ring[1:] == 1)).sum(0)  # 0->1 transitions
            P2, P4, P6, P8 = nb[0], nb[2], nb[4], nb[6]
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            if cond.any():
                img[cond] = 0
                changed = True
    return img


_OFFS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n) if n else (0.0, 0.0)


# Every step in trace_component is to an 8-neighbour, so the direction is always one
# of the 8 offsets -- precompute their unit vectors to avoid a sqrt per step.
_UNIT = {off: _unit(*off) for off in _OFFS}


def _nbrs_deg(pts: set) -> tuple[dict, dict]:
    """Build the 8-connectivity adjacency list and degree for every skeleton pixel.

    Neighbours are listed in ``_OFFS`` order -- ``trace_component`` relies on that
    order for its tie-breaking, so it must not change.
    """
    nbrs = {(x, y): [p for dx, dy in _OFFS if (p := (x + dx, y + dy)) in pts] for (x, y) in pts}
    return nbrs, {p: len(n) for p, n in nbrs.items()}


def prune_spurs(
    skel: np.ndarray, max_spur: int = 6, iters: int = 3
) -> tuple[np.ndarray, set | None, dict | None, dict | None]:
    """Remove short endpoint branches (thinning hair off a thick-ink medial axis)
    that attach to a junction; standalone small marks are left intact.

    Returns ``(skel, pts, nbrs, deg)``. When pruning settles (no spur left to cut),
    the final ``pts``/``nbrs``/``deg`` already describe the returned skeleton and are
    handed to ``trace_component`` rather than rebuilt; for an empty or still-changing
    skeleton they are ``None`` and the tracer recomputes them.
    """
    skel = skel.copy()
    for _ in range(iters):
        ys, xs = np.nonzero(skel)
        pts = set(zip(xs.tolist(), ys.tolist(), strict=False))
        if not pts:
            return skel, None, None, None
        nbrs, deg = _nbrs_deg(pts)
        remove = set()
        for ep in [p for p in pts if deg[p] == 1]:
            path, prev, cur = [ep], None, ep
            while True:
                nxt = [n for n in nbrs[cur] if n != prev]
                if len(nxt) != 1 or deg[cur] >= 3 or len(path) > max_spur:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            if len(path) <= max_spur + 1 and deg.get(path[-1], 0) >= 3:
                remove.update(path[:-1])  # drop spur, keep the junction pixel
        if not remove:
            return skel, pts, nbrs, deg  # settled: describes the final skeleton
        for x, y in remove:
            skel[y, x] = 0
    return skel, None, None, None


def trace_component(
    skel: np.ndarray,
    pts: set | None = None,
    nbrs: dict | None = None,
    deg: dict | None = None,
) -> list[list[tuple[int, int]]]:
    """Trace ONE connected skeleton into a single continuous path via depth-first
    walk with backtracking: prefer the straightest unvisited neighbour, and when a
    branch dead-ends, retrace back along already-drawn skeleton edges (invisible
    overlap -- no spurious lines, no pen-ups) until an unvisited branch is reached.
    Returns ``[path]`` (one stroke) covering the whole component, or ``[]``.

    ``pts``/``nbrs``/``deg`` may be supplied by ``prune_spurs`` to skip rebuilding
    them; they are identical to what this function derives from ``skel`` itself."""
    if pts is None:
        ys, xs = np.nonzero(skel)
        pts = set(zip(xs.tolist(), ys.tolist(), strict=False))
        if not pts:
            return []
        nbrs, deg = _nbrs_deg(pts)
    eps = [p for p in pts if deg[p] == 1]
    start = min(eps if eps else pts, key=lambda p: (p[0], p[1]))

    visited = {start}
    path = [start]
    stack = [start]
    last = (0.0, 0.0)

    def alignment(p):  # cosine of ``last`` direction with the unit step cur->p
        ux, uy = _UNIT[(p[0] - cur[0], p[1] - cur[1])]
        return last[0] * ux + last[1] * uy

    while stack:
        cur = stack[-1]
        cand = [n for n in nbrs[cur] if n not in visited]
        if cand:
            if last == (0.0, 0.0):
                nxt = min(cand, key=lambda p: (p[0], p[1]))
            else:
                nxt = max(cand, key=alignment)  # straightest continuation
            visited.add(nxt)
            stack.append(nxt)
            path.append(nxt)
            last = _UNIT[(nxt[0] - cur[0], nxt[1] - cur[1])]
        else:
            stack.pop()
            if stack:
                path.append(stack[-1])  # retrace one edge backwards
                last = _UNIT[(stack[-1][0] - cur[0], stack[-1][1] - cur[1])]
    return [path] if len(path) >= 2 else []


def _noise_min_area(binary: np.ndarray, min_area: int) -> int:
    """Resolution-aware speck threshold: ``max(min_area, round(0.75 * stroke_width**2))``.

    A fixed ``min_area`` (tuned for the experiment's ~160 px renders, stroke width ~2-3 px)
    lets real scan specks through on high-DPI diary crops (stroke width ~6 px), where a
    sub-stroke-width fragment can be 10-30 px and become a spurious pen-lift. Stroke width
    is estimated as ``2 * median(distanceTransform[ink])`` (deterministic). The 0.75 factor
    keeps the threshold below a solid i-dot (~0.78 * stroke_width**2) so real dots survive,
    while dropping thin noise fragments. For clean low-res inputs (stroke width < ~3.6 px,
    e.g. the order-recovery experiment) the scaled value is < ``min_area``, so the base
    threshold is used unchanged.
    """
    ink = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)[binary > 0]
    if ink.size == 0:
        return min_area
    stroke_width = 2.0 * float(np.median(ink))
    return max(min_area, round(0.75 * stroke_width * stroke_width))


def trace_ink(binary: np.ndarray, min_area: int = 10) -> list[list[tuple[int, int]]]:
    """Trace a binary ink image into strokes. Pen-ups come from ink CONNECTED
    COMPONENTS (real pen-lifts: separate letters, i-dots, t-crosses), each traced
    as one continuous path. Returns strokes (lists of (x, y) pixels), reading order.

    Each component is thinned and traced inside its own tight bounding box, then the
    path is shifted back to crop coordinates. The skeleton is identical to working on
    the full crop (everything outside the box is zero anyway), but the arrays stay
    small -- a one-pixel i-dot no longer gets thinned across the whole word.

    ``min_area`` is raised to a stroke-width-aware floor (see ``_noise_min_area``) so
    high-DPI scans drop sub-stroke-width specks; for clean low-res inputs it is unchanged.
    """
    min_area = _noise_min_area(binary, min_area)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    strokes = []
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue  # speck noise (resolution-aware threshold)
        comp = (labels[y : y + h, x : x + w] == lbl).astype(np.uint8)
        skel, pts, nbrs, deg = prune_spurs(zhang_suen(comp))
        for path in trace_component(skel, pts, nbrs, deg):
            strokes.append([(px + x, py + y) for px, py in path])
    strokes.sort(key=lambda s: min(p[0] for p in s))  # left-to-right reading order
    return strokes


def format_strokes(
    strokes: list[list[tuple[int, int]]], crop_width: int, crop_height: int
) -> list[list[float]]:
    """Normalize traced strokes to [0,1] and add one pen-up marker per stroke."""
    out = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        # Divide by crop size and clip to [0,1] vectorized (bit-identical IEEE ops);
        # keep Python round() per point (numpy's round differs at half-cases).
        xy = np.array(stroke, dtype=float)
        xy[:, 0] /= crop_width
        xy[:, 1] /= crop_height
        np.clip(xy, 0.0, 1.0, out=xy)
        for nx, ny in xy.tolist():
            out.append([round(nx, 4), round(ny, 4), 1])
        out.append([out[-1][0], out[-1][1], 0])  # pen-up at stroke end
    return out


# --- Clean per-word crop: strip ruled lines / bands / neighbours -------------


def remove_ruled_lines(
    binary: np.ndarray, span_frac: float = 0.8, max_thick: int = 4
) -> np.ndarray:
    """Remove TRUE ruled lines: thin, near-full-width horizontal runs that touch
    both side edges of the crop. Word strokes (thick, undulating, not edge-to-edge
    in a padded crop) are preserved."""
    _h, w = binary.shape
    klen = max(20, int(span_frac * w))
    horiz = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats((horiz > 0).astype(np.uint8), 8)
    line_mask = np.zeros_like(binary)
    for lbl in range(1, n):
        x, _y, bw, bh, _ = stats[lbl]
        if bh <= max_thick and bw >= span_frac * w and x <= 2 and x + bw >= w - 2:
            line_mask[labels == lbl] = 255
    if not line_mask.any():
        return binary
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    cleaned = cv2.subtract(binary, cv2.bitwise_and(binary, line_mask))
    return cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,  # reclose strokes a line cut
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def keep_target_components(
    binary: np.ndarray, rect: tuple[float, float, float, float], min_area: int = 14
) -> np.ndarray:
    """Keep ink components overlapping the (already-expanded) ``rect`` (x0,y0,x1,y1)
    in crop px; drop noise and components that hug a side edge as a tall band
    (scan-edge artifact)."""
    x0, y0, x1, y1 = rect
    H_, W_ = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    out = np.zeros_like(binary)
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        # drop a tall thin component glued to the left/right edge (scan-edge band)
        if (x <= 1 or x + w >= W_ - 1) and h >= 0.8 * H_ and w <= max(6, 0.05 * W_):
            continue
        if min(x + w, x1) > max(x, x0) and min(y + h, y1) > max(y, y0):
            out[labels == lbl] = 255
    return out


def clean_word(
    page_image: Image.Image,
    box_2d: list[int],
    padding: int | None = None,
    pad_frac: float | None = None,
    tighten: bool = True,
    margin: int = 6,
) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Crop a word, strip ruled lines / scan bands / neighbours, optionally tighten
    to the kept ink. Returns ``(clean_gray_PIL, clean_binary, crop_box)``.

    ``clean_gray_PIL`` is the original ink kept only where the target word is
    (neighbours whitened) -- a clean per-word image. ``clean_binary`` is its ink
    mask, ready for ``trace_ink``. Deterministic, so vectorize and package_boxes
    agree when both call it.
    """
    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    wpx, hpx = page_image.size
    # Base on the DETECTED box with margin LARGER than the band reach below, so the
    # vertical band can actually cut INSIDE the crop (not clamp to its edge). Not
    # fit_crop_to_ink -- the ink-fit over-expands into neighbours on a dense page.
    cb = box_to_crop_box(box_2d, wpx, hpx, padding, max(pad_frac, 0.85))
    left, top, _right, _bottom = cb

    gray = np.array(page_image.crop(cb).convert("L"))
    binary = remove_ruled_lines(preprocess(gray))
    sx, sy = wpx / 1000.0, hpx / 1000.0
    rx0, ry0 = box_2d[1] * sx - left, box_2d[0] * sy - top
    rx1, ry1 = box_2d[3] * sx - left, box_2d[2] * sy - top
    bw, bh = max(1.0, rx1 - rx0), max(1.0, ry1 - ry0)

    # VERTICAL band clamp (uses box height -> robust even to a bad tall box): cuts
    # a descender merging into the next line, and removes other lines entirely.
    # Horizontal stays generous (scaled to max(bw,bh)) so a narrow/mis-shaped box
    # never truncates a wide word.
    H_, _ = binary.shape
    # Band reaches up ~0.45x for ascenders, down ~0.75x for descenders; tight
    # enough to exclude an adjacent line on densely-ruled paper.
    y0c, y1c = max(0, int(ry0 - 0.5 * bh)), min(H_, int(ry1 + 0.8 * bh))
    band = np.zeros_like(binary)
    band[y0c:y1c, :] = binary[y0c:y1c, :]
    # Tight horizontal keep (the word's own x-range + small margin) so same-line
    # neighbour words are excluded. A mis-shaped detection box may truncate -- that
    # is a detection problem, not a cropping one.
    ex = 0.12 * bw
    binary = keep_target_components(band, (rx0 - ex, y0c, rx1 + ex, y1c))

    if tighten and binary.any():
        ys, xs = np.nonzero(binary)
        tx0, ty0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
        tx1 = min(binary.shape[1], int(xs.max()) + 1 + margin)
        ty1 = min(binary.shape[0], int(ys.max()) + 1 + margin)
        binary, gray = binary[ty0:ty1, tx0:tx1], gray[ty0:ty1, tx0:tx1]
        cb = (left + tx0, top + ty0, left + tx1, top + ty1)

    # Paint the ORIGINAL grayscale over a slightly grown mask, so the kept word
    # keeps its full stroke width and soft (anti-aliased) edges. Masking with the
    # bare Otsu core (`binary`) alone drops every light edge pixel, leaving thin,
    # broken, harsh letters. The thin `binary` is still returned for tracing.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out_mask = cv2.dilate(binary, grow, iterations=2)
    clean = np.full_like(gray, 255)
    clean[out_mask > 0] = gray[out_mask > 0]
    return Image.fromarray(clean), binary, cb


def vectorize_pil_crop(pil_crop: Image.Image, contrast: float = 2.0) -> list[list[float]]:
    """Vectorize a PIL crop -> list of normalized [x, y, state] points."""
    img = ImageEnhance.Contrast(pil_crop).enhance(contrast) if contrast else pil_crop
    bgr = np.array(img)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    strokes = trace_ink(preprocess(gray))
    crop_width, crop_height = pil_crop.size
    return format_strokes(strokes, crop_width, crop_height)


def vectorize_image_file(
    input_path: str, overlay_path: str | None = None
) -> list[list[tuple[int, int]]]:
    """Vectorize a standalone image file; optionally save a stroke overlay.

    Returns the traced strokes (lists of unnormalized (x, y) pixels).
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    strokes = trace_ink(preprocess(gray))

    if overlay_path:
        overlay = img.copy()
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                cv2.line(overlay, stroke[i], stroke[i + 1], (0, 255, 0), 2)
            if stroke:
                cv2.circle(overlay, stroke[0], 2, (0, 0, 255), -1)  # stroke start
        cv2.imwrite(overlay_path, overlay)
    return strokes


def vectorize_boxes(
    pdf_path: str,
    boxes: list[dict],
    page_index: int,
    dpi: int = 600,
    padding: int | None = None,
    pad_frac: float | None = None,
    fit_ink: bool | None = None,
    clean: bool | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Add {points, metadata} to each box entry, in place. Returns ``boxes``.

    With ``clean`` (default ``config.CROP_CLEAN``) each word crop is cleaned
    (ruled lines / scan bands / neighbour words removed) before tracing. A safety
    gate compares against the uncleaned trace and falls back to it when cleaning
    *increases* the stroke count (e.g. a bad detection box), so cleaning can never
    regress a word. ``metadata["cleaned"]`` records which path was used.
    ``limit`` vectorizes only the first N entries (the rest are left untouched).
    """
    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    fit_ink = config.CROP_FIT_INK if fit_ink is None else fit_ink
    clean = config.CROP_CLEAN if clean is None else clean
    page = load_page(pdf_path, page_index, dpi=dpi)
    processed = 0

    def n_strokes(pts):
        return sum(1 for p in pts if p[2] == 0)

    for entry in boxes if limit is None else boxes[:limit]:
        if "box_2d" not in entry:
            continue
        crop, _ = crop_to_box(page, entry["box_2d"], padding, pad_frac, fit_ink)
        raw_points = vectorize_pil_crop(crop)
        used_clean = False
        if clean:
            _, binary, _ = clean_word(page, entry["box_2d"], padding, pad_frac)
            ch, cw = binary.shape
            clean_points = format_strokes(trace_ink(binary), cw, ch)
            # gate: keep cleaning only if it doesn't increase the stroke count
            if n_strokes(clean_points) <= n_strokes(raw_points):
                points, crop_width, crop_height, used_clean = clean_points, cw, ch, True
            else:
                points, crop_width, crop_height = raw_points, crop.size[0], crop.size[1]
        else:
            points, crop_width, crop_height = raw_points, crop.size[0], crop.size[1]
        entry["points"] = points
        entry["metadata"] = {
            "author": "robot",
            "asciiSequence": entry.get("text", ""),
            "pointCount": len(points),
            "strokeCount": sum(1 for p in points if p[2] == 0),
            "aspectRatio": round(crop_width / crop_height, 4),
            "cleaned": used_clean,
        }
        processed += 1
        if processed % 10 == 0:
            print(f"Vectorized {processed}/{len(boxes)} words...")
    print(f"Vectorized {processed} words.")
    return boxes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorize handwriting crops into stroke points")
    # single-image mode
    p.add_argument("--image", help="Vectorize a single cropped image file instead of a page")
    p.add_argument("--save", help="(single mode) path to save a trace overlay image")
    # batch mode
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF (batch mode)")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based (batch mode)")
    p.add_argument(
        "--boxes",
        default=None,
        help="Boxes JSON to vectorize (default: canonical boxes for --pdf/--page)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output strokes JSON (default: canonical strokes for --pdf/--page)",
    )
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Page version to read/write (default: latest existing)",
    )
    p.add_argument("--limit", type=int, default=None, help="Vectorize only the first N boxes")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for crops (batch mode)")
    p.add_argument("--padding", type=int, default=config.CROP_PADDING, help="Crop padding (px)")
    p.add_argument(
        "--pad-frac",
        type=float,
        default=config.CROP_PAD_FRAC,
        help="Extra crop padding as a fraction of box size",
    )
    p.add_argument(
        "--no-fit-ink",
        action="store_true",
        help="Disable growing the box until no ink touches its borders",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.image:
        points = vectorize_image_file(args.image, args.save)
        print(f"Path with {len(points)} points.")
        if args.save:
            print(f"Overlay saved to: {args.save}")
        return

    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    if version is None and not args.boxes:
        raise SystemExit("No processed version found; run ocr.extract_boxes first or pass --boxes.")

    ensure_page_png(args.pdf, args.page, version, root)  # leave a viewable input page
    boxes_path = args.boxes or paths.boxes_json(args.pdf, args.page, version, root)
    with open(boxes_path) as f:
        boxes = json.load(f)
    vectorize_boxes(
        args.pdf,
        boxes,
        args.page - 1,
        dpi=args.dpi,
        padding=args.padding,
        pad_frac=args.pad_frac,
        fit_ink=not args.no_fit_ink,
        limit=args.limit,
    )
    if args.limit is not None:
        boxes = boxes[: args.limit]
    out = args.output or paths.strokes_json(args.pdf, args.page, version, root)
    paths.ensure_parent(out)
    with open(out, "w") as f:
        json.dump(boxes, f, indent=4)
    print(f"Wrote dataset to: {out}")


if __name__ == "__main__":
    main()
