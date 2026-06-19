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
                python -m ocr.vectorize --boxes data/content/bbox_data/page_2_data.json --page 2
  * single -- vectorize one cropped image file and (optionally) save an overlay:
                python -m ocr.vectorize --image data/content/temp_crop.jpg --save overlay.jpg
"""

import os
import json
import argparse

import cv2
import numpy as np
from PIL import ImageEnhance

from . import config, paths
from .pdf_utils import load_page, crop_to_box


def preprocess(gray):
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

def zhang_suen(binary):
    """Zhang-Suen thinning -> clean 1-px skeleton (uint8 0/1). ``binary``: ink>0."""
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            P2, P3, P4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            P5, P6, P7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            P8, P9 = P[1:-1, :-2], P[:-2, :-2]
            seq = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(seq)
            A = np.zeros_like(img)
            ring = seq + [P2]
            for a, b in zip(ring[:-1], ring[1:]):
                A += ((a == 0) & (b == 1)).astype(np.uint8)
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


def _unit(dx, dy):
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n) if n else (0.0, 0.0)


def _nbrs_deg(pts):
    nbrs = {(x, y): [(x + dx, y + dy) for dx, dy in _OFFS if (x + dx, y + dy) in pts]
            for (x, y) in pts}
    return nbrs, {p: len(n) for p, n in nbrs.items()}


def prune_spurs(skel, max_spur=6, iters=3):
    """Remove short endpoint branches (thinning hair off a thick-ink medial axis)
    that attach to a junction; standalone small marks are left intact."""
    skel = skel.copy()
    for _ in range(iters):
        ys, xs = np.nonzero(skel)
        pts = set(zip(xs.tolist(), ys.tolist()))
        if not pts:
            break
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
            break
        for (x, y) in remove:
            skel[y, x] = 0
    return skel


def trace_component(skel):
    """Trace ONE connected skeleton into a single continuous path via depth-first
    walk with backtracking: prefer the straightest unvisited neighbour, and when a
    branch dead-ends, retrace back along already-drawn skeleton edges (invisible
    overlap -- no spurious lines, no pen-ups) until an unvisited branch is reached.
    Returns ``[path]`` (one stroke) covering the whole component, or ``[]``."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []
    nbrs, deg = _nbrs_deg(pts)
    eps = [p for p in pts if deg[p] == 1]
    start = min(eps if eps else pts, key=lambda p: (p[0], p[1]))

    visited = {start}
    path = [start]
    stack = [start]
    last = (0.0, 0.0)
    while stack:
        cur = stack[-1]
        cand = [n for n in nbrs[cur] if n not in visited]
        if cand:
            if last == (0.0, 0.0):
                nxt = min(cand, key=lambda p: (p[0], p[1]))
            else:
                def cont(p):
                    d = _unit(p[0] - cur[0], p[1] - cur[1])
                    return last[0] * d[0] + last[1] * d[1]
                nxt = max(cand, key=cont)
            visited.add(nxt)
            stack.append(nxt)
            path.append(nxt)
            last = _unit(nxt[0] - cur[0], nxt[1] - cur[1])
        else:
            stack.pop()
            if stack:
                path.append(stack[-1])        # retrace one edge backwards
                last = _unit(stack[-1][0] - cur[0], stack[-1][1] - cur[1])
    return [path] if len(path) >= 2 else []


def trace_ink(binary, min_area=10):
    """Trace a binary ink image into strokes. Pen-ups come from ink CONNECTED
    COMPONENTS (real pen-lifts: separate letters, i-dots, t-crosses), each traced
    as one continuous path. Returns strokes (lists of (x, y) pixels), reading order.
    """
    n, labels = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)
    strokes = []
    for lbl in range(1, n):
        comp = (labels == lbl)
        if int(comp.sum()) < min_area:
            continue  # speck noise
        skel = prune_spurs(zhang_suen(comp.astype(np.uint8)))
        strokes.extend(trace_component(skel))
    strokes.sort(key=lambda s: min(p[0] for p in s))  # left-to-right reading order
    return strokes


def format_strokes(strokes, crop_width, crop_height):
    """Normalize traced strokes to [0,1] and add one pen-up marker per stroke."""
    out = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for (x, y) in stroke:
            nx = round(max(0.0, min(1.0, float(x) / crop_width)), 4)
            ny = round(max(0.0, min(1.0, float(y) / crop_height)), 4)
            out.append([nx, ny, 1])
        out.append([out[-1][0], out[-1][1], 0])  # pen-up at stroke end
    return out


def vectorize_pil_crop(pil_crop, contrast=2.0):
    """Vectorize a PIL crop -> list of normalized [x, y, state] points."""
    img = ImageEnhance.Contrast(pil_crop).enhance(contrast) if contrast else pil_crop
    bgr = np.array(img)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    strokes = trace_ink(preprocess(gray))
    crop_width, crop_height = pil_crop.size
    return format_strokes(strokes, crop_width, crop_height)


def vectorize_image_file(input_path, overlay_path=None):
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


def vectorize_boxes(pdf_path, boxes, page_index, dpi=600, padding=None,
                    pad_frac=None, fit_ink=None, limit=None):
    """Add {points, metadata} to each box entry, in place. Returns ``boxes``.

    ``limit`` vectorizes only the first N entries (the rest are left untouched).
    """
    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    fit_ink = config.CROP_FIT_INK if fit_ink is None else fit_ink
    page = load_page(pdf_path, page_index, dpi=dpi)
    processed = 0
    for entry in (boxes if limit is None else boxes[:limit]):
        if "box_2d" not in entry:
            continue
        crop, _ = crop_to_box(page, entry["box_2d"], padding, pad_frac, fit_ink)
        points = vectorize_pil_crop(crop)
        crop_width, crop_height = crop.size
        entry["points"] = points
        entry["metadata"] = {
            "author": "robot",
            "asciiSequence": entry.get("text", ""),
            "pointCount": len(points),
            "strokeCount": sum(1 for p in points if p[2] == 0),
            "aspectRatio": round(crop_width / crop_height, 4),
        }
        processed += 1
        if processed % 10 == 0:
            print(f"Vectorized {processed}/{len(boxes)} words...")
    print(f"Vectorized {processed} words.")
    return boxes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Vectorize handwriting crops into stroke points")
    # single-image mode
    p.add_argument("--image", help="Vectorize a single cropped image file instead of a page")
    p.add_argument("--save", help="(single mode) path to save a trace overlay image")
    # batch mode
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF (batch mode)")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based (batch mode)")
    p.add_argument("--boxes", default=None,
                   help="Boxes JSON to vectorize (default: canonical boxes for --pdf/--page)")
    p.add_argument("--output", default=None,
                   help="Output strokes JSON (default: canonical strokes for --pdf/--page)")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument("--version", type=int, default=None,
                   help="Page version to read/write (default: latest existing)")
    p.add_argument("--limit", type=int, default=None, help="Vectorize only the first N boxes")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for crops (batch mode)")
    p.add_argument("--padding", type=int, default=config.CROP_PADDING, help="Crop padding (px)")
    p.add_argument("--pad-frac", type=float, default=config.CROP_PAD_FRAC,
                   help="Extra crop padding as a fraction of box size")
    p.add_argument("--no-fit-ink", action="store_true",
                   help="Disable growing the box until no ink touches its borders")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.image:
        points = vectorize_image_file(args.image, args.save)
        print(f"Path with {len(points)} points.")
        if args.save:
            print(f"Overlay saved to: {args.save}")
        return

    root = args.output_root
    version = args.version if args.version is not None else paths.latest_version(args.pdf, args.page, root)
    if version is None and not args.boxes:
        raise SystemExit("No processed version found; run ocr.extract_boxes first or pass --boxes.")

    boxes_path = args.boxes or paths.boxes_json(args.pdf, args.page, version, root)
    with open(boxes_path) as f:
        boxes = json.load(f)
    vectorize_boxes(args.pdf, boxes, args.page - 1, dpi=args.dpi, padding=args.padding,
                    pad_frac=args.pad_frac, fit_ink=not args.no_fit_ink, limit=args.limit)
    if args.limit is not None:
        boxes = boxes[:args.limit]
    out = args.output or paths.strokes_json(args.pdf, args.page, version, root)
    paths.ensure_parent(out)
    with open(out, "w") as f:
        json.dump(boxes, f, indent=4)
    print(f"Wrote dataset to: {out}")


if __name__ == "__main__":
    main()
