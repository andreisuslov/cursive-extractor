"""Clip InkSight strokes to the word's detection box and score trace fidelity.

InkSight faithfully derenders EVERYTHING in its 224x224 letterboxed crop, and the
crops routinely contain contamination (neighbour rows/words, ruled lines). Rather
than fight this in image space, this module removes it in STROKE space: rebuild the
exact letterbox frame geometry the derenderer saw, map the word's ``box_2d`` into
frame coordinates, keep only strokes whose median point lies inside that rect
(expanded by generous margins so the target's ascenders/descenders survive while a
neighbour row a full pitch away is dropped), and drop ruled-line-shaped strokes
(mirrors ``datasets/build_diarybank.py``'s ``is_ruled``). Each word then gets an
AIoU-F1 quality score -- recall of ink pixels within a few px of the rendered
clipped strokes, precision of stroke pixels within a few px of the ink -- computed
ONLY inside the expanded rect, so contamination outside the word cannot depress
(or inflate) the score. The geometry (crop_to_box fit_ink+mask_to_box ->
``_scale_and_pad`` black letterbox) mirrors ``inksight_vectorize`` exactly; it was
visually verified against the derender overlays.

Output: a sibling json with the SAME schema, where ``points`` are the clipped
strokes re-normalized to the expanded rect (the word fills [0,1] per axis, like
``vectorize.format_strokes``' box-relative output) and ``metadata`` gains
``{"qa": <f1>, "clip": "box"}``.

    python3 -m ocr.ink_qa --pdf <pdf> --page N [--version V] [--strokes PATH]
        [--output PATH] [--dpi 600] [--limit K]
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from . import config, paths
from .inksight_vectorize import _scale_and_pad
from .pdf_utils import crop_to_box, load_page

SIZE = 224  # InkSight frame side
TOL = 2  # px tolerance for the AIoU recall/precision match
THICK = 2  # stroke raster thickness
# Rect expansion, per side, as a fraction of the rect's own size. Generous so the
# target word's ascenders/descenders survive; a neighbour row sits a full row-pitch
# away and its stroke medians still fall outside.
MARGIN_X = 0.10
MARGIN_Y = 0.25
# Ruled-line shape, as fractions of the frame (mirror build_diarybank.is_ruled).
RULED_MIN_W = 0.55
RULED_MAX_H = 0.08

STROKES_SUFFIX = "_strokes_inksight.json"
QA_SUFFIX = "_strokes_qa.json"


# --- pure geometry -----------------------------------------------------------


def letterbox_geometry(crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    """Mirror ``_scale_and_pad``: (resized_w, resized_h, off_x, off_y) in frame px."""
    ratio = min(SIZE / crop_w, SIZE / crop_h)
    rw = max(1, int(crop_w * ratio))
    rh = max(1, int(crop_h * ratio))
    return rw, rh, (SIZE - rw) // 2, (SIZE - rh) // 2


def frame_rect_for_box(
    box_2d: list[int],
    page_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    crop_w: int,
    crop_h: int,
) -> tuple[float, float, float, float]:
    """The detection box mapped page -> crop -> letterbox, as a normalized [0,1]
    frame rect ``(x0, y0, x1, y1)``. ``box_2d`` is [ymin, xmin, ymax, xmax] on the
    0-1000 page scale; ``crop_box`` is the (left, top, right, bottom) page-pixel
    crop returned by ``crop_to_box``."""
    pw, ph = page_size
    ymin, xmin, ymax, xmax = box_2d
    ratio = min(SIZE / crop_w, SIZE / crop_h)
    _, _, ox, oy = letterbox_geometry(crop_w, crop_h)
    left, top = crop_box[0], crop_box[1]

    def fx(x1000):
        return ((x1000 / 1000.0 * pw - left) * ratio + ox) / SIZE

    def fy(y1000):
        return ((y1000 / 1000.0 * ph - top) * ratio + oy) / SIZE

    return fx(xmin), fy(ymin), fx(xmax), fy(ymax)


def expand_rect(
    rect: tuple[float, float, float, float],
    margin_x: float = MARGIN_X,
    margin_y: float = MARGIN_Y,
) -> tuple[float, float, float, float]:
    """Grow ``rect`` by ``margin_x`` * width / ``margin_y`` * height on each side."""
    x0, y0, x1, y1 = rect
    mx = margin_x * (x1 - x0)
    my = margin_y * (y1 - y0)
    return x0 - mx, y0 - my, x1 + mx, y1 + my


# --- stroke clipping ---------------------------------------------------------


def split_strokes(points: list[list[float]]) -> list[list[list[float]]]:
    """Split into strokes, each keeping its rows untouched (rich channels intact)
    INCLUDING its trailing pen-up row. Stray pen-up rows with no ink are dropped."""
    strokes: list[list[list[float]]] = []
    cur: list[list[float]] = []
    for row in points:
        cur.append(row)
        if row[2] == 0:
            if len(cur) > 1:
                strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return strokes


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def clip_strokes(
    points: list[list[float]], rect: tuple[float, float, float, float]
) -> list[list[float]]:
    """Keep a stroke iff its median pen-down point lies inside ``rect`` (an already
    expanded, normalized frame rect) and it is not ruled-line-shaped. Rows pass
    through untouched; a kept stroke missing its pen-up terminator gets one (extra
    channels zeroed, matching ``strokes_to_points_rich``'s pen-up)."""
    x0, y0, x1, y1 = rect
    out: list[list[float]] = []
    for stroke in split_strokes(points):
        down = [r for r in stroke if r[2] == 1]
        if not down:
            continue
        xs = [r[0] for r in down]
        ys = [r[1] for r in down]
        if (max(xs) - min(xs)) > RULED_MIN_W and (max(ys) - min(ys)) < RULED_MAX_H:
            continue  # ruled line, even inside the rect
        if not (x0 <= _median(xs) <= x1 and y0 <= _median(ys) <= y1):
            continue  # neighbour row/word
        out.extend(list(r) for r in stroke)
        if stroke[-1][2] != 0:
            last = stroke[-1]
            out.append([last[0], last[1], 0] + [0.0] * (len(last) - 3))
    return out


def renormalize_points(
    points: list[list[float]], rect: tuple[float, float, float, float]
) -> list[list[float]]:
    """Map frame-normalized points so ``rect`` spans [0,1] per axis (independent
    x/y scaling, like ``vectorize.format_strokes``' divide-by-crop-size). Extra
    channels pass through untouched."""
    x0, y0, x1, y1 = rect
    w = max(x1 - x0, 1e-9)
    h = max(y1 - y0, 1e-9)
    out = []
    for r in points:
        nx = round(min(max((r[0] - x0) / w, 0.0), 1.0), 4)
        ny = round(min(max((r[1] - y0) / h, 0.0), 1.0), 4)
        out.append([nx, ny] + list(r[2:]))
    return out


# --- scoring -----------------------------------------------------------------


def binarize_ink(img224: Image.Image, rw: int, rh: int, ox: int, oy: int) -> np.ndarray:
    """Otsu on inverted grayscale, restricted to the letterbox content region
    (the black letterbox bars must not skew the threshold)."""
    gray = np.array(img224.convert("L"))
    content = gray[oy : oy + rh, ox : ox + rw]
    blur = cv2.GaussianBlur(content, (3, 3), 0)
    _, ink_c = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.zeros((SIZE, SIZE), np.uint8)
    ink[oy : oy + rh, ox : ox + rw] = ink_c
    return ink


def rasterize_strokes(points: list[list[float]]) -> np.ndarray:
    """Draw frame-normalized strokes ([x,y,pen,...] rows, pen=0 ends a stroke)."""
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    poly: list[tuple[int, int]] = []
    for row in points:
        poly.append((int(round(row[0] * SIZE)), int(round(row[1] * SIZE))))
        if row[2] == 0:
            if len(poly) >= 2:
                cv2.polylines(canvas, [np.array(poly, np.int32)], False, 255, THICK)
            elif len(poly) == 1:
                cv2.circle(canvas, poly[0], max(1, THICK // 2), 255, -1)
            poly = []
    if len(poly) >= 2:
        cv2.polylines(canvas, [np.array(poly, np.int32)], False, 255, THICK)
    return canvas


def _rect_to_px(rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (
        max(0, int(np.floor(x0 * SIZE))),
        max(0, int(np.floor(y0 * SIZE))),
        min(SIZE, int(np.ceil(x1 * SIZE))),
        min(SIZE, int(np.ceil(y1 * SIZE))),
    )


def aiou_f1(
    ink: np.ndarray,
    strokes: np.ndarray,
    rect: tuple[float, float, float, float] | None = None,
    tol: int = TOL,
) -> float:
    """F1 of (recall = ink pixels within ``tol`` px of a stroke pixel, precision =
    stroke pixels within ``tol`` px of an ink pixel), restricted to ``rect`` (a
    normalized frame rect) when given."""
    if rect is not None:
        px0, py0, px1, py1 = _rect_to_px(rect)
        mask = np.zeros_like(ink)
        mask[py0:py1, px0:px1] = 1
        ink = np.where(mask > 0, ink, 0)
        strokes = np.where(mask > 0, strokes, 0)
    ink_n = int((ink > 0).sum())
    st_n = int((strokes > 0).sum())
    if ink_n == 0 or st_n == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    recall = float(((ink > 0) & (cv2.dilate(strokes, kernel) > 0)).sum()) / ink_n
    precision = float(((strokes > 0) & (cv2.dilate(ink, kernel) > 0)).sum()) / st_n
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


# --- per-word pipeline -------------------------------------------------------


def qa_entry(page: Image.Image, entry: dict) -> dict:
    """Clip + score one word entry in place: ``points`` become the clipped strokes
    re-normalized to the expanded detection rect; ``metadata`` gains qa + clip."""
    points = entry.get("points") or []
    crop, crop_box = crop_to_box(
        page,
        entry["box_2d"],
        padding=config.CROP_PADDING,
        pad_frac=config.CROP_PAD_FRAC,
        fit_ink=True,
        mask_to_box=True,
        mask_hpad=config.CROP_MASK_HPAD,
        mask_vpad_up=config.CROP_MASK_VPAD_UP,
        mask_vpad_dn=config.CROP_MASK_VPAD_DN,
    )
    rect = expand_rect(
        frame_rect_for_box(entry["box_2d"], page.size, crop_box, crop.width, crop.height)
    )
    clipped = clip_strokes(points, rect)
    if clipped:
        img = _scale_and_pad(crop)
        rw, rh, ox, oy = letterbox_geometry(crop.width, crop.height)
        ink = binarize_ink(img, rw, rh, ox, oy)
        qa = round(aiou_f1(ink, rasterize_strokes(clipped), rect), 4)
    else:
        qa = 0.0
    entry["points"] = renormalize_points(clipped, rect)
    meta = entry.setdefault("metadata", {})
    meta["qa"] = qa
    meta["clip"] = "box"
    meta["pointCount"] = len(entry["points"])
    meta["strokeCount"] = sum(1 for p in entry["points"] if p[2] == 0)
    x0, y0, x1, y1 = rect
    meta["aspectRatio"] = round((x1 - x0) / max(y1 - y0, 1e-9), 4)
    return entry


# --- CLI ---------------------------------------------------------------------


def default_strokes_path(
    pdf_path: str, page: int, version: int | None, root: str | None = None
) -> str:
    """The page's ``<prefix>_strokes_inksight.json`` (what inksight runs produce)."""
    return os.path.join(
        paths.page_dir(pdf_path, page, version, root),
        f"{paths.prefix(pdf_path, page, version)}{STROKES_SUFFIX}",
    )


def default_output_path(strokes_path: str) -> str:
    if strokes_path.endswith(STROKES_SUFFIX):
        return strokes_path[: -len(STROKES_SUFFIX)] + QA_SUFFIX
    stem, _ = os.path.splitext(strokes_path)
    return stem + "_qa.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip InkSight strokes to the box + AIoU-F1 QA")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=1, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--strokes", default=None, help="InkSight strokes json (default: canonical)")
    p.add_argument("--output", default=None, help="Output json (default: sibling _strokes_qa.json)")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI (must match the derender run)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N words")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    strokes_path = args.strokes
    if strokes_path is None:
        version = (
            args.version
            if args.version is not None
            else paths.latest_version(args.pdf, args.page, args.output_root)
        )
        if version is None:
            raise SystemExit("No processed version found; pass --strokes or run the pipeline.")
        strokes_path = default_strokes_path(args.pdf, args.page, version, args.output_root)
    with open(strokes_path) as f:
        entries = json.load(f)
    if args.limit is not None:
        entries = entries[: args.limit]
    page = load_page(args.pdf, args.page - 1, dpi=args.dpi)
    for i, entry in enumerate(entries):
        if "box_2d" not in entry:
            continue
        qa_entry(page, entry)
        print(f"QA {i + 1}/{len(entries)}: {entry.get('text', '')!r} qa={entry['metadata']['qa']}")
    out_path = args.output or default_output_path(strokes_path)
    paths.ensure_parent(out_path)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=4)
    scores = [e.get("metadata", {}).get("qa", 0.0) for e in entries if "box_2d" in e]
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    passed = sum(1 for s in scores if s >= 0.85) / n if n else 0.0
    print(f"Wrote {out_path} -- n={n} mean_qa={mean:.4f} pass@0.85={passed:.3f}")


if __name__ == "__main__":
    main()
