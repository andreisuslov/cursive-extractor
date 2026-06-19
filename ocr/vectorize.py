"""Vectorize cropped handwriting ink into ordered (x, y, pen) stroke points.

Pipeline per crop: threshold -> morphological close -> skeletonize -> walk the
skeleton with a nearest-neighbor heuristic -> emit normalized points with pen-up
(state 0) markers between disconnected strokes.

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


def skeletonize(img):
    """Morphological skeletonization. ``img``: binary (white ink on black)."""
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = img.copy()
    done = False
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if size - cv2.countNonZero(temp_img) == size:
            done = True
    return skel


def order_points(skel):
    """Order skeleton pixels into a path via greedy nearest-neighbor walk."""
    y_idxs, x_idxs = np.nonzero(skel)
    points = list(zip(x_idxs, y_idxs))
    if not points:
        return []

    # Start top-left-most (heuristic for English), then hop to adjacent pixels.
    points.sort(key=lambda p: (p[0], p[1]))
    ordered_path = [points.pop(0)]
    current_point = ordered_path[0]
    while points:
        candidates = [
            p for p in points
            if max(abs(p[0] - current_point[0]), abs(p[1] - current_point[1])) <= 1
        ]
        next_point = candidates[0] if candidates else points[0]
        ordered_path.append(next_point)
        points.remove(next_point)
        current_point = next_point
    return ordered_path


def preprocess(gray):
    """Grayscale crop -> closed binary image (white ink on black)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def format_points(path_points, crop_width, crop_height):
    """Normalize an ordered path to [0,1] and insert pen-up markers on jumps.

    A gap of more than 2px between consecutive skeleton pixels is treated as the
    end of a stroke: the last point is duplicated with state 0 (pen up).
    """
    formatted = []
    if not path_points:
        return formatted

    current_stroke = []
    for j, p in enumerate(path_points):
        nx = round(max(0.0, min(1.0, float(p[0]) / crop_width)), 4)
        ny = round(max(0.0, min(1.0, float(p[1]) / crop_height)), 4)
        if j == 0:
            current_stroke.append([nx, ny, 1])
            continue
        prev_p = path_points[j - 1]
        if max(abs(p[0] - prev_p[0]), abs(p[1] - prev_p[1])) <= 2:
            current_stroke.append([nx, ny, 1])  # connected
        else:
            last_p = current_stroke[-1]  # end the stroke with a pen-up marker
            current_stroke.append([last_p[0], last_p[1], 0])
            formatted.extend(current_stroke)
            current_stroke = [[nx, ny, 1]]

    if current_stroke:
        last_p = current_stroke[-1]
        current_stroke.append([last_p[0], last_p[1], 0])
        formatted.extend(current_stroke)
    return formatted


def vectorize_pil_crop(pil_crop, contrast=2.0):
    """Vectorize a PIL crop -> list of normalized [x, y, state] points."""
    img = ImageEnhance.Contrast(pil_crop).enhance(contrast) if contrast else pil_crop
    bgr = np.array(img)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    path_points = order_points(skeletonize(preprocess(gray)))
    crop_width, crop_height = pil_crop.size
    return format_points(path_points, crop_width, crop_height)


def vectorize_image_file(input_path, overlay_path=None):
    """Vectorize a standalone image file; optionally save a green-trace overlay.

    Returns the ordered (unnormalized) path points.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path_points = order_points(skeletonize(preprocess(gray)))

    if overlay_path:
        overlay = img.copy()
        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i + 1]
            if max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) <= 2:
                cv2.line(overlay, p1, p2, (0, 255, 0), 2)
            else:
                cv2.circle(overlay, p2, 1, (0, 255, 0), -1)
        cv2.imwrite(overlay_path, overlay)
    return path_points


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
