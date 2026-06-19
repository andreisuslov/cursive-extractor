"""Shared helpers: load PDF pages and map OCR boxes to pixel crops.

OCR boxes are ``box_2d = [ymin, xmin, ymax, xmax]`` normalized to a 0-1000 scale
(as returned by ``gemini_ocr``). Cropping pads and clamps the box; the same
padding must be used when overlaying box-relative points back onto a page.
"""

import numpy as np
import cv2
from pdf2image import convert_from_path


def load_pages(pdf_path, dpi=None):
    """Return a list of PIL page images (optionally rendered at a given DPI)."""
    if dpi:
        return convert_from_path(pdf_path, dpi=dpi)
    return convert_from_path(pdf_path)


def load_page(pdf_path, page_index, dpi=None):
    """Return a single PIL page image, bounds-checked (0-based index)."""
    images = load_pages(pdf_path, dpi=dpi)
    if not 0 <= page_index < len(images):
        raise IndexError(
            f"page index {page_index} out of range (PDF has {len(images)} pages)"
        )
    return images[page_index]


def box_to_crop_box(box_2d, width, height, padding=10, pad_frac=0.0):
    """Convert a 0-1000 ``[ymin,xmin,ymax,xmax]`` box to padded, clamped pixels.

    Padding = ``padding`` pixels PLUS ``pad_frac`` of the box's own width/height
    (so generous margins scale with DPI and word size, keeping strokes uncut).
    Returns ``(left, top, right, bottom)``.
    """
    scale_x = width / 1000.0
    scale_y = height / 1000.0
    x_min = int(box_2d[1] * scale_x)
    y_min = int(box_2d[0] * scale_y)
    x_max = int(box_2d[3] * scale_x)
    y_max = int(box_2d[2] * scale_y)
    pad_x = padding + int(pad_frac * (x_max - x_min))
    pad_y = padding + int(pad_frac * (y_max - y_min))
    left = max(0, x_min - pad_x)
    top = max(0, y_min - pad_y)
    right = min(width, x_max + pad_x)
    bottom = min(height, y_max + pad_y)
    return left, top, right, bottom


def fit_crop_to_ink(page_image, box_2d, padding=10, pad_frac=0.0,
                    search_frac=0.45, overlap_vfrac=0.18, overlap_hfrac=0.03,
                    margin_frac=0.06):
    """Fit the crop to the word's own ink, capturing whole strokes (no clipping)
    while excluding neighbouring words/rows.

    Approach: render a generous region around the detected box, find connected
    ink components, keep those that overlap the detected box (expanded slightly
    so detached i/j dots and t-crossbars are included), and return the bounding
    box of *those whole components* plus a small margin. Because each kept
    component is included in full, no stroke is ever cut; because separate
    neighbour components don't overlap the box, they're left out.

    Falls back to the padded box if no ink is found. Returns ``(l, t, r, b)``.
    """
    W, H = page_image.size
    sx, sy = W / 1000.0, H / 1000.0
    bx0, by0 = int(box_2d[1] * sx), int(box_2d[0] * sy)
    bx1, by1 = int(box_2d[3] * sx), int(box_2d[2] * sy)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)

    # Search region: scale floors to TEXT HEIGHT (bh) so short/tiny boxes (a, is,
    # to, if) still search a real neighbourhood rather than their tiny width.
    sx_pad = max(int(search_frac * bw), int(0.6 * bh))
    sy_pad = max(int(search_frac * bh), int(0.4 * bh))
    L = max(0, bx0 - sx_pad)
    R = min(W, bx1 + sx_pad)
    T = max(0, by0 - sy_pad)
    B = min(H, by1 + sy_pad)
    crop = page_image.crop((L, T, R, B))
    gray = cv2.cvtColor(np.array(crop)[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop specks

    n, _labels, stats, _c = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return box_to_crop_box(box_2d, W, H, padding, pad_frac)

    # Detected box in crop coords, expanded a touch to catch detached dots/bars.
    ex, ey = int(overlap_hfrac * bw), int(overlap_vfrac * bh)
    mb0, mb1, mb2, mb3 = bx0 - L - ex, by0 - T - ey, bx1 - L + ex, by1 - T + ey

    kept = []
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < 4:
            continue
        if min(x + w, mb2) > max(x, mb0) and min(y + h, mb3) > max(y, mb1):
            kept.append((x, y, w, h))
    if not kept:
        return box_to_crop_box(box_2d, W, H, padding, pad_frac)

    xs0 = min(k[0] for k in kept)
    ys0 = min(k[1] for k in kept)
    xs1 = max(k[0] + k[2] for k in kept)
    ys1 = max(k[1] + k[3] for k in kept)

    # Clamp vertical extent so a long descender that reaches the next line (or a
    # touching neighbour row merged into one component) can't pull in a whole
    # adjacent row. Generous enough for real ascenders/descenders.
    box_top_c, box_bot_c = by0 - T, by1 - T
    ys0 = max(ys0, box_top_c - int(0.55 * bh))
    ys1 = min(ys1, box_bot_c + int(0.75 * bh))

    # Margin scaled to text height too, so a clear background gap surrounds the
    # word even for thin/short tokens (was, are, of...).
    mx = padding + max(int(margin_frac * bw), int(0.10 * bh))
    my = padding + int(0.16 * bh)
    return (max(0, L + xs0 - mx), max(0, T + ys0 - my),
            min(W, L + xs1 + mx), min(H, T + ys1 + my))


def crop_to_box(page_image, box_2d, padding=10, pad_frac=0.0, fit_ink=False):
    """Crop ``page_image`` to ``box_2d``. Returns ``(crop, (left,top,right,bottom))``.

    With ``fit_ink=True`` the crop is fitted to the word's own ink components so
    no letter strokes are clipped and neighbours are excluded (see
    ``fit_crop_to_ink``).
    """
    if fit_ink:
        crop_box = fit_crop_to_ink(page_image, box_2d, padding, pad_frac)
    else:
        width, height = page_image.size
        crop_box = box_to_crop_box(box_2d, width, height, padding, pad_frac)
    return page_image.crop(crop_box), crop_box
