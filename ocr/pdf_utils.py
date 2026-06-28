"""Shared helpers: load PDF pages and map OCR boxes to pixel crops.

OCR boxes are ``box_2d = [ymin, xmin, ymax, xmax]`` normalized to a 0-1000 scale
(as returned by ``gemini_ocr``). Cropping pads and clamps the box; the same
padding must be used when overlaying box-relative points back onto a page.
"""

import os

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from . import paths


def save_page_png(image: Image.Image, out_path: str, max_px: int = 2400) -> str:
    """Save a viewable PNG of a page image to ``out_path`` (downscaled if huge)."""
    img = image.convert("RGB")
    if max(img.size) > max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px))
    paths.ensure_parent(out_path)
    img.save(out_path)
    return out_path


def ensure_page_png(
    pdf_path: str,
    page: int,
    version: int | None = None,
    root: str | None = None,
    dpi: int = 200,
    max_px: int = 2400,
) -> str:
    """Ensure the page folder holds a PNG of the input page (idempotent).

    Renders the page only if the file isn't there yet, so any stage that touches a
    page (extract/vectorize/package) leaves a viewable input-page image behind.
    """
    out = paths.page_image(pdf_path, page, version, root)
    if os.path.exists(out):
        return out
    return save_page_png(load_page(pdf_path, page - 1, dpi=dpi), out, max_px=max_px)


def load_pages(pdf_path: str, dpi: int | None = None) -> list[Image.Image]:
    """Return a list of PIL page images (optionally rendered at a given DPI)."""
    if dpi:
        return convert_from_path(pdf_path, dpi=dpi)
    return convert_from_path(pdf_path)


def load_page(pdf_path: str, page_index: int, dpi: int | None = None) -> Image.Image:
    """Return a single PIL page image, bounds-checked (0-based index)."""
    images = load_pages(pdf_path, dpi=dpi)
    if not 0 <= page_index < len(images):
        raise IndexError(f"page index {page_index} out of range (PDF has {len(images)} pages)")
    return images[page_index]


def box_to_crop_box(
    box_2d: list[int], width: int, height: int, padding: int = 10, pad_frac: float = 0.0
) -> tuple[int, int, int, int]:
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


def fit_crop_to_ink(
    page_image: Image.Image,
    box_2d: list[int],
    padding: int = 10,
    pad_frac: float = 0.0,
    search_frac: float = 0.45,
    overlap_vfrac: float = 0.18,
    overlap_hfrac: float = 0.03,
    margin_frac: float = 0.06,
) -> tuple[int, int, int, int]:
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
    return (max(0, L + xs0 - mx), max(0, T + ys0 - my), min(W, L + xs1 + mx), min(H, T + ys1 + my))


def whiten_outside_polygon(
    gray: np.ndarray, polygon: list, page_size: tuple[int, int], left: int, top: int
) -> np.ndarray:
    """Set every pixel OUTSIDE ``polygon`` to white (255), in-place-ish on ``gray``.

    ``polygon`` is a list of [x, y] vertices in 0-1000 page coordinates (as the box
    editor saves them); ``left``/``top`` are the crop's offset on the page. Used to
    isolate a hand-outlined word from a too-wide box -- ink outside the polygon
    (ruled lines, neighbours) is erased before tracing.
    """
    h, w = gray.shape[:2]
    wpx, hpx = page_size
    pts = np.array(
        [[round(x / 1000 * wpx - left), round(y / 1000 * hpx - top)] for x, y in polygon],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    gray[mask == 0] = 255
    return gray


def box_mask_polygon(
    box_2d: list[int],
    hpad_frac: float = 0.15,
    vpad_up_frac: float = 0.45,
    vpad_dn_frac: float = 0.40,
) -> list[list[float]]:
    """A rectangle polygon (0-1000 page coords) = the detection ``box_2d`` padded
    vertically for ascenders/descenders and horizontally by a small slack.

    Used to whiten neighbour ink that bleeds into a crop rectangle in dense cursive,
    isolating one word. The vertical pad is generous (tall letters / descenders);
    the horizontal pad is tight (the box is already snug on width, so we cut the
    ligature into the next word rather than capture it). ``box_2d`` is
    ``[ymin, xmin, ymax, xmax]``.
    """
    ymin, xmin, ymax, xmax = box_2d
    bh = max(1, ymax - ymin)
    hp, vu, vd = hpad_frac * bh, vpad_up_frac * bh, vpad_dn_frac * bh
    x0, x1 = xmin - hp, xmax + hp
    y0, y1 = ymin - vu, ymax + vd
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def crop_to_box(
    page_image: Image.Image,
    box_2d: list[int],
    padding: int = 10,
    pad_frac: float = 0.0,
    fit_ink: bool = False,
    polygon: list | None = None,
    mask_to_box: bool = False,
    mask_hpad: float = 0.15,
    mask_vpad_up: float = 0.45,
    mask_vpad_dn: float = 0.40,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop ``page_image`` to ``box_2d``. Returns ``(crop, (left,top,right,bottom))``.

    With ``fit_ink=True`` the crop is fitted to the word's own ink components so
    no letter strokes are clipped and neighbours are excluded (see
    ``fit_crop_to_ink``). With ``polygon`` (0-1000 page coords) the crop's ink
    outside that hand-drawn outline is whitened, isolating an irregular word. With
    ``mask_to_box=True`` (and no explicit ``polygon``) the crop's ink is whitened
    outside the padded detection box (``box_mask_polygon``) -- isolating one word
    when the crop rectangle overlaps neighbours in dense cursive.
    """
    if fit_ink:
        crop_box = fit_crop_to_ink(page_image, box_2d, padding, pad_frac)
    else:
        width, height = page_image.size
        crop_box = box_to_crop_box(box_2d, width, height, padding, pad_frac)
    if polygon is None and mask_to_box:
        polygon = box_mask_polygon(box_2d, mask_hpad, mask_vpad_up, mask_vpad_dn)
    crop = page_image.crop(crop_box)
    if polygon:
        arr = whiten_outside_polygon(
            np.array(crop.convert("L")), polygon, page_image.size, crop_box[0], crop_box[1]
        )
        crop = Image.fromarray(arr)
    return crop, crop_box
