"""Overlay recovered strokes ON the original ink for visual inspection.

Background = the original ink, faded to light grey. Foreground = the traced
strokes, ONE COLOUR PER STROKE (so component segmentation is visible), with a
filled dot at each stroke start. Lets you see (a) does the path sit on the ink
centerline, (b) how it segments, (c) where strokes begin.

Run: python -m ocr._overlay_inspect   -> writes /tmp/overlay_*.png
"""

import json
import zipfile
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw

from ocr.vectorize import preprocess, trace_ink

# distinct, saturated BGR colours cycled per stroke
PALETTE = [(0, 0, 230), (230, 90, 0), (0, 160, 0), (200, 0, 200),
           (0, 150, 255), (255, 160, 0), (130, 0, 200), (0, 0, 0)]


def fade(gray, keep=0.32):
    """Grayscale ink -> faint light-grey BGR background."""
    g = (255 - (255 - gray.astype(float)) * keep).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def draw_overlay(gray, strokes, lw=2):
    img = fade(gray)
    for si, s in enumerate(strokes):
        col = PALETTE[si % len(PALETTE)]
        pts = [(int(round(x)), int(round(y))) for (x, y) in s]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], col, lw, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 4, col, -1)
            cv2.circle(img, pts[0], 4, (255, 255, 255), 1)
    return img


def render_scan(pts, aspect, Hpx=240, pen=4):
    """Render true online strokes to a clean grayscale 'scan'."""
    xy = pts[:, :2].copy()
    xy[:, 0] *= aspect
    mn, mx = xy.min(0), xy.max(0)
    sc = (Hpx - 8 * pen) / (mx[1] - mn[1])
    W = int((mx[0] - mn[0]) * sc + 8 * pen)
    px = (xy - mn) * sc + 4 * pen
    im = Image.new("L", (W, Hpx), 255)
    d = ImageDraw.Draw(im)
    for i in range(len(px) - 1):
        if pts[i, 2] == 1:
            d.line([tuple(px[i]), tuple(px[i + 1])], fill=0, width=pen)
    return np.array(im)


def from_bank(name, words):
    with zipfile.ZipFile(f"data/{name}.json.zip") as z:
        d = json.load(z.open(z.namelist()[0]))
    out = []
    for w in words:
        it = next((x for x in d if x["metadata"]["asciiSequence"] == w), None)
        if it:
            out.append((w, np.array(it["points"], float), it["metadata"]["aspectRatio"]))
    return out


def bank_diacritics(name, n=2):
    with zipfile.ZipFile(f"data/{name}.json.zip") as z:
        d = json.load(z.open(z.namelist()[0]))
    out = []
    for it in d:
        w = it["metadata"].get("asciiSequence", "")
        pts = np.array(it["points"], float)
        if any(c in w.lower() for c in "it") and int((pts[:, 2] == 0).sum()) >= 2 and len(pts) > 60:
            out.append((w, pts, it["metadata"].get("aspectRatio", 1)))
        if len(out) >= n:
            break
    return out


def main():
    saved = []

    # 1. easybank connected words
    for w, pts, aspect in from_bank("easybank", ["abandon", "happy", "rally"]):
        gray = render_scan(pts, aspect)
        strokes = trace_ink(preprocess(gray))
        out = f"/tmp/overlay_easy_{w}.png"
        cv2.imwrite(out, draw_overlay(gray, strokes))
        saved.append((out, len(strokes)))

    # 2. bigbank diacritic words (per-stroke colour shows i-dot / t-cross split)
    for w, pts, aspect in bank_diacritics("bigbank", 2):
        gray = render_scan(pts, aspect, Hpx=260, pen=4)
        strokes = trace_ink(preprocess(gray))
        out = f"/tmp/overlay_diac_{w[:8]}.png"
        cv2.imwrite(out, draw_overlay(gray, strokes))
        saved.append((out, len(strokes)))

    # 3. REAL scanned crops (actual ink from the PDF)
    crops = sorted(glob.glob("outputs/test_document/page_004_4/*_box_0[12345]?/box.jpg"))[:4]
    for cp in crops:
        bgr = cv2.imread(cp)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        strokes = trace_ink(preprocess(gray))
        folder = cp.rsplit("/", 2)[1]
        try:
            word = open(cp.replace("box.jpg", "text_recognized.txt")).read().strip()
        except Exception:
            word = folder
        out = f"/tmp/overlay_real_{folder.split('_box_')[-1]}_{word[:8]}.png"
        cv2.imwrite(out, draw_overlay(gray, strokes))
        saved.append((out, len(strokes)))

    for path, n in saved:
        print(f"{n:3d} strokes  ->  {path}")


if __name__ == "__main__":
    main()
