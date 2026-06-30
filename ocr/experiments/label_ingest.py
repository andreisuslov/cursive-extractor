"""Ingest human-verified labels (labels_corrected.json from letter_label.html) -> labeled letters.

Splits each word's page-box at the confirmed cuts and labels each slice from the spelling, giving
per-letter crops. This is the CLEAN, human-verified supply for (a) the font pipeline and (b) real
CRAFT weak-supervision GT (far better than self-labels, which didn't help — see WORKLOG).

    python -m ocr.experiments.label_ingest --corrected labels_corrected.json \
        --page-dir outputs/<pdf>/page_001 --out-dir letters/
"""

import argparse
import glob
import json
import os


def _boundaries(box, cuts):
    """left edge, the cut polylines (sorted top->bottom), right edge -- in crop-local coords."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    polys = [sorted(c, key=lambda p: p[1]) for c in cuts]
    return [[[0, 0], [0, bh]], *polys, [[bw, 0], [bw, bh]]]


def letter_polys(box, cuts, text):
    """Per-letter ``(char, polygon[[x,y]...] page px)`` between consecutive boundary polylines.

    Cuts are polylines (slanted/curved), so each letter is a polygon, not a rectangle."""
    x0, y0 = box[0], box[1]
    bnds = _boundaries(box, cuts)
    out = []
    for k in range(len(bnds) - 1):
        if k >= len(text):
            break
        a, b = bnds[k], bnds[k + 1]
        poly = [[x0 + px, y0 + py] for px, py in a] + [[x0 + px, y0 + py] for px, py in reversed(b)]
        out.append((text[k], poly))
    return out


def ingest(corrected, page_rgb):
    """List of ``(char, letter_crop_ndarray)`` — each letter masked to its polygon, with the
    word's eraser dabs whited out. From corrected words + the page image."""
    import cv2
    import numpy as np

    letters = []
    for w in corrected:
        erase = w.get("erase", [])
        ox, oy = w["box"][0], w["box"][1]
        for ch, poly in letter_polys(w["box"], w["cuts"], w["text"]):
            pts = np.array(poly, np.int32)
            bx0, bx1 = pts[:, 0].min(), pts[:, 0].max()
            by0, by1 = pts[:, 1].min(), pts[:, 1].max()
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            sub = page_rgb[by0:by1, bx0:bx1].copy()
            mask = np.zeros(sub.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts - [bx0, by0]], 255)
            for ex, ey, er in erase:  # erase dabs (crop-local -> sub-crop coords)
                cv2.circle(mask, (int(ox + ex - bx0), int(oy + ey - by0)), int(er), 0, -1)
            sub[mask == 0] = 255  # whiten outside the polygon + erased ink
            letters.append((ch, sub))
    return letters


def main(argv=None):
    import numpy as np
    from PIL import Image

    p = argparse.ArgumentParser(description="Ingest verified letter labels")
    p.add_argument(
        "--corrected", required=True, help="labels_corrected.json from letter_label.html"
    )
    p.add_argument("--page-dir", required=True, help="page dir with *_page.png (for the image)")
    p.add_argument("--out-dir", default="letters", help="where to write labeled letter crops")
    args = p.parse_args(argv)

    with open(args.corrected) as f:
        corrected = json.load(f)
    page = np.array(Image.open(glob.glob(f"{args.page_dir}/*_page.png")[0]).convert("RGB"))
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for n, (ch, crop) in enumerate(ingest(corrected, page)):
        if crop.size == 0:
            continue
        fn = f"{n:05d}.png"
        Image.fromarray(crop).save(os.path.join(args.out_dir, fn))
        manifest.append({"char": ch, "file": fn})
    with open(os.path.join(args.out_dir, "letters.json"), "w") as f:
        json.dump(manifest, f)
    chars = sorted({m["char"] for m in manifest})
    print(f"{len(manifest)} labeled letters, {len(chars)} distinct -> {args.out_dir}", flush=True)
    print(
        "coverage: " + ", ".join(f"{c}:{sum(m['char'] == c for m in manifest)}" for c in chars),
        flush=True,
    )


if __name__ == "__main__":
    main()
