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


def _ordered_cuts(cuts):
    """Cuts ordered left-to-right by mean x, so slice k maps to text[k] (the tool may store cuts
    out of order after a whole-line drag or a +Cut placed left of others)."""
    return sorted(cuts, key=lambda c: sum(p[0] for p in c) / len(c))


def letter_polys(cuts, bw, bh, text):
    """Per-letter ``(char, polygon[[x,y]...])`` in CROP-LOCAL coords, between consecutive
    boundary polylines (left edge, the L-1 ordered cuts, right edge). Cuts are polylines
    (slanted/curved), so each letter is a polygon, not a rectangle."""
    srt = [sorted(c, key=lambda p: p[1]) for c in _ordered_cuts(cuts)]
    bnds = [[[0, 0], [0, bh]], *srt, [[bw, 0], [bw, bh]]]
    out = []
    for k in range(len(bnds) - 1):
        if k >= len(text):
            break
        a, b = bnds[k], bnds[k + 1]
        out.append((text[k], [list(p) for p in a] + [list(p) for p in reversed(b)]))
    return out


def _decode_clean(data_url):
    """Decode an inpainted-crop data-URL (PNG) -> RGB ndarray."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def ingest(corrected, page_rgb):
    """List of ``(char, letter_crop_ndarray)`` — each letter masked to its polygon. Uses the
    word's inpainted ``clean`` crop if present (ink already removed), else the page crop with the
    eraser dabs whited out."""
    import cv2
    import numpy as np

    letters = []
    for w in corrected:
        x0, y0, x1, y1 = w["box"]
        bw, bh = x1 - x0, y1 - y0
        if w.get("clean"):
            src, erase = _decode_clean(w["clean"]), []  # ink already baked out
        else:
            src, erase = page_rgb[y0:y1, x0:x1].copy(), w.get("erase", [])
        for ch, poly in letter_polys(w["cuts"], bw, bh, w["text"]):
            pts = np.array(poly, np.int32)
            bx0, by0 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
            bx1, by1 = min(src.shape[1], pts[:, 0].max()), min(src.shape[0], pts[:, 1].max())
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            sub = src[by0:by1, bx0:bx1].copy()
            mask = np.zeros(sub.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts - [bx0, by0]], 255)
            for ex, ey, er in erase:  # eraser dabs are crop-local
                cv2.circle(mask, (int(ex - bx0), int(ey - by0)), int(er), 0, -1)
            sub[mask == 0] = 255  # whiten outside the polygon (+ erased ink)
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
