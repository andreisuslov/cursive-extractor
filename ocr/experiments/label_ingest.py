"""Ingest human-verified labels (labels_corrected.json from letter_label.html) -> labeled letters.

Splits each word's page-box at the confirmed cuts and labels each slice from the spelling, giving
per-letter crops. This is the CLEAN, human-verified supply for (a) the font pipeline and (b) real
CRAFT weak-supervision GT (far better than self-labels, which didn't help — see WORKLOG).

    python -m ocr.experiments.label_ingest --corrected labels_corrected.json --out-dir letters/
    # the exported file is self-contained (carries the page); add --page-dir only for a legacy
    # page-less edits array.
"""

import argparse
import glob
import json
import os


def _ordered_cuts(cuts):
    """Cuts ordered left-to-right by mean x, so slice k maps to text[k] (the tool may store cuts
    out of order after a whole-line drag or a +Cut placed left of others)."""
    return sorted(cuts, key=lambda c: sum(p[0] for p in c) / len(c))


def letter_polys(cuts, bw, bh, text, end_cuts=False):
    """Per-letter ``(char, polygon[[x,y]...])`` in CROP-LOCAL coords, between consecutive
    boundary polylines (left edge, the L-1 ordered cuts, right edge). Cuts are polylines
    (slanted/curved), so each letter is a polygon, not a rectangle."""
    srt = [sorted(c, key=lambda p: p[1]) for c in _ordered_cuts(cuts)]
    if end_cuts and len(srt) >= 2:                   # end-cuts mode: the first & last cut ARE the boundaries
        bnds = list(srt)
    else:                                            # classic: the box edges are the first/last boundary
        bnds = [[[0, 0], [0, bh]], *srt, [[bw, 0], [bw, bh]]]
    out = []
    for k in range(len(bnds) - 1):
        if k >= len(text):
            break
        a, b = bnds[k], bnds[k + 1]
        out.append((text[k], [list(p) for p in a] + [list(p) for p in reversed(b)]))
    return out


def ingest(corrected, page_rgb):
    """List of ``(char, letter_crop_ndarray)`` — each letter masked to its polygon. Pixels outside
    the polygon, and the eraser dabs, are filled with the word's PAPER COLOUR (median of the crop),
    so the glyph sits on a clean, matching background instead of a flat white."""
    import cv2
    import numpy as np

    words = corrected["words"] if isinstance(corrected, dict) else corrected
    abs_dabs = (
        isinstance(corrected, dict) and corrected.get("eraseSpace") == "page"
    )  # else crop-local (legacy)
    letters = []
    for w in words:
        if w.get("skip"):
            continue
        x0, y0, x1, y1 = w["box"]
        bw, bh = x1 - x0, y1 - y0
        src = page_rgb[y0:y1, x0:x1]
        paper = np.median(src.reshape(-1, src.shape[-1]), axis=0)  # ink is a minority -> paper
        for ch, poly in letter_polys(w["cuts"], bw, bh, w["text"], w.get("endCuts")):
            pts = np.array(poly, np.int32)
            bx0, by0 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
            bx1, by1 = min(src.shape[1], pts[:, 0].max()), min(src.shape[0], pts[:, 1].max())
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            sub = src[by0:by1, bx0:bx1].copy()
            mask = np.zeros(sub.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts - [bx0, by0]], 255)
            for ex, ey, er in w.get("erase", []):  # page-coords (new) or crop-local (legacy)
                cx = ex - x0 - bx0 if abs_dabs else ex - bx0
                cy = ey - y0 - by0 if abs_dabs else ey - by0
                cv2.circle(mask, (int(cx), int(cy)), int(er), 0, -1)
            sub[mask == 0] = paper  # fill outside-polygon + erased ink with paper colour
            letters.append((ch, sub))
    return letters


def main(argv=None):
    import base64
    import io

    import numpy as np
    from PIL import Image

    p = argparse.ArgumentParser(description="Ingest verified letter labels")
    p.add_argument(
        "--corrected", required=True, help="labels_corrected.json from letter_label.html"
    )
    p.add_argument(
        "--page-dir",
        default=None,
        help="page dir with *_page.png (only if the file has no embedded page)",
    )
    p.add_argument("--out-dir", default="letters", help="where to write labeled letter crops")
    args = p.parse_args(argv)

    with open(args.corrected) as f:
        corrected = json.load(f)
    if isinstance(corrected, dict) and corrected.get("page"):  # self-contained export
        raw = base64.b64decode(corrected["page"].split(",", 1)[1])
        page = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    elif args.page_dir:
        page = np.array(Image.open(glob.glob(f"{args.page_dir}/*_page.png")[0]).convert("RGB"))
    else:
        p.error("corrected file has no embedded page; pass --page-dir")
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
