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


def letter_boxes(box, cuts, text):
    """Per-letter (char, [x0,y0,x1,y1] page px) by splitting ``box`` at crop-local ``cuts``."""
    x0, y0, x1, y1 = box
    edges = [0.0, *sorted(cuts), float(x1 - x0)]
    out = []
    for k in range(len(edges) - 1):
        if k >= len(text):
            break
        lx0, lx1 = x0 + edges[k], x0 + edges[k + 1]
        if lx1 - lx0 >= 2:
            out.append((text[k], [int(lx0), int(y0), int(lx1), int(y1)]))
    return out


def ingest(corrected, page_rgb):
    """List of ``(char, letter_crop_ndarray)`` from corrected words + the page image."""
    letters = []
    for w in corrected:
        for ch, (lx0, ly0, lx1, ly1) in letter_boxes(w["box"], w["cuts"], w["text"]):
            letters.append((ch, page_rgb[ly0:ly1, lx0:lx1]))
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
