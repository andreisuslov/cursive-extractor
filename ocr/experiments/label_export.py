"""Export a page's words to label_review.json for the letter labeller (letter_label.html).

Per alphabetic word: the page-box (px) + CRAFT-proposed cut positions (crop-local px), plus the
page image once (JPEG data-URL). The human fixes cuts in the browser; labels auto-fill from the
spelling. Needs boxes_refined.json (ocr.refine_boxes) + a CRAFT checkpoint (craft_train/weaksup).

    python -m ocr.experiments.label_export --page-dir outputs/<pdf>/page_001 --out label_review.json
"""

import argparse
import base64
import glob
import io
import json
import os

import numpy as np
from PIL import Image

from .craft_segmenter import CraftSegmenter


def export_page(page_dir: str, segmenter: CraftSegmenter) -> dict:
    """Build the label_review doc: page JPEG data-URL + per-word {text, box(px), cuts(crop px)}."""
    with open(os.path.join(page_dir, "boxes_refined.json")) as f:
        boxes = json.load(f)
    page_path = glob.glob(f"{page_dir}/*_page.png")[0]
    page = Image.open(page_path).convert("RGB")
    arr = np.array(page)
    w_px, h_px = page.size
    words = []
    for b in boxes:
        t = b["text"]
        if not (t.isalpha() and 2 <= len(t) <= 12):
            continue
        ymin, xmin, ymax, xmax = b["box_2d"]
        x0, y0 = int(xmin * w_px / 1000), int(ymin * h_px / 1000)
        x1, y1 = int(xmax * w_px / 1000), int(ymax * h_px / 1000)
        if x1 - x0 < 8 or y1 - y0 < 5:
            continue
        cuts = segmenter.cut_word(arr[y0:y1, x0:x1], t)
        bh = y1 - y0
        words.append(
            {
                "text": t,
                "box": [x0, y0, x1, y1],
                # each cut is a polyline (top->bottom) so it can be slanted/curved in the tool
                "cuts": [[[round(float(c), 1), 0], [round(float(c), 1), bh]] for c in cuts],
                "erase": [],
            }
        )
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=85)
    page_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"page": page_url, "page_dir": page_dir, "words": words}


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a page for the letter labeller")
    p.add_argument(
        "--page-dir", required=True, help="page dir with boxes_refined.json + *_page.png"
    )
    p.add_argument("--out", default="label_review.json")
    p.add_argument("--weights", default=None, help="CRAFT weights (default: v4)")
    args = p.parse_args(argv)

    seg = CraftSegmenter(args.weights) if args.weights else CraftSegmenter()
    doc = export_page(args.page_dir, seg)
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(f"{len(doc['words'])} words -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
