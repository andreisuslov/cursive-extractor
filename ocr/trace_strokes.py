"""CLI: trace pen strokes for one OCR box using Gemini (not the OpenCV vectorizer).

Crops the chosen box from a page, asks Gemini to trace the strokes, saves the
resulting JSON, and writes a green overlay for inspection. This is the
model-based alternative to ``python -m ocr.vectorize``.

    export GOOGLE_API_KEY=...
    python -m ocr.trace_strokes --boxes data/content/bbox_data/page_2_data.json --index 0 --page 2
"""

import argparse
import json

from PIL import ImageDraw

from . import config, paths
from .gemini_ocr import build_model, trace_strokes
from .pdf_utils import crop_to_box, load_page


def draw_strokes(crop_image, strokes, color=(0, 255, 0), width=2):
    """Draw normalized [x, y, p] strokes onto a copy of ``crop_image``."""
    img = crop_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    pts = [(x * w, y * h, p) for x, y, p in strokes]
    for i in range(1, len(pts)):
        if pts[i][2] == 1:
            draw.line(
                [(pts[i - 1][0], pts[i - 1][1]), (pts[i][0], pts[i][1])], fill=color, width=width
            )
    return img


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Trace strokes for one box via Gemini")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based")
    p.add_argument("--index", type=int, default=0, help="Which box to trace (0-based)")
    p.add_argument(
        "--boxes", default=None, help="Boxes JSON (default: canonical boxes for --pdf/--page)"
    )
    p.add_argument("--model", default=config.GEMINI_MODEL, help="Gemini model name")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Page version to read/write (default: latest existing)",
    )
    p.add_argument("--out-json", default=None, help="Strokes JSON (default: canonical)")
    p.add_argument("--out-image", default=None, help="Overlay image (default: canonical)")
    p.add_argument("--padding", type=int, default=10, help="Crop padding in pixels")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    boxes_path = args.boxes or paths.boxes_json(args.pdf, args.page, version, root)
    with open(boxes_path) as f:
        boxes = json.load(f)

    entry = boxes[args.index]
    text = entry.get("text", "")
    print(f"Tracing box {args.index}: '{text}'")

    page = load_page(args.pdf, args.page - 1)
    crop, _ = crop_to_box(page, entry["box_2d"], args.padding)

    model = build_model(args.model)
    result = trace_strokes(model, crop, text)

    out_json = args.out_json or paths.trace_json(args.pdf, args.page, args.index, version, root)
    paths.ensure_parent(out_json)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved strokes to: {out_json}")

    out_image = args.out_image or paths.trace_overlay(
        args.pdf, args.page, args.index, version, root
    )
    draw_strokes(crop, result.get("strokes", [])).save(out_image)
    print(f"Saved overlay to: {out_image}")


if __name__ == "__main__":
    main()
