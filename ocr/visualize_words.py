"""CLI: visualize the bounding boxes of the first N words of a page.

Reads a per-page JSON produced by ``extract_boxes`` (entries of
``{text, box_2d}`` with ``box_2d`` = [ymin, xmin, ymax, xmax] on a 0-1000
scale) and draws each box, labeled with its word, over the rendered page.

    python -m ocr.visualize_words --pdf data/content/test_document.pdf \\
        --boxes data/content/bbox_data/page_2_data.json --page 2 --num 3

NOTE: this replaces the notebook's "View Sample Words" cell, which referenced
undefined globals and a different schema; it now reads the pipeline's own output.
"""

import argparse
import json

import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import ImageDraw

from . import config, paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Visualize word bounding boxes on a page")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Path to the input PDF")
    p.add_argument("--page", type=int, default=2, help="Page to render (1-based)")
    p.add_argument(
        "--boxes", default=None, help="Boxes JSON (default: canonical boxes for --pdf/--page)"
    )
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version", type=int, default=None, help="Page version to read (default: latest existing)"
    )
    p.add_argument("--num", type=int, default=3, help="How many words to show")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    images = convert_from_path(args.pdf)
    page_index = args.page - 1
    if not 0 <= page_index < len(images):
        raise SystemExit(f"Page {args.page} out of range (PDF has {len(images)} pages)")
    image = images[page_index]
    width, height = image.size

    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, args.output_root)
    )
    boxes_path = args.boxes or paths.boxes_json(args.pdf, args.page, version, args.output_root)
    with open(boxes_path) as f:
        word_data = json.load(f)
    if not word_data:
        raise SystemExit("No word data to visualize")

    for i, item in enumerate(word_data[: args.num]):
        box = item.get("box_2d")
        word = item.get("text", "")
        if not box:
            continue
        ymin, xmin, ymax, xmax = box
        left, right = (xmin / 1000) * width, (xmax / 1000) * width
        top, bottom = (ymin / 1000) * height, (ymax / 1000) * height

        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left, max(0, top - 20)), word, fill="red")

        plt.figure(figsize=(12, 8))
        plt.imshow(img_copy)
        plt.title(f"Word {i + 1}: '{word}'")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
