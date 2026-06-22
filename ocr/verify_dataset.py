"""CLI: overlay a traced stroke dataset onto its source PDF page to check accuracy.

Handles the two coordinate conventions produced by this pipeline:

  * ``--coords page`` (default): each entry's points are normalized to the whole
    page (0-1 of page width/height). This is what the browser capture tool and
    Gemini stroke tracing produce.
  * ``--coords box``: points are normalized within each word's crop and placed
    using the entry's ``box_2d``. This is what ``ocr.vectorize`` produces (the
    canonical ``*_strokes.json``), so its output must be verified in this mode.

Other options: ``--index`` to draw a single entry, ``--crop`` to crop the output
to the drawn strokes (handy with ``--index``), ``--save`` / ``--show``.

    python -m ocr.verify_dataset \\
        --dataset datasets/content/handwriting_dataset_precise.json --page 2
    python -m ocr.verify_dataset --coords box --page 4 --save overlay.jpg
"""

import argparse
import json
from collections.abc import Callable, Sequence

from PIL import Image, ImageDraw

from . import config, paths
from .pdf_utils import box_to_crop_box, load_page

STROKE_COLOR = (0, 255, 0)


def _page_mapper(width: int, height: int) -> Callable[[Sequence[float]], tuple[float, float]]:
    return lambda p: (p[0] * width, p[1] * height)


def _box_mapper(
    box_2d: list[int], width: int, height: int, padding: int
) -> Callable[[Sequence[float]], tuple[float, float]]:
    left, top, right, bottom = box_to_crop_box(box_2d, width, height, padding)
    bw, bh = right - left, bottom - top
    return lambda p: (left + p[0] * bw, top + p[1] * bh)


def overlay_strokes(
    page_image: Image.Image,
    data: list[dict],
    coords: str = "page",
    padding: int = 10,
    width: int = 2,
) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    """Draw stroke segments on a copy of ``page_image``.

    Returns ``(annotated_image, bounds)`` where bounds is the pixel bounding box
    of everything drawn, or ``None`` if nothing was drawn.
    """
    img = page_image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    require_both = coords == "box"  # box-relative data has explicit pen-up markers
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for entry in data:
        points = entry.get("points", [])
        if not points:
            continue
        if coords == "box":
            box = entry.get("box_2d")
            if not box:
                continue
            to_px = _box_mapper(box, W, H, padding)
        else:
            to_px = _page_mapper(W, H)

        for i in range(1, len(points)):
            pen_down = points[i][2] == 1 and (points[i - 1][2] == 1 or not require_both)
            if not pen_down:
                continue
            x1, y1 = to_px(points[i - 1])
            x2, y2 = to_px(points[i])
            draw.line([(x1, y1), (x2, y2)], fill=STROKE_COLOR, width=width)
            minx, miny = min(minx, x1, x2), min(miny, y1, y2)
            maxx, maxy = max(maxx, x1, x2), max(maxy, y1, y2)

    bounds = None if minx == float("inf") else (minx, miny, maxx, maxy)
    return img, bounds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay a traced dataset on a PDF page")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Path to the input PDF")
    p.add_argument("--page", type=int, default=2, help="Page to render (1-based)")
    p.add_argument(
        "--dataset", default=None, help="Dataset JSON (default: canonical strokes for --pdf/--page)"
    )
    p.add_argument(
        "--coords",
        choices=("page", "box"),
        default="box",
        help="Point coordinate convention; 'box' matches ocr.vectorize output",
    )
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Page version to read/write (default: latest existing)",
    )
    p.add_argument("--index", type=int, default=None, help="Only draw this entry (0-based)")
    p.add_argument("--crop", action="store_true", help="Crop output to drawn strokes")
    p.add_argument("--dpi", type=int, default=None, help="Render DPI (use 600 for box mode)")
    p.add_argument("--padding", type=int, default=10, help="Box crop padding (box mode)")
    p.add_argument("--save", default=None, help="Save overlay (default: canonical verify path)")
    p.add_argument("--show", action="store_true", help="Display with matplotlib")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    dpi = args.dpi or (600 if args.coords == "box" else None)
    page = load_page(args.pdf, args.page - 1, dpi=dpi)

    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    if version is None and not args.dataset:
        raise SystemExit("No processed version found; run ocr.vectorize first or pass --dataset.")

    dataset_path = args.dataset or paths.strokes_json(args.pdf, args.page, version, root)
    with open(dataset_path) as f:
        data = json.load(f)
    if args.index is not None:
        data = data[args.index : args.index + 1]
    print(f"Painting {len(data)} word(s) in '{args.coords}' coords...")

    img, bounds = overlay_strokes(page, data, coords=args.coords, padding=args.padding)

    if args.crop and bounds:
        pad = 50
        W, H = img.size
        left, top, right, bottom = bounds
        img = img.crop(
            (max(0, left - pad), max(0, top - pad), min(W, right + pad), min(H, bottom + pad))
        )

    out = args.save or paths.verify_overlay(args.pdf, args.page, args.coords, version, root)
    paths.ensure_parent(out)
    img.save(out)
    print(f"Saved overlay to: {out}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 25))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
