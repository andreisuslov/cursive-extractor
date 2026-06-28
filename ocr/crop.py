"""CLI: crop a single OCR box out of a PDF page as a high-DPI image.

For the full per-box deliverable (every box in its own folder with text + crop +
vectorized overlay) use ``ocr.package_boxes``. This is a lightweight helper to
pull one box straight from a boxes JSON, without needing a vectorize pass.

    python -m ocr.crop --pdf datasets/content/test_document.pdf --page 4 --index 0
"""

import argparse
import json

from PIL import Image, ImageEnhance

from . import config, paths
from .pdf_utils import crop_to_box, load_page


def crop_box_image(
    pdf_path: str,
    boxes: list[dict],
    index: int,
    page_index: int,
    dpi: int = 600,
    padding: int | None = None,
    pad_frac: float | None = None,
    fit_ink: bool | None = None,
    contrast: float = 1.0,
    mask_to_box: bool | None = None,
) -> tuple[Image.Image, str]:
    """Return ``(cropped_image, text)`` for the ``index``-th box on a page."""
    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    fit_ink = config.CROP_FIT_INK if fit_ink is None else fit_ink
    mask_to_box = config.CROP_MASK_TO_BOX if mask_to_box is None else mask_to_box
    page = load_page(pdf_path, page_index, dpi=dpi)
    entry = boxes[index]
    if "box_2d" not in entry:
        raise ValueError(f"Entry {index} has no 'box_2d'")
    crop, _ = crop_to_box(
        page,
        entry["box_2d"],
        padding,
        pad_frac,
        fit_ink,
        mask_to_box=mask_to_box,
        mask_hpad=config.CROP_MASK_HPAD,
        mask_vpad_up=config.CROP_MASK_VPAD_UP,
        mask_vpad_dn=config.CROP_MASK_VPAD_DN,
    )
    if contrast and contrast != 1.0:
        crop = ImageEnhance.Contrast(crop).enhance(contrast)
    return crop, entry.get("text", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop one OCR box to a high-quality image")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based")
    p.add_argument("--index", type=int, default=0, help="Which box to crop (0-based)")
    p.add_argument(
        "--boxes", default=None, help="Boxes JSON (default: canonical boxes for --pdf/--page)"
    )
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Page version to read/write (default: latest existing)",
    )
    p.add_argument("--save", default=None, help="Output path (default: the box folder's box.jpg)")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI")
    p.add_argument("--padding", type=int, default=config.CROP_PADDING, help="Crop padding (px)")
    p.add_argument(
        "--pad-frac",
        type=float,
        default=config.CROP_PAD_FRAC,
        help="Extra crop padding as a fraction of box size",
    )
    p.add_argument(
        "--no-fit-ink",
        action="store_true",
        help="Disable growing the box until no ink touches its borders",
    )
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast boost (1.0 = none)")
    p.add_argument(
        "--mask-to-box",
        action="store_true",
        default=config.CROP_MASK_TO_BOX,
        help="Whiten ink outside the (padded) detection box, isolating one word",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    if version is None and not args.boxes:
        raise SystemExit("No processed version found; run ocr.extract_boxes first or pass --boxes.")

    boxes_path = args.boxes or paths.boxes_json(args.pdf, args.page, version, root)
    with open(boxes_path) as f:
        boxes = json.load(f)

    crop, text = crop_box_image(
        args.pdf,
        boxes,
        args.index,
        args.page - 1,
        dpi=args.dpi,
        padding=args.padding,
        pad_frac=args.pad_frac,
        fit_ink=not args.no_fit_ink,
        contrast=args.contrast,
        mask_to_box=args.mask_to_box,
    )
    save = args.save or paths.box_image(args.pdf, args.page, args.index, version, root)
    paths.ensure_parent(save)
    crop.save(save, quality=95, subsampling=0)
    print(f"Saved crop of '{text}' ({crop.size}) to: {save}")


if __name__ == "__main__":
    main()
