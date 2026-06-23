"""CLI: assemble one folder per detected box.

For each word box on a page it creates a folder
``<pdf>_p<NNN>[_V]_box_<III>/`` (under the page folder) containing:

  * ``text_recognized.txt`` - the OCR'd text
  * ``box.jpg``             - the cropped word image
  * ``vectorized.jpg``      - box.jpg with the vectorized strokes drawn in teal

Reads the canonical strokes dataset (so run ``ocr.vectorize`` first) plus the
PDF. Defaults to the latest version of the page.

    python -m ocr.package_boxes --pdf datasets/content/test_document.pdf --page 4
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from . import config, paths, qa
from .pdf_utils import crop_to_box, ensure_page_png, load_page

# Teal strokes overlaid on the cursive, matching the verification colour.
TEAL = (0, 170, 160)


def _smooth(vals: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average (edge-replicated), so a per-point series varies
    gradually instead of jittering point to point."""
    if k <= 1 or vals.size < 3:
        return vals
    k = min(k | 1, vals.size if vals.size % 2 else vals.size - 1)  # odd, <= len
    pad = k // 2
    return np.convolve(np.pad(vals, pad, mode="edge"), np.ones(k) / k, mode="valid")


def draw_vectorized(
    crop_image: Image.Image,
    points: list[list[float]],
    color: tuple[int, int, int] = TEAL,
    min_width: float = 1.0,
    max_width: float = 14.0,
    gamma: float = 1.6,
    smooth: int = 11,
    path_smooth: int = 7,
) -> Image.Image:
    """Draw box-relative [x, y, pen] strokes onto an RGB copy of ``crop_image``,
    with stroke width following the ORIGINAL pen pressure.

    Pressure is read as ink DARKNESS sampled along the skeleton (a 1-D measure
    valid on strokes: a light hairline is pale, a heavy down-stroke saturated),
    normalized to this word's own ink, ``gamma``>1 so dark/heavy ink stays thick
    while light ink renders thin, capped to ``[min_width, max_width]``. The width
    series is SMOOTHED along each stroke (``smooth`` points) so fat<->thin
    transitions are gradual, and each stroke is rendered as overlapping discs (a
    brush tube) rather than jointed line segments -- no beads, gaps, or dotting.
    """
    img = crop_image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    gray = np.asarray(crop_image.convert("L"), dtype=np.float32)

    def darkness_at(x: float, y: float) -> float:
        xi, yi = min(w - 1, max(0, round(x))), min(h - 1, max(0, round(y)))
        win = gray[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2]
        return 255.0 - float(win.min())  # darkest ink in a 3x3 nbhd around the point

    px = [(p[0] * w, p[1] * h, p[2]) for p in points]
    dark = [darkness_at(x, y) for x, y, pen in px if pen == 1]
    lo, hi = (
        (float(np.percentile(dark, 20)), float(np.percentile(dark, 92))) if dark else (0.0, 1.0)
    )
    span = max(1.0, hi - lo)

    def width_of(d: float) -> float:
        t = min(1.0, max(0.0, (d - lo) / span)) ** gamma
        return min_width + t * (max_width - min_width)

    # split into pen-down strokes, then draw each as a smoothed-width disc tube
    stroke: list[tuple[float, float]] = []
    for x, y, pen in [*px, (0.0, 0.0, 0)]:  # sentinel pen-up flushes the last stroke
        if pen == 1:
            stroke.append((x, y))
            continue
        if len(stroke) >= 1:
            widths = _smooth(np.array([width_of(darkness_at(sx, sy)) for sx, sy in stroke]), smooth)
            # smooth the centreline too, so the tube curves instead of following the
            # jagged 1-px skeleton (darkness is still sampled at the true location).
            xs = _smooth(np.array([sx for sx, _ in stroke]), path_smooth)
            ys = _smooth(np.array([sy for _, sy in stroke]), path_smooth)
            for sx, sy, wd in zip(xs, ys, widths, strict=False):
                r = max(0.5, wd / 2.0)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=color)
        stroke = []
    return img


def package_boxes(
    pdf_path: str,
    data: list[dict],
    page: int,
    version: int | None = None,
    root: str | None = None,
    dpi: int = 600,
    padding: int | None = None,
    pad_frac: float | None = None,
    fit_ink: bool | None = None,
    clean: bool | None = None,
    limit: int | None = None,
) -> int:
    """Write a per-box folder for every entry with a ``box_2d``. Returns count.

    With ``clean`` (default ``config.CROP_CLEAN``) box.jpg is the cleaned per-word
    crop (matching the cleaned strokes so the teal overlay stays aligned)."""
    from .vectorize import clean_word

    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    fit_ink = config.CROP_FIT_INK if fit_ink is None else fit_ink
    clean = config.CROP_CLEAN if clean is None else clean
    page_image = load_page(pdf_path, page - 1, dpi=dpi)
    made = 0
    for i, entry in enumerate(data if limit is None else data[:limit]):
        if "box_2d" not in entry:
            continue
        bdir = paths.ensure_dir(paths.box_dir(pdf_path, page, i, version, root))
        # respect the per-word gate decision from vectorize (metadata["cleaned"])
        # so box.jpg matches the crop the strokes were normalized to.
        use_clean = clean and entry.get("metadata", {}).get("cleaned", True)
        if use_clean:
            crop, _, _ = clean_word(page_image, entry["box_2d"], padding, pad_frac)
        else:
            crop, _ = crop_to_box(page_image, entry["box_2d"], padding, pad_frac, fit_ink)

        with open(os.path.join(bdir, paths.BOX_TEXT_FILE), "w") as f:
            f.write(entry.get("text", ""))
        crop.save(os.path.join(bdir, paths.BOX_IMAGE_FILE), quality=95, subsampling=0)
        draw_vectorized(crop, entry.get("points", [])).save(
            os.path.join(bdir, paths.BOX_VECTORIZED_FILE)
        )

        made += 1
        if made % 10 == 0:
            print(f"Packaged {made} boxes...")
    print(f"Packaged {made} box folders under {paths.page_dir(pdf_path, page, version, root)}")
    return made


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble one folder per detected box")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based")
    p.add_argument(
        "--strokes", default=None, help="Strokes JSON (default: canonical strokes for --pdf/--page)"
    )
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Page version to read/write (default: latest existing)",
    )
    p.add_argument("--limit", type=int, default=None, help="Package only the first N boxes")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for crops")
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
    p.add_argument("--no-qa", action="store_true", help="Skip the transcript QA check")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    if version is None and not args.strokes:
        raise SystemExit("No processed version found; run ocr.vectorize first or pass --strokes.")

    ensure_page_png(args.pdf, args.page, version, root)  # leave a viewable input page
    strokes_path = args.strokes or paths.strokes_json(args.pdf, args.page, version, root)
    with open(strokes_path) as f:
        data = json.load(f)
    package_boxes(
        args.pdf,
        data,
        args.page,
        version=version,
        root=root,
        dpi=args.dpi,
        padding=args.padding,
        pad_frac=args.pad_frac,
        fit_ink=not args.no_fit_ink,
        limit=args.limit,
    )

    if not args.no_qa:
        _passed, report = qa.run_qa(args.pdf, args.page, version, root)
        print("\n" + report)


if __name__ == "__main__":
    main()
