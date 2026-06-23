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

import cv2
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
    cap_mult: float = 3.5,
    smooth: int = 15,
    path_smooth: int = 3,
    min_pts: int = 16,
) -> Image.Image:
    """Draw box-relative [x, y, pen] strokes onto an RGB copy of ``crop_image`` as
    smooth brush-tubes whose width COVERS the original ink.

    Width follows the local ink half-width (a distance transform of the crop's
    ink): a disc of that radius at each skeleton point exactly fills the stroke, so
    dark/heavy strokes are fully covered while light/thin strokes stay thin (and so
    width still tracks pen pressure, since heavier pressure = wider ink). The radius
    is capped at ``cap_mult`` x the median half-width (so a crossing/blob can't
    balloon) and SMOOTHED along the stroke (``smooth`` pts) for gradual fat<->thin
    transitions; the centreline is only lightly smoothed (``path_smooth``) so small
    cursive loops survive. Strokes under ``min_pts`` points are dropped as noise
    pimples. Each stroke is overlapping discs -- no beads, gaps, or dotting.
    """
    img = crop_image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    gray = np.asarray(crop_image.convert("L"))
    ink = (gray < 200).astype(np.uint8)  # cleaned crop is ink-on-white
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5) if ink.any() else np.zeros(gray.shape, "f4")

    def radius_at(x: float, y: float) -> float:
        return float(dist[min(h - 1, max(0, round(y))), min(w - 1, max(0, round(x)))])

    px = [(p[0] * w, p[1] * h, p[2]) for p in points]
    radii_all = [radius_at(x, y) for x, y, pen in px if pen == 1]
    pos = [r for r in radii_all if r > 0]
    cap = max(2.0, cap_mult * float(np.median(pos))) if pos else 6.0  # clamp crossings/blobs

    stroke: list[tuple[float, float]] = []
    for x, y, pen in [*px, (0.0, 0.0, 0)]:  # sentinel pen-up flushes the last stroke
        if pen == 1:
            stroke.append((x, y))
            continue
        if len(stroke) >= min_pts:  # else: a noise pimple -> skip
            radii = _smooth(np.array([min(cap, radius_at(sx, sy)) for sx, sy in stroke]), smooth)
            xs = _smooth(np.array([sx for sx, _ in stroke]), path_smooth)
            ys = _smooth(np.array([sy for _, sy in stroke]), path_smooth)
            for sx, sy, r in zip(xs, ys, radii, strict=False):
                rr = max(0.7, float(r))
                draw.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=color)
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
