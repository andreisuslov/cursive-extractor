"""Vectorize word crops with InkSight (offline->online derendering) instead of the
skeleton tracer in ``vectorize.py``.

InkSight (Google Research, TMLR 2024, Apache-2.0) recovers clean, ordered pen strokes
from a raster crop -- far better than the skeleton tracer on real diary ink (see
WORKLOG.md). This module is a drop-in alternative producing the SAME strokes.json schema
(each box gets ``points`` = [[x, y, pen], ...] in [0,1] + ``metadata``), so
``datasets/build_diarybank.py`` consumes it unchanged.

Heavy, SEPARATE deps (not in the core pipeline): ``tensorflow`` 2.15-2.17 +
``tensorflow-text`` + the Small-p saved_model (~0.5 GB, not in the repo). TF is imported
lazily so importing the rest of the package stays light. Point ``OCR_INKSIGHT_MODEL`` at
the unzipped ``small-p-cpu`` dir. Crops are masked to one word (``mask_to_box``) so the
model sees only the target. ~5 min/crop on CPU -- run on a GPU for a full dataset.

Setup recipe: ``ocr/experiments/_inksight_probe.py``.

    OCR_INKSIGHT_MODEL=/path/to/small-p-cpu \
        python -m ocr.inksight_vectorize --pdf <pdf> --page N [--limit K]
"""

import argparse
import json
import os
import re

import cv2
import numpy as np
from PIL import Image

from . import config, paths
from .pdf_utils import crop_to_box, load_page

_PROMPT = "Recognize and derender."
_FALLBACK_PROMPT = "Derender the ink."
_model_cache = {}


def text_to_tokens(text: str) -> list[int]:
    """Extract InkSight ink-token indices from the model's text output."""
    return [int(t) for t in re.findall(r"<ink_token_(\d+)>", text)]


def detokenize_strokes(tokens: list[int]) -> list[list[tuple[int, int]]]:
    """Decode InkSight ink tokens into strokes (lists of (x, y) in 0-224 coords).

    Mirrors the reference detokenizer: tokens interleave x and y (y offset by the
    per-dimension size); a start token separates strokes. Invalid coords are skipped.
    """
    coordinate_length = 224
    npd = coordinate_length + 1  # tokens per dimension
    start = npd * 2
    res, cur, idx = [], [], 0
    while idx < len(tokens):
        t = tokens[idx]
        if t == start:
            if cur:
                res.append(cur)
            cur = []
            idx += 1
        elif idx + 1 < len(tokens) and tokens[idx + 1] != start:
            x, y = tokens[idx], tokens[idx + 1] - npd
            if 0 <= x <= coordinate_length and 0 <= y <= coordinate_length:
                cur.append((x, y))
            idx += 2
        else:
            idx += 1
    if cur:
        res.append(cur)
    return res


def strokes_to_points(
    strokes: list[list[tuple[float, float]]], size: int = 224
) -> list[list[float]]:
    """Convert InkSight strokes (0-``size`` coords) to the strokes.json point format:
    ``[[x, y, pen], ...]`` normalized to [0,1] (divide by ``size``, matching
    ``vectorize.format_strokes``' divide-by-crop-size), one pen-up (0) per stroke end.
    """
    out: list[list[float]] = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for x, y in stroke:
            out.append(
                [round(min(max(x / size, 0.0), 1.0), 4), round(min(max(y / size, 0.0), 1.0), 4), 1]
            )
        out.append([out[-1][0], out[-1][1], 0])  # pen-up at stroke end
    return out


def strokes_to_points_rich(
    strokes: list[list[tuple[float, float]]],
    gray: np.ndarray,
    dt: np.ndarray,
    size: int = 224,
) -> list[list[float]]:
    """Like ``strokes_to_points`` but adds two channels sampled from the image at each
    point: ``[x, y, pen, width, intensity]``.

    ``width`` = local stroke thickness = ``2 * distanceTransform`` at the point, /``size``
    (the trajectory rides the stroke centerline, where the distance transform ~= half the
    stroke width). ``intensity`` = ink darkness = ``(255 - gray) / 255`` in [0,1] (a faint
    "weak" stroke reads low). The pen-up marker carries width=0, intensity=0 (no ink).
    True pen pressure/velocity is NOT recoverable from a static scan; these are its
    visual proxies. ``gray``/``dt`` are the model's 224x224 input in grayscale and its
    distance transform.
    """
    h, w = gray.shape[:2]
    out: list[list[float]] = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for x, y in stroke:
            xi = min(max(round(x), 0), w - 1)
            yi = min(max(round(y), 0), h - 1)
            width = round(min(2.0 * float(dt[yi, xi]) / size, 1.0), 4)
            intensity = round((255.0 - float(gray[yi, xi])) / 255.0, 4)
            nx = round(min(max(x / size, 0.0), 1.0), 4)
            ny = round(min(max(y / size, 0.0), 1.0), 4)
            out.append([nx, ny, 1, width, intensity])
        out.append([out[-1][0], out[-1][1], 0, 0.0, 0.0])  # pen-up: no ink
    return out


def _load_model():
    """Lazily load the Small-p saved_model's serving signature (cached)."""
    model_dir = os.environ.get("OCR_INKSIGHT_MODEL")
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError(
            "Set OCR_INKSIGHT_MODEL to the unzipped small-p-cpu dir "
            "(see ocr/experiments/_inksight_probe.py for setup)."
        )
    if model_dir not in _model_cache:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # heavy import, deferred to call time
        import tensorflow_text  # noqa: F401  (registers ops the saved_model needs)

        _model_cache[model_dir] = tf.saved_model.load(model_dir).signatures["serving_default"]
    return _model_cache[model_dir]


def _scale_and_pad(crop: Image.Image, pad_black: bool = True) -> Image.Image:
    """Resize a crop to fit 224x224 and pad to a square (InkSight's expected input)."""
    ratio = min(224 / crop.width, 224 / crop.height)
    resized = crop.convert("RGB").resize(
        (max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio)))
    )
    bg = (0, 0, 0) if pad_black else (255, 255, 255)
    out = Image.new("RGB", (224, 224), bg)
    out.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return out


def vectorize_crop_inksight(crop: Image.Image, rich: bool = False) -> list[list[float]]:
    """Derender a single word crop into points (normalized [0,1]).

    ``rich=False`` -> ``[[x, y, pen], ...]`` (drop-in for the current model).
    ``rich=True`` -> ``[[x, y, pen, width, intensity], ...]`` (thickness + faintness
    sampled from the crop; see ``strokes_to_points_rich``).
    """
    import tensorflow as tf  # deferred: only imported when actually vectorizing

    cf = _load_model()
    img = _scale_and_pad(crop)
    enc = tf.reshape(tf.io.encode_jpeg(np.array(img)[:, :, :3]), (1, 1))
    out = cf(input_text=tf.constant([_PROMPT]), **{"image/encoded": enc})
    text = out["output_0"].numpy()[0][0].decode()
    strokes = detokenize_strokes(text_to_tokens(text))
    if not strokes:  # fallback prompt
        out = cf(input_text=tf.constant([_FALLBACK_PROMPT]), **{"image/encoded": enc})
        text = out["output_0"].numpy()[0][0].decode()
        strokes = detokenize_strokes(text_to_tokens(text))
    if not rich:
        return strokes_to_points(strokes)
    gray = np.array(img.convert("L"))
    _, ink = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    return strokes_to_points_rich(strokes, gray, dt)


def vectorize_boxes_inksight(
    pdf_path: str,
    boxes: list[dict],
    page_index: int,
    dpi: int = 600,
    padding: int | None = None,
    pad_frac: float | None = None,
    fit_ink: bool | None = None,
    limit: int | None = None,
    rich: bool = False,
) -> list[dict]:
    """Add ``{points, metadata}`` to each box via InkSight derendering, in place.

    Crops are masked to the target word (``mask_to_box=True``) so the model sees one
    word. ``rich`` adds width+intensity channels (see ``strokes_to_points_rich``).
    Returns ``boxes`` (same schema as ``vectorize.vectorize_boxes``).
    """
    padding = config.CROP_PADDING if padding is None else padding
    pad_frac = config.CROP_PAD_FRAC if pad_frac is None else pad_frac
    fit_ink = config.CROP_FIT_INK if fit_ink is None else fit_ink
    page = load_page(pdf_path, page_index, dpi=dpi)
    processed = 0
    for entry in boxes if limit is None else boxes[:limit]:
        if "box_2d" not in entry:
            continue
        crop, _ = crop_to_box(
            page,
            entry["box_2d"],
            padding,
            pad_frac,
            fit_ink,
            mask_to_box=True,
            mask_hpad=config.CROP_MASK_HPAD,
            mask_vpad_up=config.CROP_MASK_VPAD_UP,
            mask_vpad_dn=config.CROP_MASK_VPAD_DN,
        )
        points = vectorize_crop_inksight(crop, rich=rich)
        entry["points"] = points
        entry["metadata"] = {
            "author": "robot",
            "asciiSequence": entry.get("text", ""),
            "pointCount": len(points),
            "strokeCount": sum(1 for p in points if p[2] == 0),
            "aspectRatio": round(crop.size[0] / max(1, crop.size[1]), 4),
            "vectorizer": "inksight",
            "channels": "xypwi" if rich else "xyp",  # xyp=[x,y,pen]; xypwi adds width,intensity
        }
        processed += 1
        total = len(boxes) if limit is None else limit
        print(f"Derendered {processed}/{total}: {entry.get('text', '')!r}")
    print(f"Derendered {processed} words with InkSight.")
    return boxes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorize OCR boxes with InkSight derendering")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=1, help="Page number, 1-based")
    p.add_argument("--boxes", default=None, help="Boxes JSON (default: canonical for --pdf/--page)")
    p.add_argument("--output", default=None, help="Output strokes.json (default: canonical path)")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI")
    p.add_argument("--limit", type=int, default=None, help="Only the first N boxes")
    p.add_argument(
        "--rich",
        action="store_true",
        help="Emit [x,y,pen,width,intensity] (thickness+faintness) instead of [x,y,pen]",
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
    vectorize_boxes_inksight(
        args.pdf, boxes, args.page - 1, dpi=args.dpi, limit=args.limit, rich=args.rich
    )
    out = args.output or paths.strokes_json(args.pdf, args.page, version, root)
    paths.ensure_parent(out)
    with open(out, "w") as f:
        json.dump(boxes, f, indent=4)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
