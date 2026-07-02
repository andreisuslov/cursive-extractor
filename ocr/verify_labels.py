"""Label read-back gate: re-read clipped strokes with Gemini, compare to the label.

WHY: detection boxes are sometimes offset by a whole word, so a *clean* derender
can be a perfectly-traced NEIGHBOUR word silently mislabeled. AIoU QA
(``ocr.ink_qa``) scores stroke geometry against the ink inside the box region,
so it cannot see that failure -- the strokes fit *some* ink perfectly. The only
check that catches it is reading the strokes back: render each word's clipped
strokes black-on-white, ask Gemini to transcribe them, and compare to the label.
Words are packed ~20 to a numbered grid image to keep API calls low; matching is
fuzzy (case-insensitive exact, or difflib ratio >= 0.7) because handwriting OCR
is fuzzy.

Input is the ``_strokes_qa.json`` that ``ocr.ink_qa`` writes (clipped points,
``metadata.qa``). Only words with ``qa >= --min-qa`` and nonempty points are
checked -- below that the strokes are too broken for a read-back to mean
anything. Each checked entry's ``metadata`` gains
``{"label_ok": bool, "label_read": "<gemini transcription>"}`` in place and the
json is rewritten.

    envchain gemini python3 -m ocr.verify_labels --pdf <pdf> --page N
        [--version V] [--strokes PATH] [--min-qa 0.85] [--limit K]
        [--dry-run] [--grid-dir DIR]
"""

import argparse
import difflib
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

from . import config, paths
from .gemini_ocr import _generate, _strip_fences, build_model
from .ink_qa import QA_SUFFIX

# Grid geometry: 4x5 = 20 cells per image, each cell one word.
CELLS_PER_IMAGE = 20
GRID_COLS = 4
CELL_W = 380
CELL_H = 200  # number strip + ink area
INK_H = 150  # nominal word height inside a cell
CELL_PAD = 12
STROKE_WIDTH = 3

MIN_QA = 0.85
MATCH_RATIO = 0.7

READBACK_PROMPT = """
Each numbered cell in this image contains ONE handwritten word rendered as black
ink on white. The red number at the top-left of each cell is the CELL NUMBER --
it is NOT part of the word. Read the handwriting in each cell and return a raw
JSON object mapping cell number to your transcription, e.g.
{"3": "wealth.", "4": "Uncle"}. Include EVERY cell number shown; if a cell is
unreadable, map it to "". Output ONLY the JSON object.
"""


# --- rendering ---------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: unsized bitmap font
        return ImageFont.load_default()


def draw_word(
    draw: ImageDraw.ImageDraw,
    points: list[list[float]],
    aspect: float,
    area: tuple[int, int, int, int],
) -> None:
    """Draw one word's strokes ([x,y,pen,...] rows normalized [0,1] per axis,
    pen=0 ends a stroke) into ``area``, un-distorting via ``aspect`` (the source
    rect's width/height) and centering."""
    ax0, ay0, ax1, ay1 = area
    avail_w, avail_h = ax1 - ax0, ay1 - ay0
    aspect = min(max(aspect, 0.2), 12.0)
    h = min(avail_h, INK_H, avail_w / aspect)
    w = aspect * h
    ox = ax0 + (avail_w - w) / 2
    oy = ay0 + (avail_h - h) / 2

    poly: list[tuple[float, float]] = []
    for row in points:
        poly.append((ox + row[0] * w, oy + row[1] * h))
        if row[2] == 0:
            if len(poly) >= 2:
                draw.line(poly, fill="black", width=STROKE_WIDTH, joint="curve")
            elif len(poly) == 1:
                x, y = poly[0]
                r = STROKE_WIDTH / 2
                draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
            poly = []
    if len(poly) >= 2:
        draw.line(poly, fill="black", width=STROKE_WIDTH, joint="curve")


def render_grid(cells: list[tuple[int, dict]]) -> Image.Image:
    """Render ``[(cell_number, entry), ...]`` as one numbered grid image."""
    rows = (len(cells) + GRID_COLS - 1) // GRID_COLS
    img = Image.new("RGB", (GRID_COLS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(img)
    font = _font(26)
    for k, (number, entry) in enumerate(cells):
        cx = (k % GRID_COLS) * CELL_W
        cy = (k // GRID_COLS) * CELL_H
        draw.rectangle([cx, cy, cx + CELL_W - 1, cy + CELL_H - 1], outline=(200, 200, 200))
        draw.text((cx + 8, cy + 4), str(number), fill=(200, 30, 30), font=font)
        aspect = float(entry.get("metadata", {}).get("aspectRatio") or 2.0)
        area = (cx + CELL_PAD, cy + 36, cx + CELL_W - CELL_PAD, cy + CELL_H - CELL_PAD)
        draw_word(draw, entry["points"], aspect, area)
    return img


# --- read-back + matching ----------------------------------------------------


def parse_readback(text: str) -> dict[int, str]:
    """Parse a ``{cell_number: transcription}`` response tolerantly."""
    text = _strip_fences(text)
    out: dict[int, str] = {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        out.setdefault(int(m.group(1)), json.loads(f'"{m.group(2)}"'))
    return out


def labels_match(label: str, read: str, ratio: float = MATCH_RATIO) -> bool:
    """Case-insensitive exact, or difflib similarity >= ``ratio``."""
    a, b = label.strip().lower(), read.strip().lower()
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


# --- CLI ---------------------------------------------------------------------


def default_qa_path(pdf_path: str, page: int, version: int | None, root: str | None = None) -> str:
    """The page's ``<prefix>_strokes_qa.json`` (what ``ocr.ink_qa`` writes)."""
    return os.path.join(
        paths.page_dir(pdf_path, page, version, root),
        f"{paths.prefix(pdf_path, page, version)}{QA_SUFFIX}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read clipped strokes back with Gemini vs the label")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=1, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--strokes", default=None, help="Strokes-QA json (default: canonical)")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument("--min-qa", type=float, default=MIN_QA, help="Only check words with qa >= this")
    p.add_argument("--limit", type=int, default=None, help="Only the first N eligible words")
    p.add_argument(
        "--dry-run", action="store_true", help="Render the grid images only; no API call"
    )
    p.add_argument(
        "--grid-dir", default=None, help="Where --dry-run saves grids (default: the page folder)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    qa_path = args.strokes
    if qa_path is None:
        version = (
            args.version
            if args.version is not None
            else paths.latest_version(args.pdf, args.page, args.output_root)
        )
        if version is None:
            raise SystemExit("No processed version found; pass --strokes or run the pipeline.")
        qa_path = default_qa_path(args.pdf, args.page, version, args.output_root)
    with open(qa_path) as f:
        entries = json.load(f)

    eligible = [
        e
        for e in entries
        if e.get("points") and e.get("metadata", {}).get("qa", 0.0) >= args.min_qa
    ]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        raise SystemExit(f"No words with qa >= {args.min_qa} and nonempty points in {qa_path}")
    print(f"{len(eligible)}/{len(entries)} words eligible (qa >= {args.min_qa})")

    # Global 1-based cell numbers so every cell across all grids is unique.
    numbered = list(enumerate(eligible, start=1))
    grids = [
        render_grid(numbered[g : g + CELLS_PER_IMAGE])
        for g in range(0, len(numbered), CELLS_PER_IMAGE)
    ]

    if args.dry_run:
        grid_dir = args.grid_dir or os.path.dirname(qa_path)
        paths.ensure_dir(grid_dir)
        stem = os.path.splitext(os.path.basename(qa_path))[0]
        for g, img in enumerate(grids):
            out = os.path.join(grid_dir, f"{stem}_label_grid_{g:02d}.png")
            img.save(out)
            print(f"[dry-run] wrote {out}")
        return

    model = build_model()
    reads: dict[int, str] = {}
    for g, img in enumerate(grids):
        response = _generate(model, [READBACK_PROMPT, img])
        got = parse_readback(response.text)
        reads.update(got)
        print(f"grid {g + 1}/{len(grids)}: {len(got)} cells read")

    matched = 0
    for number, entry in numbered:
        read = reads.get(number, "")
        ok = labels_match(entry.get("text", ""), read)
        matched += ok
        meta = entry.setdefault("metadata", {})
        meta["label_ok"] = bool(ok)
        meta["label_read"] = read
        if not ok:
            print(f"  MISMATCH cell {number}: label {entry.get('text', '')!r} read {read!r}")

    with open(qa_path, "w") as f:
        json.dump(entries, f, indent=4)
    n = len(eligible)
    print(f"Rewrote {qa_path}")
    print(f"Label match rate: {matched}/{n} = {matched / n:.3f}")


if __name__ == "__main__":
    main()
