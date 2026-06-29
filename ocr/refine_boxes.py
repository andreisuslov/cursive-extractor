"""Refine loose word boxes into a clean, non-overlapping LAYOUT PARTITION.

Gemini grounded detection gives one box per word in reading order, but on dense cursive the
boxes are loose and overlap their neighbours and -- worse -- the line above/below. That vertical
bleed is the documented P1 failure (WORKLOG): per-box `fit_crop_to_ink`/`mask_crop_to_box` can't
fix it when the *box itself* dips into the next line, because they only look inside one box.

This fixes it GLOBALLY using all boxes at once: cluster them into columns (two-up scans) and rows,
then give every word a cell bounded by its row band (vertically) and the midpoints to its row
neighbours (horizontally). Cells are non-overlapping by construction, so a crop physically can't
reach into the next line. Downstream `fit_crop_to_ink` / `mask_crop_to_box` then tighten to the
actual ink *within* the clean cell. Image-free, deterministic — operates on the 0-1000 box_2d.

    python -m ocr.refine_boxes --boxes boxes.json --out refined.json --page page.png --tighten
"""

import argparse
import itertools
import json

import numpy as np


def _yc(b):
    return (b["box_2d"][0] + b["box_2d"][2]) / 2


def _xc(b):
    return (b["box_2d"][1] + b["box_2d"][3]) / 2


def split_columns(xcs: list[float], min_gap: float = 120.0):
    """Detect a two-up gutter: the widest empty vertical band near the page middle (0-1000
    units). Returns the split x, or None for a single column."""
    xs = sorted(xcs)
    best_gap, best_at = 0.0, None
    for a, b in itertools.pairwise(xs):
        mid = (a + b) / 2
        if 300 < mid < 700 and (b - a) > best_gap:
            best_gap, best_at = b - a, mid
    return best_at if best_gap > min_gap else None


def cluster_rows(boxes: list[dict], med_h: float, frac: float = 0.6) -> list[list[dict]]:
    """Greedy 1-D clustering of boxes into rows by y-center (new row when the center jumps more
    than ``frac`` of the median box height past the current row's mean)."""
    items = sorted(boxes, key=_yc)
    rows, cur = [], [items[0]]
    for b in items[1:]:
        if _yc(b) - float(np.mean([_yc(q) for q in cur])) > frac * med_h:
            rows.append(cur)
            cur = [b]
        else:
            cur.append(b)
    rows.append(cur)
    return rows


def refine_boxes(
    boxes: list[dict],
    row_frac: float = 0.6,
    side_pad_frac: float = 0.3,
    band_pad_frac: float = 0.7,
) -> list[dict]:
    """Partition loose boxes into non-overlapping per-word cells. Preserves order, ``text`` and
    any extra keys; only ``box_2d`` (0-1000 ``[ymin,xmin,ymax,xmax]``) is rewritten."""
    if not boxes:
        return boxes
    order = {id(b): i for i, b in enumerate(boxes)}
    med_h = float(np.median([b["box_2d"][2] - b["box_2d"][0] for b in boxes])) or 1.0
    split = split_columns([_xc(b) for b in boxes])

    cols: dict[int, list[dict]] = {}
    for b in boxes:
        cols.setdefault(0 if (split is None or _xc(b) < split) else 1, []).append(b)

    out: list[dict | None] = [None] * len(boxes)
    for items in cols.values():
        rows = cluster_rows(items, med_h, row_frac)
        ryc = [float(np.mean([_yc(b) for b in r])) for r in rows]
        for ri, row in enumerate(rows):
            row = sorted(row, key=_xc)
            top = (ryc[ri - 1] + ryc[ri]) / 2 if ri > 0 else ryc[ri] - band_pad_frac * med_h
            bot = (
                (ryc[ri] + ryc[ri + 1]) / 2
                if ri < len(rows) - 1
                else ryc[ri] + band_pad_frac * med_h
            )
            for wi, b in enumerate(row):
                left = (
                    (_xc(row[wi - 1]) + _xc(b)) / 2
                    if wi > 0
                    else b["box_2d"][1] - side_pad_frac * med_h
                )
                right = (
                    (_xc(b) + _xc(row[wi + 1])) / 2
                    if wi < len(row) - 1
                    else b["box_2d"][3] + side_pad_frac * med_h
                )
                nb = dict(b)
                nb["box_2d"] = [
                    round(max(0.0, top)),
                    round(max(0.0, left)),
                    round(min(1000.0, bot)),
                    round(min(1000.0, right)),
                ]
                out[order[id(b)]] = nb
    return [b for b in out if b is not None]


def tighten_to_ink(page_image, box_2d: list[int], margin_frac: float = 0.06) -> list[int]:
    """Tighten a cell to the ink bounding box INSIDE it -- never expands beyond the cell, so the
    partition's no-bleed guarantee holds (unlike `fit_crop_to_ink`, which re-expands via connected
    components and grabs whole multi-line blobs on dense cursive). Returns box_2d (0-1000); falls
    back to the input cell if no ink is found."""
    import cv2
    import numpy as np

    w, h = page_image.size
    ymin, xmin, ymax, xmax = box_2d
    left, top = int(xmin * w / 1000), int(ymin * h / 1000)
    right, bottom = int(xmax * w / 1000), int(ymax * h / 1000)
    if right - left < 2 or bottom - top < 2:
        return box_2d
    crop = np.array(page_image.crop((left, top, right, bottom)).convert("L"))
    _t, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return box_2d
    ch, cw = crop.shape
    mx, my = margin_frac * cw, margin_frac * ch
    x0 = max(0, xs.min() - mx) + left
    x1 = min(cw, xs.max() + mx) + left
    y0 = max(0, ys.min() - my) + top
    y1 = min(ch, ys.max() + my) + top
    return [round(y0 * 1000 / h), round(x0 * 1000 / w), round(y1 * 1000 / h), round(x1 * 1000 / w)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Refine loose word boxes into a layout partition")
    p.add_argument("--boxes", required=True, help="boxes.json from detection")
    p.add_argument("--out", required=True, help="refined boxes.json")
    p.add_argument("--page", default=None, help="page image (for --tighten / --overlay)")
    p.add_argument("--tighten", action="store_true", help="tighten cells to ink (needs --page)")
    p.add_argument("--overlay", default=None, help="write a verification overlay JPG here")
    args = p.parse_args(argv)

    with open(args.boxes) as f:
        boxes = json.load(f)
    refined = refine_boxes(boxes)
    if args.tighten and args.page:
        from PIL import Image

        page = Image.open(args.page).convert("RGB")
        for b in refined:
            b["box_2d"] = tighten_to_ink(page, b["box_2d"])
    with open(args.out, "w") as f:
        json.dump(refined, f)
    print(f"{len(boxes)} boxes -> {len(refined)} refined cells -> {args.out}")

    if args.overlay and args.page:
        from PIL import Image, ImageDraw

        page = Image.open(args.page).convert("RGB")
        w, h = page.size
        d = ImageDraw.Draw(page)
        for b in refined:
            ymin, xmin, ymax, xmax = b["box_2d"]
            d.rectangle(
                [xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000],
                outline=(0, 140, 0),
                width=3,
            )
        page.save(args.overlay, quality=70)
        print(f"overlay -> {args.overlay}")


if __name__ == "__main__":
    main()
