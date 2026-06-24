"""Hybrid word detection: deterministic LINES + a real `claude` agent reads each line.

No API key: the reading is done by a `claude` CLI worker (launched in tmux, using
its own login), driven as two file-based phases with the agent in between:

  export   -- detect lines; write one PNG crop per line + manifest.json + TASK.md
              into a workdir.                                  (deterministic, no model)
  <agent>  -- a `claude` worker reads each crop and writes words.json =
              {"<line_idx>": ["word", ...]} using its own auth.
  assemble -- cut each line into exactly len(words) pieces at the lowest-ink columns,
              label each, write boxes.json (with text).         (deterministic)

    python -m ocr.label_lines --mode export   --pdf <pdf> --page N --workdir /tmp/ocr
    # ... a claude agent in /tmp/ocr reads lines/*.png and writes words.json ...
    python -m ocr.label_lines --mode assemble --pdf <pdf> --page N --workdir /tmp/ocr \
        --out /tmp/boxes.json --overlay /tmp/overlay.png
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from . import config, paths
from .pdf_utils import load_page
from .segment_words import (
    _q,
    detect_lines,
    draw_overlay,
    extract_ink,
    median_xheight,
    remove_rules,
    split_line_n,
    word_polygon,
)

TASK = """# OCR task: read handwriting line crops -> words.json (with word boxes)

This folder has `lines/` with PNG crops (line_000.png, line_001.png, ...) and
`manifest.json` listing them. Each PNG is ONE line of handwriting from a diary page.

For EACH crop listed in manifest.json:
  1. Read the PNG with the Read tool (view the image).
  2. Transcribe the HANDWRITING and locate each word, left to right. Return a JSON
     ARRAY of objects, one per word:
       {"text": "<word>", "box": [x0, y0, x1, y1]}
     where [x0, y0, x1, y1] is the word's bounding box in THIS CROP's own PIXEL
     coordinates (top-left origin; x0<x1, y0<y1). The crop's pixel size is given as
     "crop_size":[width,height] in manifest.json -- your box coords live in that space.
     - Keep punctuation attached to its word: "good.", "don't", "$5".
     - IGNORE pre-printed form text, day labels, dates, page numbers, ruled lines.
     - No handwriting on the line -> empty array [].

Write ALL results to `words.json` in THIS folder: a JSON object mapping each line
index (as a string) to its array of word objects, e.g.
  {"0": [{"text":"Went","box":[5,8,120,70]}, {"text":"to","box":[124,10,180,68]}],
   "1": [{"text":"Mr","box":[3,6,90,72]}],
   "2": []}
Include every line index from the manifest. Write words.json once when done.
"""


def _ink(rgb: np.ndarray):
    binv = remove_rules(extract_ink(rgb))
    xh = median_xheight(binv)
    return binv, xh, detect_lines(binv, xh)


def export_lines(rgb: np.ndarray, workdir: str, dpi: int) -> int:
    """Write per-line PNG crops + manifest.json + TASK.md for a claude worker."""
    _binv, xh, lines = _ink(rgb)
    h_pg, w_pg = rgb.shape[:2]
    ldir = os.path.join(workdir, "lines")
    os.makedirs(ldir, exist_ok=True)
    pad = max(2, round(0.35 * xh))
    man = {"page_w": w_pg, "page_h": h_pg, "dpi": dpi, "lines": []}
    for i, (x, y, w, h) in enumerate(lines):
        cx0, cy0 = max(0, x - pad), max(0, y - pad)
        cx1, cy1 = min(w_pg, x + w + pad), min(h_pg, y + h + pad)
        crop = rgb[cy0:cy1, cx0:cx1]
        im = Image.fromarray(crop)
        if im.width < 600:  # upscale tiny crops so faint cursive is legible
            im = im.resize((600, max(1, round(im.height * 600 / im.width))))
        name = f"lines/line_{i:03d}.png"
        im.save(os.path.join(workdir, name))
        man["lines"].append(
            {
                "idx": i,
                "box": [int(x), int(y), int(w), int(h)],
                "crop": name,
                "crop_box": [int(cx0), int(cy0), int(cx1), int(cy1)],
                "crop_size": [int(im.width), int(im.height)],  # SAVED PNG size (post-upscale)
            }
        )
    with open(os.path.join(workdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(workdir, "TASK.md"), "w") as f:
        f.write(TASK)
    return len(lines)


def assemble(rgb: np.ndarray, workdir: str, out: str) -> list[dict]:
    """Read the worker's words.json and build labelled word shapes -> boxes.json."""
    binv, xh, _ = _ink(rgb)
    h_pg, w_pg = rgb.shape[:2]
    with open(os.path.join(workdir, "manifest.json")) as f:
        man = json.load(f)
    words_map = {}  # merge words.json + any per-worker words_*.json
    for wf in sorted(glob.glob(os.path.join(workdir, "words*.json"))):
        with open(wf) as f:
            words_map.update(json.load(f))
    shapes = []
    for ln in man["lines"]:
        words = words_map.get(str(ln["idx"])) or []
        if not words:
            continue
        if isinstance(words[0], dict):  # agent-box format: {text, box} in crop pixels
            shapes.extend(_shapes_from_boxes(binv, ln, words, w_pg, h_pg))
        else:  # back-compat: list of strings -> valley-split the line into N words
            shapes.extend(_shapes_from_strings(binv, ln, words, xh, w_pg, h_pg))
    with open(paths.ensure_parent(out), "w") as f:
        json.dump(shapes, f, indent=2)
    return shapes


def _shape(binv, px, py, pw, ph, text, w_pg, h_pg) -> dict | None:
    """Build a word shape (text, box_2d, polygon in 0-1000) for a page-pixel rect,
    using word_polygon for a tight outline and falling back to the rect corners."""
    pw, ph = max(1, int(pw)), max(1, int(ph))
    px, py = int(px), int(py)
    poly = word_polygon(binv, px, py, pw, ph)
    if not poly:
        poly = [(px, py), (px + pw, py), (px + pw, py + ph), (px, py + ph)]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    box = [_q(min(ys), h_pg), _q(min(xs), w_pg), _q(max(ys), h_pg), _q(max(xs), w_pg)]
    return {
        "text": text,
        "box_2d": box,
        "polygon": [[_q(a, w_pg), _q(b, h_pg)] for a, b in poly],
    }


def _shapes_from_strings(binv, ln, words, xh, w_pg, h_pg) -> list[dict]:
    x, y, w, h = ln["box"]
    pieces = split_line_n(binv, x, y, w, h, len(words), xh)
    out = []
    for (px, py, pw, ph), text in zip(pieces, words, strict=False):
        sh = _shape(binv, px, py, pw, ph, text, w_pg, h_pg)
        if sh:
            out.append(sh)
    return out


def _shapes_from_boxes(binv, ln, words, w_pg, h_pg) -> list[dict]:
    """Map each agent {text, box} from crop pixels to page pixels and build a shape."""
    cx0, cy0, cx1, cy1 = ln["crop_box"]
    sw, sh = ln["crop_size"]
    sx = (cx1 - cx0) / sw if sw else 1.0
    sy = (cy1 - cy0) / sh if sh else 1.0
    out = []
    for wd in words:
        if not isinstance(wd, dict):  # untrusted LLM JSON: skip junk, don't abort the page
            continue
        b = wd.get("box")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue  # missing/malformed box -> skip this word, keep the rest of the page
        try:
            bx0, by0, bx1, by1 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue  # non-numeric coords
        bx0, bx1 = sorted((bx0, bx1))  # normalize inverted boxes
        by0, by1 = sorted((by0, by1))
        px, py = cx0 + bx0 * sx, cy0 + by0 * sy
        pw, ph = (bx1 - bx0) * sx, (by1 - by0) * sy
        shape = _shape(binv, px, py, pw, ph, wd.get("text", ""), w_pg, h_pg)
        if shape:
            out.append(shape)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Deterministic lines + claude-agent labelling")
    ap.add_argument("--mode", choices=["export", "assemble"], required=True)
    ap.add_argument("--pdf", default=config.PDF_PATH)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--output-root", default=paths.OUTPUT_ROOT)
    ap.add_argument("--out", default=None, help="assemble: output path (default: canonical)")
    ap.add_argument("--overlay", default=None, help="assemble: also write an overlay image")
    ap.add_argument("--force", action="store_true", help="assemble: overwrite existing boxes.json")
    a = ap.parse_args(argv)
    rgb = np.array(load_page(a.pdf, a.page - 1, dpi=a.dpi).convert("RGB"))
    if a.mode == "export":
        os.makedirs(a.workdir, exist_ok=True)
        n = export_lines(rgb, a.workdir, a.dpi)
        print(f"exported {n} line crops -> {a.workdir}/lines  (+ manifest.json, TASK.md)")
        return
    out = a.out or paths.boxes_json(a.pdf, a.page, None, a.output_root)
    if a.out is None and os.path.exists(out) and not a.force:
        raise SystemExit(f"{out} exists (hand edits?) -- pass --force or --out to a temp path")
    shapes = assemble(rgb, a.workdir, out)
    print(f"assembled {len(shapes)} word shapes -> {out}")
    if a.overlay:
        draw_overlay(rgb, shapes).save(a.overlay)
        print(f"overlay -> {a.overlay}")


if __name__ == "__main__":
    main()
