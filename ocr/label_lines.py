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

TASK = """# OCR task: read handwriting line crops -> words.json

This folder has `lines/` with PNG crops (line_000.png, line_001.png, ...) and
`manifest.json` listing them. Each PNG is ONE line of handwriting from a diary page.

For EACH crop listed in manifest.json:
  1. Read the PNG with the Read tool (view the image).
  2. Transcribe the HANDWRITING as word tokens, left to right.
     - Keep punctuation attached to its word: "good.", "don't", "$5".
     - IGNORE pre-printed form text, day labels, dates, page numbers, ruled lines.
     - No handwriting on the line -> empty list.

Write ALL results to `words.json` in THIS folder: a JSON object mapping each line
index (as a string) to its array of word-token strings, e.g.
  {"0": ["Went","to","4th","Church"], "1": ["Mr","Homick's","picked"], "2": []}
Include every line index from the manifest. Write words.json once when done.
"""


def _ink(rgb: np.ndarray):
    binv = remove_rules(extract_ink(rgb))
    xh = median_xheight(binv)
    return binv, xh, detect_lines(binv, xh)


def export_lines(rgb: np.ndarray, workdir: str, dpi: int) -> int:
    """Write per-line PNG crops + manifest.json + TASK.md for a claude worker."""
    binv, xh, lines = _ink(rgb)
    h_pg, w_pg = rgb.shape[:2]
    ldir = os.path.join(workdir, "lines")
    os.makedirs(ldir, exist_ok=True)
    pad = max(2, round(0.35 * xh))
    man = {"page_w": w_pg, "page_h": h_pg, "dpi": dpi, "lines": []}
    for i, (x, y, w, h) in enumerate(lines):
        crop = rgb[
            max(0, y - pad) : min(h_pg, y + h + pad), max(0, x - pad) : min(w_pg, x + w + pad)
        ]
        im = Image.fromarray(crop)
        if im.width < 600:  # upscale tiny crops so faint cursive is legible
            im = im.resize((600, max(1, round(im.height * 600 / im.width))))
        name = f"lines/line_{i:03d}.png"
        im.save(os.path.join(workdir, name))
        man["lines"].append({"idx": i, "box": [int(x), int(y), int(w), int(h)], "crop": name})
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
        x, y, w, h = ln["box"]
        pieces = split_line_n(binv, x, y, w, h, len(words), xh)
        for (px, py, pw, ph), text in zip(pieces, words, strict=False):
            poly = word_polygon(binv, px, py, pw, ph)
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            box = [_q(min(ys), h_pg), _q(min(xs), w_pg), _q(max(ys), h_pg), _q(max(xs), w_pg)]
            shapes.append(
                {
                    "text": text,
                    "box_2d": box,
                    "polygon": [[_q(a, w_pg), _q(b, h_pg)] for a, b in poly],
                }
            )
    with open(paths.ensure_parent(out), "w") as f:
        json.dump(shapes, f, indent=2)
    return shapes


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
