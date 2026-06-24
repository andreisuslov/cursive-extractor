"""Hybrid word detection: deterministic LINES + Claude reads each line for words+text.

Pipeline (one CLI):
  1. deterministic line detection (ocr.segment_words: colour-agnostic ink -> RLSA
     line blobs) -- reliable, slant-robust.
  2. for each line crop, ask Claude (vision) for the ordered word tokens + text.
  3. deterministic geometry: cut the line into exactly len(words) pieces at the
     lowest-ink columns (word gaps), and label each piece with Claude's word.
  4. write boxes.json (with text) -- ready for the box editor.

So Claude does what it is good at (reading cursive -> the word COUNT and TEXT) and
pixel geometry does the cutting; no full-page model detection, no Gemini.

    ANTHROPIC_API_KEY=... python -m ocr.label_lines --pdf <pdf> --page N
    envchain clawd python -m ocr.label_lines --pdf <pdf> --page N --overlay /tmp/x.png

The key is read from ANTHROPIC_API_KEY (locally: envchain `clawd` namespace).
"""

import argparse
import base64
import io
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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

PROMPT = (
    "This image is ONE line from a handwritten diary page. Transcribe the HANDWRITING "
    "on this line as word tokens in reading order (left to right). Keep punctuation "
    "attached to its word (e.g. 'good.', \"don't\", '$5'). Ignore any pre-printed form "
    "text, day labels, dates, page numbers, or ruled lines -- only the handwriting. "
    'Return ONLY a JSON array of strings, e.g. ["Went","to","4th","Church"]. '
    "If there is no handwriting on the line, return []."
)


def _claude_words(png: bytes, model: str, key: str) -> list[str]:
    """Ask Claude for the ordered word tokens in a single-line crop."""
    body = {
        "model": model,
        "max_tokens": 400,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(png).decode(),
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        txt = json.load(r)["content"][0]["text"]
    m = re.search(r"\[.*\]", txt, re.S)  # tolerate prose around the JSON
    try:
        words = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        words = []
    return [str(w).strip() for w in words if str(w).strip()]


def _line_png(rgb: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> bytes:
    H, W = rgb.shape[:2]
    crop = rgb[max(0, y - pad) : min(H, y + h + pad), max(0, x - pad) : min(W, x + w + pad)]
    im = Image.fromarray(crop)
    if im.width < 600:  # upscale tiny crops so faint cursive is legible to the model
        s = 600 / im.width
        im = im.resize((600, max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def segment_and_label(rgb: np.ndarray, model: str, workers: int = 6) -> list[dict]:
    """Detect lines, read each with Claude, cut into words, return labelled shapes."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("set ANTHROPIC_API_KEY (locally: run via `envchain clawd ...`)")
    H, W = rgb.shape[:2]
    binv = remove_rules(extract_ink(rgb))
    xh = median_xheight(binv)
    lines = detect_lines(binv, xh)
    pad = max(2, round(0.35 * xh))
    print(f"{len(lines)} lines detected; reading with {model} ...")

    def read(line):
        return _claude_words(_line_png(rgb, *line, pad), model, key)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        line_words = list(ex.map(read, lines))

    shapes = []
    for (x, y, w, h), words in zip(lines, line_words, strict=True):
        if not words:
            continue
        pieces = split_line_n(binv, x, y, w, h, len(words), xh)
        for (px, py, pw, ph), text in zip(pieces, words, strict=False):
            poly = word_polygon(binv, px, py, pw, ph)
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            box = [_q(min(ys), H), _q(min(xs), W), _q(max(ys), H), _q(max(xs), W)]
            shapes.append(
                {"text": text, "box_2d": box, "polygon": [[_q(a, W), _q(b, H)] for a, b in poly]}
            )
    print(f"-> {len(shapes)} word shapes")
    return shapes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Deterministic lines + Claude word/text labelling")
    ap.add_argument("--pdf", default=config.PDF_PATH)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Claude vision model")
    ap.add_argument("--output-root", default=paths.OUTPUT_ROOT)
    ap.add_argument("--out", default=None, help="output path (default: canonical boxes.json)")
    ap.add_argument("--overlay", default=None, help="also write an overlay image here")
    ap.add_argument("--force", action="store_true", help="overwrite an existing boxes.json")
    a = ap.parse_args(argv)
    rgb = np.array(load_page(a.pdf, a.page - 1, dpi=a.dpi).convert("RGB"))
    out = a.out or paths.boxes_json(a.pdf, a.page, None, a.output_root)
    if a.out is None and os.path.exists(out) and not a.force:
        raise SystemExit(f"{out} exists (hand edits?) -- pass --force or --out to a temp path")
    shapes = segment_and_label(rgb, a.model)
    with open(paths.ensure_parent(out), "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"wrote {out}")
    if a.overlay:
        draw_overlay(rgb, shapes).save(a.overlay)
        print(f"overlay -> {a.overlay}")


if __name__ == "__main__":
    main()
