"""EXPERIMENTAL (#2): cut a clean InkSight word TRAJECTORY into its labelled letters.

The M10-M13 segmenters cut the raster INK (geometry/recognition on image slices). Now
that InkSight gives a clean ordered pen path per word AND we know the transcription, the
problem becomes 1-D: place L-1 cuts along the left-to-right trajectory so the L pieces are
the L letters. This module starts that with a baseline + an eval/visual harness, so later
smarter cutters (recognizer forced-alignment on the path) are measured against it.

Input is the strokes.json the InkSight vectorizer writes (points = [x, y, pen, ...]; pen at
index 2; pen-up markers have pen==0). Output is L lists of points, one per letter.

    python -m ocr.experiments._letter_segment_strokes --strokes <strokes.json> [--n 12]

This is a BASELINE (equal-width x bins): honest and crude -- cursive letters vary in width
and connect, so equal bins mis-cut. It exists to anchor the metric and the render; the next
step is width-prior / recognizer-guided cuts (reusing ocr.experiments._recognizer).
"""

import argparse
import json
import re

import cv2
import numpy as np

from . import _recognizer


def _down_points(points: list[list[float]]) -> list[list[float]]:
    """Pen-down points only (drop the pen-up=0 stroke-end markers)."""
    return [p for p in points if len(p) >= 3 and p[2] == 1]


def segment_word_strokes(points: list[list[float]], n_letters: int) -> list[list[list[float]]]:
    """Split a word's trajectory into ``n_letters`` letter-point-lists by x position.

    BASELINE: equal-width bins over the ink's x-range. Each pen-down point is assigned to a
    letter by its x. Returns ``n_letters`` lists (some may be empty for a bad split). This is
    deliberately simple -- the harness/metric matters more than this cut for now.
    """
    pts = _down_points(points)
    if not pts or n_letters < 1:
        return [pts]
    xs = [p[0] for p in pts]
    x0, x1 = min(xs), max(xs)
    span = max(x1 - x0, 1e-9)
    letters: list[list[list[float]]] = [[] for _ in range(n_letters)]
    for p in pts:
        k = min(n_letters - 1, int((p[0] - x0) / span * n_letters))
        letters[k].append(p)
    return letters


def _rasterize(points: list[list[float]], height: int = 48):
    """Draw a word's pen-down trajectory to a binary image for the recognizer.

    Returns ``(binary HxW uint8, x0, x1, W)`` where x0/x1 are the normalized-x bounds
    (to map column cuts back to trajectory x), or ``None`` if too few points.
    """
    pts = _down_points(points)
    if len(pts) < 2:
        return None
    arr = np.array([[p[0], p[1]] for p in pts], float)
    x0, y0 = arr.min(0)
    x1, y1 = arr.max(0)
    sx, sy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    w = max(height, min(round(height * sx / sy), 8 * height))
    img = np.zeros((height, w), np.uint8)
    prev = None
    for p in points:  # walk in order; break the line at pen-up markers
        if len(p) >= 3 and p[2] == 1:
            px = int((p[0] - x0) / sx * (w - 1))
            py = int((p[1] - y0) / sy * (height - 1))
            if prev is not None:
                cv2.line(img, prev, (px, py), 255, 2)
            prev = (px, py)
        else:
            prev = None
    return img, float(x0), float(x1), w


def forced_align_word_strokes(
    points: list[list[float]], text: str, recognizer=None
) -> list[list[list[float]]]:
    """Cut a word's trajectory into ``len(text)`` letters by recognizer forced alignment.

    Rasterizes the path, runs ``_recognizer.align_boundaries`` (recognizer score for the
    KNOWN letter sequence + per-letter width prior; geometry term off, ``beta=0``), then
    maps the chosen column cuts back to split the trajectory points. Falls back to the
    equal-x baseline when there's nothing to align.
    """
    pts = _down_points(points)
    L = len(text)
    r = _rasterize(points)
    if r is None or L <= 1:
        return segment_word_strokes(points, max(1, L))
    img, x0, x1, w = r
    rec = recognizer or _recognizer.get_recognizer()
    cand = np.arange(w, dtype=float)
    geom = np.zeros(w, dtype=float)
    bpx = _recognizer.align_boundaries(
        img, text, 0.0, float(w - 1), cand, geom, recognizer=rec, beta=0.0
    )
    sx = max(x1 - x0, 1e-9)
    xb = sorted(x0 + (b / max(1, w - 1)) * sx for b in bpx)
    letters: list[list[list[float]]] = [[] for _ in range(L)]
    for p in pts:
        k = min(sum(1 for b in xb if p[0] >= b), L - 1)
        letters[k].append(p)
    return letters


def cut_quality(letters: list[list[list[float]]]) -> float:
    """Crude 0-1 proxy: fraction of letter-bins that are non-empty (a real cut should give
    every letter some ink). NOT a recognition metric -- a placeholder until the recognizer
    judges per-letter correctness."""
    if not letters:
        return 0.0
    return sum(1 for s in letters if s) / len(letters)


def _load_words(strokes_path: str) -> list[dict]:
    """Alphabetic words from a strokes.json, cleanest first (low strokes-per-char proxy)."""
    with open(strokes_path) as f:
        data = [b for b in json.load(f) if b.get("points")]
    words = [b for b in data if re.fullmatch(r"[A-Za-z]{2,}", b["metadata"]["asciiSequence"])]

    def spc(b):
        return b["metadata"]["strokeCount"] / max(1, len(b["metadata"]["asciiSequence"]))

    return sorted(words, key=spc)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Baseline letter segmentation of InkSight word strokes")
    p.add_argument("--strokes", required=True, help="strokes.json from inksight_vectorize")
    p.add_argument("--n", type=int, default=12, help="How many (cleanest) words to render")
    p.add_argument("--out", default="letter_segment.png", help="Output overlay image")
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    words = _load_words(args.strokes)[: args.n]
    rows = len(words)
    rec = _recognizer.get_recognizer()
    _fig, axes = plt.subplots(rows, 2, figsize=(12, 1.8 * rows))
    if rows == 1:
        axes = axes.reshape(1, 2)
    cmap = plt.colormaps["tab10"]

    def draw(ax, letters, title):
        for k, seg in enumerate(letters):
            if seg:
                a = np.array(seg, float)
                ax.scatter(a[:, 0], -a[:, 1], s=4, color=cmap(k % 10))
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    for i, b in enumerate(words):
        text = b["metadata"]["asciiSequence"]
        draw(axes[i, 0], segment_word_strokes(b["points"], len(text)), f'baseline "{text}"')
        draw(
            axes[i, 1], forced_align_word_strokes(b["points"], text, rec), f'forced-align "{text}"'
        )
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
