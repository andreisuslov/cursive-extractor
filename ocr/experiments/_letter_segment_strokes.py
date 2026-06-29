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

import numpy as np


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
    _fig, axes = plt.subplots(rows, 1, figsize=(7, 1.8 * rows))
    if rows == 1:
        axes = [axes]
    cmap = plt.colormaps["tab10"]
    qsum = 0.0
    for ax, b in zip(axes, words, strict=False):
        text = b["metadata"]["asciiSequence"]
        letters = segment_word_strokes(b["points"], len(text))
        qsum += cut_quality(letters)
        for k, seg in enumerate(letters):
            if seg:
                a = np.array(seg, float)
                ax.scatter(a[:, 0], -a[:, 1], s=4, color=cmap(k % 10))
        ax.set_title(f'"{text}"  ({len(text)} letters)', fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, facecolor="white")
    print(f"mean cut_quality (non-empty bins): {qsum / max(1, rows):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
