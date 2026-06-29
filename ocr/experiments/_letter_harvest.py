"""EXPERIMENTAL (#2): confidence-gated per-letter harvester.

Letter cutting can't be solved for *every* word (cursive overlaps; m/n/u/w have internal
valleys). But the FONT goal doesn't need every word — it needs >=3 CONFIDENT samples of each
letter. So: cut each word (``_letter_segment_strokes.trajectory_cuts``), score how trustworthy
the cut is, keep only the confident ones, and accumulate per-letter glyphs until coverage is
met. Turns "segment everything" (unsolved) into "collect the easy wins" (tractable), and feeds
N3 (variant clustering) / N4 (font).

Confidence (0-1): 0 if any letter slice is empty; else a blend of (a) the fraction of cuts that
SNAPPED to real pen structure vs fell back to the width prior, and (b) width balance (no letter
absurdly wide/narrow). Harvested glyphs are normalized to their own bbox so they're comparable.

    python -m ocr.experiments._letter_harvest --strokes <strokes.json> [--threshold 0.6]
"""

import argparse
import json
import re

import numpy as np

from . import _letter_segment_strokes as ls


def normalize_letter(seg_pts: list[list[float]]) -> list[list[float]]:
    """A letter's points normalized to its own [0,1] bounding box (x, y only)."""
    a = np.array([[p[0], p[1]] for p in seg_pts], float)
    mn, mx = a.min(0), a.max(0)
    span = np.maximum(mx - mn, 1e-9)
    return [
        [round(float((p[0] - mn[0]) / span[0]), 4), round(float((p[1] - mn[1]) / span[1]), 4)]
        for p in seg_pts
    ]


def confidence(letters: list[list[list[float]]], snapped: list[bool]) -> float:
    """How trustworthy is this word's segmentation? 0 if any letter empty."""
    if not letters or any(len(s) == 0 for s in letters):
        return 0.0
    snap_frac = (sum(snapped) / len(snapped)) if snapped else 1.0  # 1-letter word: trivially fine
    widths = [max(p[0] for p in s) - min(p[0] for p in s) for s in letters]
    mean_w = sum(widths) / len(widths)
    if mean_w <= 0:
        return 0.0
    band = sum(1 for w in widths if 0.3 * mean_w <= w <= 3.0 * mean_w) / len(widths)
    return round(0.6 * snap_frac + 0.4 * band, 3)


def segment_with_confidence(points: list[list[float]], text: str):
    """``(letters, confidence)`` for a word's trajectory + transcription."""
    letters = ls.trajectory_cut_word(points, text)
    if len(text) <= 1:
        return letters, (1.0 if letters and letters[0] else 0.0)
    if len(ls._down_points(points)) < len(text):
        return letters, 0.0
    _cuts, snapped = ls.trajectory_cuts(points, text)
    return letters, confidence(letters, snapped)


def harvest(strokes_path: str, threshold: float = 0.6):
    """Accumulate confident per-letter glyphs from a page's strokes.json.

    Returns ``(library, stats)`` where ``library`` maps each char to a list of normalized
    glyph point-lists, and ``stats`` reports words seen/kept.
    """
    with open(strokes_path) as f:
        data = [b for b in json.load(f) if b.get("points")]
    library: dict[str, list[list[list[float]]]] = {}
    total = kept = 0
    for b in data:
        text = b["metadata"]["asciiSequence"]
        if not re.fullmatch(r"[A-Za-z]+", text):  # alphabetic words only
            continue
        total += 1
        letters, conf = segment_with_confidence(b["points"], text)
        if conf < threshold:
            continue
        kept += 1
        for ch, seg_pts in zip(text, letters, strict=False):
            if seg_pts:
                library.setdefault(ch, []).append(normalize_letter(seg_pts))
    return library, {"total_words": total, "kept_words": kept}


def coverage_report(library: dict[str, list], need: int = 3) -> str:
    """Per-letter counts + how many distinct letters reach ``need`` samples."""
    counts = {c: len(v) for c, v in sorted(library.items())}
    have = sum(1 for n in counts.values() if n >= need)
    lines = [f"letters with >={need} samples: {have}/{len(counts)} distinct seen"]
    lines.append("counts: " + ", ".join(f"{c}:{n}" for c, n in counts.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Confidence-gated per-letter harvester")
    p.add_argument("--strokes", required=True, help="strokes.json from inksight_vectorize")
    p.add_argument("--threshold", type=float, default=0.6, help="min cut confidence to keep a word")
    p.add_argument("--need", type=int, default=3, help="samples per letter the font wants")
    p.add_argument("--out", default=None, help="Save the harvested library JSON here")
    p.add_argument("--render", default=None, help="Render harvested samples of a few letters here")
    args = p.parse_args(argv)

    library, stats = harvest(args.strokes, args.threshold)
    print(
        f"words: {stats['total_words']} seen, {stats['kept_words']} kept (conf>={args.threshold})"
    )
    print(coverage_report(library, args.need))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(library, f)
        print(f"wrote {args.out}")
    if args.render:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # render up to 6 of the best-covered letters, up to 6 samples each
        chars = [c for c, _ in sorted(library.items(), key=lambda kv: -len(kv[1]))[:6]]
        _fig, axes = plt.subplots(len(chars), 6, figsize=(10, 1.6 * max(1, len(chars))))
        axes = np.atleast_2d(axes)
        for r, ch in enumerate(chars):
            for col in range(6):
                ax = axes[r, col]
                ax.axis("off")
                if col < len(library[ch]):
                    a = np.array(library[ch][col], float)
                    ax.scatter(a[:, 0], -a[:, 1], s=3, color="black")
                    ax.set_aspect("equal")
                if col == 0:
                    ax.set_title(f"'{ch}' (n={len(library[ch])})", fontsize=9, loc="left")
        plt.tight_layout()
        plt.savefig(args.render, dpi=110, facecolor="white")
        print(f"wrote {args.render}")


if __name__ == "__main__":
    main()
