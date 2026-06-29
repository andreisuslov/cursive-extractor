"""Human-in-the-loop letter-cut review — the realistic path to a CLEAN alphabet.

Automatic cutting of connected cursive is unsolved (see WORKLOG), but auto-cuts are a good
*starting point* a human can fix fast. This is the data layer:

  - ``export_for_review(strokes.json)`` -> a compact review doc (per word: trajectory points +
    auto-cut x-positions) that the browser tool ``letter_review.html`` renders with draggable
    cut handles.
  - ``ingest_review(corrected)`` -> a clean per-letter library (char -> normalized glyphs) by
    splitting each word at the human-confirmed cuts. Feeds N3 (``_variant_cluster``) directly.

Round-trippable and testable without the UI: export -> ingest reproduces letters at the auto
cuts; the human only nudges the few wrong ones.

    python -m ocr.experiments._letter_review export --strokes <strokes.json> --out review.json
    # ... edit cuts in letter_review.html, save corrected.json ...
    python -m ocr.experiments._letter_review ingest --review corrected.json --out lib.json
"""

import argparse
import json
import re

from ._letter_harvest import normalize_letter
from ._letter_segment_strokes import _down_points, split_by_cuts, trajectory_cuts


def export_for_review(strokes_path: str) -> list[dict]:
    """Per alphabetic word: ``{text, points [[x,y],...], cuts [x,...]}`` (auto-cuts as a start)."""
    with open(strokes_path) as f:
        data = [b for b in json.load(f) if b.get("points")]
    out = []
    for b in data:
        text = b["metadata"]["asciiSequence"]
        if not re.fullmatch(r"[A-Za-z]+", text):
            continue
        pts = _down_points(b["points"])
        if len(pts) < len(text):
            continue
        cuts, _snapped = trajectory_cuts(b["points"], text)
        out.append(
            {
                "text": text,
                "points": [[round(p[0], 4), round(p[1], 4)] for p in pts],
                "cuts": [round(float(c), 4) for c in cuts],
            }
        )
    return out


def ingest_review(review: list[dict]) -> dict[str, list]:
    """Split each reviewed word at its (human-confirmed) cuts -> char -> [normalized glyphs]."""
    library: dict[str, list] = {}
    for item in review:
        text = item["text"]
        letters = split_by_cuts(item["points"], item.get("cuts", []), len(text))
        for ch, seg in zip(text, letters, strict=False):
            if seg:
                library.setdefault(ch, []).append(normalize_letter(seg))
    return library


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export/ingest human letter-cut review")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="strokes.json -> review.json (auto-cuts to fix)")
    e.add_argument("--strokes", required=True)
    e.add_argument("--out", required=True)
    i = sub.add_parser("ingest", help="corrected review.json -> letter library.json")
    i.add_argument("--review", required=True)
    i.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.cmd == "export":
        review = export_for_review(args.strokes)
        with open(args.out, "w") as f:
            json.dump(review, f)
        print(f"{len(review)} words -> {args.out}")
    else:
        with open(args.review) as f:
            review = json.load(f)
        library = ingest_review(review)
        with open(args.out, "w") as f:
            json.dump(library, f)
        print(f"{len(library)} letters -> {args.out}")


if __name__ == "__main__":
    main()
