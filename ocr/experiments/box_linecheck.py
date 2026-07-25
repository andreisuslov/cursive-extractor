"""Flag word boxes that swallow a neighbouring text line.

Global row-clustering does not work here: handwritten lines slope across the page, so
y-centre clustering chains rows together. This uses a LOCAL test instead — if another
word overlaps this box horizontally and its centre sits inside this box, then this box
covers that word's line as well as its own.

Tall-but-single-line words ("girl" with ascender+descender) contain no other word's
centre, so they are not flagged; a box spanning 2-3 lines is.

  python3 linecheck.py LABELS.json [--all]   # report straddling boxes (default: done=true only)
  python3 linecheck.py --selftest
"""

import json
import sys


def x_overlap(a, b):
    """Fraction of the narrower box's width that the two boxes share horizontally."""
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    narrow = min(a[2] - a[0], b[2] - b[0])
    return max(0, hi - lo) / narrow if narrow > 0 else 0.0


def intruders(i, words, min_ovl=0.4, eat=0.35):
    """Words that overlap box i horizontally and whose OWN height box i eats into.

    Centre-inside is too strict: a box two line-pitches tall covers the descenders of the
    line above and the ascenders of the line below without containing either centre.
    """
    box = words[i]["box"]
    out = []
    for j, w in enumerate(words):
        if j == i:
            continue
        b = w["box"]
        if x_overlap(box, b) < min_ovl:
            continue
        shared = max(0, min(box[3], b[3]) - max(box[1], b[1]))
        if b[3] > b[1] and shared / (b[3] - b[1]) >= eat:
            out.append(j)
    return out


def check(words, only_done=True):
    """Generous pre-filter, not a verdict — flagged boxes go to a context-aware judge."""
    return [
        (i, w["text"], intruders(i, words))
        for i, w in enumerate(words)
        if (w.get("done") or not only_done) and intruders(i, words)
    ]


def _selftest():
    # two rows, pitch 100, height 50 — tidy boxes must be clean
    ws = [
        {"text": f"r{r}", "box": [x, r * 100, x + 60, r * 100 + 50], "done": True}
        for r in range(3)
        for x in (0, 100, 200)
    ]
    assert not check(ws), check(ws)
    # a box covering rows 0-2 at x=0 swallows the r1/r2 words above it -> flagged
    ws.append({"text": "straddle", "box": [0, 10, 60, 260], "done": True})
    # a tall box on its own column (no neighbours) must NOT be flagged
    ws.append({"text": "tall", "box": [500, 0, 560, 200], "done": True})
    flagged = {t for _, t, _ in check(ws)}
    assert "straddle" in flagged, flagged
    assert "tall" not in flagged, flagged
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        with open(sys.argv[1]) as fh:
            words = json.load(fh)
        bad = check(words, only_done="--all" not in sys.argv)
        done = sum(1 for w in words if w.get("done"))
        print(f"{done} accepted; {len(bad)} straddle another word's line")
        for i, t, js in bad:
            print(f"  [{i}] {t!r:20} swallows {[words[j]['text'] for j in js][:4]}")
        print(json.dumps([i for i, _, _ in bad]))
