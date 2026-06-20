"""Assemble OCR'd diary pages into one multi-writer training dataset.

Walks every vectorized word box (the latest-version ``*_strokes.json`` under
``outputs/<diary>/page_*/``) into a single ``data/<name>.json.zip`` in the exact
``{points, metadata}`` schema the hand-collected banks use, tagging each entry's
``metadata.author`` with the diary it came from -- so the corpus is genuinely
multi-writer rather than one "robot" author. Degenerate boxes (empty/near-empty
strokes) are dropped.

    python3 data/build_diarybank.py --name diarybank
    # -> data/diarybank.json.zip   (+ per-diary counts)

Remember to allowlist the result in .gitignore (``!data/<name>.json.zip``) before
committing -- ``data/*.zip`` is ignored by default.
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def author_label(diary_slug: str) -> str:
    """Stable per-writer label from an output-dir slug.

    ``6-05-1962-black-diary-of-grace-galer_pag`` -> ``6-05-1962-black-diary-of-grace-galer``
    (strip the truncated trailing ``_p...`` page-dir fragment).
    """
    return re.sub(r"_p[a-z]*$", "", diary_slug)


def downsample_points(points: list[list[float]], target: int) -> list[list[float]]:
    """Thin a dense vectorized word to ~``target`` points, preserving structure.

    OCR skeleton-traced words carry thousands of points (median ~4.5k vs ~240 in
    the collected data); left raw, one word blows the model's token budget. Keep
    every pen-state transition (stroke boundaries) plus a uniform stride fill, so
    shape and pen-up/down structure survive the thinning.
    """
    n = len(points)
    if n <= target:
        return points
    keep = {0, n - 1}
    for i in range(1, n):
        if points[i][2] != points[i - 1][2]:  # pen lifts / lands -> stroke boundary
            keep.add(i - 1)
            keep.add(i)
    stride = max(1, n // target)
    keep.update(range(0, n, stride))
    return [points[i] for i in sorted(keep)]


def stroke_count(points: list[list[float]]) -> int:
    """Number of pen-down segments (runs of state==1)."""
    count = prev = 0
    for p in points:
        if p[2] == 1 and prev != 1:
            count += 1
        prev = p[2]
    return count


def _split_strokes(points: list[list[float]]) -> list[list[list[float]]]:
    """Split a point list into pen-down strokes (runs of state==1)."""
    strokes, cur = [], []
    for p in points:
        if p[2] == 1:
            cur.append(p)
        elif cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return strokes


def _rebuild(strokes: list[list[list[float]]]) -> list[list[float]]:
    """Reassemble strokes into a point list, ending each with a pen-up marker."""
    out = []
    for s in strokes:
        out.extend([p[0], p[1], 1] for p in s)
        out.append([s[-1][0], s[-1][1], 0])  # pen lift
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _dominant_band(ys: list[float], bins: int = 24, frac: float = 0.30) -> tuple[float, float]:
    """The target word's horizontal ink band, in [0,1] y.

    OCR crops routinely capture the lines above/below the target word; those show
    up as separate y-density humps with a gap between. Because the crop is built
    *around the target's detection box*, the target row sits near the vertical
    centre -- so seed at the densest **central** bin (density weighted by nearness
    to 0.5, NOT raw density: a denser neighbour line must not win, or the word gets
    mislabelled), then grow outward while density stays above ``frac`` of the seed,
    stopping at the inter-line gap. Returns the (lo, hi) y-range of the word's row.
    """
    if not ys:
        return 0.0, 1.0
    hist = [0] * bins
    for y in ys:
        hist[min(bins - 1, max(0, int(y * bins)))] += 1

    def score(i):  # density, biased toward the centre of the crop
        center = (i + 0.5) / bins
        return hist[i] * math.exp(-(((center - 0.5) / 0.22) ** 2))

    seed = max(range(bins), key=score)
    thresh = frac * hist[seed]
    lo = hi = seed
    while lo - 1 >= 0 and hist[lo - 1] >= thresh:
        lo -= 1
    while hi + 1 < bins and hist[hi + 1] >= thresh:
        hi += 1
    pad = 0.5 / bins  # half a bin of slack for ascenders/descenders
    return lo / bins - pad, (hi + 1) / bins + pad


def clean_strokes(points: list[list[float]]) -> list[list[float]]:
    """Strip the contamination the generative model visibly learned: flat-wide
    ruled lines, and the neighbour text lines bled into the crop. Conservative:
    never touches a word with <=2 strokes (nothing to separate)."""
    strokes = _split_strokes(points)
    if len(strokes) <= 2:
        return points

    def is_ruled(s):
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        return (max(xs) - min(xs)) > 0.55 and (max(ys) - min(ys)) < 0.08

    strokes = [s for s in strokes if not is_ruled(s)]
    if len(strokes) <= 2:
        return _rebuild(strokes) if strokes else points
    # Keep only strokes whose y-centre sits in the dominant ink band (target row),
    # dropping the partial lines above/below that share the crop.
    lo, hi = _dominant_band([p[1] for s in strokes for p in s])
    kept = [s for s in strokes if lo <= _median([p[1] for p in s]) <= hi]
    return _rebuild(kept) if kept else _rebuild(strokes)


def _version(page_dir: str) -> int:
    """Version suffix of a page dir: ``page_001`` -> 0, ``page_001_2`` -> 2."""
    m = re.search(r"page_\d+(?:_(\d+))?$", os.path.basename(page_dir))
    return int(m.group(1)) if m and m.group(1) else 0


def latest_strokes(output_root: str) -> list[str]:
    """One strokes.json per (diary, page), choosing the highest page version."""
    best: dict[tuple, tuple] = {}
    for sj in glob.glob(os.path.join(output_root, "*", "page_*", "*strokes.json")):
        page_dir = os.path.dirname(sj)
        diary = os.path.relpath(sj, output_root).split(os.sep)[0]
        page_num = int(re.search(r"page_(\d+)", page_dir).group(1))
        key = (diary, page_num)
        ver = _version(page_dir)
        if key not in best or ver > best[key][0]:
            best[key] = (ver, sj)
    return [v[1] for v in best.values()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="diarybank", help="dataset name -> data/<name>.json.zip")
    ap.add_argument("--output-root", default=os.path.join(REPO, "outputs"))
    ap.add_argument("--min-points", type=int, default=8, help="drop boxes with fewer points")
    ap.add_argument(
        "--max-points",
        type=int,
        default=300,
        help="thin words denser than this toward the collected-data density",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove ruled-line + adjacent-line-bleed strokes within each word",
    )
    ap.add_argument(
        "--max-ink-per-char",
        type=float,
        default=0.0,
        help="drop words with raw points/char above this (0=off; ~3900 = p90)",
    )
    ap.add_argument(
        "--max-strokes-per-char",
        type=float,
        default=0.0,
        help="drop words with strokes/char above this (0=off; ~6 = p90)",
    )
    ap.add_argument("--exclude", nargs="*", default=["test_document"], help="diary slugs to skip")
    args = ap.parse_args()

    entries: list[dict] = []
    per_author: collections.Counter = collections.Counter()
    dropped = 0
    for sj in sorted(latest_strokes(args.output_root)):
        diary = os.path.relpath(sj, args.output_root).split(os.sep)[0]
        if diary in args.exclude:
            continue
        author = author_label(diary)
        with open(sj) as f:
            boxes = json.load(f)
        for box in boxes:
            pts = box.get("points") or []
            meta = dict(box.get("metadata") or {})
            text = box.get("text") or meta.get("asciiSequence") or ""
            if len(pts) < args.min_points or not text.strip():
                dropped += 1
                continue
            nchar = max(1, len(text))
            if args.max_ink_per_char and len(pts) / nchar > args.max_ink_per_char:
                dropped += 1
                continue
            if args.max_strokes_per_char and stroke_count(pts) / nchar > args.max_strokes_per_char:
                dropped += 1
                continue
            if args.clean:
                pts = clean_strokes(pts)
                if len(pts) < args.min_points:
                    dropped += 1
                    continue
            pts = downsample_points(pts, args.max_points)
            meta["author"] = author
            meta["asciiSequence"] = text
            meta["pointCount"] = len(pts)
            meta["strokeCount"] = stroke_count(pts)
            entries.append({"points": pts, "metadata": meta})
            per_author[author] += 1

    if not entries:
        raise SystemExit("No usable strokes found -- has anything been vectorized yet?")

    out_json = os.path.join(REPO, "data", f"{args.name}.json")
    with open(out_json, "w") as f:
        json.dump(entries, f)
    out_zip = out_json + ".zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname=f"{args.name}.json")
    os.remove(out_json)

    print(f"Wrote {out_zip}: {len(entries)} words, {len(per_author)} writers (dropped {dropped}).")
    for author, n in per_author.most_common():
        print(f"  {author:50s} {n}")


if __name__ == "__main__":
    main()
