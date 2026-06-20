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
    ap.add_argument("--max-points", type=int, default=300,
                    help="thin words denser than this toward the collected-data density")
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
        for box in json.load(open(sj)):
            pts = box.get("points") or []
            meta = dict(box.get("metadata") or {})
            text = box.get("text") or meta.get("asciiSequence") or ""
            if len(pts) < args.min_points or not text.strip():
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
    json.dump(entries, open(out_json, "w"))
    out_zip = out_json + ".zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname=f"{args.name}.json")
    os.remove(out_json)

    print(f"Wrote {out_zip}: {len(entries)} word samples, {len(per_author)} writers (dropped {dropped}).")
    for author, n in per_author.most_common():
        print(f"  {author:50s} {n}")


if __name__ == "__main__":
    main()
