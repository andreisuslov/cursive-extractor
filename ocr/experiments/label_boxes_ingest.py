"""Ingest human-adjusted WORD labels (labels_corrected.json from letter_label.html) ->
clean, deskewed, ruled-line-free word crops ready for InkSight.

This closes the labelling round-trip. The labeller exports a self-contained JSON (the page
image is embedded as a data-url, plus one entry per word carrying its box, tilt angle and
eraser dabs). For every kept word we crop it with a little padding, deskew it by its angle,
paint out the eraser dabs with the paper colour, strip ruled-line remnants with ``deline``,
and drop a grayscale PNG + a manifest line. A QA montage of the first crops is written too.

    python3 ocr/experiments/label_boxes_ingest.py --labels labels_corrected.json
    python3 ocr/experiments/label_boxes_ingest.py --labels FILE --out outputs/handwriting
    python3 ocr/experiments/label_boxes_ingest.py --selftest      # assert-based self-check

Input JSON (exported by letter_label.html):
    {"page": "data:image/jpeg;base64,...", "eraseSpace": "page",
     "words": [{"text": str, "box": [x0,y0,x1,y1] in PAGE px, "angle": deg (may be absent),
                "done": bool, "skip": bool, "erase": [[px,py,r],...] page-coords, "cuts": []}]}
"""

import argparse
import base64
import io
import json
import os
import re

import cv2
import numpy as np
from PIL import Image


# --- ruled-line remover: body copied verbatim from Downloads/data_pilot/deline.py (self-contained) ---
def deline(gray):
    """Whole-page/crop ruled-line removal: connect the line mask so faint/dashed lines are
    caught fully, keep ink at crossings, then INPAINT the removed pixels for a clean paper fill.
    Returns (cleaned_gray, removed_mask)."""
    H, W = gray.shape
    paper = int(np.percentile(gray, 85))
    mark = (gray < paper - 12).astype(np.uint8) * 255      # fainter threshold -> catch pale line bits
    ink = (gray < paper - 70).astype(np.uint8) * 255
    inkd = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 25)))
    hl = cv2.morphologyEx(mark, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(mark, cv2.MORPH_OPEN, vk)
    # bridge gaps so a dashed line becomes one continuous run before we subtract ink
    hl = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 40), 1)))
    vl = cv2.morphologyEx(vl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 40))))
    lines = cv2.bitwise_or(hl, vl)
    remove = cv2.bitwise_and(lines, cv2.bitwise_not(inkd))
    remove = cv2.dilate(remove, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(gray, remove, 3, cv2.INPAINT_TELEA)
    return out, remove


def _sanitize(text):
    """Filesystem-safe slug of a word (letters/digits kept, everything else -> '_')."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s or "word"


def _decode_page(data_url):
    """data-url ('data:image/...;base64,XXXX') -> (gray ndarray, BGR ndarray)."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.array(img)
    gray = np.array(img.convert("L"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return gray, bgr


# normalized output: 1 baseline-unit -> STD_UNIT px, then every crop is forced to STD_H tall so the
# collected data is uniform height regardless of how big/small the box was drawn (width still varies).
GUIDE_UNIT = 0.30   # must match letter_label.html: box height spans 1/GUIDE_UNIT baseline-units
STD_UNIT = 36
STD_H = 144         # box + 0.10 pad each side spans ~(1.2/GUIDE_UNIT)=4 units -> 4*STD_UNIT


def _process_word(gray, w):
    """Turn one word entry into a cleaned grayscale crop, or None if it degenerates to nothing.
    Steps: pad-crop -> paint eraser dabs (un-rotated frame) -> deskew -> deline."""
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in w["box"])
    bh = y1 - y0
    if bh <= 0 or x1 - x0 <= 0:
        return None
    pad = int(round(0.10 * bh))

    # crop the box with padding, clamped to the page. cx0/cy0 is the crop's true top-left in
    # page coords -> equals (box_x0-pad, box_y0-pad) when unclamped (the spec's formula), and
    # stays correct at the page edges where the pad is clipped.
    # full box+pad rect (may hang off-page now that the labeller allows it): build a white canvas and
    # paste the on-page part, so off-page area is blank and the word keeps its position in the frame.
    cx0, cy0, cx1, cy1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    crop = np.full((cy1 - cy0, cx1 - cx0), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, cx0), max(0, cy0), min(W, cx1), min(H, cy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0] = gray[sy0:sy1, sx0:sx1]

    # paint out eraser dabs FIRST, in the un-rotated crop frame where the stored page-coord dab
    # centres map directly (px-cx0, py-cy0); doing this before deskew keeps them on the ink.
    paper = float(np.median(crop))
    for dab in w.get("erase") or []:
        px, py, r = dab
        cv2.circle(crop, (int(round(px - cx0)), int(round(py - cy0))), int(round(r)), paper, -1)

    # deskew by -angle about the crop centre, filling exposed corners with white
    ang = float(w.get("angle") or 0)
    if ang:
        h, ww = crop.shape
        M = cv2.getRotationMatrix2D((ww / 2.0, h / 2.0), -ang, 1.0)
        crop = cv2.warpAffine(
            crop, M, (ww, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )

    clean, _ = deline(crop)
    # normalize height: scale so one baseline-unit (GUIDE_UNIT of the box height) == STD_UNIT px,
    # then pad/crop to the fixed STD_H so every crop is exactly the same height (uniform training data).
    unit_px = GUIDE_UNIT * bh
    scale = STD_UNIT / unit_px if unit_px > 0 else 1.0
    nw, nh = max(1, round(clean.shape[1] * scale)), max(1, round(clean.shape[0] * scale))
    r = cv2.resize(clean, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if nh < STD_H:
        top = (STD_H - nh) // 2
        canvas = np.full((STD_H, nw), 255, np.uint8)
        canvas[top:top + nh] = r
        return canvas
    if nh > STD_H:
        top = (nh - STD_H) // 2
        return r[top:top + STD_H]
    return r


def run(labels_path, out_dir):
    """Ingest a labels_corrected.json file. Returns list of (crop_path, text) for kept words."""
    with open(labels_path) as f:
        labels = json.load(f)
    page_stem = os.path.splitext(os.path.basename(labels_path))[0]
    gray, _bgr = _decode_page(labels["page"])

    page_dir = os.path.join(out_dir, page_stem)
    os.makedirs(page_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    results = []
    with open(manifest_path, "a") as mf:
        for idx, w in enumerate(labels.get("words", [])):
            text = w.get("text") or ""
            if w.get("skip") or not text.strip() or text.strip() == "?":
                continue
            clean = _process_word(gray, w)
            if clean is None:
                continue
            fn = f"{idx}_{_sanitize(text)}.png"
            path = os.path.join(page_dir, fn)
            cv2.imwrite(path, clean)
            mf.write(json.dumps({"crop": path, "text": text}) + "\n")
            results.append((path, text))

    _montage(results, os.path.join(out_dir, "_montage.png"))
    return results


def _montage(results, path, cols=6, cell=(200, 120), label_h=22, max_n=24):
    """QA grid of the first ``max_n`` crops, each with its text label above it."""
    items = results[:max_n]
    cw, ch = cell
    if not items:
        cv2.imwrite(path, np.full((ch + label_h, cw, 3), 255, np.uint8))
        return
    cols = min(cols, len(items))
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * (ch + label_h), cols * cw, 3), 255, np.uint8)
    for k, (cpath, text) in enumerate(items):
        img = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        thumb = cv2.resize(img, (nw, nh))
        r, c = divmod(k, cols)
        cell_y, cell_x = r * (ch + label_h), c * cw
        oy, ox = cell_y + label_h + (ch - nh) // 2, cell_x + (cw - nw) // 2
        canvas[oy:oy + nh, ox:ox + nw] = thumb
        cv2.putText(canvas, text[:18], (cell_x + 4, cell_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def demo():
    """Self-contained assert-based check: synthesize a tiny {page, words} export, ingest it,
    and assert the kept-crop count + manifest lines match (skip / '?' / empty are dropped)."""
    import tempfile

    tmp = tempfile.mkdtemp(prefix="label_boxes_ingest_")

    # synthetic page: near-white with a little dark ink inside the two real word boxes
    page = np.full((300, 400), 245, np.uint8)
    cv2.putText(page, "cat", (55, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 20, 3)
    cv2.putText(page, "fox", (228, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 20, 3)
    cv2.line(page, (0, 75), (400, 75), 200, 1)      # a faint ruled line for deline to chew on
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    assert ok, "failed to encode synthetic page"
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    labels = {
        "page": data_url,
        "eraseSpace": "page",
        "words": [
            {"text": "cat", "box": [50, 50, 150, 100], "angle": 0,
             "erase": [[60, 60, 5]], "skip": False, "cuts": []},          # kept (idx 0)
            {"text": "dog", "box": [200, 200, 300, 260],
             "erase": [], "skip": True, "cuts": []},                       # dropped: skip
            {"text": "?", "box": [50, 150, 150, 200],
             "erase": [], "skip": False, "cuts": []},                      # dropped: '?'
            {"text": "   ", "box": [10, 10, 60, 40],
             "erase": [], "skip": False, "cuts": []},                      # dropped: empty
            {"text": "fox", "box": [220, 50, 340, 110], "angle": 6.0,
             "erase": [], "skip": False, "cuts": []},                      # kept (idx 4, deskew)
        ],
    }
    lp = os.path.join(tmp, "page_007.json")
    with open(lp, "w") as f:
        json.dump(labels, f)
    out = os.path.join(tmp, "handwriting")

    results = run(lp, out)

    assert len(results) == 2, f"expected 2 kept crops, got {len(results)}"
    page_dir = os.path.join(out, "page_007")
    pngs = sorted(f for f in os.listdir(page_dir) if f.endswith(".png"))
    assert len(pngs) == 2, f"expected 2 PNGs on disk, got {pngs}"
    assert any(f.startswith("0_cat") for f in pngs), f"missing 0_cat*: {pngs}"
    assert any(f.startswith("4_fox") for f in pngs), f"missing 4_fox*: {pngs}"

    lines = [ln for ln in open(os.path.join(out, "manifest.jsonl")).read().splitlines() if ln.strip()]
    assert len(lines) == 2, f"expected 2 manifest lines, got {len(lines)}"
    recs = [json.loads(ln) for ln in lines]
    assert {r["text"] for r in recs} == {"cat", "fox"}, recs
    assert all(os.path.exists(r["crop"]) for r in recs), "manifest points at a missing crop"

    assert os.path.exists(os.path.join(out, "_montage.png")), "no montage written"

    # every kept crop is a valid, non-empty grayscale image
    for r in recs:
        im = cv2.imread(r["crop"], cv2.IMREAD_GRAYSCALE)
        assert im is not None and im.size > 0, f"unreadable crop {r['crop']}"

    print(f"selftest OK: 2 crops kept (cat, fox), 3 dropped (skip/?/empty) -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", help="labels_corrected.json exported by letter_label.html")
    p.add_argument("--out", default="outputs/handwriting", help="output dir (default outputs/handwriting)")
    p.add_argument("--selftest", action="store_true", help="run the built-in assert-based demo and exit")
    args = p.parse_args(argv)

    if args.selftest:
        demo()
        return
    if not args.labels:
        p.error("--labels is required (or pass --selftest)")

    results = run(args.labels, args.out)
    print(f"{len(results)} crops -> {args.out}")


if __name__ == "__main__":
    main()
