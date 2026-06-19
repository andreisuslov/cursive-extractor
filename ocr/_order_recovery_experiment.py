"""Order-recovery experiment: can ocr.vectorize recover usable strokes from a raster?

Ground truth = easybank words (captured online, true pen order). For each word we
render a CLEAN raster (best-case 'scan'), run it back through the vectorizer, and
compare recovered vs true on:
  - VISUAL fidelity  (IoU of inked pixels)            -> matters for static postcards
  - PEN-UP fidelity  (recovered lifts / true lifts)   -> fabricated lifts
  - PATH-LENGTH ratio(recovered / true)               -> retracing from bad order
  - X-REVERSALS      (backtracks in recovered path)   -> order scrambling

Run: python -m ocr._order_recovery_experiment
"""

import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from ocr.vectorize import vectorize_pil_crop

H = 160  # render height (px)
PEN = 3  # pen width (px)
PAD = 12
N_WORDS = 40
SAVE_SAMPLES = [0, 5, 12, 25]  # indices to save side-by-side images


def load_words(name="easybank", n=N_WORDS):
    with zipfile.ZipFile(f"data/{name}.json.zip") as z:
        data = json.load(z.open(z.namelist()[0]))
    # spread across the set; skip tiny words
    out = []
    for it in data[:: max(1, len(data) // (n * 2))]:
        pts = np.array(it["points"], dtype=float)
        if len(pts) > 30:
            out.append(
                (it["metadata"].get("asciiSequence", ""), pts, it["metadata"].get("aspectRatio", 1))
            )
        if len(out) >= n:
            break
    return out


def to_canvas(pts, aspect):
    """Map true points (x*aspect, y) into a (W,H) canvas; returns pixel pts + size."""
    xy = pts[:, :2].copy()
    xy[:, 0] *= aspect
    mn, mx = xy.min(0), xy.max(0)
    span = np.maximum(mx - mn, 1e-6)
    inner_h = H - 2 * PAD
    scale = inner_h / span[1]
    W = int(span[0] * scale + 2 * PAD)
    px = (xy - mn) * scale + PAD
    return px, W


def render(px, pen_flags, W, width=PEN):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    pts = [tuple(p) for p in px.tolist()]  # Python-float coords (same values as px)
    flags = pen_flags.tolist()
    for i in range(len(pts) - 1):
        if flags[i] == 1:  # pen down at i -> draw to i+1
            d.line((pts[i], pts[i + 1]), fill=0, width=width)
    return img


def mask(img):
    return np.array(img) < 128


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def path_len(px, pen_flags):
    """Total pen-down segment length. Per-segment lengths are computed in one numpy
    call; summing the in-order list reproduces the original sequential float sum."""
    deltas = px[1:] - px[:-1]
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    return float(sum(seg[pen_flags[:-1] == 1].tolist()))


def x_reversals(px, pen_flags, thresh=3.0):
    """Count backtracks (x decreasing) within pen-down runs -- a sign of bad order."""
    dx = px[:-1, 0] - px[1:, 0]
    return int(((pen_flags[:-1] == 1) & (dx > thresh)).sum())


def run(name, save_prefix=None, save_idx=()):
    words = load_words(name)
    print(f"--- {name}: testing {len(words)} words (clean render -> vectorize -> compare) ---")
    rows = []
    for k, (word, pts, aspect) in enumerate(words):
        px, W = to_canvas(pts, aspect)
        true_pen = pts[:, 2]
        true_img = render(px, true_pen, W)

        # recover via the OCR vectorizer
        rec = np.array(
            vectorize_pil_crop(true_img.convert("RGB")), dtype=float
        )  # [nx,ny,state] in [0,1]
        if len(rec) < 5:
            continue
        rec_px = rec[:, :2] * [W, H]
        rec_pen = rec[:, 2]
        rec_img = render(rec_px, rec_pen, W)

        m_true, m_rec = mask(true_img), mask(rec_img)
        true_ups = int((true_pen == 0).sum())
        rec_ups = int((rec_pen == 0).sum())
        tl, rl = path_len(px, true_pen), path_len(rec_px, rec_pen)
        rows.append(
            {
                "word": word,
                "iou": iou(m_true, m_rec),
                "true_ups": true_ups,
                "rec_ups": rec_ups,
                "len_ratio": (rl / tl) if tl else 0,
                "true_rev": x_reversals(px, true_pen),
                "rec_rev": x_reversals(rec_px, rec_pen),
                "npts_true": len(px),
                "npts_rec": len(rec),
            }
        )
        if save_prefix is not None and k in save_idx:
            combo = Image.new("L", (W, 2 * H + 4), 200)
            combo.paste(true_img, (0, 0))
            combo.paste(rec_img, (0, H + 4))
            combo.save(f"/tmp/{save_prefix}_{k:02d}_{word[:10]}.png")

    def a(key):
        return np.mean([r[key] for r in rows])

    print(f"  visual IoU:        {a('iou'):.3f}   (1.0 = perfect overlap; held = good)")
    print(
        f"  pen-ups true->rec: {a('true_ups'):.1f} -> {a('rec_ups'):.1f}   "
        f"(x{a('rec_ups') / max(a('true_ups'), 1e-6):.1f} fabricated; target ~1x)"
    )
    print(f"  path-length ratio: {a('len_ratio'):.2f}   (>1 = retracing detours)")
    print(f"  x-reversals:       {a('true_rev'):.1f} -> {a('rec_rev'):.1f}")
    return rows


def diacritic_check(rows):
    """For words with i/j/t/x, the writer lifts the pen for dots/crosses, so true
    pen-ups > 1. Check the recovered pen-ups track the true count (= diacritics
    recovered as separate strokes)."""
    dia = [r for r in rows if any(c in r["word"].lower() for c in "ijtx") and r["true_ups"] > 1]
    if not dia:
        print("  (no multi-stroke i/j/t/x words found)")
        return
    exact = sum(1 for r in dia if r["rec_ups"] == r["true_ups"])
    close = sum(1 for r in dia if abs(r["rec_ups"] - r["true_ups"]) <= 1)
    tu = np.mean([r["true_ups"] for r in dia])
    ru = np.mean([r["rec_ups"] for r in dia])
    print(f"  diacritic words (i/j/t/x, true pen-ups>1): {len(dia)}")
    print(f"    avg pen-ups true->rec: {tu:.1f} -> {ru:.1f}")
    print(f"    recovered pen-ups EXACT match: {exact}/{len(dia)}   within 1: {close}/{len(dia)}")
    for r in sorted(dia, key=lambda r: -abs(r["rec_ups"] - r["true_ups"]))[:8]:
        print(
            f"      {r['word'][:16]:16} true={r['true_ups']} rec={r['rec_ups']} iou={r['iou']:.2f}"
        )


if __name__ == "__main__":
    print("==================== EASYBANK (connected, no i/j/t/x) ====================")
    run("easybank", save_prefix="order_recovery", save_idx=(0, 5, 12, 25))
    print("\n==================== BIGBANK (has i/j/t/x diacritics) ====================")
    big = run("bigbank", save_prefix="bigbank_diac", save_idx=())
    print("  --- diacritic recovery ---")
    diacritic_check(big)
    # save a few i/j/t/x sample images for visual inspection
    import json as _json
    import zipfile as _zip

    with _zip.ZipFile("data/bigbank.json.zip") as z:
        _d = _json.load(z.open(z.namelist()[0]))
    shown = 0
    for it in _d:
        w = it["metadata"].get("asciiSequence", "")
        pts = np.array(it["points"], float)
        if shown >= 4 or not any(c in w.lower() for c in "it") or int((pts[:, 2] == 0).sum()) < 2:
            continue
        aspect = it["metadata"].get("aspectRatio", 1)
        px, W = to_canvas(pts, aspect)
        true_img = render(px, pts[:, 2], W)
        rec = np.array(vectorize_pil_crop(true_img.convert("RGB")), float)
        rec_img = render(rec[:, :2] * [W, H], rec[:, 2], W)
        combo = Image.new("L", (W, 2 * H + 4), 200)
        combo.paste(true_img, (0, 0))
        combo.paste(rec_img, (0, H + 4))
        combo.save(f"/tmp/bigbank_diac_{shown}_{w[:10]}.png")
        shown += 1
    print(f"  saved {shown} i/t diacritic samples to /tmp/bigbank_diac_*.png")
