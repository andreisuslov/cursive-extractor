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

H = 160            # render height (px)
PEN = 3            # pen width (px)
PAD = 12
N_WORDS = 40
SAVE_SAMPLES = [0, 5, 12, 25]   # indices to save side-by-side images


def load_words(name="easybank", n=N_WORDS):
    with zipfile.ZipFile(f"data/{name}.json.zip") as z:
        data = json.load(z.open(z.namelist()[0]))
    # spread across the set; skip tiny words
    out = []
    for it in data[:: max(1, len(data) // (n * 2))]:
        pts = np.array(it["points"], dtype=float)
        if len(pts) > 30:
            out.append((it["metadata"].get("asciiSequence", ""), pts, it["metadata"].get("aspectRatio", 1)))
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
    for i in range(len(px) - 1):
        if pen_flags[i] == 1:                      # pen down at i -> draw to i+1
            d.line([tuple(px[i]), tuple(px[i + 1])], fill=0, width=width)
    return img


def mask(img):
    return np.array(img) < 128


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def path_len(px, pen_flags):
    s = 0.0
    for i in range(len(px) - 1):
        if pen_flags[i] == 1:
            s += float(np.hypot(*(px[i + 1] - px[i])))
    return s


def x_reversals(px, pen_flags, thresh=3.0):
    """Count backtracks (x decreasing) within pen-down runs -- a sign of bad order."""
    n = 0
    for i in range(len(px) - 1):
        if pen_flags[i] == 1 and (px[i, 0] - px[i + 1, 0]) > thresh:
            n += 1
    return n


def main():
    words = load_words()
    print(f"Testing {len(words)} words (clean render -> vectorize -> compare)\n")
    rows = []
    for k, (word, pts, aspect) in enumerate(words):
        px, W = to_canvas(pts, aspect)
        true_pen = pts[:, 2]
        true_img = render(px, true_pen, W)

        # recover via the OCR vectorizer
        rec = np.array(vectorize_pil_crop(true_img.convert("RGB")), dtype=float)  # [nx,ny,state] in [0,1]
        if len(rec) < 5:
            continue
        rec_px = rec[:, :2] * [W, H]
        rec_pen = rec[:, 2]
        rec_img = render(rec_px, rec_pen, W)

        m_true, m_rec = mask(true_img), mask(rec_img)
        true_ups = int((true_pen == 0).sum())
        rec_ups = int((rec_pen == 0).sum())
        tl, rl = path_len(px, true_pen), path_len(rec_px, rec_pen)
        rows.append(dict(
            word=word,
            iou=iou(m_true, m_rec),
            true_ups=true_ups, rec_ups=rec_ups,
            len_ratio=(rl / tl) if tl else 0,
            true_rev=x_reversals(px, true_pen), rec_rev=x_reversals(rec_px, rec_pen),
            npts_true=len(px), npts_rec=len(rec),
        ))
        if k in SAVE_SAMPLES:
            # stack true (top) over recovered (bottom)
            combo = Image.new("L", (W, 2 * H + 4), 200)
            combo.paste(true_img, (0, 0))
            combo.paste(rec_img, (0, H + 4))
            path = f"/tmp/order_recovery_{k:02d}_{word[:10]}.png"
            combo.save(path)

    a = lambda key: np.mean([r[key] for r in rows])
    print(f"{'word':14} {'IoU':>5} {'trueUp':>6} {'recUp':>6} {'lenRatio':>8} {'trueRev':>7} {'recRev':>6}")
    for r in rows[:25]:
        print(f"{r['word'][:14]:14} {r['iou']:5.2f} {r['true_ups']:6d} {r['rec_ups']:6d} "
              f"{r['len_ratio']:8.2f} {r['true_rev']:7d} {r['rec_rev']:6d}")
    print("\n=== AVERAGES over", len(rows), "words ===")
    print(f"  visual IoU (recovered vs true ink): {a('iou'):.3f}   (1.0 = perfect overlap)")
    print(f"  pen-ups true -> recovered:          {a('true_ups'):.1f} -> {a('rec_ups'):.1f}   "
          f"(x{a('rec_ups')/max(a('true_ups'),1e-6):.1f} fabricated)")
    print(f"  path-length ratio (rec/true):       {a('len_ratio'):.2f}   (>1 = retracing/detours)")
    print(f"  x-reversals true -> recovered:      {a('true_rev'):.1f} -> {a('rec_rev'):.1f}   "
          f"(backtracks = scrambled order)")
    print(f"\nSaved side-by-side samples (true top / recovered bottom) to /tmp/order_recovery_*.png")


if __name__ == "__main__":
    main()
