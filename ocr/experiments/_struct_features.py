"""Structural (stroke-like) column features for the cut predictor, from the image alone.

The shipped model sees ONE channel: slant-corrected column ink SUM. It therefore cannot tell a
ligature (one thin low stroke) from a letter body (tall, several vertical crossings) when both
happen to carry similar total ink. These channels add that vertical structure:

  0 ink      total ink per column          (the current feature)
  1 runs     number of separate ink runs   ligature ~1, letter body >=2   <- the classic cursive cue
  2 extent   vertical extent of ink        ligatures are short
  3 lowy     lowest ink row (lower contour) ligatures sit at the baseline
"""
import numpy as np
from ocr.experiments.cut_predictor import N, WIN, EPS

C = 4

def _shear(crop):
    """Ink image sheared to the angle that most sharply collapses letters into columns."""
    H, W = crop.shape
    ink = 255.0 - crop.astype(np.float64)
    ymid = H / 2
    best, bs = ink, -1.0
    for sl in np.linspace(-0.7, 0.7, 15):
        img = np.stack([np.roll(ink[y], int(round(sl * (y - ymid)))) for y in range(H)])
        p = img.sum(0)
        pk = (p ** 2).sum() / (p.sum() ** 2 + 1e-9)
        if pk > bs:
            bs, best = pk, img
    return best

def _norm(v):
    v = np.asarray(v, float)
    m = v.max()
    return v / m if m > 0 else v

def struct_prof(crop, n=N):
    """(n, C) channel profiles, each peak-normalised, resampled to n columns."""
    img = _shear(crop)
    H, W = img.shape
    thr = img.max() * 0.30 if img.max() > 0 else 1.0
    binary = img > thr
    ink = img.sum(0)
    runs, extent, lowy = np.zeros(W), np.zeros(W), np.zeros(W)
    for x in range(W):
        col = binary[:, x]
        idx = np.flatnonzero(col)
        if idx.size:
            runs[x] = 1 + np.count_nonzero(np.diff(idx) > 1)
            extent[x] = idx[-1] - idx[0]
            lowy[x] = idx[-1] / max(1, H - 1)
    chans = [ink, runs, extent, lowy]
    out = np.zeros((n, C))
    xs = np.linspace(0, W - 1, n)
    for i, ch in enumerate(chans):
        r = np.interp(xs, np.arange(W), ch)
        r = np.convolve(r, np.ones(3) / 3, mode="same")
        out[:, i] = _norm(r)
    return out

def cols_multi(sp, bnds, bw):
    """(features[N, C*(2WIN+1)], labels[N]) from an (N,C) structural profile."""
    p = np.pad(sp, ((WIN, WIN), (0, 0)), mode="edge")
    true = [b / bw * N for b in bnds[1:-1]]
    feats = np.stack([p[c:c + 2 * WIN + 1].reshape(-1) for c in range(N)])
    labs = np.array([1 if any(abs(c - t) < EPS * N for t in true) else 0 for c in range(N)])
    return feats, labs

def train_multi(items, iters=400, lr=0.3):
    X = np.vstack([cols_multi(i["sp"], i["bnds"], i["bw"])[0] for i in items])
    y = np.concatenate([cols_multi(i["sp"], i["bnds"], i["bw"])[1] for i in items])
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    sw = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(Xb @ w)))
        w -= lr * (Xb.T @ ((pr - y) * sw)) / len(Xb)
    return w

def score_multi(sp, w):
    p = np.pad(sp, ((WIN, WIN), (0, 0)), mode="edge")
    X = np.stack([p[c:c + 2 * WIN + 1].reshape(-1) for c in range(N)])
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-(Xb @ w)))
