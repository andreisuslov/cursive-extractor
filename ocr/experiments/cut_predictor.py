"""Cut-predictor: given a word crop + its text, place the between-letter cuts.

We KNOW the letter count from the transcript, so it's "place L-1 boundaries", not "detect N cuts".
Pipeline (all numpy, no torch — the data is tiny):

  1. SLANT-aware ink profile  — shear the crop to the angle that collapses letters into sharp columns,
     sum ink per column. (A plain vertical sum smears slanted cursive and hides the connectors.)
  2. per-column BOUNDARY model — logistic regression on the local profile window, trained on your cuts
     (a column is positive if it's within 6% of a human cut). Learns what a boundary looks like instead
     of assuming "lowest ink".
  3. DP layout               — place exactly L-1 cuts maximizing boundary-score with an even-spacing prior.

Beats uniform guessing leave-one-file-out (unseen page): MAE 0.163 vs 0.187 /bh, 84% vs 80% within
0.25*bh (64 cut-words, 2026-07). Grows with data.

    python -m ocr.experiments.cut_predictor            # leave-one-file-out eval vs uniform
    python -m ocr.experiments.cut_predictor --train    # fit on all data -> cut_model.json
    python -m ocr.experiments.cut_predictor --selftest
"""

import base64
import glob
import io
import json
import os

import numpy as np
from PIL import Image

N = 160        # profile resolution (columns)
WIN = 12       # feature half-window; feature dim = 2*WIN+1
EPS = 0.06     # a column is a "boundary" if within EPS*N of a human cut
LAM = 0.30     # even-spacing prior weight in the DP
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cut_model.json")


# --- data -------------------------------------------------------------------
def _prep(d):
    """int boxes + white-pad the page so off-page boxes (labeller off-page pan) stay aligned."""
    page = np.array(Image.open(io.BytesIO(base64.b64decode(d["page"].split(",", 1)[1]))).convert("L"))
    H, W = page.shape
    ws = [w for w in d["words"] if w.get("cuts") and not w.get("skip")]
    for w in ws:
        w["box"] = [int(round(v)) for v in w["box"]]
    P = max([0] + [max(-w["box"][0], -w["box"][1], w["box"][2] - W, w["box"][3] - H) for w in ws])
    if P > 0:
        page = np.pad(page, ((P, P), (P, P)), constant_values=255)
        for w in ws:
            w["box"] = [c + P for c in w["box"]]
    return ws, page


def boundaries_x(w):
    """Ordered L+1 boundary x-positions in box-local px (same rule as label_ingest / the labeller)."""
    xs = sorted(sum(p[0] for p in c) / len(c) for c in w["cuts"])
    bw = w["box"][2] - w["box"][0]
    return xs if (w.get("endCuts") and len(xs) >= 2) else [0.0] + xs + [float(bw)]


def load_words(folder):
    """[{text, bw, bh, bnds, prof}] for every complete cut-word (L>=2, cut count matches spelling)."""
    out = []
    for f in sorted(glob.glob(folder + "/*.json")):
        d = json.load(open(f))
        if not any(w.get("cuts") for w in d.get("words", [])):
            continue
        ws, page = _prep(d)
        for w in ws:
            B, text = boundaries_x(w), w["text"]
            if len(B) != len(text) + 1 or len(text) < 2:
                continue
            x0, y0, x1, y1 = w["box"]
            out.append(dict(text=text, bw=x1 - x0, bh=y1 - y0, bnds=B,
                            prof=slant_prof(page[y0:y1, x0:x1]), file=f))
    return out


# --- features ---------------------------------------------------------------
def slant_prof(crop, n=N):
    """Slant-aware, peak-normalised column-ink profile (letters collapsed to sharp columns)."""
    H, W = crop.shape
    ink = 255.0 - crop.astype(np.float64)
    ymid = H / 2
    best, bs = None, -1.0
    for sl in np.linspace(-0.7, 0.7, 15):
        p = np.zeros(W)
        for y in range(H):
            p += np.roll(ink[y], int(round(sl * (y - ymid))))
        pk = (p ** 2).sum() / (p.sum() ** 2 + 1e-9)     # peakiness -> best shear aligns strokes
        if pk > bs:
            bs, best = pk, p
    p = np.interp(np.linspace(0, len(best) - 1, n), np.arange(len(best)), best) if len(best) != n else best
    p = np.convolve(p, np.ones(3) / 3, mode="same")
    return p / p.max() if p.max() > 0 else p


def _cols(w):
    """(features[N, 2WIN+1], labels[N]) — one row per column of the word's profile."""
    p = np.pad(w["prof"], WIN, mode="edge")
    true = [b / w["bw"] * N for b in w["bnds"][1:-1]]
    feats = np.stack([p[c:c + 2 * WIN + 1] for c in range(N)])
    labs = np.array([1 if any(abs(c - t) < EPS * N for t in true) else 0 for c in range(N)])
    return feats, labs


# --- model ------------------------------------------------------------------
def train(words, iters=400, lr=0.3):
    """Weighted logistic regression over all columns; returns weight vector (dim 2WIN+2 incl bias)."""
    X = np.vstack([_cols(w)[0] for w in words])
    y = np.concatenate([_cols(w)[1] for w in words])
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    sw = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)   # upweight rare positives
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(Xb @ w)))
        w -= lr * (Xb.T @ ((pr - y) * sw)) / len(Xb)
    return w


def _score(prof, w):
    p = np.pad(prof, WIN, mode="edge")
    X = np.hstack([np.stack([p[c:c + 2 * WIN + 1] for c in range(N)]), np.ones((N, 1))])
    return 1 / (1 + np.exp(-(X @ w)))


def predict(text, prof, bw, w, lam=LAM):
    """Cut x-positions (box-local px): DP maximizing boundary-score + even-spacing prior."""
    L = len(text)
    if L < 2:
        return np.array([])
    cost = 1.0 - _score(prof, w)
    t = N / L
    pen = lambda g: ((g - t) / t) ** 2
    DP = np.full((L - 1, N), 1e18)
    bk = np.full((L - 1, N), -1, int)
    DP[0, 1:] = cost[1:] + lam * pen(np.arange(1, N))
    for k in range(1, L - 1):
        for x in range(k + 1, N):
            xs = np.arange(k, x)
            v = DP[k - 1, xs] + lam * pen(x - xs)
            j = int(np.argmin(v))
            DP[k, x], bk[k, x] = cost[x] + v[j], xs[j]
    xs = np.arange(L - 1, N)
    x = int(xs[np.argmin(DP[L - 2, xs] + lam * pen(N - xs))])
    cuts, k = [], L - 2
    while k >= 0:
        cuts.append(x)
        x, k = bk[k, x], k - 1
    return np.sort(np.array(cuts, float)) * bw / N


# --- eval / cli -------------------------------------------------------------
def _mae(words, w=None, lam=LAM):
    e = []
    for x in words:
        true = np.array(x["bnds"][1:-1])
        pred = (np.arange(1, len(x["text"])) * x["bw"] / len(x["text"])) if w is None \
            else predict(x["text"], x["prof"], x["bw"], w, lam)
        if len(pred) == len(true):
            e.extend(np.abs(pred - true) / x["bh"])
    return np.array(e)


def evaluate(folder):
    words = load_words(folder)
    files = sorted({w["file"] for w in words})
    print(f"{len(words)} complete cut-words, {len(files)} files, leave-one-file-out:\n")
    eu = _mae(words)
    el = []
    for f in files:
        wt = train([x for x in words if x["file"] != f])
        el.extend(_mae([x for x in words if x["file"] == f], wt))
    el = np.array(el)
    print(f"  uniform:  MAE/bh {eu.mean():.3f}   within .25bh {(eu < 0.25).mean() * 100:.1f}%")
    print(f"  learned:  MAE/bh {el.mean():.3f}   within .25bh {(el < 0.25).mean() * 100:.1f}%")


def _selftest():
    # learned model must beat uniform on a tiny 2-file set (train on one, test the other).
    import types
    def mk(prof, bnds, bw, file):
        return dict(text="ab", bw=bw, bh=20, bnds=bnds, prof=np.array(prof), file=file)
    prof = [0.0] * 70 + [1.0] * 20 + [0.0] * 70                 # a clear valley at col ~80
    w = mk(prof, [0, 80 * 100 / N, 100], 100, "a")
    wt = train([w])
    p = predict("ab", np.array(prof), 100, wt)
    assert abs(p[0] - 80 * 100 / N) < 12, f"should place the cut in the learned valley, got {p}"
    print(f"selftest OK: learned cut at {p[0]:.0f}px (target {80*100/N:.0f})")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    elif "--train" in sys.argv:
        words = load_words("/Users/ansuslov/Downloads/label_boxes")
        w = train(words)
        json.dump({"N": N, "WIN": WIN, "lam": LAM, "w": w.tolist()}, open(MODEL_PATH, "w"))
        print(f"trained on {len(words)} words -> {MODEL_PATH} ({len(w)} weights)")
    else:
        evaluate("/Users/ansuslov/Downloads/label_boxes")
