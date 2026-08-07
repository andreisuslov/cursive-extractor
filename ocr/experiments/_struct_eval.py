"""LOFO: structural multi-channel features vs the shipped single-channel ink profile."""
import glob, json, numpy as np
from PIL import Image
from ocr.experiments.cut_predictor import (
    _prep, boundaries_x, slant_prof, train, predict, _score, FOLDER, N, LAM)
from ocr.experiments._struct_features import struct_prof, train_multi, score_multi
from ocr.experiments._width_prior_eval import errs

def load_all(folder):
    out = []
    for f in sorted(glob.glob(folder + "/*.json")):
        d = json.load(open(f))
        if not any(w.get("cuts") for w in d.get("words", [])): continue
        ws, page = _prep(d)
        for w in ws:
            B, text = boundaries_x(w), w["text"]
            if len(B) != len(text) + 1 or len(text) < 2: continue
            x0, y0, x1, y1 = w["box"]
            crop = page[y0:y1, x0:x1]
            out.append(dict(text=text, bw=x1-x0, bh=y1-y0, bnds=B, file=f,
                            prof=slant_prof(crop), sp=struct_prof(crop)))
    return out

def predict_multi(text, sp, bw, w, lam=LAM):
    L = len(text)
    if L < 2: return np.array([])
    cost = 1.0 - score_multi(sp, w)
    t = N / L
    pen = lambda g: ((g - t) / t) ** 2
    DP = np.full((L-1, N), 1e18); bk = np.full((L-1, N), -1, int)
    DP[0, 1:] = cost[1:] + lam * pen(np.arange(1, N))
    for k in range(1, L-1):
        for x in range(k+1, N):
            xs = np.arange(k, x)
            v = DP[k-1, xs] + lam * pen(x - xs)
            j = int(np.argmin(v)); DP[k, x], bk[k, x] = cost[x] + v[j], xs[j]
    xe = np.arange(L-1, N)
    x = int(xe[np.argmin(DP[L-2, xe] + lam * pen(N - xe))])
    cuts, k = [], L-2
    while k >= 0:
        cuts.append(x); x, k = bk[k, x], k-1
    return np.sort(np.array(cuts, float)) * bw / N

words = load_all(FOLDER)
files = sorted({x["file"] for x in words})
print(f"  {len(words)} cut-words, {len(files)} files\n")
eb, em = [], []
for f in files:
    tr = [x for x in words if x["file"] != f]; te = [x for x in words if x["file"] == f]
    wb = train(tr); wm = train_multi(tr)
    eb.extend(errs(te, lambda x: predict(x["text"], x["prof"], x["bw"], wb)))
    em.extend(errs(te, lambda x: predict_multi(x["text"], x["sp"], x["bw"], wm)))
for name, e in (("ink only (shipped)", eb), ("+ structural channels", em)):
    e = np.array(e)
    print(f"  {name:24} MAE/bh {e.mean():.4f}   within .25bh {(e<0.25).mean()*100:.1f}%")
