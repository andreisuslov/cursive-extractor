"""Letter-width priors in the cut DP, evaluated leave-one-file-out against the current model."""
import numpy as np, collections
from ocr.experiments.cut_predictor import (
    load_words, train, _score, FOLDER, N, LAM)

def fit_widths(words, shrink=5.0):
    """char -> relative width (1.0 = average letter), shrunk toward 1.0 for rare chars."""
    acc = collections.defaultdict(list)
    for x in words:
        b = np.array(x["bnds"], float); t = x["text"]
        if len(b) != len(t) + 1: continue
        g = np.diff(b); m = g.mean()
        if m <= 0: continue
        for ch, gi in zip(t, g): acc[ch].append(gi / m)
    out = {}
    for c, v in acc.items():
        n = len(v)
        out[c] = (n * float(np.mean(v)) + shrink * 1.0) / (n + shrink)
    return out

def predict_wp(text, prof, bw, w, widths, lam=LAM):
    """Same DP, but each letter's target gap comes from its own expected width."""
    L = len(text)
    if L < 2: return np.array([])
    cost = 1.0 - _score(prof, w)
    r = np.array([widths.get(c, 1.0) for c in text], float)
    t = N * r / r.sum()                       # per-letter target widths, sum to N
    pen = lambda g, k: ((g - t[k]) / t[k]) ** 2
    DP = np.full((L - 1, N), 1e18); bk = np.full((L - 1, N), -1, int)
    xs0 = np.arange(1, N)
    DP[0, 1:] = cost[1:] + lam * pen(xs0, 0)
    for k in range(1, L - 1):
        for x in range(k + 1, N):
            xs = np.arange(k, x)
            v = DP[k - 1, xs] + lam * pen(x - xs, k)
            j = int(np.argmin(v)); DP[k, x], bk[k, x] = cost[x] + v[j], xs[j]
    xe = np.arange(L - 1, N)
    x = int(xe[np.argmin(DP[L - 2, xe] + lam * pen(N - xe, L - 1))])
    cuts, k = [], L - 2
    while k >= 0:
        cuts.append(x); x, k = bk[k, x], k - 1
    return np.sort(np.array(cuts, float)) * bw / N

def errs(words, fn):
    e = []
    for x in words:
        true = np.array(x["bnds"][1:-1]); pred = fn(x)
        if len(pred) == len(true): e.extend(np.abs(pred - true) / x["bh"])
    return np.array(e)

words = load_words(FOLDER)
files = sorted({x["file"] for x in words})
eu, el, ew = [], [], []
for f in files:
    tr = [x for x in words if x["file"] != f]
    te = [x for x in words if x["file"] == f]
    wt = train(tr); wd = fit_widths(tr)
    from ocr.experiments.cut_predictor import predict
    eu.extend(errs(te, lambda x: np.arange(1, len(x["text"])) * x["bw"] / len(x["text"])))
    el.extend(errs(te, lambda x: predict(x["text"], x["prof"], x["bw"], wt)))
    ew.extend(errs(te, lambda x: predict_wp(x["text"], x["prof"], x["bw"], wt, wd)))
for name, e in (("uniform", eu), ("learned (current)", el), ("learned + width prior", ew)):
    e = np.array(e)
    print(f"  {name:24} MAE/bh {e.mean():.4f}   within .25bh {(e < 0.25).mean()*100:.1f}%   (n={len(e)})")
