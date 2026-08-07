"""Does the width prior pay off if the DP trusts the spacing prior more? Sweep lam for both."""
import numpy as np
from ocr.experiments.cut_predictor import load_words, train, predict, FOLDER
from ocr.experiments._width_prior_eval import fit_widths, predict_wp, errs

words = load_words(FOLDER)
files = sorted({x["file"] for x in words})
folds = []
for f in files:
    tr = [x for x in words if x["file"] != f]
    folds.append((tr, [x for x in words if x["file"] == f], train(tr), fit_widths(tr)))

print(f"  {'lam':>6}  {'even-spacing prior':>26}   {'width prior':>26}")
print(f"  {'':>6}  {'MAE':>10}{'within.25':>12}   {'MAE':>10}{'within.25':>12}")
best = None
for lam in (0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00):
    eu, ew = [], []
    for tr, te, wt, wd in folds:
        eu.extend(errs(te, lambda x: predict(x["text"], x["prof"], x["bw"], wt, lam=lam)))
        ew.extend(errs(te, lambda x: predict_wp(x["text"], x["prof"], x["bw"], wt, wd, lam=lam)))
    eu, ew = np.array(eu), np.array(ew)
    print(f"  {lam:6.2f}  {eu.mean():10.4f}{(eu<0.25).mean()*100:11.1f}%   "
          f"{ew.mean():10.4f}{(ew<0.25).mean()*100:11.1f}%")
    for tag, e in (("even", eu), ("width", ew)):
        if best is None or e.mean() < best[0]: best = (e.mean(), (e<0.25).mean()*100, lam, tag)
print(f"\n  best: MAE {best[0]:.4f}  within.25 {best[1]:.1f}%  at lam={best[2]} with the {best[3]} prior")
print(f"  current shipped config: lam=0.30, even prior -> MAE 0.1504, 84.4%")
