"""Is the structural loss capacity or signal? Match feature counts by shrinking the window."""
import numpy as np
from ocr.experiments.cut_predictor import train, predict, FOLDER, N, LAM
from ocr.experiments._struct_features import struct_prof, C
from ocr.experiments._width_prior_eval import errs
from ocr.experiments._struct_eval import load_all, predict_multi
import ocr.experiments._struct_features as SF

def run(win, chans):
    """LOFO with a given half-window and channel subset."""
    old = SF.WIN; SF.WIN = win
    words = WORDS
    files = sorted({x["file"] for x in words})
    e = []
    def sub(sp): return sp[:, chans]
    for f in files:
        tr = [x for x in words if x["file"] != f]; te = [x for x in words if x["file"] == f]
        items = [dict(sp=sub(x["sp"]), bnds=x["bnds"], bw=x["bw"]) for x in tr]
        w = SF.train_multi(items)
        for x in te:
            true = np.array(x["bnds"][1:-1])
            pred = predict_multi(x["text"], sub(x["sp"]), x["bw"], w)
            if len(pred) == len(true): e.extend(np.abs(pred - true) / x["bh"])
    SF.WIN = old
    return np.array(e)

WORDS = load_all(FOLDER)
print(f"  {'config':38}{'feats':>7}{'MAE':>9}{'within.25':>11}")
print(f"  {'ink only, WIN=12 (shipped)':38}{25:>7}{0.1504:>9.4f}{84.4:>10.1f}%")
for win, chans, name in [
    (12, [0,1,2,3], "4 channels, WIN=12"),
    (3,  [0,1,2,3], "4 channels, WIN=3  (capacity-matched)"),
    (12, [0,1],     "ink+runs, WIN=12"),
    (12, [0,3],     "ink+lowy, WIN=12"),
]:
    e = run(win, chans)
    print(f"  {name:38}{len(chans)*(2*win+1):>7}{e.mean():>9.4f}{(e<0.25).mean()*100:>10.1f}%")
