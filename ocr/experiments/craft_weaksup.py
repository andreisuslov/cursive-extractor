"""Real-domain weak-supervision for the CRAFT letter cutter (EXPERIMENTAL).

Synthetic-only fine-tuning (craft_train.py) transfers to real ink but the peaks stay broad/merged.
This sharpens them: run the synthetic model on real word crops, keep only the CONFIDENT ones (it
fired ~L well-separated peaks), turn those peaks into tight pseudo-GT, and fine-tune on a
synthetic+real mix. On page 1 this took the per-letter peak counts much closer to the true letter
counts (e.g. Isabel 2->6, John 1->4) and produced visibly crisper, separated blobs (v4).

Needs a page dir with ``boxes_refined.json`` (see ocr/refine_boxes.py) + ``*_page.png``, plus a
base checkpoint from craft_train.py. CPU, ~20-30 min. Produces craft_finetuned_v4.pth.

    python -m ocr.experiments.craft_weaksup --page-dir outputs/<pdf>/page_001 \
        --base craft_weights/craft_finetuned_v3.pth --out craft_weights/craft_finetuned_v4.pth
"""

import argparse
import glob
import itertools
import json
import os
import random

import cv2
import numpy as np
from PIL import Image

from ._recognizer import font_bank
from .craft_segmenter import _MEAN, _STD, _clahe, fit_to_canvas, local_maxima
from .craft_train import OH, OW, _gauss, _vocab
from .craft_train import sample as synth_sample


def collect_pseudo_labels(net, torch, page_dir):
    """Run ``net`` on every page word; keep words where it fired ~L separated peaks, as tight GT."""
    with open(os.path.join(page_dir, "boxes_refined.json")) as f:
        boxes = json.load(f)
    page = np.array(Image.open(glob.glob(f"{page_dir}/*_page.png")[0]).convert("RGB"))
    h, w = page.shape[:2]
    real = []
    for b in boxes:
        t = b["text"]
        if not (t.isalpha() and 2 <= len(t) <= 9):
            continue
        ymin, xmin, ymax, xmax = b["box_2d"]
        crop = page[
            int(ymin * h / 1000) : int(ymax * h / 1000), int(xmin * w / 1000) : int(xmax * w / 1000)
        ]
        if crop.shape[0] < 5 or crop.shape[1] < 10:
            continue
        L = len(t)
        canvas, _sc, _ox = fit_to_canvas(crop)
        g = _clahe.apply(cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY))
        x = (np.stack([g] * 3, -1).astype(np.float32) - _MEAN) / _STD
        with torch.no_grad():
            y, _ = net(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0))
        region = y[0, :, :, 0].cpu().numpy()
        prof = np.convolve(region.max(0).astype(np.float32), np.ones(5) / 5, mode="same")
        pk = local_maxima(prof, max(3, region.shape[1] // (L * 2)))
        if not (L <= len(pk) <= L + 2):  # confidence gate: ~L separated peaks
            continue
        pk = sorted(sorted(pk, key=lambda i: -prof[i])[:L])  # top-L by value, ordered by x
        rg = np.zeros((OH, OW), np.float32)
        af = np.zeros((OH, OW), np.float32)
        cents = []
        for hx in pk:
            hy = int(region[:, hx].argmax())
            cents.append((hx, hy))
            rg = np.maximum(rg, _gauss(OH, OW, hx, hy, 4, 5))  # tight pseudo-GT -> crisp peaks
        for (a0, b0), (a1, b1) in itertools.pairwise(cents):
            af = np.maximum(
                af, _gauss(OH, OW, (a0 + a1) / 2, (b0 + b1) / 2, max(abs(a1 - a0) / 3, 2), 4)
            )
        real.append((x.astype(np.float32), rg, af))
    return real


def main(argv=None):
    from collections import OrderedDict

    import torch

    from ._craft_model import CRAFT

    here = os.path.dirname(__file__)
    p = argparse.ArgumentParser(description="CRAFT real-domain weak-supervision")
    p.add_argument(
        "--page-dirs",
        required=True,
        nargs="+",
        help="page dirs with boxes_refined.json + *_page.png",
    )
    p.add_argument(
        "--base",
        default=os.path.join(here, "craft_weights", "craft_finetuned_v3.pth"),
        help="checkpoint to fine-tune FROM",
    )
    p.add_argument(
        "--collect-from",
        default=None,
        help="checkpoint used to pseudo-label real words (default: --base)",
    )
    p.add_argument("--out", default=os.path.join(here, "craft_weights", "craft_finetuned_v4.pth"))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args(argv)

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    def load(path):
        net = CRAFT()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()))
        return net.to("cpu")

    cnet = load(args.collect_from or args.base).eval()
    real = []
    for pdir in args.page_dirs:
        got = collect_pseudo_labels(cnet, torch, pdir)
        real += got
        print(f"  {os.path.basename(pdir)}: +{len(got)} pseudo-labels", flush=True)
    print(
        f"collected {len(real)} confident real pseudo-labels from {len(args.page_dirs)} pages",
        flush=True,
    )
    net = load(args.base)
    words, fonts = _vocab(), font_bank()

    def ohem(pred, gt, ratio=3):
        loss = (pred - gt) ** 2
        pos = gt > 0.1
        npos = int(pos.sum().item())
        if npos == 0:
            return loss.mean()
        neg = loss[~pos].flatten()
        k = min(neg.numel(), npos * ratio)
        nl = torch.topk(neg, k).values.sum() if k > 0 else loss.new_zeros(())
        return (loss[pos].sum() + nl) / (npos + k + 1e-6)

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    for s in range(args.steps):
        ex = [
            random.choice(real) if (real and random.random() < 0.5) else synth_sample(words, fonts)
            for _ in range(args.batch)
        ]
        xs, rs, as_ = zip(*ex, strict=False)
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
        y, _ = net(x)
        yc = y.contiguous()
        loss = ohem(yc[:, :, :, 0], torch.from_numpy(np.stack(rs))) + ohem(
            yc[:, :, :, 1], torch.from_numpy(np.stack(as_))
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 100 == 0:
            print(f"step {s} loss {loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
