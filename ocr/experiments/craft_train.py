"""Fine-tune CRAFT to detect cursive character regions, from FREE synthetic data (EXPERIMENTAL).

Recipe (reproduces craft_weights/craft_finetuned_v3.pth):
  - render words from cursive fonts -> perfect per-character boxes (font advances) -> CRAFT
    region (per-char) + affinity (between-char) Gaussian ground-truth heatmaps;
  - diary-style augmentation (grey ink, off-white textured paper, blur, faded, low-res) + CLAHE
    so it transfers to real diary ink (the synthetic->real domain gap, see WORKLOG);
  - OHEM MSE loss so the sparse-zero background can't dominate and peaks stay crisp.

Runs on CPU (the MPS bilinear-interpolate backward has a stride bug). ~20-40 min for 1400 steps.
Needs torch + torchvision + a base checkpoint (craft_mlt_25k.pth); see README for the download.

    python -m ocr.experiments.craft_train --base craft_weights/craft_mlt_25k.pth \
        --out craft_weights/craft_finetuned_v3.pth --steps 1400
"""

import argparse
import itertools
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._recognizer import font_bank

CW, CH = 384, 96
OW, OH = CW // 2, CH // 2
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], np.float32)
_STD = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], np.float32)


def _vocab():
    words = ["Uncle", "Aunt", "Isabel", "Music", "Helen", "daughter", "married", "School"]
    try:
        with open("/usr/share/dict/words") as f:
            dw = [w.strip() for w in f if w.strip().isalpha() and 3 <= len(w.strip()) <= 9]
        words += random.sample(dw, min(3000, len(dw)))
        words += [w.capitalize() for w in random.sample(dw, min(1200, len(dw)))]
    except FileNotFoundError:
        pass
    return words


def _gauss(h, w, cx, cy, sx, sy):
    ys, xs = np.mgrid[0:h, 0:w]
    return np.exp(
        -(((xs - cx) ** 2) / (2 * sx * sx + 1e-6) + ((ys - cy) ** 2) / (2 * sy * sy + 1e-6))
    ).astype(np.float32)


def sample(words, fonts):
    """One training example: (normalized CLAHE input HxWx3, region GT, affinity GT)."""
    word, fp, size = random.choice(words), random.choice(fonts), random.randint(38, 70)
    font = ImageFont.truetype(fp, size)
    left, top, _r, bottom = font.getbbox(word)
    ww, wh = font.getlength(word), bottom - top
    if ww > CW - 20 or wh > CH - 20:
        sc = min((CW - 20) / max(ww, 1), (CH - 20) / max(wh, 1))
        font = ImageFont.truetype(fp, max(12, int(size * sc)))
        left, top, _r, bottom = font.getbbox(word)
        ww, wh = font.getlength(word), bottom - top
    ox, oy = int((CW - ww) / 2), int((CH - wh) / 2)
    img = Image.new("L", (CW, CH), 255)
    ImageDraw.Draw(img).text((ox - left, oy - top), word, font=font, fill=0)
    arr = np.array(img)
    region = np.zeros((OH, OW), np.float32)
    affinity = np.zeros((OH, OW), np.float32)
    cents = []
    for i in range(len(word)):
        x0 = ox - left + font.getlength(word[:i])
        x1 = ox - left + font.getlength(word[: i + 1])
        sl = arr[:, max(0, int(x0)) : max(int(x0) + 1, int(x1))]
        rows = np.where((sl < 128).any(1))[0]
        if len(rows) == 0:
            continue
        y0, y1 = rows.min(), rows.max()
        cx, cy = (x0 + x1) / 4, (y0 + y1) / 4
        cents.append((cx, cy))
        region = np.maximum(
            region, _gauss(OH, OW, cx, cy, max((x1 - x0) / 6, 2), max((y1 - y0) / 6, 2))
        )
    for (a0, b0), (a1, b1) in itertools.pairwise(cents):
        affinity = np.maximum(
            affinity, _gauss(OH, OW, (a0 + a1) / 2, (b0 + b1) / 2, max(abs(a1 - a0) / 3, 2), 4)
        )
    # diary-style augmentation
    ink, paper = random.randint(40, 125), random.randint(188, 236)
    aug = np.where(arr < 128, ink, paper).astype(np.float32)
    aug += np.random.normal(0, random.uniform(3, 15), aug.shape)
    k = random.choice([0, 3, 3, 5])
    if k:
        aug = cv2.GaussianBlur(aug, (k, k), 0)
    if random.random() < 0.6:
        f = random.uniform(0.45, 0.85)
        h, w = aug.shape
        aug = cv2.resize(cv2.resize(aug, (max(1, int(w * f)), max(1, int(h * f)))), (w, h))
    aug = _clahe.apply(np.clip(aug, 0, 255).astype(np.uint8))
    rgb = (np.stack([aug] * 3, -1).astype(np.float32) - _MEAN) / _STD
    return rgb, region, affinity


def main(argv=None):
    import torch

    from ._craft_model import CRAFT

    p = argparse.ArgumentParser(description="Fine-tune CRAFT on synthetic cursive")
    here = os.path.dirname(__file__)
    p.add_argument("--base", default=os.path.join(here, "craft_weights", "craft_mlt_25k.pth"))
    p.add_argument("--out", default=os.path.join(here, "craft_weights", "craft_finetuned_v3.pth"))
    p.add_argument("--steps", type=int, default=1400)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args(argv)

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    words, fonts = _vocab(), font_bank()

    from collections import OrderedDict

    net = CRAFT()
    sd = torch.load(args.base, map_location="cpu", weights_only=True)
    net.load_state_dict(OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()))
    net = net.to("cpu").train()

    def ohem(pred, gt, ratio=3):
        loss = (pred - gt) ** 2
        pos = gt > 0.1
        npos = int(pos.sum().item())
        if npos == 0:
            return loss.mean()
        neg = loss[~pos].flatten()
        k = min(neg.numel(), npos * ratio)
        neg_loss = torch.topk(neg, k).values.sum() if k > 0 else loss.new_zeros(())
        return (loss[pos].sum() + neg_loss) / (npos + k + 1e-6)

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    for s in range(args.steps):
        xs, rs, as_ = zip(*[sample(words, fonts) for _ in range(args.batch)], strict=False)
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
        rr = torch.from_numpy(np.stack(rs))
        aa = torch.from_numpy(np.stack(as_))
        y, _ = net(x)
        yc = y.contiguous()
        loss = ohem(yc[:, :, :, 0], rr) + ohem(yc[:, :, :, 1], aa)
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
