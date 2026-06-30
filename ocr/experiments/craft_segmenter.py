"""CRAFT-based cursive letter cutter (EXPERIMENTAL).

Fine-tuned CRAFT predicts a per-character region score on a word crop; we then place EXACTLY
``len(text)-1`` cuts using the known spelling (model peaks where confident, width-prior spacing in
the gaps where the model is faint). This is the first letter-cutter in the project that works on
real cursive — geometry, recognizer gates, and pretrained CRAFT all failed (see WORKLOG).

The extraction (`fit_to_canvas`, `local_maxima`, `region_to_cuts`, `ink_xrange`) is pure
numpy/cv2 and unit-tested. Torch + the vendored CRAFT model are lazy-loaded only for inference,
so this module imports without torch. Train/produce weights with ``craft_train.py``.

    from ocr.experiments.craft_segmenter import CraftSegmenter
    seg = CraftSegmenter()                 # loads craft_weights/craft_finetuned_v3.pth
    cuts = seg.cut_word(crop_rgb, "Uncle")  # -> 4 x-positions (crop coords)
"""

import itertools
import os

import cv2
import numpy as np

CW, CH = 384, 96  # model input canvas (must stay /32-divisible)
_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], np.float32)
_STD = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], np.float32)
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "craft_weights", "craft_finetuned_v4.pth")


def fit_to_canvas(crop_rgb: np.ndarray):
    """Center a word crop in the CWxCH canvas. Returns ``(canvas_rgb, scale, x_offset)``."""
    h, w = crop_rgb.shape[:2]
    sc = min((CW - 20) / max(w, 1), (CH - 20) / max(h, 1))
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    ox, oy = (CW - nw) // 2, (CH - nh) // 2
    canvas = np.full((CH, CW, 3), 255, np.uint8)
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(crop_rgb, (nw, nh))
    return canvas, sc, ox


def local_maxima(prof: np.ndarray, min_gap: int, thr: float = 0.2, want: int | None = None):
    """1-D local maxima above ``thr``, kept strongest-first with ``min_gap`` spacing (NMS)."""
    peaks = [
        i
        for i in range(1, len(prof) - 1)
        if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > thr
    ]
    peaks.sort(key=lambda i: -prof[i])
    kept: list[int] = []
    for i in peaks:
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(i)
        if want and len(kept) >= want:
            break
    return sorted(kept)


def ink_xrange(crop_rgb: np.ndarray):
    """x-extent of the dark ink in a crop (for width-prior backfill bounds)."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    cols = np.where((gray < 160).any(0))[0]
    return (int(cols.min()), int(cols.max())) if len(cols) else (0, crop_rgb.shape[1])


def region_to_cuts(region, n_letters, scale, x_offset, crop_w, ink_range):
    """Place exactly ``n_letters-1`` cut x-positions (crop coords) from a region-score map.

    Letter centers = the ``n_letters`` strongest region peaks (mapped heatmap->crop x); if the
    model fired fewer, backfill by splitting the largest gaps (width prior). Cuts are the
    midpoints between consecutive centers."""
    if n_letters <= 1:
        return []
    prof = np.convolve(region.max(0).astype(np.float32), np.ones(5) / 5, mode="same")
    centers_hx = local_maxima(prof, max(3, region.shape[1] // (n_letters * 2)), want=n_letters)
    cx = sorted(c for c in ((hx * 2 - x_offset) / scale for hx in centers_hx) if 0 <= c <= crop_w)
    xmin, xmax = ink_range
    while len(cx) < n_letters:  # width-prior backfill into the largest gap
        bounds = [xmin, *cx, xmax]
        gi = max(range(len(bounds) - 1), key=lambda i: bounds[i + 1] - bounds[i])
        cx.append((bounds[gi] + bounds[gi + 1]) / 2)
        cx.sort()
    return [(a + b) / 2 for a, b in itertools.pairwise(cx)]


class CraftSegmenter:
    """Lazy-loading CRAFT inference wrapper (imports torch only on first use)."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        self.weights = weights
        self._net = None
        self._torch = None

    def _load(self):
        from collections import OrderedDict

        import torch

        from ._craft_model import CRAFT

        if not os.path.exists(self.weights):
            raise FileNotFoundError(f"CRAFT weights not found: {self.weights} (run craft_train.py)")
        net = CRAFT()
        sd = torch.load(self.weights, map_location="cpu", weights_only=True)
        net.load_state_dict(OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()))
        net.eval()
        self._net, self._torch = net, torch

    def region_score(self, crop_rgb: np.ndarray):
        """Return ``(region_map, scale, x_offset)`` for a word crop (CLAHE-normalized input)."""
        if self._net is None:
            self._load()
        canvas, sc, ox = fit_to_canvas(crop_rgb)
        g = _clahe.apply(cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY))
        x = (np.stack([g] * 3, -1).astype(np.float32) - _MEAN) / _STD
        t = self._torch
        with t.no_grad():
            y, _ = self._net(t.from_numpy(x).permute(2, 0, 1).unsqueeze(0))
        return y[0, :, :, 0].cpu().numpy(), sc, ox

    def cut_word(self, crop_rgb: np.ndarray, text: str):
        """Cut x-positions (crop coords) splitting ``crop_rgb`` into ``len(text)`` letters."""
        region, sc, ox = self.region_score(crop_rgb)
        return region_to_cuts(region, len(text), sc, ox, crop_rgb.shape[1], ink_xrange(crop_rgb))
