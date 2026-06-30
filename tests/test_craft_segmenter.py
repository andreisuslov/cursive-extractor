"""Tests for the CRAFT letter-cut extraction (pure numpy/cv2 — no torch needed)."""

import numpy as np

from ocr.experiments import craft_segmenter as cs


def test_fit_to_canvas_shape_and_scale():
    canvas, sc, ox = cs.fit_to_canvas(np.full((40, 200, 3), 255, np.uint8))
    assert canvas.shape == (cs.CH, cs.CW, 3)
    assert sc > 0 and ox >= 0  # small crops upscale (sc may exceed 1)


def test_local_maxima_finds_separated_peaks():
    prof = np.zeros(100, np.float32)
    prof[20], prof[50], prof[80] = 1.0, 0.9, 0.8
    assert cs.local_maxima(prof, min_gap=5) == [20, 50, 80]


def test_local_maxima_nms_suppresses_close_peaks_and_honors_want():
    prof = np.zeros(100, np.float32)
    prof[20], prof[23], prof[60] = 1.0, 0.95, 0.8  # 20 & 23 within min_gap
    pk = cs.local_maxima(prof, min_gap=10, want=2)
    assert 20 in pk and 23 not in pk and len(pk) <= 2


def test_region_to_cuts_count_equals_L_minus_1():
    region = np.zeros((48, 192), np.float32)
    for cx in (30, 90, 150):
        region[19:29, cx - 4 : cx + 4] = 1.0
    cuts = cs.region_to_cuts(region, 3, scale=1.0, x_offset=0, crop_w=384, ink_range=(0, 384))
    assert len(cuts) == 2 and cuts[0] < cuts[1]


def test_region_to_cuts_backfills_when_model_underfires():
    region = np.zeros((48, 192), np.float32)
    region[20:28, 40:48] = 1.0  # only one blob, but ask for 4 letters
    cuts = cs.region_to_cuts(region, 4, scale=1.0, x_offset=0, crop_w=384, ink_range=(10, 300))
    assert len(cuts) == 3 and all(cuts[i] < cuts[i + 1] for i in range(2))


def test_ink_xrange_finds_dark_span():
    crop = np.full((20, 100, 3), 255, np.uint8)
    crop[:, 30:70] = 0
    lo, hi = cs.ink_xrange(crop)
    assert lo <= 30 and hi >= 69
