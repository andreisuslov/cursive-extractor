"""Tests for verified-label ingest: ordered polyline polygons, eraser, inpainted-crop path."""

import base64
import io

import numpy as np
from PIL import Image

from ocr.experiments.label_ingest import ingest, letter_polys


def test_letter_polys_splits_and_labels_local():
    out = letter_polys([[[40, 0], [40, 50]]], 100, 50, "ab")
    assert [c for c, _ in out] == ["a", "b"]
    assert min(p[0] for p in out[0][1]) == 0 and max(p[0] for p in out[0][1]) == 40
    assert min(p[0] for p in out[1][1]) == 40 and max(p[0] for p in out[1][1]) == 100


def test_letter_polys_orders_cuts_left_to_right():
    # cuts given out of x-order -> must still label slices left-to-right
    out = letter_polys([[[60, 0], [60, 50]], [[30, 0], [30, 50]]], 100, 50, "abc")
    assert [c for c, _ in out] == ["a", "b", "c"]
    assert max(p[0] for p in out[0][1]) == 30  # 'a' ends at the leftmost cut, not 60


def test_letter_polys_caps_at_text_length():
    cuts = [[[25, 0], [25, 20]], [[50, 0], [50, 20]], [[75, 0], [75, 20]]]
    assert len(letter_polys(cuts, 100, 20, "ab")) == 2


def test_letter_polys_respects_slanted_cut():
    out = letter_polys([[[40, 0], [60, 50]]], 100, 50, "ab")
    ax = [p[0] for p in out[0][1]]
    assert 40 in ax and 60 in ax


def test_ingest_masks_polygon_and_applies_eraser():
    page = np.full((40, 140, 3), 100, np.uint8)
    corrected = [
        {
            "text": "ab",
            "box": [10, 5, 110, 25],
            "cuts": [[[50, 0], [50, 20]]],
            "erase": [[20, 10, 6]],
        }
    ]
    letters = ingest(corrected, page)
    assert [c for c, _ in letters] == ["a", "b"]
    a_crop = letters[0][1]
    assert (a_crop == 255).any()  # eraser dab whited out part of 'a'
    assert (a_crop == 100).any()  # paper elsewhere kept


def test_ingest_uses_inpainted_clean_crop_when_present():
    clean = np.full((20, 100, 3), 200, np.uint8)
    clean[:, :50] = 50  # left half (the 'a') is dark in the cleaned crop
    buf = io.BytesIO()
    Image.fromarray(clean).save(buf, "PNG")
    durl = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    corrected = [
        {"text": "ab", "box": [0, 0, 100, 20], "cuts": [[[50, 0], [50, 20]]], "clean": durl}
    ]
    # page is all black; if the clean crop is used, 'a' carries its 50s, not the page's 0s
    letters = ingest(corrected, np.zeros((20, 100, 3), np.uint8))
    assert (letters[0][1] == 50).any()
