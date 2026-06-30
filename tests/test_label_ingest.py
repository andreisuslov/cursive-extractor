"""Tests for verified-label ingest: polyline polygon split + eraser."""

import numpy as np

from ocr.experiments.label_ingest import ingest, letter_polys


def test_letter_polys_splits_and_labels():
    out = letter_polys([100, 0, 200, 50], [[[40, 0], [40, 50]]], "ab")
    assert [c for c, _ in out] == ["a", "b"]
    ax = [p[0] for p in out[0][1]]
    assert min(ax) == 100 and max(ax) == 140  # 'a' between left edge and the cut
    bx = [p[0] for p in out[1][1]]
    assert min(bx) == 140 and max(bx) == 200  # 'b' between the cut and right edge


def test_letter_polys_caps_at_text_length():
    cuts = [[[25, 0], [25, 20]], [[50, 0], [50, 20]], [[75, 0], [75, 20]]]
    assert len(letter_polys([0, 0, 100, 20], cuts, "ab")) == 2


def test_letter_polys_respects_slanted_cut():
    # cut slanted: top at x=40, bottom at x=60 -> the boundary polygon carries both
    out = letter_polys([0, 0, 100, 50], [[[40, 0], [60, 50]]], "ab")
    ax = [p[0] for p in out[0][1]]
    assert 40 in ax and 60 in ax


def test_ingest_masks_polygon_and_applies_eraser():
    page = np.full((40, 140, 3), 100, np.uint8)  # uniform gray "page"
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
    assert (a_crop == 100).any()  # but ink/paper elsewhere kept
