"""Tests for verified-label ingest: ordered polyline polygons + paper-colour eraser."""

import numpy as np

from ocr.experiments.label_ingest import ingest, letter_polys


def test_letter_polys_splits_and_labels_local():
    out = letter_polys([[[40, 0], [40, 50]]], 100, 50, "ab")
    assert [c for c, _ in out] == ["a", "b"]
    assert min(p[0] for p in out[0][1]) == 0 and max(p[0] for p in out[0][1]) == 40
    assert min(p[0] for p in out[1][1]) == 40 and max(p[0] for p in out[1][1]) == 100


def test_letter_polys_orders_cuts_left_to_right():
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


def test_ingest_eraser_fills_with_paper_colour():
    page = np.full((40, 140, 3), 200, np.uint8)
    page[12:18, 30:42] = 30  # a dark ink blob inside the word (crop-local x≈20..32)
    corrected = [
        {
            "text": "ab",
            "box": [10, 5, 110, 25],
            "cuts": [[[60, 0], [60, 20]]],
            "erase": [[26, 9, 8]],
        }
    ]
    a_crop = ingest(corrected, page)[0][1]  # 'a' spans crop-local x 0..60, contains the blob
    assert not (a_crop == 30).any()  # ink erased
    assert (a_crop == 200).any()  # filled with the paper colour (not flat white)


def test_ingest_eraser_page_coords_when_flagged():
    page = np.full((40, 140, 3), 200, np.uint8)
    page[12:18, 30:42] = 30  # ink blob at page x 30..42 (inside word box x0=10)
    doc = {
        "eraseSpace": "page",
        "words": [
            {
                "text": "ab",
                "box": [10, 5, 110, 25],
                "cuts": [[[60, 0], [60, 20]]],
                "erase": [[36, 15, 8]],
            }
        ],
    }
    a_crop = ingest(doc, page)[0][1]  # page-coord dab (36,15) lands on the blob inside 'a'
    assert not (a_crop == 30).any() and (a_crop == 200).any()


def test_ingest_accepts_dict_doc_and_skips_skipped_words():
    page = np.full((40, 140, 3), 200, np.uint8)
    doc = {
        "words": [
            {"text": "ab", "box": [10, 5, 110, 25], "cuts": [[[50, 0], [50, 20]]], "skip": True},
            {"text": "cd", "box": [10, 5, 110, 25], "cuts": [[[50, 0], [50, 20]]]},
        ]
    }
    assert [c for c, _ in ingest(doc, page)] == ["c", "d"]  # 'ab' skipped
