"""Tests for verified-label ingest (pure box-splitting logic)."""

from ocr.experiments.label_ingest import letter_boxes


def test_letter_boxes_splits_and_labels():
    # box x 100..200 (width 100), one cut at crop-local 40 -> 'a' [100..140], 'b' [140..200]
    out = letter_boxes([100, 0, 200, 50], [40], "ab")
    assert [c for c, _ in out] == ["a", "b"]
    assert out[0][1] == [100, 0, 140, 50]
    assert out[1][1] == [140, 0, 200, 50]


def test_letter_boxes_caps_at_text_length():
    # 3 cuts -> 4 slices but only 2 letters -> 2 boxes
    out = letter_boxes([0, 0, 100, 20], [25, 50, 75], "ab")
    assert len(out) == 2


def test_letter_boxes_drops_slivers():
    # a cut 1px from the edge -> first slice <2px is dropped
    out = letter_boxes([0, 0, 100, 20], [1], "ab")
    assert all(b[2] - b[0] >= 2 for _, b in out)
