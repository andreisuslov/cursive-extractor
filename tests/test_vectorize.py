"""Unit tests for ocr.vectorize core behavior (tiny synthetic rasters, no real data)."""

import numpy as np
from PIL import Image

from ocr import vectorize as v


def _has_2x2_block(skel):
    """True if the skeleton has any 2x2 all-ones block (i.e. it is not 1-px thin)."""
    return bool((skel[:-1, :-1] & skel[1:, :-1] & skel[:-1, 1:] & skel[1:, 1:]).any())


def test_zhang_suen_thins_filled_block_to_one_px():
    skel = v.zhang_suen(np.ones((6, 6), np.uint8))
    assert skel.dtype == np.uint8
    assert skel.max() <= 1
    assert not _has_2x2_block(skel)


def test_zhang_suen_idempotent_on_skeleton():
    bar = np.zeros((9, 16), np.uint8)
    bar[3:6, 2:14] = 1
    skel = v.zhang_suen(bar)
    assert not _has_2x2_block(skel)
    assert np.array_equal(skel, v.zhang_suen(skel))  # skeleton of a skeleton is itself


def _bars_image(bars, w=44, h=44):
    """White RGB image with black filled bars at the given (r0, r1, c0, c1) rects."""
    arr = np.full((h, w), 255, np.uint8)
    for r0, r1, c0, c1 in bars:
        arr[r0:r1, c0:c1] = 0
    return Image.fromarray(arr).convert("RGB")


def test_vectorize_pil_crop_one_component_one_penup():
    pts = v.vectorize_pil_crop(_bars_image([(8, 11, 6, 36)]))
    assert sum(1 for p in pts if p[2] == 0) == 1  # one connected ink blob -> one pen-up
    assert all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in pts)  # normalized


def test_vectorize_pil_crop_two_components_two_penups():
    pts = v.vectorize_pil_crop(_bars_image([(8, 11, 6, 36), (30, 33, 6, 36)]))
    assert sum(1 for p in pts if p[2] == 0) == 2  # two separate blobs -> two pen-ups


def test_format_strokes_shape_and_penup_marker():
    out = v.format_strokes([[(0, 0), (10, 0), (10, 10)]], 20, 20)
    assert out == [[0.0, 0.0, 1], [0.5, 0.0, 1], [0.5, 0.5, 1], [0.5, 0.5, 0]]


def test_format_strokes_rounds_to_4dp():
    out = v.format_strokes([[(1, 0), (2, 0)]], 3, 3)  # 1/3 -> 0.3333, 2/3 -> 0.6667
    assert out == [[0.3333, 0.0, 1], [0.6667, 0.0, 1], [0.6667, 0.0, 0]]


def test_format_strokes_skips_single_point_strokes():
    assert v.format_strokes([[(0, 0)]], 10, 10) == []
