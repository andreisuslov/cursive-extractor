"""Tracer robustness on REAL diary handwriting crops (tests/fixtures/diary_*.jpg).

Locks ocr/vectorize.py against regressions on real scanned cursive: the tracer must be
deterministic (same crop -> byte-identical strokes) and produce sane output (non-empty,
normalized points, pen-up markers, stroke counts in a plausible range). These are real
word crops from page 4 of the test diary, including previously over-segmented (noisy) ones.
"""

import glob
import os

import pytest
from PIL import Image

from ocr.vectorize import vectorize_pil_crop

_FIXTURES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "diary_*.jpg")))
_IDS = [os.path.basename(p) for p in _FIXTURES]


def _load(path):
    return Image.open(path).convert("RGB")


def _n_strokes(points):
    return sum(1 for p in points if p[2] == 0)


def test_fixtures_present():
    assert len(_FIXTURES) >= 5  # a handful of representative real diary crops are tracked


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_real_crop_deterministic(path):
    a = vectorize_pil_crop(_load(path))
    b = vectorize_pil_crop(_load(path))  # fresh load, traced again
    assert a == b  # identical strokes on the same real crop


@pytest.mark.parametrize("path", _FIXTURES, ids=_IDS)
def test_real_crop_sane_output(path):
    pts = vectorize_pil_crop(_load(path))
    assert pts, "trace of real ink must be non-empty"
    assert all(len(p) == 3 for p in pts)
    assert all(p[2] in (0, 1) for p in pts)  # pen state is 0 (up) or 1 (down)
    assert all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in pts)  # normalized to [0,1]
    assert pts[-1][2] == 0  # the sequence ends with a pen-up marker
    assert 1 <= _n_strokes(pts) <= 25  # plausible pen-lift count for one diary word
