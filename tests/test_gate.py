"""Property test: the crop-cleaning safety gate never increases a word's stroke count.

``vectorize_boxes`` traces each word both cleaned and uncleaned and keeps cleaning
only when it does not increase the stroke count. These tests drive that gate on a
synthetic page (via a monkeypatched ``load_page``) and assert the invariant holds,
including a word where cleaning helps and a bad-box word that falls back.
"""

import copy

import numpy as np
import pytest
from PIL import Image

import ocr.vectorize as v


def _synthetic_page():
    """A 400x400 page with two words:

    A = a blob with a full-width ruled line below it (cleaning removes the line, so
        the cleaned trace has fewer strokes -> cleaning helps).
    B = an H whose full-width crossbar looks like a ruled line; removing it splits the
        glyph into more pieces, so cleaning would *increase* strokes -> gate falls back.
    """
    pg = np.full((400, 400), 255, np.uint8)
    pg[80:95, 120:260] = 0  # A: word blob
    pg[120:123, 0:400] = 0  # A: full-width ruled line below the blob
    pg[200:330, 60:75] = 0  # B: left bar
    pg[200:330, 320:335] = 0  # B: right bar
    pg[262:265, 0:400] = 0  # B: full-width crossbar (looks ruled)
    return Image.fromarray(pg).convert("RGB")


# box_2d = [ymin, xmin, ymax, xmax] on a 0-1000 scale (the page is 400x400 px)
BOX_A = [175, 250, 337, 700]  # over the blob + the ruled line
BOX_B = [475, 100, 850, 900]  # over the whole H


def _run(clean):
    boxes = [
        {"box_2d": copy.deepcopy(BOX_A), "text": "A"},
        {"box_2d": copy.deepcopy(BOX_B), "text": "B"},
    ]
    # fit_ink=False so word A's uncleaned crop includes the ruled line below it.
    return v.vectorize_boxes("fake.pdf", boxes, 0, fit_ink=False, clean=clean)


@pytest.fixture
def gate_results(monkeypatch):
    monkeypatch.setattr(v, "load_page", lambda *a, **k: _synthetic_page())
    raw = {b["text"]: b["metadata"] for b in _run(clean=False)}
    cleaned = {b["text"]: b["metadata"] for b in _run(clean=True)}
    return raw, cleaned


def test_gate_never_increases_stroke_count(gate_results):
    raw, cleaned = gate_results
    for word in raw:
        assert cleaned[word]["strokeCount"] <= raw[word]["strokeCount"], word


def test_gate_cleaning_helps_case(gate_results):
    raw, cleaned = gate_results
    # box A: the ruled line is stripped -> fewer strokes, so cleaning is kept.
    assert cleaned["A"]["cleaned"] is True
    assert cleaned["A"]["strokeCount"] < raw["A"]["strokeCount"]


def test_gate_bad_box_falls_back(gate_results):
    raw, cleaned = gate_results
    # box B: cleaning would split the glyph (more strokes) -> gate falls back to raw.
    assert cleaned["B"]["cleaned"] is False
    assert cleaned["B"]["strokeCount"] == raw["B"]["strokeCount"]
