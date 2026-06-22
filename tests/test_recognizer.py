"""Unit tests for ocr._recognizer (template recognizer + forced-alignment cut).

Pure-logic / synthetic only -- no PDF render. The recognizer is built once (cached
``get_recognizer``); the font bank renders from system fonts, so a self-match test
(a glyph rendered from a bank font must recognise as itself) is exact and
deterministic. The alignment tests assert structural invariants (count, monotonic,
in-span), not which boundary -- the absolute cut quality is measured by the CLI eval,
not asserted here.
"""

import itertools

import numpy as np

from ocr.experiments import _recognizer as r


def _any_glyph(ch: str) -> np.ndarray:
    """First system-font render of ``ch`` (skips fonts absent on this machine)."""
    for fp in r.FONT_PATHS:
        m = r.render_glyph(ch, fp)
        if m is not None:
            return m
    raise AssertionError(f"no font rendered {ch!r}")


def test_render_glyph_returns_binary_mask():
    m = _any_glyph("a")
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 255})
    assert m.max() == 255


def test_descriptor_is_l2_normalised_and_empty_is_zero():
    d = r.descriptor(_any_glyph("m"))
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5
    assert not r.descriptor(np.zeros((10, 10), np.uint8)).any()  # empty -> zero vector


def test_recognizer_self_matches_rendered_glyph():
    # a glyph rendered from one of the bank's own fonts has cosine 1.0 with its own
    # template, so the free top-1 must be that char (sanity: the recognizer works
    # on in-domain input; its weakness is the font->handwriting gap, not the code).
    rec = r.get_recognizer()
    for ch in "mwsoe":
        assert rec.top(_any_glyph(ch), 1)[0] == ch


def test_recognizer_scores_in_unit_range():
    rec = r.get_recognizer()
    sc = rec.scores(_any_glyph("a"))
    assert sc and all(-1e-6 <= v <= 1.0 + 1e-6 for v in sc.values())


def test_slice_columns_count_and_tighten():
    b = np.zeros((20, 100), np.uint8)
    b[5:15, 10:20] = b[5:15, 50:60] = b[5:15, 80:90] = 255
    slices = r.slice_columns(b, [0, 35, 70, 100])
    assert len(slices) == 3  # one per gap between consecutive cuts
    assert all(s.max() == 255 and s.shape[0] == 10 for s in slices)  # tightened to ink


def test_align_boundaries_count_monotone_in_span():
    b = np.zeros((40, 150), np.uint8)
    for x in (10, 60, 110):  # three separated letter-ish blobs
        b[10:30, x : x + 20] = 255
    x_min, x_max = 10.0, 129.0
    cand = np.arange(x_min, x_max, 3.0)
    bxs = r.align_boundaries(b, "abc", x_min, x_max, cand, np.zeros(150))
    assert len(bxs) == 2  # L-1 internal boundaries
    assert x_min < bxs[0] < bxs[1] < x_max  # strictly monotone, inside span


def test_align_boundaries_single_char_is_empty():
    b = np.zeros((20, 40), np.uint8)
    b[5:15, 10:30] = 255
    assert r.align_boundaries(b, "x", 10.0, 30.0, np.arange(10, 30, 2.0), np.zeros(40)) == []


def test_align_respects_min_letter_width():
    # boundaries must be at least ALIGN_MIN_WFRAC * (span / L) apart (no slivers).
    b = np.zeros((40, 150), np.uint8)
    for x in (10, 60, 110):
        b[10:30, x : x + 20] = 255
    x_min, x_max = 10.0, 129.0
    bxs = r.align_boundaries(b, "abc", x_min, x_max, np.arange(x_min, x_max, 3.0), np.zeros(150))
    min_w = r.ALIGN_MIN_WFRAC * (x_max - x_min) / 3
    cuts = [x_min, *bxs, x_max]
    assert all(hi - lo >= min_w for lo, hi in itertools.pairwise(cuts))
