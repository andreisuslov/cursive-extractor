"""Unit tests for N4 dynamic-font rendering (variant selection logic)."""

from ocr.experiments import _font_render as fr

_G = [[0.0, 0.0], [1.0, 1.0]]  # a trivial glyph


def test_variant_sequence_cycles_through_variants():
    variants = {"a": [_G, _G, _G]}  # 3 variants
    idxs = [i for _, i in fr.variant_sequence(variants, "aaaa")]
    assert idxs == [0, 1, 2, 0]  # repeats cycle -> "dynamic"


def test_variant_sequence_case_fallback_and_gaps():
    variants = {"a": [_G]}
    seq = fr.variant_sequence(variants, "A z")
    assert seq[0] == ("a", 0)  # 'A' -> 'a' via swapcase
    assert seq[1][1] is None  # space: no glyph
    assert seq[2][1] is None  # 'z' absent


def test_render_text_writes_file(tmp_path):
    out = tmp_path / "r.png"
    fr.render_text({"a": [_G, _G]}, "aa", str(out))
    assert out.exists() and out.stat().st_size > 0
