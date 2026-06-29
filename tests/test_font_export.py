"""Tests for SVG glyph export (N4 export)."""

from ocr.experiments import _font_export as fe


def test_glyph_to_path_format():
    p = fe.glyph_to_path([[0.0, 0.0], [1.0, 0.5]], scale=100)
    assert p.startswith("M 0.0 0.0")
    assert " L 100.0 50.0" in p


def test_variants_to_paths_shape():
    paths = fe.variants_to_paths({"a": [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.5, 1.0]]]})
    assert len(paths["a"]) == 2
    assert all(s.startswith("M ") for s in paths["a"])


def test_specimen_svg_has_one_path_per_glyph():
    v = {"a": [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.5, 1.0]]], "b": [[[0.0, 0.0], [1.0, 1.0]]]}
    svg = fe.specimen_svg(v)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<path") == 3  # 2 'a' + 1 'b'
