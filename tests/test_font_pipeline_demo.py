"""Smoke test for the font-backend demo (clean font-traced letters -> pipeline)."""

from ocr.experiments import _font_pipeline_demo as d


def test_font_letter_library_builds_traced_glyphs():
    lib = d.font_letter_library(letters="aeio")  # uses font_bank (DejaVu fallback on CI)
    assert set(lib) and set(lib) <= set("aeio")
    assert all(len(g) >= 2 for variants in lib.values() for g in variants)


def test_demo_renders_to_file(tmp_path):
    out = tmp_path / "demo.png"
    d.demo(text="ab", out=str(out), k=2)
    assert out.exists() and out.stat().st_size > 0
