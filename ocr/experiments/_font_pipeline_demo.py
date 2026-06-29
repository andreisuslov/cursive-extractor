"""Validate the N3/N4 font backend on CLEAN letters traced from cursive fonts.

The diary path can't yet supply clean per-letter glyphs (cutting/derender unsolved -- see
WORKLOG). To prove the rest of the pipeline is sound, this feeds it a *clean* letter source:
render each letter from a few cursive fonts, trace to strokes (the repo's own
``vectorize.trace_ink``), then run the real N3 (cluster -> variants) + N4 (render) stages.

Result: readable, joined, variant-rotated cursive text -- demonstrating that GIVEN clean
letters, harvest -> cluster -> render produces the goal. The only missing piece for the full
"font from a person's document" goal is clean letter SUPPLY (human-in-the-loop review, a
letter template, or a much stronger recogniser/derenderer).

    python -m ocr.experiments._font_pipeline_demo --text "the quick brown fox" --out demo.png
"""

import argparse
import string

from ocr.vectorize import format_strokes, trace_ink

from ._font_render import render_text
from ._letter_harvest import normalize_letter
from ._letter_segment_strokes import _down_points
from ._recognizer import font_bank, render_glyph
from ._variant_cluster import build_variants


def font_letter_library(fonts: list[str] | None = None, letters: str = string.ascii_lowercase):
    """Build a clean per-letter glyph library by tracing each letter from each font."""
    fonts = fonts or font_bank()
    lib: dict[str, list] = {}
    for ch in letters:
        for fp in fonts:
            g = render_glyph(ch, fp)
            if g is None:
                continue
            pts = _down_points(format_strokes(trace_ink(g), g.shape[1], g.shape[0]))
            if len(pts) >= 2:
                lib.setdefault(ch, []).append(normalize_letter(pts))
    return lib


def demo(text: str = "the quick brown fox", out: str = "font_backend_demo.png", k: int = 3) -> str:
    """Build a clean font library, cluster into variants, render ``text``. Returns ``out``."""
    variants = build_variants(font_letter_library(), k=k)
    render_text(variants, text, out)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Demo the font backend on clean font-traced letters")
    p.add_argument("--text", default="the quick brown fox", help="text to render")
    p.add_argument("--out", default="font_backend_demo.png", help="output image")
    p.add_argument("--k", type=int, default=3, help="variants per letter")
    args = p.parse_args(argv)
    print(f"wrote {demo(args.text, args.out, args.k)}")


if __name__ == "__main__":
    main()
