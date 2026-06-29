"""EXPERIMENTAL (N4): render typed text from a per-letter VARIANT library -- the "dynamic font".

Given the variant library from N3 (char -> list of variant glyphs), lay the glyphs left-to-right
along a baseline, **cycling through each letter's variants** so a repeated letter never looks
stamped (the "dynamic" in dynamic font), and draw a join between consecutive glyphs (cursive
ligature). This is the font BACKEND / generation step (roadmap N4-N5): type text, get it in the
writer's hand.

Prototype limitations (honest): glyphs were bbox-normalized in harvesting, so per-letter width
and ascender/descender height are lost -- every glyph renders in a unit cell (even advance, even
height). Real kerning/baseline needs those preserved upstream. And output quality is gated by
the harvested glyphs (currently rough -- see WORKLOG). What this proves: the variant-rotation +
placement + join machinery works end-to-end.

    python -m ocr.experiments._font_render --variants variants.json --text "the baby" --out out.png
"""

import argparse
import json


def variant_sequence(variants: dict[str, list], text: str) -> list[tuple[str, int | None]]:
    """Per character, the (library-key, variant-index) chosen -- cycling a letter's variants so
    repeats vary. Falls back to swapped case; index ``None`` = no glyph (space/gap)."""
    counter: dict[str, int] = {}
    out: list[tuple[str, int | None]] = []
    for ch in text:
        key = ch if ch in variants else (ch.swapcase() if ch.swapcase() in variants else None)
        if key is None:
            out.append((ch, None))
            continue
        i = counter.get(key, 0) % len(variants[key])
        counter[key] = counter.get(key, 0) + 1
        out.append((key, i))
    return out


def render_text(variants: dict[str, list], text: str, out_path: str, join: bool = True) -> str:
    """Render ``text`` in the harvested hand to ``out_path``. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seq = variant_sequence(variants, text)
    advance, gap = 1.15, 0.7  # per-glyph x advance; blank-space width
    fig_w = max(4.0, 0.5 * advance * max(1, len(text)))
    fig, ax = plt.subplots(figsize=(fig_w, 1.8))
    x_off = 0.0
    prev_end = None
    for key, idx in seq:
        if idx is None:  # space / unavailable -> gap, break the join
            x_off += gap
            prev_end = None
            continue
        a = np.array(variants[key][idx], float)
        gx = a[:, 0] + x_off
        gy = -a[:, 1]
        if join and prev_end is not None:  # cursive ligature to this glyph's start
            ax.plot(
                [prev_end[0], gx[0]], [prev_end[1], gy[0]], "-", color="black", lw=1.0, alpha=0.7
            )
        ax.plot(gx, gy, "-", color="black", lw=1.3, solid_capstyle="round")
        prev_end = (gx[-1], gy[-1])
        x_off += advance
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.3, x_off + 0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render text from a variant library (N4 dynamic font)")
    p.add_argument("--variants", required=True, help="variant library JSON from N3")
    p.add_argument("--text", required=True, help="text to render in the harvested hand")
    p.add_argument("--out", default="rendered.png", help="output image")
    p.add_argument("--no-join", dest="join", action="store_false", help="don't draw cursive joins")
    args = p.parse_args(argv)

    with open(args.variants) as f:
        variants = json.load(f)
    missing = sorted(
        {c for c in args.text if c.strip() and c not in variants and c.swapcase() not in variants}
    )
    if missing:
        print(f"note: no glyph for {missing} (gaps left)")
    render_text(variants, args.text, args.out, args.join)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
