"""EXPERIMENTAL (N4 export): variant glyph library -> reusable SVG glyph assets.

The genuine "font" deliverable beyond the stroke renderer: turn each variant glyph into an SVG
path string. These paths drop straight into a font editor (Glyphs/FontForge), Calligraphr, or a
web renderer (opentype.js). Dep-free -- just string building; SVG y-down matches our glyph y.

Outputs: ``paths.json`` (char -> [svg path strings]) and ``specimen.svg`` (a viewable grid).

ponytail: paths are single open polylines (our glyphs are one connected stroke -- pen-up breaks
within a letter were dropped upstream in `_down_points`). For a real installable OTF with filled
outlines + multi-contour letters, add `fontTools` and stroke-to-outline; not worth the dep yet.

    python -m ocr.experiments._font_export --variants variants.json --out-dir fontout/
"""

import argparse
import json


def glyph_to_path(glyph: list[list[float]], scale: float = 100.0) -> str:
    """One glyph's normalized points -> an SVG path ``M x y L x y ...`` (y-down, * scale)."""
    pts = [(round(x * scale, 1), round(y * scale, 1)) for x, y in glyph]
    return "M " + " L ".join(f"{x} {y}" for x, y in pts)


def variants_to_paths(variants: dict[str, list], scale: float = 100.0) -> dict[str, list[str]]:
    """char -> list of SVG path strings (one per variant)."""
    return {ch: [glyph_to_path(g, scale) for g in gl] for ch, gl in variants.items()}


def specimen_svg(variants: dict[str, list], cell: int = 120, scale: float = 100.0) -> str:
    """A grid SVG: one row per letter, one column per variant -- for eyeballing the font."""
    chars = sorted(variants)
    k = max((len(v) for v in variants.values()), default=1)
    w, h = k * cell, max(1, len(chars)) * cell
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">'
    ]
    for r, ch in enumerate(chars):
        parts.append(f'<text x="4" y="{r * cell + 14}" font-size="12" fill="#888">{ch}</text>')
        for c, g in enumerate(variants[ch]):
            tx, ty = c * cell + cell * 0.1, r * cell + cell * 0.1
            d = glyph_to_path(g, scale * (cell * 0.8 / scale))  # scale glyph to ~0.8 cell
            parts.append(
                f'<g transform="translate({tx:.1f},{ty:.1f})">'
                f'<path d="{d}" fill="none" stroke="black" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round"/></g>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    import os

    p = argparse.ArgumentParser(description="Export a variant library to SVG glyph assets")
    p.add_argument("--variants", required=True, help="variant library JSON from N3")
    p.add_argument("--out-dir", default="fontout", help="directory for paths.json + specimen.svg")
    args = p.parse_args(argv)

    with open(args.variants) as f:
        variants = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = variants_to_paths(variants)
    with open(os.path.join(args.out_dir, "paths.json"), "w") as f:
        json.dump(paths, f)
    with open(os.path.join(args.out_dir, "specimen.svg"), "w") as f:
        f.write(specimen_svg(variants))
    total = sum(len(v) for v in paths.values())
    print(f"{len(paths)} letters, {total} glyph paths -> {args.out_dir}/(paths.json, specimen.svg)")


if __name__ == "__main__":
    main()
