"""EXPERIMENTAL (N3): cluster a letter's harvested glyphs into <=K variants.

A dynamic font wants >=3 *distinct* forms per letter (so generated text doesn't look stamped).
Given the per-letter glyph library from ``_letter_harvest``, this rasterizes each glyph to a
small shape descriptor, k-means clusters them (deterministic init), and returns the medoid of
each cluster as a representative variant. Output: char -> list of <=K representative glyphs,
ready for N4 (font assembly).

Independent of cut quality: this just groups whatever glyphs it's given. Garbage in -> garbage
variants; the point is the pipeline (N3) is built and tested, gated only on clean letter supply.

    python -m ocr.experiments._variant_cluster --library lib.json [--k 3]
"""

import argparse
import json

import cv2
import numpy as np


def glyph_descriptor(glyph: list[list[float]], size: int = 24, blur: float = 1.2) -> np.ndarray:
    """Rasterize a normalized [0,1] glyph (connected polyline) to a size*size blurred,
    L2-normalised vector -- a shape descriptor robust to small variation."""
    img = np.zeros((size, size), np.uint8)
    a = np.array(glyph, float)
    mn, mx = a.min(0), a.max(0)
    span = np.maximum(mx - mn, 1e-9)
    pts = (a - mn) / span  # map this glyph's own bbox to the unit square (aspect-agnostic)
    prev = None
    for x, y in pts:
        cur = (int(x * (size - 1)), int(y * (size - 1)))
        if prev is not None:
            cv2.line(img, prev, cur, 255, 1)
        prev = cur
    v = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur).flatten()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def kmeans(x: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    """Tiny deterministic Lloyd's k-means -> per-row cluster label. Init = k rows spread
    evenly by index (deterministic, so variants/tests are reproducible)."""
    n = len(x)
    k = max(1, min(k, n))
    centroids = x[np.linspace(0, n - 1, k).round().astype(int)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centroids[c] = members.mean(0)
    return labels


def cluster_letter(glyphs: list[list[list[float]]], k: int = 3) -> list[int]:
    """Indices of the medoid glyph of each cluster (<=k), for one letter's samples."""
    if len(glyphs) <= k:
        return list(range(len(glyphs)))
    x = np.stack([glyph_descriptor(g) for g in glyphs])
    labels = kmeans(x, k)
    medoids = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        centroid = x[idx].mean(0)
        medoids.append(int(idx[np.linalg.norm(x[idx] - centroid, axis=1).argmin()]))
    return medoids


def build_variants(
    library: dict[str, list], k: int = 3, min_samples: int = 1
) -> dict[str, list[list[list[float]]]]:
    """char -> up to ``k`` representative variant glyphs (medoids of k-means clusters)."""
    variants = {}
    for ch, glyphs in library.items():
        if len(glyphs) < min_samples:
            continue
        variants[ch] = [glyphs[i] for i in cluster_letter(glyphs, k)]
    return variants


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cluster harvested letters into <=K variants (N3)")
    p.add_argument("--library", required=True, help="harvested library JSON (char -> [glyphs])")
    p.add_argument("--k", type=int, default=3, help="max variants per letter")
    p.add_argument("--out", default=None, help="Save the variant library JSON here")
    p.add_argument("--render", default=None, help="Render each letter's variants here")
    args = p.parse_args(argv)

    with open(args.library) as f:
        library = json.load(f)
    variants = build_variants(library, args.k)
    covered = sum(1 for v in variants.values() if len(v) >= args.k)
    print(f"letters: {len(variants)} | with full {args.k} variants: {covered}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(variants, f)
        print(f"wrote {args.out}")
    if args.render:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chars = sorted(variants, key=lambda c: -len(library[c]))[:10]
        _fig, axes = plt.subplots(
            len(chars), args.k, figsize=(2 * args.k, 1.5 * max(1, len(chars)))
        )
        axes = np.atleast_2d(axes)
        for r, ch in enumerate(chars):
            for col in range(args.k):
                ax = axes[r, col]
                ax.axis("off")
                if col < len(variants[ch]):
                    a = np.array(variants[ch][col], float)
                    ax.plot(a[:, 0], -a[:, 1], "-", color="black", lw=1.2)
                    ax.set_aspect("equal")
                if col == 0:
                    ax.set_title(f"'{ch}'", fontsize=10, loc="left")
        plt.tight_layout()
        plt.savefig(args.render, dpi=110, facecolor="white")
        print(f"wrote {args.render}")


if __name__ == "__main__":
    main()
