"""EXPERIMENTAL per-writer ALLOGRAPH LIBRARY: the multiple FORMS the diarist uses for
each letter.

Stands on the two pieces that already work: the ~91% per-letter segmenter
(``ocr._segment_prototype``) and the bootstrap CNN recognizer
(``ocr._bootstrap_recognizer``, ~44% real held-out top-1, weights in ``runs/recog/``).
The flow, all driven from ``main``::

    python -m ocr._allograph_library --pdf test_document --pages 1 2 3 4

  1. EXTRACT  -- every word (pooled across ``--pages``) -> per-letter slices labeled by the
     transcription char (reuses ``_bootstrap_recognizer.extract_words``: geometry cut,
     recognizer-free, so the label set is not circular w.r.t. the CNN that filters it next).
  2. NOISE-REJECT -- drop a slice when the CNN recognizer does NOT classify it as its
     labeled letter (case-insensitive). Those are the mis-cuts / neighbour-row ink the
     91% segmenter leaves behind, so the library stays clean. This is a SELF-CONSISTENCY
     filter (the CNN was trained on these same slices), not an independent judge -- it
     removes what the net itself cannot fit as the label; reported as such.
  3. CLUSTER  -- each clean letter's slices into up to ``MAX_VARIANTS`` forms by a tiny
     numpy k-means over the blurred shape descriptor (``_recognizer.descriptor``); K is
     chosen by silhouette (``SIL_MIN`` floor, else a single form). Raw SHAPE is clustered
     on purpose -- the CNN's penultimate features are trained to be allograph-INVARIANT,
     the opposite of what we want to surface here.
  4. LIBRARY  -- ``letter -> [variant exemplars]`` where each exemplar is the cluster
     MEDOID (mask + traced strokes + member count + source word); saved to
     ``runs/recog/allograph_library.pt`` (git-ignored).
  5. VISUALISE -- a grid of the discovered forms for the well-sampled letters ->
     ``runs/allograph_grid.png``.

HONEST: clustering raw shape means a "variant" can be a real allograph OR a recurring
mis-cut; the per-letter silhouette (printed, and on the grid) is the evidence for which.
The KEY QUESTION -- is 91% segmentation good enough for a usable per-letter variant
library, or is segmentation the bottleneck -- is answered from those numbers in
``_print_report``, not asserted.
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import _bootstrap_recognizer as B
from . import _recognizer as R
from . import _segment_prototype as sp
from . import paths, vectorize

# --- knobs ------------------------------------------------------------------
MAX_VARIANTS = 5  # at most this many allograph forms per letter
MIN_CLUSTER = 3  # a variant must hold >= this many slices (else it's an outlier, not a form)
MIN_FOR_VARIANTS = 8  # need >= this many clean slices before we even try to split a letter
SIL_MIN = 0.05  # accept K>1 only if its silhouette beats this; else the letter is one form

DEFAULT_WEIGHTS = B.DEFAULT_WEIGHTS  # runs/recog/bootstrap_cnn.pt
LIBRARY_PATH = "runs/recog/allograph_library.pt"
GRID_PATH = "runs/allograph_grid.png"
VIZ_LETTERS = "e a o t l r n s i"  # well-sampled letters to show side by side


# --- recognizer (noise filter) ----------------------------------------------


def load_recognizer(
    weights: str | None, records: list[dict], epochs: int, seed: int
) -> tuple[B.CNNRecognizer, str]:
    """Get the CNN recognizer used as the noise filter. Prefer saved ``weights`` (lazy --
    don't re-burn CPU); else train on ALL records (no held-out split: it's a filter, not
    an eval). Returns ``(recognizer, source)`` where source is ``"loaded"``/``"trained"``."""
    if weights and os.path.exists(weights):
        ck = torch.load(weights, map_location="cpu", weights_only=True)
        classes = ck["classes"]
        model = B.LetterCNN(len(classes))
        model.load_state_dict(ck["state_dict"])
        return B.CNNRecognizer(model, classes), "loaded"
    classes = sorted(c for c, n in B.class_counts(records).items() if n >= B.MIN_CLASS)
    model = B.train_cnn(records, classes, epochs, seed)
    return B.CNNRecognizer(model, classes), "trained"


# --- 1-2. collect + noise-reject --------------------------------------------


def collect_slices(records: list[dict]) -> dict[str, list[tuple[np.ndarray, str]]]:
    """``letter (lowercased) -> [(mask, source_word), ...]`` over every alpha slice. Case
    is folded because allographs are a property of the letter, not the case."""
    by_letter: dict[str, list[tuple[np.ndarray, str]]] = defaultdict(list)
    for rec in records:
        for ch, sl in rec["slices"]:
            by_letter[ch.lower()].append((sl, rec["word"]))
    return dict(by_letter)


def reject_black_blocks(
    by_letter: dict[str, list[tuple[np.ndarray, str]]],
) -> tuple[dict[str, list[tuple[np.ndarray, str]]], dict]:
    """Drop slices that are BLACK-BLOCKS (solid over-inked blobs / degenerate fragments)
    before clustering, via the cut module's slice-level detector (``sp.is_black_block``).
    These are mis-cuts that recur as a fake dominant 'variant'; this is a cut-quality
    filter, not a relabel. Runs BEFORE the CNN noise filter so the report separates the two.
    Returns ``(filtered_by_letter, stats)``."""
    out: dict[str, list[tuple[np.ndarray, str]]] = {}
    by_letter_drops: dict[str, int] = {}
    n_total = n_blocks = 0
    for letter, items in by_letter.items():
        kept = [(m, w) for m, w in items if not sp.is_black_block(m)]
        out[letter] = kept
        by_letter_drops[letter] = len(items) - len(kept)
        n_total += len(items)
        n_blocks += len(items) - len(kept)
    return out, {"n_total": n_total, "n_blocks": n_blocks, "by_letter": by_letter_drops}


def noise_reject(
    by_letter: dict[str, list[tuple[np.ndarray, str]]], recognizer: B.CNNRecognizer
) -> tuple[dict[str, list[tuple[np.ndarray, str]]], dict]:
    """Keep slices whose CNN top-1 (case-insensitive) matches the label; drop the rest as
    likely mis-cuts/noise. Letters the recognizer never trained on can't be judged, so
    they are kept whole and flagged ``unjudged``. Returns ``(clean_by_letter, stats)``."""
    known = {c.lower() for c in recognizer.classes}
    clean: dict[str, list[tuple[np.ndarray, str]]] = {}
    n_total = n_rejected = 0
    unjudged: set[str] = set()
    rej_by_letter: dict[str, int] = {}
    for letter, items in by_letter.items():
        n_total += len(items)
        if letter not in known:  # CNN never saw this class -> can't filter it
            clean[letter] = list(items)
            rej_by_letter[letter] = 0
            unjudged.add(letter)
            continue
        kept = [(m, w) for m, w in items if (t := recognizer.top(m, 1)) and t[0].lower() == letter]
        clean[letter] = kept
        rej_by_letter[letter] = len(items) - len(kept)
        n_rejected += len(items) - len(kept)
    stats = {
        "n_total": n_total,
        "n_rejected": n_rejected,
        "rej_by_letter": rej_by_letter,
        "unjudged": unjudged,
    }
    return clean, stats


# --- 3. cluster (pure-numpy k-means + silhouette) ---------------------------


def descriptors(masks: list[np.ndarray]) -> np.ndarray:
    """Blurred, aspect-normalised, L2-normalised shape vectors (reuse the recognizer's
    descriptor) -- one row per slice. Euclidean distance on these ~ cosine on shape."""
    return np.stack([R.descriptor(m) for m in masks])


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Tiny k-means++ over rows of ``x``. Deterministic given ``seed``. Returns
    ``(labels, centroids)``. Sizes here are small (n<~120, k<=5, dim 576), so the naive
    O(n*k*dim) loop is plenty."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):  # k-means++ seeding
        d2 = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(x[rng.choice(n, p=probs)])
    c = np.stack(centers).astype(np.float64)
    labels = np.full(n, -1)
    for step in range(iters):
        dist = ((x[:, None, :] - c[None]) ** 2).sum(2)  # (n, k)
        new = dist.argmin(1)
        if step and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
    return labels, c


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (euclidean). +1 = tight, well-separated clusters; ~0 =
    overlapping; <0 = wrong assignments. Undefined (<2 clusters) -> -1.0."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    g = x @ x.T
    sq = np.diag(g)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * g, 0.0))
    sils = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        a = d[i, same].mean() if same.any() else 0.0
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        sils.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(sils))


def choose_k(x: np.ndarray, seed: int) -> tuple[np.ndarray, int, float]:
    """Pick the number of allograph forms by best silhouette over ``K in 2..Kmax``, where
    ``Kmax`` keeps every variant at >= ``MIN_CLUSTER`` members. Falls back to a single form
    (``K=1``) when the best structure is weaker than ``SIL_MIN``. Returns
    ``(labels, k, best_silhouette)`` -- the silhouette is reported even when it loses, as
    the honest evidence of how distinct the forms are."""
    n = len(x)
    kmax = min(MAX_VARIANTS, n // MIN_CLUSTER)
    best_labels, best_k, best_sil = np.zeros(n, int), 1, 0.0
    for k in range(2, kmax + 1):
        labels, _ = kmeans(x, k, seed)
        if len(np.unique(labels)) < k or np.bincount(labels, minlength=k).min() < MIN_CLUSTER:
            continue  # collapsed or an undersized cluster -> not a real split
        s = silhouette(x, labels)
        if s > best_sil:
            best_labels, best_k, best_sil = labels, k, s
    if best_sil < SIL_MIN:
        return np.zeros(n, int), 1, best_sil
    return best_labels, best_k, best_sil


# --- 4. build the library ---------------------------------------------------


def build_variants(
    masks: list[np.ndarray], words: list[str], labels: np.ndarray, x: np.ndarray
) -> list[dict]:
    """One exemplar per cluster = the MEDOID (descriptor nearest the cluster mean), kept as
    mask + traced strokes + member count + source word. Sorted most-common first."""
    variants = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        mean = x[idx].mean(0)
        ex = idx[((x[idx] - mean) ** 2).sum(1).argmin()]
        variants.append(
            {
                "count": len(idx),
                "exemplar_mask": masks[ex],
                "exemplar_strokes": vectorize.trace_ink(masks[ex]),
                "source_word": words[ex],
            }
        )
    variants.sort(key=lambda v: -v["count"])
    return variants


def build_library(records: list[dict], recognizer: B.CNNRecognizer, seed: int) -> tuple[dict, dict]:
    """Full pipeline 1-4. Returns ``(library, stats)``. ``library[letter]`` =
    ``{n_raw, n_clean, n_rejected, unjudged, silhouette, variants}``; a letter with
    fewer than ``MIN_FOR_VARIANTS`` clean slices gets a single medoid form (no split)."""
    raw_by_letter = collect_slices(records)
    by_letter, block_stats = reject_black_blocks(raw_by_letter)
    clean, stats = noise_reject(by_letter, recognizer)
    stats["n_blocks"] = block_stats["n_blocks"]
    stats["blocks_by_letter"] = block_stats["by_letter"]
    library: dict[str, dict] = {}
    for letter, items in sorted(clean.items()):
        masks = [m for m, _ in items]
        words = [w for _, w in items]
        n = len(items)
        entry = {
            "n_raw": len(raw_by_letter[letter]),
            "n_blocks": block_stats["by_letter"].get(letter, 0),
            "n_clean": n,
            "n_rejected": stats["rej_by_letter"][letter],
            "unjudged": letter in stats["unjudged"],
            "silhouette": None,
            "variants": [],
        }
        if n == 0:
            library[letter] = entry
            continue
        x = descriptors(masks)
        if n < MIN_FOR_VARIANTS:  # too few to split: one form
            labels = np.zeros(n, int)
        else:
            labels, _, entry["silhouette"] = choose_k(x, seed)
        entry["variants"] = build_variants(masks, words, labels, x)
        library[letter] = entry
    return library, stats


# --- 5. visualise -----------------------------------------------------------


def visualize(library: dict, letters: list[str], path: str) -> list[str]:
    """Grid: one row per letter, one column per discovered variant (medoid exemplar),
    titled with its member count; the row is labelled with the letter and its silhouette.
    Returns the letters actually drawn (those with >= 2 variants are the interesting ones,
    but any letter with >= 1 variant is shown)."""
    shown = [c for c in letters if library.get(c) and library[c]["variants"]]
    if not shown:
        return []
    ncol = max(len(library[c]["variants"]) for c in shown)
    fig, axes = plt.subplots(
        len(shown), ncol, figsize=(ncol * 1.5, len(shown) * 1.5), squeeze=False
    )
    for r, letter in enumerate(shown):
        vs = library[letter]["variants"]
        sil = library[letter]["silhouette"]
        for col in range(ncol):
            ax = axes[r][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(vs):
                ax.imshow(B.slice_to_canvas(vs[col]["exemplar_mask"]), cmap="gray_r")
                ax.set_title(f"n={vs[col]['count']}", fontsize=8)
            else:
                ax.axis("off")
        label = f"'{letter}'" + (f"\nsil {sil:.2f}" if sil is not None else "\n(few)")
        axes[r][0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=22)
    fig.suptitle("Discovered allograph variants (per-letter forms, medoid exemplars)")
    paths.ensure_parent(path)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return shown


# --- CLI + report -----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a per-writer allograph (letter-form) library")
    p.add_argument("--pdf", default="test_document", help="Source PDF (slug or path)")
    p.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=[4],
        help="Page numbers, 1-based; multiple pages are pooled into one library",
    )
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for the source page")
    p.add_argument("--seed", type=int, default=0, help="k-means seed")
    p.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"CNN noise-filter weights (default {DEFAULT_WEIGHTS}); else trained on all slices",
    )
    p.add_argument("--epochs", type=int, default=150, help="Epochs if the CNN must be trained")
    p.add_argument("--save", default=LIBRARY_PATH, help="Where to save the library .pt")
    p.add_argument("--grid", default=GRID_PATH, help="Where to save the variant grid PNG")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    pages = args.pages
    print(
        f"\n=== EXTRACT (pages {' '.join(map(str, pages))}, geometry cut, "
        "label = transcription char) ==="
    )
    records: list[dict] = []
    for pg in pages:
        recs, _, _ = B.extract_words(args.pdf, pg, args.version, args.dpi)
        print(f"  page {pg}: {len(recs)} words, {sum(B.class_counts(recs).values())} alpha slices")
        records.extend(recs)
    counts = B.class_counts(records)
    print(
        f"pooled: {len(records)} words   {sum(counts.values())} alpha slices   "
        f"{len(counts)} classes"
    )

    recognizer, source = load_recognizer(args.weights, records, args.epochs, args.seed)
    print(f"noise-filter CNN: {source} ({len(recognizer.classes)} classes)")

    library, stats = build_library(records, recognizer, args.seed)

    if args.save:
        paths.ensure_parent(args.save)
        torch.save(
            {"library": library, "stats": {**stats, "unjudged": sorted(stats["unjudged"])}},
            args.save,
        )
        print(f"saved library -> {os.path.abspath(args.save)}")
    shown = visualize(library, VIZ_LETTERS.split(), args.grid)
    if shown:
        print(f"saved grid    -> {os.path.abspath(args.grid)}  (letters: {' '.join(shown)})")

    _print_report(library, stats)
    return library


def _print_report(library: dict, stats: dict) -> None:
    n_total, n_rej = stats["n_total"], stats["n_rejected"]
    rate = n_rej / n_total if n_total else 0.0
    n_blocks = stats.get("n_blocks", 0)
    raw_total = n_total + n_blocks  # noise_reject's n_total is already post-black-block
    print("\n=== 1b. BLACK-BLOCK REJECTION (solid over-inked blobs / degenerate fragments) ===")
    brate = n_blocks / raw_total if raw_total else 0.0
    print(f"slices: {raw_total}   black-blocks dropped: {n_blocks} ({brate:.0%})   kept: {n_total}")

    print("\n=== 2. NOISE REJECTION (CNN top-1 != labeled letter -> dropped) ===")
    print(
        f"slices: {n_total}   rejected: {n_rej} ({rate:.0%})   kept: {n_total - n_rej}   "
        f"unjudged letters (CNN never trained): {' '.join(sorted(stats['unjudged'])) or '(none)'}"
    )

    print("\n=== 3-4. PER-LETTER LIBRARY (clean slices -> variant forms) ===")
    print(
        f"{'L':<3}{'raw':>5}{'blk':>5}{'clean':>7}{'rej':>5}{'forms':>7}{'sil':>7}  variant counts"
    )
    have_variants = 0
    multi = 0
    sils = []
    for letter, e in sorted(library.items()):
        if e["n_clean"] == 0:
            continue
        nvar = len(e["variants"])
        if e["n_clean"] >= MIN_FOR_VARIANTS:
            have_variants += 1
        if nvar >= 2:
            multi += 1
            sils.append(e["silhouette"])
        sil = f"{e['silhouette']:.2f}" if e["silhouette"] is not None else "  -"
        vc = " ".join(str(v["count"]) for v in e["variants"])
        print(
            f"{letter:<3}{e['n_raw']:>5}{e.get('n_blocks', 0):>5}{e['n_clean']:>7}"
            f"{e['n_rejected']:>5}{nvar:>7}{sil:>7}  {vc}"
        )

    mean_sil = float(np.mean(sils)) if sils else 0.0
    print("\n=== HONEST READ -- is ~91% segmentation enough for a usable variant library? ===")
    print(
        f"- letters with enough clean slices for variant analysis (>= {MIN_FOR_VARIANTS}): "
        f"{have_variants}"
    )
    print(f"- letters where >1 distinct form was found: {multi}  (mean silhouette {mean_sil:.2f})")
    print(
        f"- noise rejection dropped {rate:.0%} of slices -- the segmenter's ~9% cut error "
        "plus the\n  CNN's own ~56% miss rate compound here, so this is a coarse de-noise, "
        "not a clean cut."
    )
    verdict = (
        "PARTIAL: enough samples for a handful of common letters, but the per-form "
        "silhouettes are\n  low -- the 'variants' are as much recurring mis-cut as true "
        "allograph. Segmentation\n  quality (and slice count) is the bottleneck; a cleaner "
        "cut or more pages would sharpen them."
        if mean_sil < 0.25
        else "USABLE: common letters split into silhouette-distinct forms that read as real "
        "allographs."
    )
    print(f"- VERDICT: {verdict}")


if __name__ == "__main__":
    main()
