"""EXPERIMENTAL recognition-guided per-letter cut for traced cursive words.

The geometric cutter (``ocr._segment_prototype``) places ``L-1`` boundaries from
INK SHAPE alone (thin/compact/low columns, skeleton envelope valleys). That tops
out around ~91% on the page-4 diary crops: geometry cannot tell a real edge
letter from neighbour-row noise, nor split gap-free connected cursive
(*translate*, *creates*) where the ink profile is flat. Both need to know what a
letter LOOKS like -- a recognizer.

This module adds that, in the cheapest form that the measurement justifies:

  1. RECOGNIZER (``LetterRecognizer``) -- a template matcher. Each of ``a-z A-Z``
     is rendered from several macOS cursive/script fonts to a binary glyph, blurred
     and aspect-normalised into a fixed descriptor; a slice is scored by the MAX
     cosine over a char's font templates. No training, no torch, deterministic.
     It is WEAK on real handwriting (the font->hand domain gap is large -- see the
     eval, ~25% top-1), so it is used only as a RELATIVE signal under a known char.

  2. RECOGNITION-GUIDED CUT (``align_boundaries``) -- forced alignment. We KNOW the
     word's transcription, so a DP places the ``L-1`` internal cuts to maximise the
     recognizer's score for the KNOWN letter sequence (sum of per-slice scores for
     ``word[i]``), blended with the geometry cut-score + width prior so a word
     geometry already cuts well is NOT regressed. Forced alignment scores each slice
     against its KNOWN char, so it sidesteps the case/confusion errors that wreck the
     recognizer's free top-1 -- a weak recognizer can still have a usable per-char
     response over cut position.

  3. MEASURE (``main`` / ``evaluate_word``) -- the recognizer gives a REAL metric:
     per-letter recognition accuracy (top-1, case-insensitive top-1, mean true-char
     rank) on the page-4 slices produced by the geometry cutter vs the
     recognition-guided cutter, beside the old geometry confidence proxy. Reported
     honestly with numbers -- forced alignment only "wins" where the second table
     beats the first.

CLI::

    python -m ocr._recognizer --pdf test_document --page 4 [--version N] [--dpi 600]

Reuses ``ocr._segment_prototype`` for the whole clean/trace/slant/seam pipeline and
only swaps the boundary-SELECTION step; nothing in that module changes, so its tests
stay green.
"""

import argparse
import itertools
import json
import os
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import _segment_prototype as sp
from . import paths, vectorize

CHARS = string.ascii_lowercase + string.ascii_uppercase

# macOS cursive / script / handwriting fonts. A mix on purpose: no single font
# matches the diary hand, so the per-char MAX over fonts is the best a template bank
# can do. Any path that fails to load is skipped (kept robust across OS versions).
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Savoye LET.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/ChalkboardSE.ttc",
]

DESC_SIZE = 24  # descriptor canvas side (px)
DESC_BLUR = 1.2  # gaussian sigma for the descriptor (tolerates small shape variance)

# --- forced-alignment DP weights -------------------------------------------
ALIGN_ALPHA = 1.5  # recognition weight; page-4 sweep peaks at 1.5 (30/74), 3.0 regresses
ALIGN_BETA = 0.35  # weight on the geometry cut-score at each chosen boundary
ALIGN_LAMBDA = 0.8  # weight on the squared width-prior residual (regulariser)
ALIGN_MAX_CAND = 56  # cap candidate columns so the O(L * n^2) DP stays small
ALIGN_MIN_WFRAC = 0.30  # min letter width as a fraction of (span / L)


# --- 1. recognizer ----------------------------------------------------------


def render_glyph(ch: str, font_path: str, px: int = 64) -> np.ndarray | None:
    """Render ``ch`` from ``font_path`` to a tight binary glyph mask (uint8 0/255),
    or ``None`` if the font cannot be loaded or the glyph is empty."""
    try:
        ft = ImageFont.truetype(font_path, px)
    except OSError:
        return None
    img = Image.new("L", (px * 3, px * 3), 0)
    ImageDraw.Draw(img).text((px, px // 2), ch, fill=255, font=ft)
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 40)
    if xs.size == 0:
        return None
    return ((a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 40).astype(np.uint8)) * 255


def descriptor(mask: np.ndarray, size: int = DESC_SIZE) -> np.ndarray:
    """Aspect-preserving normalise an ink mask into a ``size x size`` blurred,
    L2-normalised vector. Empty/degenerate input -> a zero vector (cosine 0)."""
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.zeros(size * size, dtype=np.float32)
    m = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = m.shape
    scale = (size - 6) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    r = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = r
    canvas = cv2.GaussianBlur(canvas, (0, 0), DESC_BLUR)
    v = canvas.flatten()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class LetterRecognizer:
    """Multi-font template letter recognizer. ``scores`` maps a slice to a per-char
    cosine in [0, 1] (max over that char's font templates); ``score_char`` is the
    single-char query forced alignment uses; ``top`` is the free classification used
    by the honest top-1 metric."""

    def __init__(self, font_paths: list[str] = FONT_PATHS, size: int = DESC_SIZE):
        self.size = size
        self.templates: dict[str, np.ndarray] = {}
        for c in CHARS:
            descs = [
                d
                for fp in font_paths
                if (g := render_glyph(c, fp)) is not None and (d := descriptor(g, size)).any()
            ]
            if descs:
                self.templates[c] = np.stack(descs)  # (n_fonts, size*size)

    def scores(self, slice_binary: np.ndarray) -> dict[str, float]:
        d = descriptor(slice_binary, self.size)
        if not d.any():
            return dict.fromkeys(self.templates, 0.0)
        return {c: float((T @ d).max()) for c, T in self.templates.items()}

    def score_char(self, slice_binary: np.ndarray, ch: str) -> float:
        T = self.templates.get(ch)
        if T is None:
            return 0.0
        d = descriptor(slice_binary, self.size)
        return float((T @ d).max()) if d.any() else 0.0

    def top(self, slice_binary: np.ndarray, k: int = 1) -> list[str]:
        ranked = sorted(self.scores(slice_binary).items(), key=lambda kv: -kv[1])
        return [c for c, _ in ranked[:k]]

    def ranking(self, slice_binary: np.ndarray) -> list[str]:
        return [c for c, _ in sorted(self.scores(slice_binary).items(), key=lambda kv: -kv[1])]


_RECOGNIZER: LetterRecognizer | None = None


def get_recognizer() -> LetterRecognizer:
    """Lazily build and cache the module-level recognizer (template render is cheap
    but we only want it once per process)."""
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = LetterRecognizer()
    return _RECOGNIZER


# --- 2. recognition-guided cut (forced alignment over candidate cut columns) -


def slice_columns(binary: np.ndarray, cuts: list[float]) -> list[np.ndarray]:
    """Cut ``binary`` into the column windows between consecutive ``cuts`` (vertical
    strips), each tightened to its own ink bbox. ``cuts`` includes the outer x_min and
    x_max, so ``len(cuts) - 1`` slices come back."""
    out: list[np.ndarray] = []
    for lo, hi in itertools.pairwise(sorted(cuts)):
        a, b = round(lo), max(round(lo) + 1, round(hi) + 1)
        strip = (binary[:, a:b] > 0).astype(np.uint8) * 255
        ys, xs = np.nonzero(strip)
        out.append(strip[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] if xs.size else strip)
    return out


def align_boundaries(
    binary: np.ndarray,
    word: str,
    x_min: float,
    x_max: float,
    cand_x: np.ndarray,
    geom_score: np.ndarray,
    recognizer: LetterRecognizer | None = None,
    alpha: float = ALIGN_ALPHA,
    beta: float = ALIGN_BETA,
    lam: float = ALIGN_LAMBDA,
) -> list[float]:
    """Forced-alignment cut: choose the ``L-1`` internal boundary x-positions
    maximising

        alpha * sum_i recog(slice_i, word[i])
      + beta  * sum_b geom_cut_score(b)
      - lam   * sum_b ((b - x_expected) / span)^2

    over monotone candidate columns. ``binary`` is the (de-sheared) ink the geometry
    cutter sliced; ``cand_x`` are candidate columns (sorted) with ``geom_score`` the
    geometry cut-score sampled at every image column. Outer boundaries are fixed to
    ``x_min``/``x_max`` (no edge trim in v1). Returns the internal boundary x-list.
    """
    rec = recognizer or get_recognizer()
    L = len(word)
    span = max(1.0, x_max - x_min)
    if L <= 1:
        return []

    # candidate columns: dedupe, force the outer x_min/x_max as the first/last, and
    # downsample to keep the O(L * n^2) DP small.
    cols = sorted({round(x_min), *(round(c) for c in cand_x), round(x_max)})
    cols = [c for c in cols if x_min - 1 <= c <= x_max + 1]
    if len(cols) > ALIGN_MAX_CAND:
        keep = np.linspace(0, len(cols) - 1, ALIGN_MAX_CAND).round().astype(int)
        cols = sorted({cols[0], cols[-1], *(cols[i] for i in keep)})
    p = np.asarray(cols, dtype=float)
    n = len(p)
    if n < L + 1:  # not enough columns to place L letters -> even split
        return list(np.linspace(x_min, x_max, L + 1)[1:-1])

    x_expected = sp.expected_boundary_x(word, x_min, x_max)  # L-1 expected positions
    min_w = ALIGN_MIN_WFRAC * span / L
    gmax = len(geom_score) - 1

    # Emission cache: per-char score dict of each window [a, b] (cand indices a < b),
    # computed lazily on demand. Going through ``rec.scores`` (not the template matmul
    # directly) keeps this recognizer-agnostic -- any object exposing
    # ``scores(binary) -> {char: score}`` plugs in (the font templates OR a trained
    # CNN, see ``ocr._bootstrap_recognizer``). For the font recognizer this is
    # behaviour-identical to the old descriptor-cosine path.
    score_cache: dict[tuple[int, int], dict[str, float]] = {}

    def window_scores(a: int, b: int) -> dict[str, float]:
        key = (a, b)
        s = score_cache.get(key)
        if s is None:
            lo, hi = int(p[a]), max(int(p[a]) + 1, int(p[b]) + 1)
            s = rec.scores((binary[:, lo:hi] > 0).astype(np.uint8) * 255)
            score_cache[key] = s
        return s

    def emission(ch: str, a: int, b: int) -> float:
        return window_scores(a, b).get(ch, 0.0)

    def reg(boundary_ix: int, k: int) -> float:
        """geometry + width-prior term attached to internal boundary ``boundary_ix``
        (1-based, 1..L-1) chosen at candidate index ``k``."""
        g = beta * float(geom_score[min(gmax, max(0, round(p[k])))])
        resid = (p[k] - x_expected[boundary_ix - 1]) / span
        return g - lam * resid * resid

    NEG = -1e18
    # dp[i][k]: best score for the first i letters, internal boundary i at cand k.
    dp = np.full((L, n), NEG)
    back = np.full((L, n), -1, dtype=int)
    for k in range(1, n - 1):  # boundary 1 (between letter 0 and 1)
        if p[k] - p[0] >= min_w:
            dp[1][k] = alpha * emission(word[0], 0, k) + reg(1, k)
    for i in range(2, L):  # boundaries 2 .. L-1
        for k in range(i, n - 1):
            best, bj = NEG, -1
            for j in range(i - 1, k):
                if p[k] - p[j] < min_w or dp[i - 1][j] <= NEG / 2:
                    continue
                cand = dp[i - 1][j] + alpha * emission(word[i - 1], j, k)
                if cand > best:
                    best, bj = cand, j
            if bj >= 0:
                dp[i][k] = best + reg(i, k)
                back[i][k] = bj

    # close the last letter [boundary L-1, x_max] and read the best path end.
    best, bk = NEG, -1
    for k in range(L - 1, n - 1):
        if dp[L - 1][k] <= NEG / 2 or p[n - 1] - p[k] < min_w:
            continue
        cand = dp[L - 1][k] + alpha * emission(word[L - 1], k, n - 1)
        if cand > best:
            best, bk = cand, k
    if bk < 0:  # no valid alignment -> even split
        return list(np.linspace(x_min, x_max, L + 1)[1:-1])

    chosen: list[float] = []
    k, i = bk, L - 1
    while i >= 1 and k >= 0:
        chosen.append(float(p[k]))
        k = back[i][k]
        i -= 1
    chosen.reverse()
    return sp._ensure_count_x(chosen, L - 1, x_min, x_max)


# --- 3. shared prep + geometry boundaries (mirror of _segment_prototype) -----
# ponytail: ~15 lines duplicated from _segment_prototype.segment_word so that module
# (1400 lines, fully tested) is not refactored. Both must stay in step; if the clean
# pipeline there changes, mirror it here.


def prepare(page_image, box_2d: list[int], word: str, pitch_px: float) -> dict | None:
    """Clean + trace + de-slant one word; return the intermediates both cut paths
    need, or ``None`` if the word is empty/degenerate."""
    clean_pil, binary, crop_box = vectorize.clean_word(page_image, box_2d)
    gray = np.asarray(clean_pil)
    binary, gray, crop_box = sp.restrict_to_word_band(
        binary, gray, box_2d, crop_box, page_image.size, pitch_px
    )
    binary = sp.strip_ruled_line(binary, box_2d, crop_box, page_image.size)
    binary, gray, crop_box = sp.strip_binding_and_neighbors(
        binary, gray, box_2d, crop_box, page_image.size
    )
    binary, gray, _ = sp.reject_noise(binary, gray)
    h, w = binary.shape
    strokes = vectorize.trace_ink(binary)
    if not strokes:
        return None
    L = max(1, len(word))
    body, diacritics = sp.classify_components(strokes, (h, w), L)
    s_slant, prom = sp.estimate_slant(binary)
    if prom < sp.SLANT_MIN_PROM:
        s_slant = 0.0
    shift = sp._shear_shifts(h, s_slant)
    sbin = sp._shear_binary(binary, shift) if s_slant else binary
    body_s = sp._shear_points(body, shift) if s_slant else body
    body_pts = np.concatenate([np.asarray(pp, dtype=float) for pp in body_s])
    x_min, x_max = float(body_pts[:, 0].min()), float(body_pts[:, 0].max())
    if x_max - x_min < 2:
        return None
    return {
        "binary": binary,
        "gray": gray,
        "sbin": sbin,
        "body": body,
        "diacritics": diacritics,
        "shift": shift,
        "s_slant": s_slant,
        "x_min": x_min,
        "x_max": x_max,
        "h": h,
        "w": w,
    }


def geometry_boundaries(prep: dict, word: str) -> tuple[list[float], np.ndarray]:
    """The existing geometry cut: x-density profile (topology fallback on a connected
    blob) + width-prior DP. Returns ``(boundaries, geom_score_profile)``."""
    sbin, x_min, x_max = prep["sbin"], prep["x_min"], prep["x_max"]
    L = len(word)
    min_gap = 0.40 * (x_max - x_min) / L

    def cuts_for(profile: np.ndarray) -> list[float]:
        cand_x, cand_score = sp.grid_candidates(profile, x_min, x_max)
        b = sp.dp_boundaries_x(
            cand_x, cand_score, sp.expected_boundary_x(word, x_min, x_max), x_min, x_max, min_gap
        )
        return sp._ensure_count_x(b, L - 1, x_min, x_max)

    score = sp.cut_score_profile(sbin, x_min, x_max)
    bxs = cuts_for(score)
    if (
        sp.is_connected_blob(sbin, x_min, x_max)
        and sp.max_cut_gap(bxs, x_min, x_max) > sp.BLOB_XDENS_MAXGAP * (x_max - x_min) / L
    ):
        score = sp.topology_cut_score(sbin, x_min, x_max)
        bxs = cuts_for(score)
    return bxs, score


def recognition_boundaries(
    prep: dict,
    word: str,
    geom_score: np.ndarray,
    alpha: float = ALIGN_ALPHA,
    recognizer: LetterRecognizer | None = None,
) -> list[float]:
    """Recognition-guided cut: forced alignment over candidate columns, geometry +
    width prior blended in. Falls back to geometry candidates for the column grid."""
    sbin, x_min, x_max = prep["sbin"], prep["x_min"], prep["x_max"]
    cand_x, _ = sp.grid_candidates(geom_score, x_min, x_max)
    return align_boundaries(
        sbin, word, x_min, x_max, cand_x, geom_score, recognizer=recognizer, alpha=alpha
    )


# --- 4. measurement ---------------------------------------------------------


def recognition_accuracy(
    prep: dict, word: str, boundaries: list[float], recognizer: LetterRecognizer | None = None
) -> dict:
    """Recognise each letter slice cut at ``boundaries`` (in de-sheared space) and
    score against the KNOWN char: top-1, case-insensitive top-1, mean true-char rank.
    This is the real metric -- a higher top-1 here means the cut produced slices that
    genuinely look more like their letters. ``recognizer`` defaults to the font-template
    bank; pass a trained ``CNNRecognizer`` to measure with the bootstrap recognizer."""
    rec = recognizer or get_recognizer()
    cuts = [prep["x_min"], *sorted(boundaries), prep["x_max"]]
    slices = slice_columns(prep["sbin"], cuts)
    top1 = ci_top1 = 0
    ranks: list[int] = []
    detail: list[str] = []
    for ch, sl in zip(word, slices, strict=False):
        ranking = rec.ranking(sl) if sl.size else []
        pred = ranking[0] if ranking else "?"
        rank = ranking.index(ch) + 1 if ch in ranking else len(CHARS)
        top1 += pred == ch
        ci_top1 += pred.lower() == ch.lower()
        ranks.append(rank)
        detail.append(f"{ch}->{pred}({rank})")
    n = max(1, len(word))
    return {
        "top1": top1,
        "ci_top1": ci_top1,
        "n": len(word),
        "mean_rank": float(np.mean(ranks)) if ranks else float(len(CHARS)),
        "top1_frac": top1 / n,
        "ci_top1_frac": ci_top1 / n,
        "detail": " ".join(detail),
    }


def evaluate_word(
    page_image,
    box_2d,
    word,
    pitch_px,
    overlay_dir=None,
    alpha: float = ALIGN_ALPHA,
    recognizer: LetterRecognizer | None = None,
) -> dict | None:
    """Cut one word both ways and measure. Optionally writes geometry/recognition
    overlays for visual QA. ``recognizer`` (font templates by default) is used BOTH to
    guide the recognition cut AND to score per-letter top-1 in both rows."""
    word = word.strip()
    prep = prepare(page_image, box_2d, word, pitch_px)
    if prep is None or len(word) < 2:
        return None
    geo_bxs, geom_score = geometry_boundaries(prep, word)
    rec_bxs = recognition_boundaries(prep, word, geom_score, alpha=alpha, recognizer=recognizer)
    geo = recognition_accuracy(prep, word, geo_bxs, recognizer=recognizer)
    rec = recognition_accuracy(prep, word, rec_bxs, recognizer=recognizer)
    conf = sp.compute_confidence(geom_score, geo_bxs, word, prep["x_min"], prep["x_max"])
    if overlay_dir:
        paths.ensure_dir(overlay_dir)
        _write_overlay(
            prep, word, geo_bxs, os.path.join(overlay_dir, f"geo_{paths.slugify(word)}.png")
        )
        _write_overlay(
            prep, word, rec_bxs, os.path.join(overlay_dir, f"rec_{paths.slugify(word)}.png")
        )
    return {
        "word": word,
        "L": len(word),
        "geom_conf": conf,
        "geo": geo,
        "rec": rec,
        "geo_bxs": geo_bxs,
        "rec_bxs": rec_bxs,
    }


def _write_overlay(prep: dict, word: str, bxs: list[float], path: str) -> None:
    """Reuse the geometry cutter's seam + overlay machinery for a like-for-like PNG."""
    seams = sp.build_cut_seams(
        prep["binary"],
        bxs,
        prep["shift"],
        prep["s_slant"],
        prep["x_min"],
        prep["x_max"],
        len(word),
    )
    groups = sp.slice_by_seams(prep["body"], seams, len(word))
    groups = sp.attach_diacritics_seams(groups, seams, prep["diacritics"], len(word))
    paths.ensure_parent(path)
    cv2.imwrite(
        path, sp.draw_overlay(prep["gray"], groups, word, bxs, shift=prep["shift"], seams=seams)
    )


# --- CLI --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recognition-guided cut eval (geometry vs forced align)"
    )
    p.add_argument("--pdf", default=sp.PDF_PATH, help="Source PDF (slug or path)")
    p.add_argument("--page", type=int, default=4, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for the source page")
    p.add_argument("--overlay-dir", default="runs/recog", help="Where to write geo/rec overlays")
    p.add_argument("--no-overlays", action="store_true", help="Skip writing overlay PNGs")
    p.add_argument(
        "--alpha",
        type=float,
        default=ALIGN_ALPHA,
        help="Forced-alignment recognition (emission) weight",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pdf_path = sp._resolve_pdf(args.pdf)
    version = (
        args.version if args.version is not None else paths.latest_version(pdf_path, args.page)
    )
    page_image = sp.render_page(pdf_path, args.page, dpi=args.dpi)
    with open(paths.boxes_json(pdf_path, args.page, version)) as f:
        boxes = json.load(f)
    pitch_px = sp._row_pitch(boxes, page_image.size[1])
    overlay_dir = None if args.no_overlays else args.overlay_dir

    rows: list[dict] = []
    for index, expected in sp.DEFAULT_SAMPLE.items():
        try:
            entry = boxes[index]
            word = sp._read_word(paths.box_text(pdf_path, args.page, index, version), expected)
        except (IndexError, KeyError):
            continue
        try:
            r = evaluate_word(
                page_image, entry["box_2d"], word or expected, pitch_px, overlay_dir, args.alpha
            )
        except Exception as e:  # eval CLI: report the word and keep going
            print(f"  {word}: ERROR {type(e).__name__}: {e}")
            r = None
        if r:
            rows.append(r)
    _print_report(rows, overlay_dir)


def _print_report(rows: list[dict], overlay_dir: str | None) -> None:
    print(f"\n{'word':<11}{'L':>2}  {'conf':>5} | {'GEOMETRY cut':^22} | {'RECOGNITION cut':^22}")
    print(
        f"{'':<13}{'':>5} | {'top1':>5} {'ci':>5} {'rank':>6} | {'top1':>5} {'ci':>5} {'rank':>6}"
    )
    print("-" * 74)
    g1 = c1 = r1 = rc1 = tot = 0
    gr_sum = rr_sum = 0.0
    for r in rows:
        g, rc = r["geo"], r["rec"]
        print(
            f"{r['word']:<11}{r['L']:>2}  {r['geom_conf']:>5.2f} | "
            f"{g['top1']:>2}/{g['n']:<2} {g['ci_top1']:>2}/{g['n']:<2} {g['mean_rank']:>6.1f} | "
            f"{rc['top1']:>2}/{rc['n']:<2} {rc['ci_top1']:>2}/{rc['n']:<2} {rc['mean_rank']:>6.1f}"
        )
        g1 += g["top1"]
        c1 += g["ci_top1"]
        r1 += rc["top1"]
        rc1 += rc["ci_top1"]
        tot += g["n"]
        gr_sum += g["mean_rank"] * g["n"]
        rr_sum += rc["mean_rank"] * rc["n"]
    print("-" * 74)
    if tot:
        print(
            f"{'TOTAL':<11}{'':>2}  {'':>5} | "
            f"{g1:>2}/{tot:<2} {c1:>2}/{tot:<2} {gr_sum / tot:>6.1f} | "
            f"{r1:>2}/{tot:<2} {rc1:>2}/{tot:<2} {rr_sum / tot:>6.1f}"
        )
        print(
            f"\ntop-1 recognition: geometry {g1}/{tot}={g1 / tot:.0%}  "
            f"recognition-guided {r1}/{tot}={r1 / tot:.0%}  (delta {(r1 - g1) / tot:+.0%})"
        )
        print(
            f"case-insensitive : geometry {c1}/{tot}={c1 / tot:.0%}  "
            f"recognition-guided {rc1}/{tot}={rc1 / tot:.0%}  (delta {(rc1 - c1) / tot:+.0%})"
        )
    if overlay_dir:
        print(f"\noverlays: {os.path.abspath(overlay_dir)}/ (geo_*.png vs rec_*.png)")


if __name__ == "__main__":
    main()
