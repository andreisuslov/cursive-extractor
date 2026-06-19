"""EXPERIMENTAL transcription-forced per-letter segmentation of a traced word.

Given a word crop (``box.jpg``) and the KNOWN text (``text_recognized.txt`` /
``metadata.asciiSequence``), cut the live-traced ink into exactly ``L = len(word)``
contiguous per-letter stroke groups. The transcription supplies the letter count
(the hard constraint) and a per-character width prior; a dynamic program then
places the ``L-1`` internal boundaries on the highest-scoring cut candidates near
their expected arc-length positions.

Why a NEW module (and why ``_``-prefixed = prototype): this reuses ``ocr.vectorize``
(live re-trace -- NOT the stale fragmented ``*_strokes.json``) and ``ocr.paths``
only, adds numpy + cv2, and writes per-letter ``{points, metadata}`` files that
round-trip through ``data.decompose_offsets``/``strokes_to_offsets`` unchanged.

Algorithm (see the module docstring of the parent spec for the full rationale):

  0. LOAD + LETTER SLOTS   -- ``L = len(word)``; ``L == 1`` -> emit whole word.
  1. LIVE TRACE            -- ``vectorize.preprocess`` + ``vectorize.trace_ink``.
  2. CLASSIFY components   -- split BODY runs from DIACRITICS (dots / crosses).
  3. CONCATENATE           -- one polyline P + free pen-lift boundary candidates
                              + cumulative arc-length (same ``hypot`` as data.py).
  4. CUT-CANDIDATE SCORES  -- baseline fit + weighted cut score c[i] in [0,1].
  5. TRANSCRIPTION DP      -- choose L-1 monotone boundaries maximising
                              sum(c) - lambda * (arc-length residual)^2.
  6. SLICE + RE-ATTACH     -- cut P; assign each diacritic to a slot by x-centre.
  7. EMIT                  -- format_strokes per slot -> letter_<kk>_<char>.json.
  8. CONFIDENCE + OVERLAY  -- conf gate (low_confidence flag) + colour overlay PNG.

CLI::

    python -m ocr._segment_prototype --pdf test_document --page 4 \
        [--version N] [--box III] [--min-conf 0.4]

With no ``--box`` it segments a default sample of page-4 words and also writes the
colour overlays to ``runs/seg_proto_<word>.png`` for visual QA.
"""

import argparse
import json
import os
from itertools import pairwise

import cv2
import numpy as np

from . import paths, vectorize
from .config import PDF_PATH

# --- tunables (Step 4 weights + Step 5 lambda) ------------------------------
W1_PENLIFT = 1.0  # free real pen-lift between joined body runs (strongest)
W2_BASELINE = 0.5  # proximity to the fitted baseline (ligatures ride it)
W3_YMAX = 0.5  # local y-maximum (pen low on the page = baseline trough)
W4_HORIZ = 0.4  # near-horizontal local tangent (thin connector)
W5_THIN = 0.4  # thin ink column (a join, not a stem)
LOOP_PENALTY = 0.6  # subtract inside a bowl (o/e/a) so we never cut a loop
DP_LAMBDA = 6.0  # arc-length residual weight in the DP objective
DP_BAND = 0.30  # search band +-30% of S around each expected boundary

# Per-character width prior (relative). Wide letters get more arc-length; thin
# stems (i, l, t, ...) get less. Anything not listed is 1.0.
WIDTH_PRIOR = {
    "m": 1.8,
    "w": 1.8,
    "M": 1.8,
    "W": 1.8,
    "i": 0.6,
    "l": 0.6,
    "t": 0.6,
    "I": 0.6,
    "j": 0.6,
    ".": 0.5,
    ",": 0.5,
    "'": 0.4,
}

# Distinct, saturated BGR colours cycled per letter slot (matches the teal-style
# overlay palette used in vectorize / _overlay_inspect).
PALETTE = [
    (0, 0, 230),
    (200, 130, 0),
    (0, 160, 0),
    (200, 0, 200),
    (0, 150, 255),
    (255, 160, 0),
    (130, 0, 200),
    (0, 110, 110),
    (110, 70, 0),
    (0, 0, 0),
]


# --- Step 1 helpers ---------------------------------------------------------


def load_gray(box_image_path: str) -> np.ndarray:
    """Read ``box.jpg`` as grayscale; raise if missing."""
    gray = cv2.imread(box_image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read box image: {box_image_path}")
    return gray


# --- Step 2: classify components into BODY vs DIACRITIC ---------------------


def _ipt(x: float, y: float) -> tuple[int, int]:
    """Round a (possibly numpy-float) coordinate to a Python int pixel tuple."""
    return (int(np.rint(x)), int(np.rint(y)))


def _stroke_bbox(stroke: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in stroke]
    ys = [p[1] for p in stroke]
    return min(xs), min(ys), max(xs), max(ys)


def classify_components(
    strokes: list[list[tuple[int, int]]], shape: tuple[int, int], n_slots: int
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Split traced strokes into BODY runs (kept, in reading order) and DIACRITICS.

    A stroke is a DIACRITIC when it is tiny (``< max(12, 0.04 * total_pts)``) OR
    sits entirely in the top ~30% ascender band while being narrow
    (``x-width < 0.5 * W / L``). Everything else is body ink.
    """
    h, _w = shape
    total_pts = sum(len(s) for s in strokes)
    tiny = max(12, 0.04 * total_pts)
    median_letter_width = shape[1] / max(1, n_slots)
    band_y = 0.30 * h

    body: list[list[tuple[int, int]]] = []
    diacritics: list[list[tuple[int, int]]] = []
    for stroke in strokes:
        x0, _y0, x1, y1 = _stroke_bbox(stroke)
        width = x1 - x0
        is_tiny = len(stroke) < tiny
        in_band = y1 <= band_y and width < 0.5 * median_letter_width
        if is_tiny or in_band:
            diacritics.append(stroke)
        else:
            body.append(stroke)
    # Guard: never route ALL ink to diacritics (e.g. a short word of dots).
    if not body and strokes:
        body = [max(strokes, key=len)]
        diacritics = [s for s in strokes if s is not body]
    return body, diacritics


# --- Step 3: concatenate body runs + free pen-lift boundaries + arc-length --


def concatenate_body(
    body: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, set[int], np.ndarray, float]:
    """Concatenate body runs (reading order) into one polyline ``P``.

    Returns ``(P, idx_penlift, s, S)``:
      * ``P``           -- (N, 2) float array of (x, y) pixels.
      * ``idx_penlift`` -- P-indices that are the LAST point of each body run
                           except the last (FREE candidate boundaries).
      * ``s``           -- cumulative arc-length (same ``hypot`` as data.py).
      * ``S``           -- total arc-length.
    """
    pts: list[tuple[int, int]] = []
    idx_penlift: set[int] = set()
    for k, run in enumerate(body):
        pts.extend(run)
        if k < len(body) - 1:
            idx_penlift.add(len(pts) - 1)  # last point of this run
    P = np.asarray(pts, dtype=float)
    if len(P) < 2:
        return P, idx_penlift, np.zeros(len(P)), 0.0
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return P, idx_penlift, s, float(s[-1])


# --- Step 4: baseline fit + cut-candidate scores ----------------------------


def fit_baseline(P: np.ndarray, n_bins: int = 24) -> np.ndarray:
    """Baseline y as a function of x: bin P by x, take p90 of y per bin (y is
    top-down so the baseline is HIGH y), smooth, then evaluate per P-index.
    """
    x = P[:, 0]
    y = P[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        return np.full(len(P), float(np.percentile(y, 90)))
    edges = np.linspace(xmin, xmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(n_bins, np.nan)
    bidx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        yb = y[bidx == b]
        if yb.size:
            vals[b] = np.percentile(yb, 90)
    # fill empty bins, then 3-tap box smooth
    good = ~np.isnan(vals)
    if good.any():
        vals = np.interp(centers, centers[good], vals[good])
    else:
        vals[:] = float(np.percentile(y, 90))
    kernel = np.array([0.25, 0.5, 0.25])
    vals = np.convolve(np.pad(vals, 1, mode="edge"), kernel, mode="valid")
    return np.interp(x, centers, vals)


def _ink_column_counts(binary: np.ndarray) -> np.ndarray:
    """Per-column ink-pixel count (Step 4 w5: thin column == a join, not a stem)."""
    return (binary > 0).sum(axis=0).astype(float)


def _local_window(arr: np.ndarray, i: int, half: int) -> np.ndarray:
    lo = max(0, i - half)
    hi = min(len(arr), i + half + 1)
    return arr[lo:hi]


def cut_scores(
    P: np.ndarray,
    idx_penlift: set[int],
    baseline_y: np.ndarray,
    binary: np.ndarray,
) -> np.ndarray:
    """Per-P-index cut score ``c[i]`` in [0, 1] = weighted sum of the Step-4 cues
    minus a loop penalty. Higher = more likely a true between-letter boundary.
    """
    n = len(P)
    c = np.zeros(n)
    if n < 3:
        return c
    x, y = P[:, 0], P[:, 1]
    _h, w = binary.shape

    # w2: proximity to baseline (1 when on the baseline, decaying with distance).
    span = max(1.0, float(y.max() - y.min()))
    base_prox = np.clip(1.0 - np.abs(y - baseline_y) / (0.4 * span), 0.0, 1.0)

    # w3: local y-maximum (low pen on the page). Window scaled to point density.
    half = max(3, n // 60)

    # w4: horizontality of local tangent (|dy/dx| small with forward dx>0).
    dx = np.gradient(x)
    dy = np.gradient(y)
    horiz = np.where(
        dx > 0.05,
        np.clip(1.0 - np.abs(dy) / (np.abs(dx) + 1e-6), 0.0, 1.0),
        0.0,
    )

    # w5: thin ink column (normalised against the median non-trivial column).
    col = _ink_column_counts(binary)
    nz = col[col > 0]
    med_col = float(np.median(nz)) if nz.size else 1.0
    xi = np.clip(np.round(x).astype(int), 0, w - 1)
    thin = np.clip(1.0 - col[xi] / (2.0 * med_col + 1e-6), 0.0, 1.0)

    # loop detection: x reverses then returns within a local window (bowl of o/e/a).
    for i in range(n):
        win = _local_window(x, i, half)
        is_ymax = 1.0 if y[i] >= _local_window(y, i, half).max() - 1e-9 else 0.0
        # local x non-monotonic (sign change of dx) => inside a loop
        wdx = np.diff(win)
        in_loop = 1.0 if (wdx > 0).any() and (wdx < 0).any() else 0.0
        penlift = 1.0 if i in idx_penlift else 0.0
        c[i] = (
            W1_PENLIFT * penlift
            + W2_BASELINE * base_prox[i]
            + W3_YMAX * is_ymax
            + W4_HORIZ * horiz[i]
            + W5_THIN * thin[i]
            - LOOP_PENALTY * in_loop
        )
    # normalise to [0,1] by the max achievable (pen-lift dominates the scale).
    denom = W1_PENLIFT + W2_BASELINE + W3_YMAX + W4_HORIZ + W5_THIN
    c = np.clip(c / denom, 0.0, 1.0)
    # pen-lift points are FREE boundaries: pin them to a high score.
    for i in idx_penlift:
        c[i] = max(c[i], 0.95)
    return c


def non_max_suppress(c: np.ndarray, s: np.ndarray, min_sep: float) -> np.ndarray:
    """Suppress candidates within ``min_sep`` arc-length of a higher-scoring one.
    Returns a boolean mask of surviving candidate indices."""
    n = len(c)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-c)
    for i in order:
        if not keep[i]:
            continue
        near = np.abs(s - s[i]) < min_sep
        near[i] = False
        keep[near] = False
    return keep


# --- Step 5: transcription-forced DP ---------------------------------------


def char_width_priors(word: str) -> list[float]:
    return [WIDTH_PRIOR.get(ch, 1.0) for ch in word]


def expected_boundary_arclengths(word: str, S: float) -> np.ndarray:
    """Expected cumulative arc-length at each of the ``L-1`` internal boundaries,
    from the per-char width prior."""
    w = np.asarray(char_width_priors(word), dtype=float)
    total = w.sum()
    cum = np.cumsum(w)[:-1]  # boundaries are between chars
    return S * cum / total


def dp_boundaries(
    c: np.ndarray,
    s: np.ndarray,
    S: float,
    candidate_mask: np.ndarray,
    s_expected: np.ndarray,
    lam: float = DP_LAMBDA,
    band: float = DP_BAND,
) -> list[int]:
    """Choose exactly ``L-1`` monotone boundary P-indices maximising
    ``sum_j [ c[b_j] - lam * ((s[b_j]-s_expected_j)/S)^2 ]``.

    Banded DP over (boundary j, candidate index): O(L * N). Each boundary may
    only land within ``+-band*S`` of its expected arc-length, which keeps the
    search small and the boundaries near their prior positions; pen-lift
    candidates (c ~= 0.95) dominate so the DP lands on free boundaries first.
    """
    m = len(s_expected)  # number of internal boundaries = L-1
    cand = np.flatnonzero(candidate_mask)
    if m == 0:
        return []
    if len(cand) < m:
        # not enough disconnected candidates -> fall back to every interior index
        cand = np.arange(1, len(s) - 1)

    def local_score(i: int, j: int) -> float:
        resid = (s[i] - s_expected[j]) / (S + 1e-9)
        return c[i] - lam * resid * resid

    NEG = -1e18
    # dp[j][k] = best total score using boundary j placed at candidate cand[k],
    # with boundaries 0..j monotonically increasing in index.
    n_cand = len(cand)
    dp = np.full((m, n_cand), NEG)
    back = np.full((m, n_cand), -1, dtype=int)

    for k, idx in enumerate(cand):
        if abs(s[idx] - s_expected[0]) <= band * S:
            dp[0, k] = local_score(idx, 0)
    for j in range(1, m):
        for k, idx in enumerate(cand):
            if abs(s[idx] - s_expected[j]) > band * S:
                continue
            base = local_score(idx, j)
            # best previous boundary strictly before this index
            best_prev = NEG
            best_pk = -1
            for pk in range(k):
                if cand[pk] < idx and dp[j - 1, pk] > best_prev:
                    best_prev = dp[j - 1, pk]
                    best_pk = pk
            if best_prev > NEG / 2:
                dp[j, k] = base + best_prev
                back[j, k] = best_pk

    # If the banded DP failed (no valid path), widen the band progressively.
    if dp[m - 1].max() <= NEG / 2:
        if band < 1.0:
            return dp_boundaries(c, s, S, candidate_mask, s_expected, lam, min(1.0, band * 2))
        # last resort: evenly spaced interior indices
        return [int(x) for x in np.round(np.linspace(1, len(s) - 2, m))]

    last_k = int(np.argmax(dp[m - 1]))
    chosen: list[int] = []
    j = m - 1
    k = last_k
    while j >= 0 and k >= 0:
        chosen.append(int(cand[k]))
        k = back[j, k]
        j -= 1
    chosen.reverse()
    return chosen


# --- Step 6: slice + re-attach diacritics -----------------------------------


def slice_polyline(P: np.ndarray, boundaries: list[int]) -> list[np.ndarray]:
    """Cut ``P`` at the boundary indices into contiguous per-char pixel runs."""
    cuts = [0, *sorted(boundaries), len(P)]
    slots: list[np.ndarray] = []
    for a, b in pairwise(cuts):
        b = max(b, a + 1)
        slots.append(P[a:b])
    return slots


def slot_xspans(slots: list[np.ndarray]) -> list[tuple[float, float]]:
    spans = []
    for sl in slots:
        spans.append((float(sl[:, 0].min()), float(sl[:, 0].max())))
    return spans


def attach_diacritics(
    slots: list[np.ndarray],
    diacritics: list[list[tuple[int, int]]],
) -> list[list[np.ndarray]]:
    """Build per-slot sub-stroke lists: the body slice + any diacritic whose
    x-centre falls within (or nearest to) the slot's x-span."""
    spans = slot_xspans(slots)
    groups: list[list[np.ndarray]] = [[sl] for sl in slots]
    for d in diacritics:
        dx0, _dy0, dx1, _dy1 = _stroke_bbox(d)
        xc = 0.5 * (dx0 + dx1)
        slot = None
        for i, (lo, hi) in enumerate(spans):
            if lo <= xc <= hi:
                slot = i
                break
        if slot is None:  # nearest by span midpoint
            slot = int(np.argmin([abs(xc - 0.5 * (lo + hi)) for lo, hi in spans]))
        groups[slot].append(np.asarray(d, dtype=float))
    return groups


# --- Step 7/8: emit per-letter files + confidence + overlay -----------------


def compute_confidence(
    c: np.ndarray, boundaries: list[int], slots: list[np.ndarray], word: str, S: float
) -> float:
    """conf = mean(c at boundaries) * (1 - mean width-prior residual)."""
    # single letter (no cut) scores cut-quality 1.0
    cut_quality = float(np.mean([c[b] for b in boundaries])) if boundaries else 1.0
    priors = np.asarray(char_width_priors(word), dtype=float)
    target = priors / priors.sum()
    actual = np.asarray([len(sl) for sl in slots], dtype=float)
    actual = actual / max(1.0, actual.sum())
    width_resid = float(np.mean(np.abs(actual - target)))
    return float(np.clip(cut_quality * (1.0 - width_resid), 0.0, 1.0))


def emit_letter_files(
    groups: list[list[np.ndarray]],
    word: str,
    shape: tuple[int, int],
    box_id: str,
    boundary_fracs: list[float],
    conf: float,
    low_conf: bool,
    out_dir: str,
    author: str = "robot",
) -> list[str]:
    """Write ``letters/letter_<kk>_<char>.json`` for every slot. Each file is in
    the native ``{points, metadata}`` schema via ``vectorize.format_strokes`` so it
    round-trips through ``data`` unchanged."""
    h, w = shape
    letters_dir = paths.ensure_dir(os.path.join(out_dir, "letters"))
    written: list[str] = []
    for k, (ch, substrokes) in enumerate(zip(word, groups, strict=False)):
        sub_px = [[_ipt(x, y) for x, y in sub] for sub in substrokes]
        points = vectorize.format_strokes(sub_px, w, h)
        safe_ch = _safe_char(ch)
        path = os.path.join(letters_dir, f"letter_{k:02d}_{safe_ch}.json")
        payload = {
            "points": points,
            "metadata": {
                "asciiSequence": ch,
                "author": author,
                "parent_box": box_id,
                "boundary_fracs": boundary_fracs,
                "confidence": round(conf, 4),
                "low_confidence": low_conf,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        written.append(path)
    return written


def _safe_char(ch: str) -> str:
    """Filesystem-safe single-char tag (digits/letters pass; others -> name)."""
    if ch.isalnum():
        return ch
    names = {
        ".": "dot",
        ",": "comma",
        "'": "apos",
        "-": "dash",
        "/": "slash",
        ";": "semi",
        ":": "colon",
        "!": "bang",
        "?": "qmark",
        '"': "quote",
        "(": "lparen",
        ")": "rparen",
    }
    return names.get(ch, f"u{ord(ch)}")


def _fade(gray: np.ndarray, keep: float = 0.32) -> np.ndarray:
    g = (255 - (255 - gray.astype(float)) * keep).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def draw_overlay(
    gray: np.ndarray,
    groups: list[list[np.ndarray]],
    word: str,
    boundaries: list[int],
    P: np.ndarray,
    lw: int = 3,
) -> np.ndarray:
    """Colour overlay: faded ink background, one colour per letter slot, a dashed
    vertical line at each boundary x, and the slot char labelled above it."""
    img = _fade(gray)
    h = img.shape[0]
    for k, substrokes in enumerate(groups):
        col = PALETTE[k % len(PALETTE)]
        for sub in substrokes:
            pts = [_ipt(x, y) for x, y in sub]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], col, lw, cv2.LINE_AA)
            if pts:
                cv2.circle(img, pts[0], 4, col, -1)
                cv2.circle(img, pts[0], 4, (255, 255, 255), 1)
        # label this slot's char near its start
        ch = word[k] if k < len(word) else "?"
        if substrokes and len(substrokes[0]):
            lx = int(substrokes[0][0][0])
            cv2.putText(
                img, ch, (lx, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA
            )
    # dashed vertical boundary lines
    for b in boundaries:
        bx = int(np.rint(P[b, 0]))
        for y0 in range(0, h, 12):
            cv2.line(img, (bx, y0), (bx, min(h, y0 + 6)), (40, 40, 40), 1, cv2.LINE_AA)
    return img


# --- Orchestration ----------------------------------------------------------


def segment_word(
    box_image_path: str,
    word: str,
    box_id: str,
    out_dir: str | None,
    min_conf: float,
    overlay_path: str | None = None,
    write_files: bool = True,
) -> dict:
    """Segment one word crop into ``L = len(word)`` per-letter stroke groups.

    Returns a QA dict with the word, L, free-pen-lift / internal-cut counts,
    confidence, low_confidence flag, and the count-gate result
    (``segments == L``). Writes per-letter files (if ``write_files`` and
    ``out_dir``) and a colour overlay (if ``overlay_path``).
    """
    word = word.strip()
    L = len(word)
    gray = load_gray(box_image_path)  # Step 0/1
    binary = vectorize.preprocess(gray)
    h, w = binary.shape

    if L <= 1:  # Step 0: single char -> whole word is one letter
        strokes = vectorize.trace_ink(binary)
        groups = [[np.asarray(s, dtype=float) for s in strokes]] if strokes else [[]]
        conf = 1.0
        if overlay_path:
            paths.ensure_parent(overlay_path)
            cv2.imwrite(overlay_path, draw_overlay(gray, groups, word or "?", [], np.zeros((1, 2))))
        if write_files and out_dir:
            emit_letter_files(groups, word or "?", (h, w), box_id, [], conf, False, out_dir)
        return _qa(word, max(1, L), 0, 0, conf, False, len(groups))

    strokes = vectorize.trace_ink(binary)  # Step 1
    body, diacritics = classify_components(strokes, (h, w), L)  # Step 2
    P, idx_penlift, s, S = concatenate_body(body)  # Step 3
    if len(P) < 2 or S <= 0:
        return _qa(word, L, 0, 0, 0.0, True, 0, error="empty trace")

    baseline_y = fit_baseline(P)  # Step 4
    c = cut_scores(P, idx_penlift, baseline_y, binary)
    min_sep = 0.5 * (S / L)
    keep = non_max_suppress(c, s, min_sep)
    # always keep pen-lift candidates (they survive NMS by construction, but be safe)
    for i in idx_penlift:
        keep[i] = True

    s_expected = expected_boundary_arclengths(word, S)  # Step 5
    boundaries = dp_boundaries(c, s, S, keep, s_expected)
    boundaries = sorted({int(b) for b in boundaries if 0 < b < len(P) - 1})
    # If DP returned fewer than L-1 (degenerate), pad with evenly spaced indices.
    boundaries = _ensure_count(boundaries, L - 1, len(P))

    slots = slice_polyline(P, boundaries)  # Step 6
    groups = attach_diacritics(slots, diacritics)

    n_free = sum(1 for b in boundaries if b in idx_penlift)
    n_internal = len(boundaries) - n_free
    boundary_fracs = [round(float(s[b] / S), 4) for b in boundaries]
    conf = compute_confidence(c, boundaries, slots, word, S)  # Step 8
    low_conf = conf < min_conf

    if overlay_path:
        paths.ensure_parent(overlay_path)
        cv2.imwrite(overlay_path, draw_overlay(gray, groups, word, boundaries, P))
    if write_files and out_dir:
        emit_letter_files(
            groups, word, (h, w), box_id, boundary_fracs, conf, low_conf, out_dir
        )

    # Step 8 count-reconciliation gate (mirrors ocr.qa): emitted == L.
    assert len(groups) == L, f"segment count {len(groups)} != L {L} for {word!r}"
    return _qa(word, L, n_free, n_internal, conf, low_conf, len(groups))


def _ensure_count(boundaries: list[int], want: int, n: int) -> list[int]:
    """Force exactly ``want`` distinct interior boundary indices."""
    b = sorted(set(boundaries))
    if len(b) == want:
        return b
    if len(b) > want:
        return b[:want]  # keep the earliest (DP already ranked by score+position)
    # pad with evenly spaced interior indices not already present
    existing = set(b)
    for cand in np.linspace(1, n - 2, want + 2)[1:-1]:
        ci = int(np.rint(cand))
        while ci in existing and ci < n - 2:
            ci += 1
        if ci not in existing and 0 < ci < n - 1:
            existing.add(ci)
            b.append(ci)
        if len(b) == want:
            break
    return sorted(set(b))[:want]


def _qa(
    word: str,
    L: int,
    n_free: int,
    n_internal: int,
    conf: float,
    low_conf: bool,
    n_segments: int,
    error: str | None = None,
) -> dict:
    return {
        "word": word,
        "L": L,
        "free_penlift_boundaries": n_free,
        "dp_internal_cuts": n_internal,
        "confidence": round(conf, 4),
        "low_confidence": low_conf,
        "n_segments": n_segments,
        "count_gate_ok": n_segments == L,
        "error": error,
    }


# Default sample of page-4 words to segment + eye-check when no --box is given.
DEFAULT_SAMPLE = {
    1: "Realized",
    2: "that",
    4: "could",
    5: "translate",
    10: "savings",
    30: "This",
    22: "the",
    8: "wealth",
    16: "pleasure",
    24: "without",
    50: "through",
    37: "period",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXPERIMENTAL transcription-forced per-letter segmentation"
    )
    p.add_argument("--pdf", default=PDF_PATH, help="Source PDF (slug or path)")
    p.add_argument("--page", type=int, default=4, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--box", default=None, help="Box index (e.g. 004). Omit for a sample.")
    p.add_argument("--min-conf", type=float, default=0.4, help="Confidence gate")
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Where to also write seg_proto_<word>.png overlays (sample mode)",
    )
    return p.parse_args(argv)


def _read_word(text_path: str) -> str:
    with open(text_path) as f:
        return f.read().strip()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page)
    )

    def box_paths(index: int) -> tuple[str, str, str, str]:
        img = paths.box_image(args.pdf, args.page, index, version)
        txt = paths.box_text(args.pdf, args.page, index, version)
        out_dir = paths.box_dir(args.pdf, args.page, index, version)
        box_id = os.path.basename(out_dir)
        return img, txt, out_dir, box_id

    results: list[dict] = []
    overlays: list[str] = []

    if args.box is not None:
        index = int(args.box)
        img, txt, out_dir, box_id = box_paths(index)
        word = _read_word(txt)
        overlay = os.path.join(out_dir, "letters_overlay.png")
        qa = segment_word(img, word, box_id, out_dir, args.min_conf, overlay_path=overlay)
        results.append(qa)
        overlays.append(overlay)
    else:
        paths.ensure_dir(args.runs_dir)
        for index, expected in DEFAULT_SAMPLE.items():
            img, txt, out_dir, box_id = box_paths(index)
            if not os.path.exists(img):
                continue
            word = _read_word(txt) if os.path.exists(txt) else expected
            run_overlay = os.path.join(args.runs_dir, f"seg_proto_{paths.slugify(word)}.png")
            box_overlay = os.path.join(out_dir, "letters_overlay.png")
            qa = segment_word(
                img, word, box_id, out_dir, args.min_conf, overlay_path=box_overlay
            )
            # also drop a copy of the overlay into runs/ for QA
            seg = segment_word(
                img, word, box_id, None, args.min_conf, overlay_path=run_overlay,
                write_files=False,
            )
            qa.update({k: seg[k] for k in ("confidence",)})
            results.append(qa)
            overlays.append(run_overlay)

    _print_summary(results, overlays)


def _print_summary(results: list[dict], overlays: list[str]) -> None:
    print(f"\n{'word':<12} {'L':>2} {'free':>4} {'cuts':>4} {'conf':>6} {'low?':>5} {'gate':>5}")
    print("-" * 46)
    n_ok = 0
    for r in results:
        gate = "OK" if r["count_gate_ok"] else "FAIL"
        n_ok += r["count_gate_ok"]
        low = "yes" if r["low_confidence"] else ""
        print(
            f"{r['word']:<12} {r['L']:>2} {r['free_penlift_boundaries']:>4} "
            f"{r['dp_internal_cuts']:>4} {r['confidence']:>6.3f} {low:>5} {gate:>5}"
            + (f"  ERR={r['error']}" if r.get("error") else "")
        )
    print("-" * 46)
    print(f"count-gate passed: {n_ok}/{len(results)}")
    if overlays:
        print("overlays:")
        for o in overlays:
            print(f"  {os.path.abspath(o)}")


if __name__ == "__main__":
    main()
