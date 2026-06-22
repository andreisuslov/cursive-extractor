"""Unit tests for the x-position cutting logic in ocr._segment_prototype.

These guard the two fixes that distinguish this prototype from the first cut:
boundaries are placed by HORIZONTAL x-position (not path arc-length), and every
body point is assigned to a slot purely by its x -- so a non-monotonic trace
order (the backtracking DFS in vectorize.trace_component) cannot mislabel points.
Pure-numpy logic only; no PDF render or cv2 trace needed.
"""

import cv2
import numpy as np

from ocr.experiments import _segment_prototype as sp


def test_expected_boundary_x_monotone_and_in_span():
    xs = sp.expected_boundary_x("could", 100.0, 300.0)
    assert len(xs) == len("could") - 1
    assert np.all(np.diff(xs) > 0)  # strictly increasing
    assert xs.min() > 100.0 and xs.max() < 300.0


def test_expected_boundary_x_width_prior_widens_m():
    # 'm' has a larger width prior than 'i', so the slot after 'm' is wider:
    # the first boundary of "mi" sits past the midpoint, the first of "im" before it.
    mi = sp.expected_boundary_x("mi", 0.0, 100.0)[0]
    im = sp.expected_boundary_x("im", 0.0, 100.0)[0]
    assert mi > 50.0 > im


def test_dp_picks_score_peaks_over_uniform_prior():
    # Two sharp score peaks; with L-1 = 2 boundaries the DP should land on them
    # rather than on the (evenly spaced) expected positions.
    score = np.zeros(101)
    score[30] = 1.0
    score[70] = 1.0
    cand_x = np.arange(101, dtype=float)
    cand_score = score[cand_x.astype(int)]
    x_expected = sp.expected_boundary_x("abc", 0.0, 100.0)  # ~33, 67
    bxs = sp.dp_boundaries_x(cand_x, cand_score, x_expected, 0.0, 100.0)
    assert sorted(round(b) for b in bxs) == [30, 70]


def test_dp_min_gap_prevents_piling_on_adjacent_peaks():
    # Two near-adjacent peaks (50, 52) plus a far one (80): with no min-gap the two
    # boundaries pile on 50/52; a min-gap forces the second onto the far peak.
    score = np.full(101, 0.1)
    score[50], score[52], score[80] = 1.0, 0.9, 0.6
    cand_x = np.arange(101, dtype=float)
    cand_score = score[cand_x.astype(int)]
    x_expected = sp.expected_boundary_x("abc", 0.0, 100.0)  # ~33, 67
    piled = sp.dp_boundaries_x(cand_x, cand_score, x_expected, 0.0, 100.0, 0.0)
    spaced = sp.dp_boundaries_x(cand_x, cand_score, x_expected, 0.0, 100.0, 25.0)
    assert max(abs(a - b) for i, a in enumerate(piled) for b in piled[i + 1 :]) < 25.0
    assert all(abs(a - b) >= 25.0 for i, a in enumerate(spaced) for b in spaced[i + 1 :])


def test_slice_by_x_assigns_by_x_not_trace_order():
    # A stroke whose POINTS jump backwards in x (mimics the backtracking DFS):
    # arc-length order is non-monotonic, but slicing must still bucket by x.
    stroke = [(5, 0), (45, 0), (15, 1), (55, 1), (25, 2), (35, 3)]
    groups = sp.slice_by_x([stroke], [30.0], n_slots=2)
    left_x = [p[0] for sub in groups[0] for p in sub]
    right_x = [p[0] for sub in groups[1] for p in sub]
    assert left_x and right_x
    assert all(x < 30 for x in left_x)
    assert all(x >= 30 for x in right_x)


def test_slice_by_x_preserves_all_points_and_slot_count():
    rng = np.random.default_rng(0)
    stroke = [(int(rng.integers(0, 100)), int(rng.integers(0, 10))) for _ in range(200)]
    boundaries = [25.0, 50.0, 75.0]
    groups = sp.slice_by_x([stroke], boundaries, n_slots=4)
    assert len(groups) == 4
    total = sum(len(sub) for g in groups for sub in g)
    assert total == len(stroke)  # no point dropped


def test_attach_diacritics_by_x_centre():
    groups = [[], [], []]
    dot_left = [(8, 1), (9, 1)]  # x-centre ~8.5 -> slot 0
    dot_right = [(72, 1), (74, 1)]  # x-centre ~73 -> slot 2
    out = sp.attach_diacritics_x(groups, [30.0, 60.0], [dot_left, dot_right], n_slots=3)
    assert len(out[0]) == 1 and len(out[1]) == 0 and len(out[2]) == 1


def test_is_connected_blob_true_for_one_fullwidth_component():
    # one full-height bar spanning the whole width -> a single ligature-joined blob.
    b = np.zeros((40, 100), np.uint8)
    b[8:32, 5:95] = 255
    assert sp.is_connected_blob(b, 5.0, 94.0)
    # two well-separated components -> normal multi-component word, not a blob.
    b2 = np.zeros((40, 100), np.uint8)
    b2[8:32, 5:35] = 255
    b2[8:32, 65:95] = 255
    assert not sp.is_connected_blob(b2, 5.0, 94.0)


def test_max_cut_gap_flags_an_uncut_letter_run():
    # cuts packed at the left leave a wide run on the right -> large gap (x-density's
    # failure signature on a blob); evenly spaced cuts stay near one letter-width.
    assert sp.max_cut_gap([10.0, 20.0, 30.0], 0.0, 100.0) == 70.0
    assert sp.max_cut_gap([25.0, 50.0, 75.0], 0.0, 100.0) == 25.0


def test_topology_cut_score_peaks_at_baseline_connector():
    # Two tall "letters" joined by a thin low baseline connector: the skeleton upper
    # envelope dips to the baseline only at the connector, so the score must peak
    # there (a ligature cut), not on the tall stems.
    h, w = 50, 120
    b = np.zeros((h, w), np.uint8)
    b[8:41, 15:28] = 255  # letter 1 (tall)
    b[8:41, 92:105] = 255  # letter 2 (tall)
    b[36:40, 28:92] = 255  # thin near-baseline connector
    score = sp.topology_cut_score(b, 15.0, 104.0)
    peak = int(np.argmax(score))
    assert 30 < peak < 90  # falls inside the connector span
    assert score[peak] > score[21]  # connector scores above the stem centre


def test_shear_roundtrip_recovers_original_x():
    # de-slant a stroke then map it back: the per-row integer shift is invertible,
    # so emitted/overlaid points must land on the original x exactly.
    shift = sp._shear_shifts(50, -0.3)
    strokes = [[(10, 0), (12, 25), (15, 49)]]
    sheared = sp._shear_points(strokes, shift)
    back = sp._unshear_groups([[np.asarray(sheared[0], dtype=float)]], shift)
    np.testing.assert_allclose(back[0][0][:, 0], [10, 12, 15])


def test_estimate_slant_recovers_known_lean():
    # four parallel strokes leaning at tan = -0.3 -> de-shear should recover ~-0.3
    # with a clear (gate-passing) prominence.
    h, w, s0 = 60, 200, -0.3
    b = np.zeros((h, w), np.uint8)
    for base in (40, 90, 140, 180):
        for y in range(8, 52):
            x = round(base + s0 * y)
            b[y, x - 1 : x + 2] = 255
    s, prom = sp.estimate_slant(b)
    assert prom >= sp.SLANT_MIN_PROM
    assert abs(s - s0) < 0.07


def test_estimate_slant_gates_flat_horizontal_ink():
    # a horizontal bar has no near-vertical structure: every shear leaves the same
    # short vertical runs, so the peak is not prominent -> gated off (s stays 0).
    b = np.zeros((40, 200), np.uint8)
    b[18:22, 10:190] = 255
    s, prom = sp.estimate_slant(b)
    assert prom < sp.SLANT_MIN_PROM or s == 0.0


def test_reject_noise_drops_speck_keeps_letters():
    # two thin letter strokes + a 3x3 speck: the stroke-width-aware floor drops the
    # speck (sub-letter) and keeps both strokes, so a speck is never a cut landmark.
    b = np.zeros((60, 120), np.uint8)
    b[10:50, 20:24] = 255  # stroke 1
    b[10:50, 60:64] = 255  # stroke 2
    b[2:5, 100:103] = 255  # 3x3 speck (area 9, below the floor)
    out, _g, changed = sp.reject_noise(b, b.copy())
    assert changed
    assert out.sum() == b[:, :80].sum()  # only the two strokes survive (speck removed)


def test_reject_noise_keeps_real_diacritic():
    # an i-dot sized blob (~5*stroke_width^2) sits ABOVE the floor and is kept.
    b = np.zeros((60, 80), np.uint8)
    b[10:50, 30:34] = 255  # a stem (stroke width ~4)
    b[3:9, 31:37] = 255  # a 6x6 dot (area 36 ~ 2.3*stroke_width^2, a real diacritic)
    out, _g, changed = sp.reject_noise(b, b.copy())
    assert not changed  # nothing dropped: the dot is above the speck floor
    assert out.sum() == b.sum()


def test_is_black_block_flags_solid_block_and_tiny_fragment():
    # a solid filled block is deeply over-inked, and a sub-letter fragment is too small:
    # both are mis-cuts, never letters.
    solid = np.zeros((44, 44), np.uint8)
    solid[6:38, 6:38] = 255  # 32x32 fully-filled block -> deep over-ink core
    assert sp.is_black_block(solid)
    tiny = np.zeros((24, 24), np.uint8)
    tiny[6:12, 6:15] = 255  # 6x9 fragment, max dim < BLOCK_SMALL_PX
    assert sp.is_black_block(tiny)


def test_is_black_block_keeps_thin_stroke_and_open_loop():
    # a real pen stroke is thin (high skeleton/area, no deep core) -> never a black-block,
    # even when its tight bbox is nearly full (a straight stem) or it is a closed loop.
    stem = np.zeros((90, 40), np.uint8)
    stem[8:82, 16:23] = 255  # a thin tall stem (l / i body)
    assert not sp.is_black_block(stem)
    ring = np.zeros((70, 70), np.uint8)
    cv2.circle(ring, (35, 35), 26, 255, 5)  # an open 'o' ring: hollow, thin stroke
    assert not sp.is_black_block(ring)


def test_drop_offrow_drops_floating_sliver_keeps_diacritic_over_stem():
    # a free-floating off-row sliver (no body column under it) is dropped; an i-dot sitting
    # above a body stem is kept -- so adjacent-row bleed goes but real diacritics stay.
    b = np.zeros((130, 180), np.uint8)
    for x in (20, 55, 90):  # word body: thin stems in the x-height band (rows 55-105)
        b[55:105, x : x + 10] = 255
    b[24:32, 55:62] = 255  # an i-dot ABOVE a body stem (x~55) -> kept
    frag = np.zeros_like(b)
    frag[12:20, 135:159] = 255  # free-floating off-row sliver (no stem under it) -> dropped
    b |= frag
    out, _g, changed = sp.drop_offrow_components(b, np.full_like(b, 255))
    assert changed
    assert out.sum() == b.sum() - frag.sum()  # only the floating sliver removed (dot kept)


def test_drop_offrow_noop_on_single_component():
    # one connected word -> nothing to strip (the guard returns it unchanged).
    b = np.zeros((60, 80), np.uint8)
    b[10:50, 10:70] = 255
    out, _g, changed = sp.drop_offrow_components(b, np.full_like(b, 255))
    assert not changed and np.array_equal(out, b)


def test_local_slope_returns_fallback_when_unreliable():
    # an empty window has no near-vertical ink -> keep the global slant (fallback).
    b = np.zeros((40, 60), np.uint8)
    assert sp._local_slope(b, 30.0, 12.0, -0.4) == -0.4


def test_local_slope_finds_local_stem_lean():
    # a window over a stem leaning tan=-0.3 returns that LOCAL angle, not the fallback.
    b = np.zeros((60, 80), np.uint8)
    for y in range(8, 52):
        x = round(40 - 0.3 * y)
        b[y, x - 1 : x + 2] = 255
    s = sp._local_slope(b, 30.0, 40.0, 0.0)  # fallback 0; the stem must override it
    assert abs(s - (-0.3)) < 0.1


def test_carve_seam_bends_around_an_ink_bar():
    # a straight preferred line runs through a vertical ink bar; the min-ink seam must
    # bend AROUND the bar (the descender-routing the global straight cut cannot do),
    # while riding the line where there is no ink.
    ink = np.zeros((40, 60), dtype=float)
    ink[10:30, 28:33] = 1.0  # the bar to route around
    pref = np.full(40, 30.0)
    cut = sp._carve_seam(ink, pref, band=12)
    assert np.all((cut[10:30] < 28) | (cut[10:30] >= 33))  # routed off the bar
    assert abs(cut[0] - 30) <= 3  # rides the preferred line away from the bar


def test_slice_by_seams_assigns_by_curved_boundary():
    # a BENT seam (jumps right at row 5): the same x lands in different slots by row --
    # exactly what a straight global-slant line cannot express.
    seams = np.array([[5] * 5 + [15] * 5])  # one boundary, bent
    groups = sp.slice_by_seams([[(8, 2), (8, 7)]], seams, n_slots=2)
    g0 = [(p[0], p[1]) for sub in groups[0] for p in sub]
    g1 = [(p[0], p[1]) for sub in groups[1] for p in sub]
    assert (8.0, 2.0) in g1  # row 2: 8 >= seam(5) -> right slot
    assert (8.0, 7.0) in g0  # row 7: 8 <  seam(15) -> left slot


def test_ensure_count_x_forces_exact_distinct_count():
    assert sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0) == sorted(
        sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0)
    )
    out = sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0)
    assert len(out) == 3 and len(set(out)) == 3
    assert all(0.0 < b < 100.0 for b in out)


# A box_2d 200px wide on a 1000px page -> morph-open kernel ~120px; the line spans
# the whole 400px crop and is removed, the narrow word stems survive.
_RULE_BOX = [0, 100, 100, 300]  # ymin, xmin, ymax, xmax (0-1000 scale)
_RULE_PAGE = (1000, 200)


def _word_with_rule(rule_row_frac: float) -> np.ndarray:
    """A binary with three narrow vertical stems and one wide horizontal bar whose
    ink fill spans ``rule_row_frac`` of the width."""
    b = np.zeros((60, 400), np.uint8)
    for x in (60, 160, 260):  # word stems (narrow, kept)
        b[5:45, x : x + 4] = 255
    b[50:53, : int(rule_row_frac * 400)] = 255  # horizontal bar (the rule)
    return b


def test_strip_ruled_line_removes_full_width_rule():
    b = _word_with_rule(1.0)
    out = sp.strip_ruled_line(b, _RULE_BOX, (0, 0, 400, 60), _RULE_PAGE)
    assert out[50:53, :].sum() == 0  # the edge-to-edge rule is gone
    assert out[5:45, 60:64].any() and out[5:45, 260:264].any()  # word stems survive


def test_strip_ruled_line_noop_below_rowfrac_gate():
    # a 0.90-full row mimics translate's wavy baseline: below the 0.97 gate -> untouched
    b = _word_with_rule(0.90)
    out = sp.strip_ruled_line(b, _RULE_BOX, (0, 0, 400, 60), _RULE_PAGE)
    assert np.array_equal(out, b)
