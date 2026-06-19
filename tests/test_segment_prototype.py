"""Unit tests for the x-position cutting logic in ocr._segment_prototype.

These guard the two fixes that distinguish this prototype from the first cut:
boundaries are placed by HORIZONTAL x-position (not path arc-length), and every
body point is assigned to a slot purely by its x -- so a non-monotonic trace
order (the backtracking DFS in vectorize.trace_component) cannot mislabel points.
Pure-numpy logic only; no PDF render or cv2 trace needed.
"""

import numpy as np

from ocr import _segment_prototype as sp


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
