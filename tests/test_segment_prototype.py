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


def test_ensure_count_x_forces_exact_distinct_count():
    assert sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0) == sorted(
        sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0)
    )
    out = sp._ensure_count_x([20.0, 20.0, 80.0], 3, 0.0, 100.0)
    assert len(out) == 3 and len(set(out)) == 3
    assert all(0.0 < b < 100.0 for b in out)
