"""Unit tests for data.py pure stroke transforms (fixed seeds, tiny synthetic inputs)."""

import random

import numpy as np

import data as d

# --- offset <-> stroke round-trips ----------------------------------------------


def test_strokes_offsets_round_trip():
    pts = np.array([[0.0, 0.0, 1], [1.0, 2.0, 1], [3.0, 1.0, 1], [3.0, 1.0, 0], [5.0, 5.0, 1]])
    rt = d.offsets_to_strokes(d.strokes_to_offsets(pts))
    assert rt.shape == pts.shape
    # the round-trip preserves geometry relative to the first point (translated to origin)
    assert np.allclose(rt[:, :2], pts[:, :2] - pts[0, :2])
    assert np.array_equal(rt[:, 2], pts[:, 2])  # pen state preserved exactly


def test_decompose_reconstruct_round_trip():
    offs = np.array([[1.0, 2.0, 1], [-0.5, 0.3, 1], [0.0, 0.0, 0], [2.0, -1.0, 1]])
    back = d.reconstruct_offsets(d.decompose_offsets(offs))
    assert np.allclose(back[:, :2], offs[:, :2])
    assert np.array_equal(back[:, 2], offs[:, 2])


def test_decompose_offsets_is_polar():
    offs = np.array([[3.0, 4.0, 1]])  # r = 5, theta = atan2(4, 3)
    polar = d.decompose_offsets(offs)
    assert np.isclose(polar[0, 0], 5.0)  # radius
    assert np.isclose(polar[0, 1], np.arctan2(4, 3))  # angle


# --- augmentation: shear / rotate ----------------------------------------------


def _sample_stroke():
    return np.array([[1.0, 3.0, 1], [2.0, -4.0, 1], [-1.0, 2.0, 1], [0.0, 0.0, 0]])


def test_horizontal_shear_preserves_y_shape_and_bounds():
    np.random.seed(0)
    orig = _sample_stroke()
    out = d.random_horizontal_shear(orig.copy(), shear_range=(-0.4, 0.4))
    assert out.shape == orig.shape
    assert np.array_equal(out[:, 1], orig[:, 1])  # shear leaves y unchanged
    assert np.array_equal(out[:, 2], orig[:, 2])  # pen state unchanged
    # x' = x + shear * y with |shear| <= 0.4
    bound = 0.4 * np.max(np.abs(orig[:, 1])) + 1e-9
    assert np.all(np.abs(out[:, 0] - orig[:, 0]) <= bound)


def test_horizontal_shear_deterministic_under_seed():
    orig = _sample_stroke()
    np.random.seed(7)
    a = d.random_horizontal_shear(orig.copy())
    np.random.seed(7)
    b = d.random_horizontal_shear(orig.copy())
    assert np.array_equal(a, b)


def test_rotate_preserves_norms_and_shape():
    np.random.seed(1)
    orig = _sample_stroke()
    out = d.random_rotate(orig.copy(), angle_range=(-0.08, 0.08))
    assert out.shape == orig.shape
    assert np.array_equal(out[:, 2], orig[:, 2])  # pen state unchanged
    # rotation is an isometry -> each point's xy magnitude is preserved
    assert np.allclose(np.hypot(out[:, 0], out[:, 1]), np.hypot(orig[:, 0], orig[:, 1]))


def test_rotate_deterministic_under_seed():
    orig = _sample_stroke()
    np.random.seed(3)
    a = d.random_rotate(orig.copy())
    np.random.seed(3)
    b = d.random_rotate(orig.copy())
    assert np.array_equal(a, b)


# --- downsample -----------------------------------------------------------------


def _ten_point_stroke():
    # 10 pen-down points followed by a pen-up marker
    return np.array([[float(i), 0.0, 1] for i in range(10)] + [[9.0, 0.0, 0]])


def test_downsample_fraction_one_is_identity():
    arr = _ten_point_stroke()
    assert d.downsample(arr, 1) is arr


def test_downsample_reduces_and_keeps_endpoints():
    arr = _ten_point_stroke()
    out = d.downsample(arr, 0.5, drop_prob=0.0)  # deterministic (no random drop)
    assert out.shape == (6, 3)  # 5 reduced pen-down points + 1 pen-up marker
    assert out[0, 0] == 0.0  # first pen-down point kept
    assert out[-2, 0] == 9.0  # last pen-down point kept
    assert out[-1, 2] == 0  # pen-up marker preserved


def test_downsample_drop_prob_reduces_further():
    arr = _ten_point_stroke()
    base = d.downsample(arr, 0.5, drop_prob=0.0)
    random.seed(0)
    dropped = d.downsample(arr, 0.5, drop_prob=0.5)
    assert len(dropped) <= len(base)  # random drop removes some middle points
    assert dropped[0, 0] == 0.0 and dropped[-2, 0] == 9.0  # endpoints always kept
