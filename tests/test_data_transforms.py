"""Unit tests for data.py pure stroke transforms (fixed seeds, tiny synthetic inputs)."""

import random
from types import SimpleNamespace

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


# --- augment_stroke (widened training augmentation: slant/incline/height/width/jitter) ---


def _aug_dataset():
    args = SimpleNamespace(
        alphabet=" abcdefghijklmnopqrstuvwxyz",
        augment=True,
        max_seq_length=512,
        seed=0,
        downsample_mean=0.65,
        downsample_width=0.1,
    )
    word = np.array([[i * 0.01, np.sin(i * 0.3) * 0.1, 1] for i in range(40)] + [[0.4, 0.0, 0]])
    return d.StrokeDataset([[word]], ["w"], args), word


def test_augment_stroke_finite_shape_and_pen_valid():
    ds, word = _aug_dataset()
    np.random.seed(1)
    random.seed(1)
    out = ds.augment_stroke(word.copy())
    assert out.ndim == 2 and out.shape[1] == 3  # (M, 3)
    assert out.shape[0] >= 2  # downsample keeps at least 2 points
    assert np.isfinite(out).all()  # no NaN/inf from the wider transforms
    assert set(np.unique(out[:, 2]).tolist()).issubset({0.0, 1.0})  # pen states stay 0/1


def test_augment_stroke_deterministic_under_seed():
    ds, word = _aug_dataset()
    np.random.seed(7)
    random.seed(7)
    a = ds.augment_stroke(word.copy())
    np.random.seed(7)
    random.seed(7)
    b = ds.augment_stroke(word.copy())
    assert np.array_equal(a, b)  # identical given the same RNG seeds


def test_augment_stroke_identity_ranges_only_downsample():
    # Identity geometric ranges -> no shear/rotate/scale/jitter, only downsampling, so
    # every output point is one of the input points (geometry untouched).
    ds, word = _aug_dataset()
    np.random.seed(3)
    random.seed(3)
    out = ds.augment_stroke(
        word.copy(),
        shear_range=(0.0, 0.0),
        rotate_range=(0.0, 0.0),
        height_scale_range=(1.0, 1.0),
        width_scale_range=(1.0, 1.0),
        height_jitter=0.0,
    )
    inset = {tuple(np.round(p, 6)) for p in word}
    assert all(tuple(np.round(p, 6)) in inset for p in out)  # output is a subset of input


def test_augment_default_slant_is_symmetric_and_wider_than_old():
    # Old slant was one-directional and narrow (-0.22..-0.18); the new default shear range
    # (-0.3, 0.3) tilts x BOTH ways and reaches a larger magnitude. dx = 2 * shear_factor.
    base = np.array([[0.0, 1.0, 1], [0.0, -1.0, 1]])  # vertical segment: x' = x + factor*y
    dx = []
    for s in range(200):
        np.random.seed(s)
        o = d.random_horizontal_shear(base.copy(), shear_range=(-0.3, 0.3))
        dx.append(o[0, 0] - o[1, 0])
    dx = np.array(dx)
    assert dx.min() < -0.3 and dx.max() > 0.3  # both directions, wider than the old range
    assert np.all(np.abs(dx) <= 0.6 + 1e-9)  # bounded by 2 * 0.3
