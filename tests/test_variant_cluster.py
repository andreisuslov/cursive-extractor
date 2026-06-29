"""Unit tests for N3 variant clustering."""

import numpy as np

from ocr.experiments import _variant_cluster as vc


def test_kmeans_separates_two_clusters():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = vc.kmeans(x, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_glyph_descriptor_shape_and_unit_norm():
    d = vc.glyph_descriptor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], size=24)
    assert d.shape == (24 * 24,)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5


def test_cluster_letter_returns_at_most_k_medoids():
    glyphs = [[[0.0, 0.0], [0.1, 0.1]]] * 2 + [[[1.0, 1.0], [0.9, 0.9]]] * 2
    med = vc.cluster_letter(glyphs, k=2)
    assert len(med) <= 2
    assert all(0 <= i < 4 for i in med)


def test_build_variants_caps_at_k_and_skips_sparse():
    lib = {"a": [[[0.0, 0.0], [0.1, 0.1]] for _ in range(5)], "q": [[[0.0, 0.0], [0.1, 0.1]]]}
    v = vc.build_variants(lib, k=3, min_samples=2)
    assert len(v["a"]) <= 3
    assert "q" not in v  # only 1 sample, below min_samples
