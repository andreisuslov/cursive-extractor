"""Unit tests for ocr._allograph_library (per-writer allograph clustering).

Pure-logic / synthetic only -- NO PDF render and NO CNN training (the real numbers come
from the module CLI, reported honestly there). These pin the clustering contracts: that
k-means + silhouette actually find two separated blobs and not split one, that the
noise filter drops mislabeled slices, and that a built variant carries a medoid exemplar.
"""

import numpy as np

from ocr import _allograph_library as A


def _two_blobs(seed=0, n=24, dim=576):
    """Two well-separated unit-ish gaussian blobs in descriptor space."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.02, (n, dim))
    b = rng.normal(0, 0.02, (n, dim))
    a[:, 0] += 1.0
    b[:, 1] += 1.0
    return np.vstack([a, b]).astype(np.float32)


def test_kmeans_separates_two_blobs():
    x = _two_blobs()
    labels, c = A.kmeans(x, 2, seed=0)
    assert c.shape == (2, x.shape[1])
    # each true blob (first/second half) should land in a single cluster
    assert len(set(labels[:24])) == 1 and len(set(labels[24:])) == 1
    assert labels[0] != labels[-1]


def test_silhouette_high_for_separated_low_for_random():
    x = _two_blobs()
    good, _ = A.kmeans(x, 2, seed=0)
    assert A.silhouette(x, good) > 0.5  # clearly separated
    rng = np.random.default_rng(1)
    rand = rng.integers(0, 2, len(x))
    assert A.silhouette(x, rand) < 0.2
    assert A.silhouette(x, np.zeros(len(x), int)) == -1.0  # <2 clusters undefined


def test_choose_k_finds_two_and_refuses_to_split_one():
    labels, k, sil = A.choose_k(_two_blobs(), seed=0)
    assert k == 2 and sil >= A.SIL_MIN and len(np.unique(labels)) == 2

    rng = np.random.default_rng(3)
    blob = rng.normal(0, 0.02, (24, 576)).astype(np.float32)  # single cloud
    labels1, k1, _ = A.choose_k(blob, seed=0)
    assert k1 == 1 and np.all(labels1 == 0)  # no spurious variants


def test_build_variants_medoid_and_counts():
    x = _two_blobs(n=10)
    labels, _ = A.kmeans(x, 2, seed=0)
    masks = [np.ones((5, 5), np.uint8) * 255 for _ in range(len(x))]
    words = [f"w{i}" for i in range(len(x))]
    variants = A.build_variants(masks, words, labels, x)
    assert len(variants) == 2
    assert sum(v["count"] for v in variants) == len(x)
    assert all("exemplar_mask" in v and "exemplar_strokes" in v for v in variants)
    assert variants[0]["count"] >= variants[1]["count"]  # most-common first


class _StubRecognizer:
    """Predicts the first char of the source word's letter; used to test the filter."""

    classes = ("a", "b")

    def top(self, mask, k=1):
        return ["a"] if mask[0, 0] else ["b"]  # flag pixel decides the prediction


def test_noise_reject_drops_mislabeled_and_flags_unjudged():
    good = np.ones((3, 3), np.uint8)  # -> predicted "a"
    bad = np.zeros((3, 3), np.uint8)  # -> predicted "b"
    by_letter = {
        "a": [(good, "apple"), (bad, "apple")],  # one true 'a', one mis-cut
        "z": [(good, "zoo")],  # 'z' not in recognizer.classes -> unjudged, kept
    }
    clean, stats = A.noise_reject(by_letter, _StubRecognizer())
    assert len(clean["a"]) == 1 and stats["rej_by_letter"]["a"] == 1
    assert len(clean["z"]) == 1 and "z" in stats["unjudged"]
    assert stats["n_total"] == 3 and stats["n_rejected"] == 1


def test_collect_slices_folds_case():
    records = [
        {
            "word": "The",
            "slices": [("T", np.ones((2, 2), np.uint8)), ("h", np.ones((2, 2), np.uint8))],
        }
    ]
    by_letter = A.collect_slices(records)
    assert set(by_letter) == {"t", "h"} and len(by_letter["t"]) == 1
