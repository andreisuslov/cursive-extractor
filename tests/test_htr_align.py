"""Unit tests for ocr._htr_align (holistic CTC forced-alignment cut).

Pure-logic / synthetic only -- NO PDF render and NO training (the real cut quality is
measured by the module CLI, reported honestly there). These assert the structural
contracts: canvas shape/range + min-frame guarantee, CRNN output time-length, that CTC
forced alignment returns a monotone all-chars-visited path, that boundaries are L-1,
monotone and in-span, and that the whole-word split never leaks.
"""

import itertools

import numpy as np
import torch

from ocr import _htr_align as H


def _region(h=20, w=120) -> np.ndarray:
    """Three ink blobs across the width -> a 3-'letter' word region (ink=1.0)."""
    r = np.zeros((h, w), np.float32)
    for x in (10, 55, 100):
        r[4 : h - 4, x : x + 12] = 1.0
    return r


def test_region_to_canvas_shape_range_and_min_frames():
    c = H.region_to_canvas(_region(), height=32, min_frames=4)
    assert c.shape[0] == 32 and c.dtype == np.float32
    assert c.min() >= 0.0 and c.max() <= 1.0 and c.any()
    # min_frames is honoured: the CNN's W//WIDTH_DS time-steps cover the chars.
    assert c.shape[1] // H.WIDTH_DS >= 4


def test_region_to_canvas_augment_is_deterministic_and_keeps_ink():
    a = H.region_to_canvas(_region(), augment=True, rng=np.random.default_rng(3))
    b = H.region_to_canvas(_region(), augment=True, rng=np.random.default_rng(3))
    assert a.any() and np.array_equal(a, b)  # same seed -> same canvas, ink preserved


def test_build_vocab_and_split():
    recs = [{"word": "cat", "region": _region()}, {"word": "dog", "region": _region()}]
    assert H.build_vocab(recs) == ["a", "c", "d", "g", "o", "t"]
    many = [{"word": f"w{i:02d}", "region": _region()} for i in range(20)]
    train, test = H.split_records(many, 0.25, seed=0)
    assert len(test) == 5 and len(train) == 15
    train_ids = {id(r) for r in train}
    assert not any(id(r) in train_ids for r in test)
    assert H.split_records(many, 0.25, seed=0)[1] == test  # deterministic


def test_crnn_forward_time_length():
    torch.manual_seed(0)
    model = H.CRNN(n_classes=7, height=32)
    w = 120
    x = torch.zeros(2, 1, 32, w)
    out = model(x)
    assert out.shape[0] == 2 and out.shape[2] == 7
    assert out.shape[1] == w // H.WIDTH_DS  # T = W / WIDTH_DS


def test_ctc_forced_align_monotone_all_chars_visited():
    # logp peaks on chars 0,1,2 in three frame thirds -> alignment must visit each.
    blank = 3
    t_len = 30
    logp = np.full((t_len, blank + 1), -5.0)
    logp[:, blank] = -1.0  # mild blank everywhere
    for j, sl in enumerate((slice(0, 10), slice(10, 20), slice(20, 30))):
        logp[sl, j] = 5.0
    states = H.ctc_forced_align(logp, [0, 1, 2], blank)
    assert states is not None and len(states) == t_len
    assert states == sorted(states)  # monotone non-decreasing
    emitted = {(s - 1) // 2 for s in states if s % 2 == 1}
    assert emitted == {0, 1, 2}  # every char got at least one frame
    assert H.ctc_forced_align(np.full((2, 4), -1.0), [0, 1, 2], blank=3) is None  # T<L


def test_boundaries_from_states_count_monotone_inspan():
    blank = 3
    t_len = 30
    logp = np.full((t_len, blank + 1), -5.0)
    for j, sl in enumerate((slice(0, 10), slice(10, 20), slice(20, 30))):
        logp[sl, j] = 5.0
    states = H.ctc_forced_align(logp, [0, 1, 2], blank)
    bxs = H.boundaries_from_states(states, t_len, 3, x_min=5.0, x_max=125.0)
    assert len(bxs) == 2  # L-1
    assert 5.0 < bxs[0] < bxs[1] < 125.0  # monotone, strictly in-span
    assert all(lo < hi for lo, hi in itertools.pairwise([5.0, *bxs, 125.0]))


def test_align_word_returns_lminus1_inspan_boundaries():
    # End-to-end on an untrained tiny model: contract is L-1 monotone in-span cuts.
    torch.manual_seed(0)
    vocab = ["a", "b", "c"]
    model = H.CRNN(len(vocab) + 1, height=32)
    bxs = H.align_word(model, _region(), "abc", x_min=5.0, x_max=125.0, vocab=vocab)
    assert len(bxs) == 2
    assert 5.0 < bxs[0] < bxs[1] < 125.0
    # an out-of-vocab char falls back to an even split (still L-1 in-span)
    bxs2 = H.align_word(model, _region(), "axc", x_min=5.0, x_max=125.0, vocab=vocab)
    assert len(bxs2) == 2 and 5.0 < bxs2[0] < bxs2[1] < 125.0
