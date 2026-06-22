"""Unit tests for ocr._bootstrap_recognizer (CNN bootstrap recognizer).

Pure-logic / synthetic only -- NO PDF render and NO training (the real accuracy is
measured by the module CLI, reported honestly there). These assert the structural
contracts: the canvas/augment shape invariants, that ``CNNRecognizer`` is a drop-in
for ``LetterRecognizer`` (so it plugs into ``align_boundaries``), and that the
whole-word split never leaks a word into both halves.
"""

import itertools

import numpy as np
import torch

from ocr.experiments import _bootstrap_recognizer as B
from ocr.experiments import _recognizer as R


def _blob_mask(h=24, w=16) -> np.ndarray:
    m = np.zeros((h + 8, w + 8), np.uint8)
    m[4 : 4 + h, 4 : 4 + w] = 255
    return m


def test_slice_to_canvas_shape_range_and_empty():
    c = B.slice_to_canvas(_blob_mask())
    assert c.shape == (B.CNN_SIZE, B.CNN_SIZE)
    assert c.dtype == np.float32
    assert c.min() >= 0.0 and c.max() <= 1.0 and c.any()
    assert not B.slice_to_canvas(np.zeros((10, 10), np.uint8)).any()  # empty -> zeros


def test_augment_mask_keeps_ink_and_is_deterministic():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = B.augment_mask(_blob_mask(), rng1)
    b = B.augment_mask(_blob_mask(), rng2)
    assert a.dtype == np.uint8 and a.any()  # augmentation never erases all ink here
    assert np.array_equal(a, b)  # same seed -> same augmentation


def _tiny_recognizer(classes=("a", "b", "c"), seed=0) -> B.CNNRecognizer:
    torch.manual_seed(seed)
    return B.CNNRecognizer(B.LetterCNN(len(classes)), list(classes))


def test_cnn_recognizer_is_letterrecognizer_dropin():
    rec = _tiny_recognizer()
    sc = rec.scores(_blob_mask())
    assert set(sc) == {"a", "b", "c"}
    assert abs(sum(sc.values()) - 1.0) < 1e-4  # softmax over the trained classes
    assert rec.ranking(_blob_mask())[0] in sc  # ranking covers the classes
    assert rec.top(_blob_mask(), 2) == rec.ranking(_blob_mask())[:2]
    assert rec.score_char(_blob_mask(), "a") == sc["a"]
    assert rec.score_char(_blob_mask(), "z") == 0.0  # untrained char -> 0
    assert not any(B.CNNRecognizer(rec.model, ["a"]).scores(np.zeros((5, 5), np.uint8)).values())


def test_cnn_recognizer_plugs_into_align_boundaries():
    # the whole point of the drop-in: align_boundaries must accept it and still return
    # L-1 monotone in-span boundaries (cut quality is measured by the CLI, not here).
    b = np.zeros((40, 150), np.uint8)
    for x in (10, 60, 110):
        b[10:30, x : x + 20] = 255
    x_min, x_max = 10.0, 129.0
    rec = _tiny_recognizer()
    bxs = R.align_boundaries(
        b, "abc", x_min, x_max, np.arange(x_min, x_max, 3.0), np.zeros(150), recognizer=rec
    )
    assert len(bxs) == 2
    assert x_min < bxs[0] < bxs[1] < x_max
    cuts = [x_min, *bxs, x_max]
    min_w = R.ALIGN_MIN_WFRAC * (x_max - x_min) / 3
    assert all(hi - lo >= min_w for lo, hi in itertools.pairwise(cuts))


def test_split_words_holds_out_whole_words_no_leak():
    records = [{"word": f"w{i}", "slices": [("a", _blob_mask())]} for i in range(20)]
    train, test = B.split_words(records, 0.25, seed=0)
    assert len(test) == 5 and len(train) == 15
    train_ids = {id(r) for r in train}
    assert not any(id(r) in train_ids for r in test)  # no record in both halves
    assert B.split_words(records, 0.25, seed=0)[1] == test  # deterministic


def test_top1_accuracy_counts():
    class Stub:
        def top(self, sl, k=1):
            return ["a"]  # always predicts 'a'

    slices = [("a", _blob_mask()), ("a", _blob_mask()), ("b", _blob_mask())]
    acc, correct, total = B.top1_accuracy(Stub(), slices)
    assert correct["a"] == 2 and correct["b"] == 0 and total["a"] == 2 and total["b"] == 1
    assert abs(acc - 2 / 3) < 1e-9
