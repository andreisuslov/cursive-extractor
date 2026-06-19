"""Unit tests for StrokeDataset.count_truncated (silent-truncation visibility)."""

from types import SimpleNamespace

import numpy as np

from data import StrokeDataset

ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def _line(n):
    """An n-point pen-down stroke. One word of n points tokenizes to ~2n tokens."""
    return np.array([[i * 0.01, 0.0, 1] for i in range(n)], dtype=float)


def _dataset(max_seq_length):
    # augment off -> deterministic encoding (a single word of P points -> 2P tokens)
    args = SimpleNamespace(
        alphabet=ALPHABET,
        augment=False,
        max_seq_length=max_seq_length,
        seed=0,
        downsample_mean=0.65,
        downsample_width=0.1,
    )
    words = [[_line(5)], [_line(30)], [_line(4)]]  # ~10, ~60, ~8 tokens
    texts = ["a", "b", "c"]
    return StrokeDataset(words, texts, args, name="t")


def test_count_truncated_flags_the_long_example():
    ds = _dataset(max_seq_length=20)  # token budget = 19
    n_trunc, n_checked = ds.count_truncated()
    assert n_checked == 3
    assert n_trunc == 1  # only the 30-point (~60-token) example exceeds 19


def test_count_truncated_zero_when_budget_large():
    ds = _dataset(max_seq_length=200)  # budget 199 -> all three fit
    n_trunc, n_checked = ds.count_truncated()
    assert n_trunc == 0
    assert n_checked == 3


def test_count_truncated_respects_sample_size():
    ds = _dataset(max_seq_length=20)
    n_trunc, n_checked = ds.count_truncated(sample_size=2)
    assert n_checked == 2  # only the first two examples are checked
    assert n_trunc == 1  # the 30-point example is index 1, within the first two
