"""CPU-only, deterministic unit tests for sample.py generation/decoding.

Uses a tiny Transformer built from a SimpleNamespace (like test_model_smoke), greedy
decoding (do_sample=False) for determinism, headless matplotlib (set in conftest), and
no W&B / network.
"""

import dataclasses
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import Transformer
from sample import GenerationParams, generate, plot_strokes, word_offsets_to_points

# Tiny CPU config (n_embd_context == n_embd is required by the model).
TINY = SimpleNamespace(
    n_embd=16,
    n_embd_context=16,
    n_ctx_head=2,
    n_layer=2,
    block_size=12,
    vocab_size=30,
    context_vocab_size=10,
    context_block_size=8,
)


def _tiny_model():
    torch.manual_seed(0)  # deterministic init
    model = Transformer(TINY)
    model.eval()
    return model


def _inputs(t0=3, t_ctx=5):
    idx = torch.randint(0, TINY.vocab_size, (1, t0))
    context = torch.randint(0, TINY.context_vocab_size, (1, t_ctx))
    return idx, context


# --- generate -------------------------------------------------------------------


def test_generate_output_shape_and_prefix():
    model = _tiny_model()
    idx, context = _inputs(t0=3)
    out = generate(model, idx, context, max_new_tokens=8, do_sample=False)
    assert out.shape == (1, 8)  # grows the sequence to max_new_tokens
    assert torch.equal(out[:, :3], idx)  # keeps the conditioning prefix
    assert out.device.type == "cpu"


def test_generate_deterministic_greedy():
    model = _tiny_model()
    idx, context = _inputs(t0=3)
    a = generate(model, idx, context, max_new_tokens=10, do_sample=False)
    b = generate(model, idx, context, max_new_tokens=10, do_sample=False)
    assert torch.equal(a, b)  # greedy (do_sample=False) is deterministic


def test_generate_no_growth_when_budget_below_prefix():
    model = _tiny_model()
    idx, context = _inputs(t0=3)
    out = generate(model, idx, context, max_new_tokens=2, do_sample=False)  # 2 < 3
    assert out.shape == (1, 3)  # steps = max(0, 2 - 3) = 0, nothing appended


# --- GenerationParams -----------------------------------------------------------


def test_generation_params_defaults():
    p = GenerationParams()
    assert p.temperature == 1.0
    assert p.do_sample is False
    assert p.num_steps == 1050
    assert p.warmup_steps == 50
    assert p.n_at_a_time == 2
    assert p.n_words == 4
    assert p.seed == 42
    assert p.letter_height == 0.35
    assert p.space_width == 0.16


def test_generation_params_linewidth_is_class_attr_not_field():
    # `linewidth = 1.3` has no type annotation, so it is a plain class attribute
    # (readable as 1.3) but NOT a dataclass field. Documenting observed behavior.
    p = GenerationParams()
    assert p.linewidth == 1.3
    assert "linewidth" not in {f.name for f in dataclasses.fields(p)}


# --- word_offsets_to_points -----------------------------------------------------


def _polar(*rows):
    return np.array(rows, dtype=float)  # rows are (r, theta, pen)


def test_word_offsets_to_points_structure_and_pen():
    w1 = _polar([0, 0, 1], [0.1, 0, 1], [0.1, 0, 1], [0.1, 0, 0])
    w2 = _polar([0, 0, 1], [0.2, 0, 1], [0.2, 0, 0])
    out = word_offsets_to_points([w1, w2], GenerationParams())
    assert len(out) == 2  # one points-array per input word
    assert out[0].shape == (4, 3)
    assert out[1].shape == (3, 3)
    assert np.array_equal(out[0][:, 2], w1[:, 2])  # pen states preserved
    assert np.array_equal(out[1][:, 2], w2[:, 2])


def test_word_offsets_to_points_advances_x_and_clips_y():
    w1 = _polar([0, 0, 1], [0.1, 0, 1], [0.1, 0, 0])
    w2 = _polar([0, 0, 1], [0.2, 0, 1], [0.2, 0, 0])
    p = GenerationParams()
    out = word_offsets_to_points([w1, w2], p)
    assert out[1][0, 0] >= out[0][-1, 0]  # the second word starts to the right
    assert np.all(np.abs(out[0][:, 1]) <= p.letter_height + 1e-9)  # y clamped to letter_height


# --- plot_strokes ---------------------------------------------------------------


def test_plot_strokes_renders_without_error():
    stroke = np.array([[0.0, 0.0, 1], [0.1, 0.1, 1], [0.2, 0.0, 0], [0.3, 0.2, 1]])
    fig, ax = plot_strokes(stroke, "unit test")
    assert fig is not None and ax is not None
    assert len(ax.get_lines()) == 2  # the pen-up at index 2 splits it into 2 segments
    plt.close(fig)


def test_plot_strokes_all_pen_up_draws_nothing():
    stroke = np.array([[0.0, 0.0, 0], [0.1, 0.1, 0]])  # no pen-down points
    fig, ax = plot_strokes(stroke, "empty")
    assert len(ax.get_lines()) == 0
    plt.close(fig)
