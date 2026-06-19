"""CPU forward/backward smoke test for the Transformer (tiny config, no wandb/network).

Builds the model from a hand-made SimpleNamespace config (NOT get_all_args, which
prompts for a W&B key), runs a forward pass on a dummy token batch, and does one
backward to confirm gradients are finite. Everything is tiny, CPU-only, deterministic.
"""

from types import SimpleNamespace

import torch

from model import Transformer

# Tiny CPU config. n_embd_context MUST equal n_embd: the context position embedding is
# added to the context token embedding, and the cross-attention LayerNorm is over
# n_embd_context while operating on n_embd-wide activations.
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

B, T, T_CTX = 2, 6, 5


def _tiny_model():
    torch.manual_seed(0)  # deterministic init
    return Transformer(TINY)


def _dummy_batch():
    idx = torch.randint(0, TINY.vocab_size, (B, T))
    context = torch.randint(0, TINY.context_vocab_size, (B, T_CTX))
    return idx, context


def test_forward_logits_shape():
    model = _tiny_model()
    idx, context = _dummy_batch()
    logits, loss = model(idx, context)  # no targets
    assert logits.shape == (B, T, TINY.vocab_size)
    assert loss is None  # no targets -> no loss
    assert logits.device.type == "cpu"


def test_loss_and_backward_finite_grads():
    model = _tiny_model()
    idx, context = _dummy_batch()
    targets = torch.randint(0, TINY.vocab_size, (B, T))
    logits, loss = model(idx, context, targets)
    assert logits.shape == (B, T, TINY.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss).item()  # isfinite output carries no grad -> safe to read
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads  # at least some parameters received gradients
    assert all(torch.isfinite(g).all().item() for g in grads)
