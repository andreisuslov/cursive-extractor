"""Shared test setup, applied before any test module imports.

Forces a headless matplotlib backend and fully disables Weights & Biases (no
network / no login) so importing sample.py / model.py is side-effect-free in CI.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_API_KEY", "disabled-for-tests")
