"""Regression test: the order-recovery experiment's locked metrics.

Runs the real easybank/bigbank vectorization (a few seconds) and asserts the
recovered pen-up and IoU means stay at their committed values. Depends on the
data/*.json.zip banks, so it must run from the repo root.
"""

import numpy as np

from ocr.experiments._order_recovery_experiment import run


def _means(rows):
    rec_ups = float(np.mean([r["rec_ups"] for r in rows]))
    iou = float(np.mean([r["iou"] for r in rows]))
    return rec_ups, iou


def test_easybank_locked_metrics():
    rec_ups, iou = _means(run("easybank"))
    assert abs(rec_ups - 1.0) < 0.05  # connected words -> ~1 pen-up each
    assert abs(iou - 0.666) < 0.01


def test_bigbank_locked_metrics():
    rec_ups, iou = _means(run("bigbank"))
    assert abs(rec_ups - 3.1) < 0.1  # i/j/t/x diacritics -> a few pen-ups
    assert abs(iou - 0.668) < 0.01
