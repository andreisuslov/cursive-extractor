"""Unit tests for the confidence-gated per-letter harvester (#2)."""

import json

from ocr.experiments import _letter_harvest as h


def test_normalize_letter_aspect_preserved():
    # x 2..4 (w=2), y 4..8 (h=4): divide both by height -> y in [0,1], x in [0, 0.5]
    n = h.normalize_letter([[2.0, 4.0], [4.0, 4.0], [4.0, 8.0]])
    assert n[0] == [0.0, 0.0]
    assert n[1] == [0.5, 0.0]  # width/height = 0.5 preserved
    assert n[2] == [0.5, 1.0]


def test_confidence_zero_when_a_letter_is_empty():
    assert h.confidence([[[0.0, 0.0, 1]], []], [True]) == 0.0


def test_confidence_high_when_snapped_and_balanced():
    letters = [[[0.0, 0.5, 1], [0.1, 0.5, 1]], [[0.5, 0.5, 1], [0.6, 0.5, 1]]]
    assert h.confidence(letters, [True]) > 0.9  # snap_frac 1, width band 1 -> ~1.0


def test_confidence_drops_when_cut_is_a_fallback():
    letters = [[[0.0, 0.5, 1], [0.1, 0.5, 1]], [[0.5, 0.5, 1], [0.6, 0.5, 1]]]
    assert h.confidence(letters, [False]) < 0.6  # 0.6*0 + 0.4*1 = 0.4


def test_harvest_keeps_a_confident_word(tmp_path):
    # 'a' ink x 0..0.40 then a pen-lift, 'b' ink x 0.55..1.0 -> a clean snap near the middle
    pts = (
        [[x / 100, 0.5, 1] for x in range(0, 41, 5)]
        + [[0.40, 0.5, 0]]
        + [[x / 100, 0.5, 1] for x in range(55, 101, 5)]
        + [[1.0, 0.5, 0]]
    )
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps([{"points": pts, "metadata": {"asciiSequence": "ab"}}]))
    library, stats = h.harvest(str(sj), threshold=0.6)
    assert stats["kept_words"] == 1
    assert len(library["a"]) == 1 and len(library["b"]) == 1
