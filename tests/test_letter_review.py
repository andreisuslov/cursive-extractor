"""Tests for the HITL letter-cut review data layer (export/ingest round-trip)."""

import json

from ocr.experiments import _letter_review as lr


def test_export_emits_points_and_cuts(tmp_path):
    pts = (
        [[x / 100, 0.5, 1] for x in range(0, 41, 5)]
        + [[0.40, 0.5, 0]]
        + [[x / 100, 0.5, 1] for x in range(55, 101, 5)]
        + [[1.0, 0.5, 0]]
    )
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps([{"points": pts, "metadata": {"asciiSequence": "ab"}}]))
    review = lr.export_for_review(str(sj))
    assert len(review) == 1
    item = review[0]
    assert item["text"] == "ab"
    assert len(item["points"]) >= 2 and len(item["cuts"]) == 1  # L-1 cuts for 2 letters


def test_ingest_splits_words_into_letters():
    review = [
        {
            "text": "ab",
            "points": [[0.0, 0.5], [0.2, 0.5], [0.8, 0.5], [1.0, 0.5]],
            "cuts": [0.5],
        }
    ]
    library = lr.ingest_review(review)
    assert set(library) == {"a", "b"}
    assert len(library["a"]) == 1 and len(library["b"]) == 1


def test_export_then_ingest_round_trips(tmp_path):
    pts = (
        [[x / 100, 0.5, 1] for x in range(0, 41, 5)]
        + [[0.40, 0.5, 0]]
        + [[x / 100, 0.5, 1] for x in range(55, 101, 5)]
        + [[1.0, 0.5, 0]]
    )
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps([{"points": pts, "metadata": {"asciiSequence": "ab"}}]))
    library = lr.ingest_review(lr.export_for_review(str(sj)))
    assert "a" in library and "b" in library  # both letters recovered at the auto cut
