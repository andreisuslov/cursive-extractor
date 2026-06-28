"""Unit tests for the InkSight vectorizer's pure token/format logic (no TF, no model)."""

from ocr import inksight_vectorize as iv


def test_text_to_tokens_extracts_indices():
    assert iv.text_to_tokens("x<ink_token_5>y<ink_token_300>z") == [5, 300]
    assert iv.text_to_tokens("no tokens here") == []


def test_detokenize_strokes_decodes_points():
    # npd = 225, start token = 450. A point (x, y) -> tokens [x, y + 225].
    # Two strokes: [(10,20),(30,40)] and [(5,6)], separated by the start token.
    npd = 225
    tokens = [10, 20 + npd, 30, 40 + npd, npd * 2, 5, 6 + npd]
    strokes = iv.detokenize_strokes(tokens)
    assert strokes == [[(10, 20), (30, 40)], [(5, 6)]]


def test_detokenize_skips_out_of_range():
    npd = 225
    tokens = [10, 20 + npd, 999, 5 + npd]  # x=999 invalid -> skipped
    assert iv.detokenize_strokes(tokens) == [[(10, 20)]]


def test_strokes_to_points_normalizes_and_marks_penups():
    strokes = [[(0, 0), (112, 112), (224, 224)], [(224, 0)]]  # 2nd stroke too short
    pts = iv.strokes_to_points(strokes, size=224)
    # only the 3-point stroke survives (len>=2); ends with a pen-up
    assert pts == [
        [0.0, 0.0, 1],
        [0.5, 0.5, 1],
        [1.0, 1.0, 1],
        [1.0, 1.0, 0],
    ]


def test_strokes_to_points_clips_to_unit_range():
    pts = iv.strokes_to_points([[(-10, 250), (224, 224)]], size=224)
    xs_ys = [(p[0], p[1]) for p in pts]
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in xs_ys)
