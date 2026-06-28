"""Unit tests for the InkSight vectorizer's pure token/format logic (no TF, no model)."""

import cv2
import numpy as np

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


def test_strokes_to_points_rich_adds_width_and_intensity():
    # A dark thick blob on white; sample a stroke through its centre.
    gray = np.full((224, 224), 255, np.uint8)
    gray[100:120, 50:90] = 0  # dark band, ~20px tall
    _, ink = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    pts = iv.strokes_to_points_rich([[(70, 110), (70, 111)]], gray, dt, size=224)
    assert all(len(p) == 5 for p in pts)  # x,y,pen,width,intensity
    assert pts[0][4] > 0.9  # on dark ink -> high intensity (strong)
    assert pts[0][3] > 0.0  # inside the blob -> positive width
    assert pts[-1][2] == 0 and pts[-1][3] == 0.0 and pts[-1][4] == 0.0  # pen-up = no ink


def test_strokes_to_points_rich_faint_reads_low_intensity():
    gray = np.full((224, 224), 230, np.uint8)  # very faint ink everywhere
    dt = np.ones((224, 224), np.float32)
    pts = iv.strokes_to_points_rich([[(10, 10), (12, 12)]], gray, dt, size=224)
    assert pts[0][4] < 0.2  # faint -> low intensity ("weak" stroke)
