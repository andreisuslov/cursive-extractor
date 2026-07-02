"""Pure-logic tests for ocr.ink_qa (no PDF/page rendering).

Everything runs on synthetic frames and rects: stroke clipping (keep target,
drop neighbour row, drop ruled lines), rect renormalization, and the AIoU-F1
score on small numpy images built directly.
"""

import cv2
import numpy as np

from ocr.ink_qa import (
    SIZE,
    aiou_f1,
    clip_strokes,
    expand_rect,
    rasterize_strokes,
    renormalize_points,
)

RECT = expand_rect((0.3, 0.4, 0.7, 0.6))  # expanded detection rect used below


def _stroke(pts, extra=()):
    """Pen-down rows + one pen-up terminator, optionally with rich channels."""
    rows = [[x, y, 1, *extra] for x, y in pts]
    rows.append([pts[-1][0], pts[-1][1], 0, *([0.0] * len(extra))])
    return rows


def test_clip_keeps_inside_drops_neighbour_row():
    inside = _stroke([(0.4, 0.5), (0.5, 0.52), (0.6, 0.5)])
    neighbour = _stroke([(0.4, 0.92), (0.5, 0.9), (0.6, 0.93)])  # a row-pitch below
    clipped = clip_strokes(inside + neighbour, RECT)
    assert clipped == inside


def test_clip_drops_ruled_line_even_inside_rect():
    # Median sits inside the rect, but the shape is a ruled line: wide + flat.
    ruled = _stroke([(0.05, 0.5), (0.5, 0.505), (0.95, 0.51)])
    word = _stroke([(0.45, 0.45), (0.5, 0.55), (0.55, 0.45)])
    clipped = clip_strokes(ruled + word, RECT)
    assert clipped == word


def test_clip_preserves_rich_channels_untouched():
    rich = _stroke([(0.4, 0.5), (0.5, 0.5)], extra=(0.12, 0.87))
    clipped = clip_strokes(rich, RECT)
    assert clipped == rich
    assert all(len(r) == 5 for r in clipped)


def test_clip_empty_when_all_strokes_outside():
    neighbour = _stroke([(0.4, 0.05), (0.6, 0.06)])
    assert clip_strokes(neighbour, RECT) == []


def test_renormalize_maps_rect_corners_to_unit():
    rect = (0.2, 0.3, 0.6, 0.8)
    pts = [[0.2, 0.3, 1], [0.6, 0.8, 0]]
    out = renormalize_points(pts, rect)
    assert out == [[0.0, 0.0, 1], [1.0, 1.0, 0]]


def test_renormalize_clips_and_keeps_extra_channels():
    rect = (0.2, 0.3, 0.6, 0.8)
    out = renormalize_points([[0.1, 0.9, 1, 0.5, 0.5]], rect)
    assert out == [[0.0, 1.0, 1, 0.5, 0.5]]


def _line_points(x0, y0, x1, y1, n=30):
    return [
        (x0 + (x1 - x0) * t / (n - 1), y0 + (y1 - y0) * t / (n - 1)) for t in range(n)
    ]


def test_aiou_high_for_perfect_trace_low_for_displaced():
    # Ink: a diagonal line drawn directly on the frame.
    ink = np.zeros((SIZE, SIZE), np.uint8)
    cv2.line(ink, (60, 100), (160, 130), 255, 3)
    rect = (0.0, 0.0, 1.0, 1.0)

    perfect = rasterize_strokes(_stroke(_line_points(60 / SIZE, 100 / SIZE, 160 / SIZE, 130 / SIZE)))
    displaced = rasterize_strokes(
        _stroke(_line_points(60 / SIZE, 130 / SIZE, 160 / SIZE, 160 / SIZE))  # +30 px in y
    )
    assert aiou_f1(ink, perfect, rect) > 0.9
    assert aiou_f1(ink, displaced, rect) < 0.2


def test_aiou_restricted_to_rect():
    # Ink exists only OUTSIDE the rect; strokes only inside -> both masked away -> 0.
    ink = np.zeros((SIZE, SIZE), np.uint8)
    cv2.line(ink, (10, 10), (60, 10), 255, 3)
    strokes = rasterize_strokes(_stroke([(0.5, 0.5), (0.6, 0.5)]))
    assert aiou_f1(ink, strokes, (0.4, 0.4, 0.7, 0.7)) == 0.0
