"""Unit tests for the baseline trajectory letter-segmenter (#2 start)."""

from ocr.experiments import _letter_segment_strokes as ls


def test_down_points_drops_penups():
    pts = [[0.0, 0.0, 1], [0.1, 0.1, 1], [0.1, 0.1, 0]]  # last = pen-up marker
    assert ls._down_points(pts) == [[0.0, 0.0, 1], [0.1, 0.1, 1]]


def test_segment_splits_by_x():
    # points spanning x 0..1, two letters -> left half / right half
    pts = [[0.0, 0.5, 1], [0.2, 0.5, 1], [0.6, 0.5, 1], [1.0, 0.5, 1]]
    letters = ls.segment_word_strokes(pts, 2)
    assert len(letters) == 2
    assert [p[0] for p in letters[0]] == [0.0, 0.2]  # x < 0.5
    assert [p[0] for p in letters[1]] == [0.6, 1.0]  # x >= 0.5
    assert ls.cut_quality(letters) == 1.0


def test_segment_handles_rich_points():
    # 5-channel points (x,y,pen,width,intensity) still split on pen index 2
    pts = [[0.0, 0.5, 1, 0.01, 0.8], [0.9, 0.5, 1, 0.01, 0.8], [0.9, 0.5, 0, 0.0, 0.0]]
    letters = ls.segment_word_strokes(pts, 2)
    assert len(letters) == 2 and letters[0] and letters[1]


def test_cut_quality_flags_empty_bins():
    # only the two x-extremes -> with 3 letters the middle bin is empty -> quality 2/3
    pts = [[0.0, 0.5, 1], [1.0, 0.5, 1]]
    q = ls.cut_quality(ls.segment_word_strokes(pts, 3))
    assert abs(q - 2 / 3) < 1e-9


def test_rasterize_returns_binary():
    r = ls._rasterize([[0.0, 0.5, 1], [0.5, 0.4, 1], [1.0, 0.5, 1]], height=32)
    assert r is not None
    img, _x0, _x1, w = r
    assert img.shape[0] == 32 and img.ndim == 2 and int(img.max()) == 255 and w >= 32


def test_forced_align_returns_L_segments_preserving_points():
    # two separated x-clusters, 2 letters -> 2 segments, all 4 down-points kept
    pts = [[0.0, 0.5, 1], [0.1, 0.5, 1], [0.1, 0.5, 0], [0.9, 0.5, 1], [1.0, 0.5, 1], [1.0, 0.5, 0]]
    letters = ls.forced_align_word_strokes(pts, "ab")  # builds recognizer (DejaVu fallback on CI)
    assert len(letters) == 2
    assert sum(len(s) for s in letters) == 4
