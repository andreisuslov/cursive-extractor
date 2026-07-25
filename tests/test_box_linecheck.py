from ocr.experiments.box_linecheck import check, x_overlap


def _grid():
    """Three tidy rows of one-line boxes: pitch 100, height 50."""
    return [
        {"text": f"r{r}", "box": [x, r * 100, x + 60, r * 100 + 50], "done": True}
        for r in range(3)
        for x in (0, 100, 200)
    ]


def test_tidy_single_line_boxes_are_clean():
    assert check(_grid()) == []


def test_flags_box_that_eats_neighbouring_lines():
    ws = [*_grid(), {"text": "straddle", "box": [0, 10, 60, 260], "done": True}]
    assert "straddle" in {t for _, t, _ in check(ws)}


def test_tall_but_isolated_box_is_not_flagged():
    """A word tall from its own ascender+descender must survive — that was the false-alarm risk."""
    ws = [*_grid(), {"text": "tall", "box": [500, 0, 560, 200], "done": True}]
    assert "tall" not in {t for _, t, _ in check(ws)}


def test_only_done_words_are_checked_by_default():
    ws = [*_grid(), {"text": "straddle", "box": [0, 10, 60, 260], "done": False}]
    assert check(ws) == []
    assert check(ws, only_done=False)


def test_x_overlap_is_fraction_of_narrower_box():
    assert x_overlap([0, 0, 100, 10], [50, 0, 150, 10]) == 0.5
    assert x_overlap([0, 0, 100, 10], [200, 0, 300, 10]) == 0
