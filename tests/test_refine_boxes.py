"""Tests for layout-partition box refinement."""

from ocr.refine_boxes import cluster_rows, refine_boxes, split_columns, tighten_to_ink


def _b(text, ymin, xmin, ymax, xmax):
    return {"text": text, "box_2d": [ymin, xmin, ymax, xmax]}


def test_refine_makes_adjacent_cells_touch_not_overlap():
    # two words in one row, two in the next
    boxes = [_b("a", 100, 100, 200, 200), _b("b", 100, 300, 200, 400)]
    r = refine_boxes(boxes)
    a, b = r[0], r[1]
    assert abs(a["box_2d"][3] - b["box_2d"][1]) <= 1  # share the midpoint boundary -> no overlap
    assert a["box_2d"][3] <= b["box_2d"][1] + 1  # a ends where b begins


def test_refine_clamps_each_row_to_a_uniform_band():
    # same row, different individual heights -> both get the row band (kills per-box vertical bleed)
    boxes = [_b("a", 100, 100, 200, 200), _b("b", 120, 300, 180, 400)]
    r = refine_boxes(boxes)
    assert r[0]["box_2d"][0] == r[1]["box_2d"][0]  # same top
    assert r[0]["box_2d"][2] == r[1]["box_2d"][2]  # same bottom


def test_refine_separates_rows_vertically():
    boxes = [_b("a", 100, 100, 200, 200), _b("c", 400, 100, 500, 200)]  # two rows, one column
    r = refine_boxes(boxes)
    assert r[0]["box_2d"][2] <= r[1]["box_2d"][0] + 1  # row 1 bottom <= row 2 top


def test_refine_preserves_order_text_and_count():
    boxes = [_b("one", 100, 100, 200, 200), _b("two", 100, 300, 200, 400)]
    r = refine_boxes(boxes)
    assert [x["text"] for x in r] == ["one", "two"]
    assert len(r) == 2


def test_split_columns_detects_gutter_and_single_column():
    # two clusters around 200 and 800 -> gutter near 500
    assert 400 < split_columns([180, 200, 220, 780, 800, 820]) < 600
    # one tight cluster -> no split
    assert split_columns([480, 500, 520]) is None


def test_tighten_to_ink_shrinks_to_ink_and_never_expands():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(img).rectangle([40, 45, 60, 55], fill="black")
    ymin, xmin, ymax, xmax = tighten_to_ink(img, [0, 0, 1000, 1000])
    assert xmin > 200 and xmax < 800  # shrank horizontally toward the ink
    assert ymin > 200 and ymax < 800  # shrank vertically toward the ink


def test_tighten_to_ink_blank_cell_returns_input():
    from PIL import Image

    img = Image.new("RGB", (50, 50), "white")
    assert tighten_to_ink(img, [100, 100, 200, 200]) == [100, 100, 200, 200]


def test_cluster_rows_groups_by_y_center():
    boxes = [_b("a", 100, 100, 200, 200), _b("b", 110, 300, 210, 400), _b("c", 400, 100, 500, 200)]
    rows = cluster_rows(boxes, med_h=100.0)
    assert len(rows) == 2  # a,b together; c separate
