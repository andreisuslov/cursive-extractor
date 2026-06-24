"""Unit tests for the row-projection line detector (ocr.segment_words.detect_lines)
and the agent-box-per-line labelling mapper (ocr.label_lines._shapes_from_boxes).

These exercise the coordinate math and the giant-blob regression guard directly, so
the changed code paths are covered by CI (the rest of the gate never imports them).
All inputs are synthetic numpy arrays -- no PDF, no network, no model.
"""

from itertools import pairwise

import numpy as np

from ocr import label_lines
from ocr.segment_words import _column_blocks, detect_lines


def _two_up_binv(h: int = 600, w: int = 1000, n_lines: int = 8, xh: int = 30) -> np.ndarray:
    """Synthetic two-up page: a left text block and a right text block separated by a
    blank binding gutter, each with ``n_lines`` evenly spaced horizontal text rows."""
    binv = np.zeros((h, w), np.uint8)
    pitch = h // (n_lines + 1)
    for page_x0, page_x1 in ((60, 420), (580, 940)):  # left block, right block
        for i in range(n_lines):
            y = pitch * (i + 1)
            binv[y : y + xh, page_x0:page_x1] = 255  # a solid text row
    return binv


# --- detect_lines: giant-blob regression guard + two-up column split -----------


def test_detect_lines_no_full_width_box():
    binv = _two_up_binv()
    xh = 30.0
    lines = detect_lines(binv, xh)
    w = binv.shape[1]
    assert lines, "expected line boxes"
    # The RLSA giant-blob bug merged most of the page into one >half-width box.
    assert all(bw <= 0.55 * w for _, _, bw, _ in lines), [bw for _, _, bw, _ in lines]


def test_detect_lines_two_up_yields_two_column_blocks():
    binv = _two_up_binv()
    blocks = _column_blocks(binv, 30.0)
    # left page block, right page block -> at least two distinct x-ranges.
    assert len(blocks) >= 2, blocks
    assert blocks == sorted(blocks)  # left-to-right reading order
    # blocks must not overlap (a clean gutter cut)
    for (_a0, a1), (b0, _b1) in pairwise(blocks):
        assert a1 <= b0, blocks


def test_detect_lines_reading_order_left_page_first():
    binv = _two_up_binv()
    lines = detect_lines(binv, 30.0)
    mid = binv.shape[1] // 2
    sides = ["L" if (x + bw / 2) < mid else "R" for x, _, bw, _ in lines]
    # every left-page line precedes every right-page line in reading order
    assert "RL" not in "".join(sides), "".join(sides)


def test_detect_lines_no_box_exceeds_height_cap():
    # densely stacked rows (tight inter-line gap) must still split into single lines.
    binv = _two_up_binv(n_lines=12)
    xh = 30.0
    lines = detect_lines(binv, xh)
    assert lines
    assert all(bh <= 2.2 * xh for _, _, _, bh in lines), [bh for _, _, _, bh in lines]


# --- _shapes_from_boxes: crop -> page -> 0-1000 round-trip ---------------------


def _line(crop_box, crop_size):
    return {"idx": 0, "box": [0, 0, 1, 1], "crop_box": crop_box, "crop_size": crop_size}


def test_shapes_from_boxes_roundtrip_non_upscaled():
    """A crop the same size as its page region: an agent box maps straight back to its
    page-pixel location (and thence to 0-1000) within rounding tolerance."""
    w_pg = h_pg = 1000
    binv = np.zeros((h_pg, w_pg), np.uint8)  # no ink -> rect-corner fallback polygon
    cx0, cy0, cx1, cy1 = 200, 300, 600, 360  # a line region on the page
    ln = _line([cx0, cy0, cx1, cy1], [cx1 - cx0, cy1 - cy0])  # crop NOT upscaled
    bx0, by0, bx1, by1 = 40, 5, 120, 50  # agent word box in crop pixels
    word = {"text": "hi", "box": [bx0, by0, bx1, by1]}
    shapes = label_lines._shapes_from_boxes(binv, ln, [word], w_pg, h_pg)
    assert len(shapes) == 1
    ymin, xmin, ymax, xmax = shapes[0]["box_2d"]
    # expected page pixels -> 0-1000
    exp_x0 = round((cx0 + bx0) / w_pg * 1000)
    exp_x1 = round((cx0 + bx1) / w_pg * 1000)
    exp_y0 = round((cy0 + by0) / h_pg * 1000)
    exp_y1 = round((cy0 + by1) / h_pg * 1000)
    assert abs(xmin - exp_x0) <= 1 and abs(xmax - exp_x1) <= 1, shapes[0]["box_2d"]
    assert abs(ymin - exp_y0) <= 1 and abs(ymax - exp_y1) <= 1, shapes[0]["box_2d"]


def test_shapes_from_boxes_roundtrip_upscaled():
    """A tiny crop upscaled to width 600: crop pixels are downscaled by the saved-vs-
    region ratio before mapping to the page, so the box still lands on its region."""
    w_pg = h_pg = 1000
    binv = np.zeros((h_pg, w_pg), np.uint8)
    cx0, cy0, cx1, cy1 = 100, 100, 400, 150  # 300x50 page region
    region_w, region_h = cx1 - cx0, cy1 - cy0
    saved_w = 600  # upscaled
    saved_h = round(region_h * saved_w / region_w)
    ln = _line([cx0, cy0, cx1, cy1], [saved_w, saved_h])
    # agent box covering the whole saved crop -> should map back to the whole region
    box = [0, 0, saved_w, saved_h]
    shapes = label_lines._shapes_from_boxes(binv, ln, [{"text": "w", "box": box}], w_pg, h_pg)
    assert len(shapes) == 1
    ymin, xmin, ymax, xmax = shapes[0]["box_2d"]
    assert abs(xmin - round(cx0 / w_pg * 1000)) <= 1
    assert abs(xmax - round(cx1 / w_pg * 1000)) <= 1
    assert abs(ymin - round(cy0 / h_pg * 1000)) <= 1
    assert abs(ymax - round(cy1 / h_pg * 1000)) <= 1


# --- _shapes_from_boxes: defensive parsing of untrusted LLM JSON --------------


def test_shapes_from_boxes_skips_missing_box_keeps_good_words():
    w_pg = h_pg = 1000
    binv = np.zeros((h_pg, w_pg), np.uint8)
    ln = _line([0, 0, 100, 20], [100, 20])
    words = [
        {"text": "NOBOX"},  # missing box -> would KeyError before the fix
        {"box": [1, 2, 3]},  # wrong arity -> would ValueError before the fix
        {"text": "junk", "box": "nope"},  # non-list box
        {"text": "ok", "box": [10, 2, 50, 18]},  # the one good word
    ]
    shapes = label_lines._shapes_from_boxes(binv, ln, words, w_pg, h_pg)
    assert [s["text"] for s in shapes] == ["ok"]  # bad words skipped, page survives


def test_shapes_from_boxes_normalizes_inverted_box():
    w_pg = h_pg = 1000
    binv = np.zeros((h_pg, w_pg), np.uint8)
    ln = _line([0, 0, 100, 20], [100, 20])
    # x1<x0 and y1<y0: must be normalized, not dropped, and not produce a negative box
    shapes = label_lines._shapes_from_boxes(
        binv, ln, [{"text": "z", "box": [50, 18, 10, 2]}], w_pg, h_pg
    )
    assert len(shapes) == 1
    ymin, xmin, ymax, xmax = shapes[0]["box_2d"]
    assert xmin <= xmax and ymin <= ymax


# --- assemble dual-format dispatch (strings vs dicts) -------------------------


def test_assemble_dispatches_string_vs_dict_format(monkeypatch):
    """assemble() routes a list-of-strings line through the valley splitter and a
    list-of-dicts line through the agent-box mapper."""
    calls = {"strings": 0, "boxes": 0}

    def fake_strings(binv, ln, words, xh, w_pg, h_pg):
        calls["strings"] += 1
        return [{"text": w, "box_2d": [0, 0, 1, 1], "polygon": []} for w in words]

    def fake_boxes(binv, ln, words, w_pg, h_pg):
        calls["boxes"] += 1
        return [{"text": w["text"], "box_2d": [0, 0, 1, 1], "polygon": []} for w in words]

    monkeypatch.setattr(label_lines, "_shapes_from_strings", fake_strings)
    monkeypatch.setattr(label_lines, "_shapes_from_boxes", fake_boxes)
    monkeypatch.setattr(label_lines, "_ink", lambda rgb: (np.zeros((4, 4), np.uint8), 1.0, []))

    import json
    import tempfile

    rgb = np.zeros((10, 10, 3), np.uint8)
    with tempfile.TemporaryDirectory() as wd:
        man = {
            "lines": [
                {"idx": 0, "box": [0, 0, 5, 5], "crop_box": [0, 0, 5, 5], "crop_size": [5, 5]},
                {"idx": 1, "box": [0, 5, 5, 5], "crop_box": [0, 5, 5, 10], "crop_size": [5, 5]},
            ]
        }
        with open(f"{wd}/manifest.json", "w") as f:
            json.dump(man, f)
        with open(f"{wd}/words.json", "w") as f:
            json.dump(
                {
                    "0": ["went", "to"],  # string format -> valley splitter
                    "1": [{"text": "Mr", "box": [0, 0, 3, 3]}],  # dict format -> box mapper
                },
                f,
            )
        shapes = label_lines.assemble(rgb, wd, f"{wd}/out.json")

    assert calls["strings"] == 1 and calls["boxes"] == 1
    assert sorted(s["text"] for s in shapes) == ["Mr", "to", "went"]
