"""Unit tests for the pure response parsers in ocr.gemini_ocr (no API calls)."""

from ocr import gemini_ocr as g


def test_strip_fences():
    assert g._strip_fences("```json\n[1,2]\n```") == "[1,2]"
    assert g._strip_fences("```\n[3]\n```") == "[3]"
    assert g._strip_fences("  [3,4]  ") == "[3,4]"


def test_box_of_keys():
    assert g._box_of({"box_2d": [1, 2, 3, 4]}) == [1, 2, 3, 4]
    assert g._box_of({"html_2d": [5, 6, 7, 8]}) == [5, 6, 7, 8]  # model sometimes emits html_2d
    assert g._box_of({"foo_2d": [9, 9, 9, 9]}) == [9, 9, 9, 9]  # any *_2d key
    assert g._box_of({"text": "x"}) is None
    assert g._box_of("not a dict") is None
    assert g._box_of({"box_2d": [1, 2]}) is None  # fewer than 4 elements


def test_box_of_prefers_box_2d_over_html_2d():
    assert g._box_of({"html_2d": [5, 6, 7, 8], "box_2d": [1, 2, 3, 4]}) == [1, 2, 3, 4]


def test_parse_word_boxes_strict_json():
    assert g.parse_word_boxes('[{"text":"hi","box_2d":[1,2,3,4]}]') == [
        {"text": "hi", "box_2d": [1, 2, 3, 4]}
    ]


def test_parse_word_boxes_normalizes_html_2d():
    assert g.parse_word_boxes('[{"text":"yo","html_2d":[5,6,7,8]}]') == [
        {"text": "yo", "box_2d": [5, 6, 7, 8]}
    ]


def test_parse_word_boxes_regex_fallback():
    out = g.parse_word_boxes('junk {"text":"z","box_2d":[0,0,1,1]} trailing')
    assert out == [{"text": "z", "box_2d": [0, 0, 1, 1]}]


def test_parse_word_boxes_no_match_returns_empty():
    assert g.parse_word_boxes("no json here") == []


def test_parse_indexed_boxes():
    out = g.parse_indexed_boxes(
        '[{"index":1,"box_2d":[1,2,3,4]},{"index":3,"box_2d":[5,6,7,8]}]', 5
    )
    assert out == {1: [1, 2, 3, 4], 3: [5, 6, 7, 8]}


def test_parse_indexed_boxes_filters_out_of_range():
    assert g.parse_indexed_boxes('[{"index":9,"box_2d":[1,1,1,1]}]', 5) == {}
