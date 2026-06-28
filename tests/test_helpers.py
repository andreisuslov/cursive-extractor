"""Unit tests for assorted non-torch ocr helpers (synthetic inputs, no network)."""

import os

import numpy as np
import pytest
from PIL import Image

from ocr import crop, extract_boxes, package_boxes, paths, qa, reconcile, verify_dataset
from ocr.pdf_utils import box_to_crop_box

# --- reconcile.reconcile_indexed -----------------------------------------------


def test_reconcile_indexed_maps_and_infills():
    result, n_infilled = reconcile.reconcile_indexed(
        {1: [10, 10, 20, 20], 3: [50, 50, 60, 60]}, ["a", "b", "c"]
    )
    assert [e["text"] for e in result] == ["a", "b", "c"]  # tokens kept in order
    assert n_infilled == 1
    assert result[0]["box_2d"] == [10, 10, 20, 20]
    assert result[2]["box_2d"] == [50, 50, 60, 60]
    assert result[1].get("estimated") is True  # the missing token is infilled
    assert result[1]["box_2d"] == [30, 30, 40, 40]  # interpolated midpoint


def test_reconcile_indexed_all_present_none_infilled():
    result, n_infilled = reconcile.reconcile_indexed({1: [0, 0, 1, 1], 2: [2, 2, 3, 3]}, ["x", "y"])
    assert n_infilled == 0
    assert all("estimated" not in e for e in result)


# --- verify_dataset mappers ----------------------------------------------------


def test_page_mapper():
    f = verify_dataset._page_mapper(200, 100)
    assert f((0.5, 0.5)) == (100.0, 50.0)
    assert f((0.0, 1.0)) == (0.0, 100.0)


def test_box_mapper_spans_crop_box():
    box = [100, 200, 300, 400]
    left, top, right, bottom = box_to_crop_box(box, 1000, 1000, 10)
    f = verify_dataset._box_mapper(box, 1000, 1000, 10)
    assert f((0, 0)) == (left, top)  # normalized (0,0) -> crop top-left
    assert f((1, 1)) == (right, bottom)  # normalized (1,1) -> crop bottom-right


# --- extract_boxes.parse_args --------------------------------------------------


def test_extract_boxes_parse_args_defaults():
    args = extract_boxes.parse_args([])
    assert args.grounded is True
    assert args.limit is None
    assert args.start_page == 1


def test_extract_boxes_parse_args_overrides():
    args = extract_boxes.parse_args(
        ["--start-page", "2", "--end-page", "5", "--no-grounded", "--limit", "10"]
    )
    assert (args.start_page, args.end_page, args.grounded, args.limit) == (2, 5, False, 10)


# --- package_boxes.draw_vectorized ---------------------------------------------


def test_draw_vectorized_draws_without_mutating_input():
    img = Image.new("RGB", (40, 20), (255, 255, 255))
    before = np.array(img).copy()
    points = [[0.1, 0.5, 1], [0.9, 0.5, 1], [0.9, 0.5, 0]]  # one pen-down segment
    out = package_boxes.draw_vectorized(img, points)
    assert out is not img
    assert out.size == img.size
    assert np.array_equal(np.array(img), before)  # input image not mutated
    assert not np.array_equal(np.array(out), before)  # something was drawn


def test_draw_vectorized_pen_up_draws_nothing():
    img = Image.new("RGB", (40, 20), (255, 255, 255))
    points = [[0.1, 0.5, 0], [0.9, 0.5, 0]]  # all pen-up -> no segments drawn
    out = package_boxes.draw_vectorized(img, points)
    assert np.array_equal(np.array(out), np.array(img))


def test_draw_vectorized_grayscale_crop():
    # clean_word() returns a mode-"L" (grayscale) crop; drawing the RGB teal
    # overlay on it must not crash and must yield an RGB image with the stroke.
    img = Image.new("L", (40, 20), 255)
    points = [[0.1, 0.5, 1], [0.9, 0.5, 1], [0.9, 0.5, 0]]
    out = package_boxes.draw_vectorized(img, points)
    assert out.mode == "RGB"
    arr = np.array(out)
    assert (arr[..., 0] != arr[..., 1]).any()  # a coloured (non-gray) pixel was drawn


# --- qa.run_qa -----------------------------------------------------------------


def _setup_qa(root, tokens, box_texts, pdf="x/doc.pdf", page=4):
    tpath = paths.ensure_parent(paths.transcript_txt(pdf, page, None, root))
    with open(tpath, "w") as f:
        f.write(" ".join(tokens) + "\n")
    for i, text in enumerate(box_texts):
        bdir = paths.ensure_dir(paths.box_dir(pdf, page, i, None, root))
        with open(os.path.join(bdir, paths.BOX_TEXT_FILE), "w") as f:
            f.write(text)


def test_run_qa_passes_when_transcript_matches_boxes(tmp_path):
    root = str(tmp_path)
    _setup_qa(root, ["hello", "world."], ["hello", "world."])
    passed, report = qa.run_qa("x/doc.pdf", 4, None, root, write=False)
    assert passed is True
    assert "exact-order match: PASS" in report


def test_run_qa_fails_on_extra_box(tmp_path):
    root = str(tmp_path)
    _setup_qa(root, ["hello", "world."], ["hello", "world.", "extra"])
    passed, report = qa.run_qa("x/doc.pdf", 4, None, root, write=False)
    assert passed is False
    assert "extra" in report


# --- crop.crop_box_image (synthetic page via a monkeypatched load_page) ---------


def _synthetic_page():
    arr = np.full((300, 300), 255, np.uint8)
    arr[100:140, 80:200] = 0  # a single ink blob
    return Image.fromarray(arr).convert("RGB")


def test_crop_box_image_returns_crop_and_text(monkeypatch):
    monkeypatch.setattr(crop, "load_page", lambda *a, **k: _synthetic_page())
    boxes = [{"box_2d": [316, 233, 483, 700], "text": "word"}]  # over the blob
    crop_img, text = crop.crop_box_image("fake.pdf", boxes, 0, 0, dpi=100, fit_ink=False)
    assert text == "word"
    assert isinstance(crop_img, Image.Image)
    assert crop_img.size[0] > 0 and crop_img.size[1] > 0


def test_crop_box_image_missing_box_raises(monkeypatch):
    monkeypatch.setattr(crop, "load_page", lambda *a, **k: _synthetic_page())
    with pytest.raises(ValueError, match="box_2d"):
        crop.crop_box_image("fake.pdf", [{"text": "x"}], 0, 0)


def test_crop_mask_to_box_whitens_neighbour(monkeypatch):
    # Page with a target word and a separate neighbour word to its right.
    arr = np.full((300, 300), 255, np.uint8)
    arr[120:150, 80:160] = 0  # target ink
    arr[120:150, 180:240] = 0  # neighbour ink (outside the target box)
    page = Image.fromarray(arr).convert("RGB")
    monkeypatch.setattr(crop, "load_page", lambda *a, **k: page)
    boxes = [{"box_2d": [400, 267, 500, 533], "text": "target"}]  # 0-1000, snug on target
    # Large pad_frac makes the crop rectangle include the neighbour, so masking has work to do.
    plain, _ = crop.crop_box_image(
        "f.pdf", boxes, 0, 0, dpi=100, pad_frac=1.0, fit_ink=False, mask_to_box=False
    )
    masked, _ = crop.crop_box_image(
        "f.pdf", boxes, 0, 0, dpi=100, pad_frac=1.0, fit_ink=False, mask_to_box=True
    )
    pa, ma = np.array(plain.convert("L")), np.array(masked.convert("L"))
    assert ma.min() == 0  # target ink kept
    assert (ma < 128).sum() < (pa < 128).sum()  # neighbour ink whitened out
