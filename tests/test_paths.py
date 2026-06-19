"""Unit tests for the pure path builders in ocr.paths."""

import os

from ocr import paths


def test_slugify():
    assert paths.slugify("12/21/00") == "12-21-00"
    assert paths.slugify("test_document") == "test_document"
    assert paths.slugify("") == "untitled"
    assert paths.slugify("a b!c") == "a-b-c"


def test_slugify_maxlen():
    assert len(paths.slugify("x" * 100, maxlen=10)) == 10


def test_prefix():
    assert paths.prefix("a/test_document.pdf", 4) == "test_document_p004"
    assert paths.prefix("a/test_document.pdf", 4, 1) == "test_document_p004_1"
    # version 0 is the base run -> no suffix
    assert paths.prefix("a/test_document.pdf", 4, 0) == "test_document_p004"


def test_page_file_and_builders():
    pdf = "x/doc.pdf"
    assert paths.boxes_json(pdf, 4, None, "outputs") == "outputs/doc/page_004/doc_p004_boxes.json"
    assert (
        paths.strokes_json(pdf, 4, None, "outputs")
        == "outputs/doc/page_004/doc_p004_strokes.json"
    )
    assert (
        paths._page_file(pdf, 4, 1, "outputs", "x.json")
        == "outputs/doc/page_004_1/doc_p004_1_x.json"
    )


def test_box_family():
    pdf = "x/doc.pdf"
    bd = paths.box_dir(pdf, 4, 7, 2, "outputs")
    assert bd == "outputs/doc/page_004_2/doc_p004_2_box_007"
    assert paths.box_text(pdf, 4, 7, 2, "outputs") == os.path.join(bd, "text_recognized.txt")
    assert paths.box_image(pdf, 4, 7, 2, "outputs") == os.path.join(bd, "box.jpg")
    assert paths.box_vectorized(pdf, 4, 7, 2, "outputs") == os.path.join(bd, "vectorized.jpg")


def test_version_discovery(tmp_path):
    pdf = "x/doc.pdf"
    root = str(tmp_path)
    rd = paths.run_dir(pdf, root)
    for nm in ("page_004", "page_004_1", "page_004_3", "page_002"):
        os.makedirs(os.path.join(rd, nm))
    with open(os.path.join(rd, "page_004_9_afile"), "w"):  # a file, must be ignored
        pass
    assert paths.existing_versions(pdf, 4, root) == [0, 1, 3]
    assert paths.latest_version(pdf, 4, root) == 3
    assert paths.next_version(pdf, 4, root) == 4
    assert paths.existing_versions(pdf, 2, root) == [0]
    assert paths.next_version(pdf, 2, root) == 1


def test_version_discovery_unprocessed(tmp_path):
    assert paths.latest_version("x/doc.pdf", 7, str(tmp_path)) is None
    assert paths.next_version("x/doc.pdf", 7, str(tmp_path)) == 0
