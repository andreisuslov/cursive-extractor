"""Canonical output layout for extracted data.

Everything lands under a single root (default ``outputs/``), namespaced by the
source PDF and page so multiple PDFs and multiple pages never collide. Filenames
repeat the origin (pdf slug + page [+ version], plus box index + word for crops)
so a file stays self-describing even if it's moved out of its folder.

Re-processing a page does not overwrite: the page folder is *versioned*. The
first run is ``page_<NNN>`` (no suffix); subsequent runs become ``page_<NNN>_1``,
``page_<NNN>_2``, ... The filename prefix carries the same suffix
(``<pdf>_p<NNN>_1_...``).

Each detected word gets its own *box folder* with fixed-name contents.

Layout::

    outputs/
      <pdf_slug>/
        <pdf_slug>_tool.html                          # capture tool (spans a run)
        page_<NNN>[_V]/
          <pdf_slug>_p<NNN>[_V]_boxes.json            # detected words + box_2d
          <pdf_slug>_p<NNN>[_V]_boxes_overlay.jpg     # page with boxes drawn
          <pdf_slug>_p<NNN>[_V]_strokes.json          # vectorized {points, metadata}
          <pdf_slug>_p<NNN>[_V]_box_<III>/            # one folder per word box
            text_recognized.txt                       # the OCR'd text
            box.jpg                                    # cropped word image
            vectorized.jpg                             # box.jpg + teal strokes overlay

Page and box numbers are 1-based / 0-based respectively and zero-padded to 3
digits so lexical sort matches numeric order.
"""

import os
import re

# Default output root; override with --output-root or OCR_OUTPUT_ROOT.
OUTPUT_ROOT = os.environ.get("OCR_OUTPUT_ROOT", "outputs")


def slugify(text, maxlen=40):
    """Filesystem-safe slug, preserving _ and - ('12/21/00' -> '12-21-00',
    'test_document' -> 'test_document', '' -> 'untitled')."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-_")
    return slug[:maxlen] or "untitled"


def pdf_slug(pdf_path):
    """Slug derived from a PDF's filename stem."""
    return slugify(os.path.splitext(os.path.basename(pdf_path))[0])


def _pp(page):
    return f"{int(page):03d}"


def _suffix(version):
    """'' for the base run (version 0/None), '_N' for later versions."""
    return f"_{int(version)}" if version else ""


def prefix(pdf_path, page, version=None):
    """Origin filename prefix, e.g. 'test_document_p004' or 'test_document_p004_1'."""
    return f"{pdf_slug(pdf_path)}_p{_pp(page)}{_suffix(version)}"


def run_dir(pdf_path, root=None):
    return os.path.join(root or OUTPUT_ROOT, pdf_slug(pdf_path))


def page_dir(pdf_path, page, version=None, root=None):
    return os.path.join(run_dir(pdf_path, root), f"page_{_pp(page)}{_suffix(version)}")


def ensure_dir(path):
    """Create ``path`` (a directory) if needed; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent(path):
    """Create the parent directory of a *file* ``path`` if needed; return it."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


# --- version discovery ------------------------------------------------------


def existing_versions(pdf_path, page, root=None):
    """Sorted list of existing version numbers for a page (0 == base 'page_NNN')."""
    rd = run_dir(pdf_path, root)
    if not os.path.isdir(rd):
        return []
    base = f"page_{_pp(page)}"
    pat = re.compile(re.escape(base) + r"(?:_(\d+))?$")
    versions = []
    for name in os.listdir(rd):
        m = pat.fullmatch(name)
        if m and os.path.isdir(os.path.join(rd, name)):
            versions.append(int(m.group(1)) if m.group(1) else 0)
    return sorted(versions)


def latest_version(pdf_path, page, root=None):
    """Highest existing version (0 == base), or None if not processed yet."""
    versions = existing_versions(pdf_path, page, root)
    return versions[-1] if versions else None


def next_version(pdf_path, page, root=None):
    """Version to use for a new run: 0 (base) if none exists, else max+1."""
    versions = existing_versions(pdf_path, page, root)
    return (max(versions) + 1) if versions else 0


# --- artifact path builders -------------------------------------------------


def boxes_json(pdf_path, page, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root), f"{prefix(pdf_path, page, version)}_boxes.json"
    )


def boxes_overlay(pdf_path, page, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root),
        f"{prefix(pdf_path, page, version)}_boxes_overlay.jpg",
    )


def strokes_json(pdf_path, page, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root), f"{prefix(pdf_path, page, version)}_strokes.json"
    )


def transcript_txt(pdf_path, page, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root), f"{prefix(pdf_path, page, version)}_transcript.txt"
    )


def qa_report(pdf_path, page, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root), f"{prefix(pdf_path, page, version)}_qa.txt"
    )


def verify_overlay(pdf_path, page, coords, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root),
        f"{prefix(pdf_path, page, version)}_verify_{coords}.jpg",
    )


# Fixed filenames inside each per-box folder.
BOX_TEXT_FILE = "text_recognized.txt"
BOX_IMAGE_FILE = "box.jpg"
BOX_VECTORIZED_FILE = "vectorized.jpg"


def box_dir(pdf_path, page, index, version=None, root=None):
    """Per-box folder, e.g. .../page_004/test_document_p004_box_000/."""
    return os.path.join(
        page_dir(pdf_path, page, version, root),
        f"{prefix(pdf_path, page, version)}_box_{int(index):03d}",
    )


def box_text(pdf_path, page, index, version=None, root=None):
    return os.path.join(box_dir(pdf_path, page, index, version, root), BOX_TEXT_FILE)


def box_image(pdf_path, page, index, version=None, root=None):
    return os.path.join(box_dir(pdf_path, page, index, version, root), BOX_IMAGE_FILE)


def box_vectorized(pdf_path, page, index, version=None, root=None):
    return os.path.join(box_dir(pdf_path, page, index, version, root), BOX_VECTORIZED_FILE)


def trace_json(pdf_path, page, index, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root),
        f"{prefix(pdf_path, page, version)}_trace_box_{int(index):03d}.json",
    )


def trace_overlay(pdf_path, page, index, version=None, root=None):
    return os.path.join(
        page_dir(pdf_path, page, version, root),
        f"{prefix(pdf_path, page, version)}_trace_box_{int(index):03d}.jpg",
    )


def tool_html(pdf_path, root=None):
    return os.path.join(run_dir(pdf_path, root), f"{pdf_slug(pdf_path)}_tool.html")
