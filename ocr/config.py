"""Default configuration for the OCR pipeline.

Every value can be overridden by an environment variable, and the CLI entry
points expose the commonly-changed ones as flags. Paths default to the sample
document that already lives in ``datasets/content/``.
"""

import os

# Strings that count as "off" for a boolean env flag (anything else is "on").
_FALSEY = ("0", "false", "False", "")


def _env_bool(name: str, default: bool = True) -> bool:
    """Read a boolean flag from the environment; 0/false/empty means off."""
    return os.environ.get(name, "1" if default else "0") not in _FALSEY


# --- Paths ---
# Run the CLIs from the repo root; this default is relative to it.
#
# PDF_PATH is the SLUG anchor: the stored page-4 box data lives under
# ``outputs/test_document/`` and its filenames carry the ``test_document`` slug, so
# this path must keep that basename for box lookups to resolve. The slug PDF itself
# is git-ignored and usually absent. The real, renderable diary PDF lives in
# ``outputs/`` -- PDF_RENDER_FALLBACK points at it so a missing slug PDF renders
# from there (no manual symlink needed). Override either via env.
PDF_PATH = os.environ.get("OCR_PDF_PATH", "datasets/content/test_document.pdf")
PDF_RENDER_FALLBACK = os.environ.get(
    "OCR_PDF_RENDER_FALLBACK", "outputs/diary_partial_pages_1-4/diary_partial_pages_1-4.pdf"
)

# Output layout (root, per-PDF/per-page folders, filenames) lives in ocr/paths.py.

# --- Cropping ---
# Boxes are padded so no glyph strokes (ascenders/descenders/dots) get clipped.
# Padding = a fixed pixel margin PLUS a fraction of the box's own size (so it
# scales with DPI and word size). The SAME values must be used by vectorize and
# package_boxes so the stroke overlay aligns with the crop.
CROP_PADDING = int(os.environ.get("OCR_CROP_PADDING", "10"))
CROP_PAD_FRAC = float(os.environ.get("OCR_CROP_PAD_FRAC", "0.12"))
# After padding, grow the box until no ink touches its borders (no clipped
# strokes). Vectorize and package_boxes MUST agree on this so overlays align.
CROP_FIT_INK = _env_bool("OCR_CROP_FIT_INK")
# Clean each word crop before vectorizing: strip ruled lines / scan-edge bands and
# keep only the target word's ink (drop neighbouring words/lines). See
# vectorize.clean_word. Disable with OCR_CROP_CLEAN=0.
CROP_CLEAN = _env_bool("OCR_CROP_CLEAN")
# Mask each crop's ink to the detection box: whiten ink outside the box (padded for
# ascenders/descenders + a small horizontal slack), so a crop rectangle that overlaps
# neighbouring words/rows in dense cursive yields ONLY the target word's ink. Critical
# for derendering (InkSight traces all ink in the crop). Opt-in (off by default so the
# existing crop/overlay outputs and tests are unchanged): OCR_CROP_MASK_TO_BOX=1.
CROP_MASK_TO_BOX = _env_bool("OCR_CROP_MASK_TO_BOX", False)
CROP_MASK_HPAD = float(os.environ.get("OCR_CROP_MASK_HPAD", "0.15"))  # x slack, frac of box height
CROP_MASK_VPAD_UP = float(os.environ.get("OCR_CROP_MASK_VPAD_UP", "0.45"))  # ascenders
CROP_MASK_VPAD_DN = float(os.environ.get("OCR_CROP_MASK_VPAD_DN", "0.40"))  # descenders

# The HTML template ships alongside this package.
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "handwriting_tool.html")
# Placeholder in the template that gets replaced with the JSON word bank.
WORD_BANK_PLACEHOLDER = "__WORD_BANK_JSON__"

# --- Page range (0-based; END is exclusive) ---
# Defaults process pages 1-3 (indices 0, 1, 2).
START_PAGE_INDEX = int(os.environ.get("OCR_START_PAGE_INDEX", "0"))
END_PAGE_INDEX = int(os.environ.get("OCR_END_PAGE_INDEX", "3"))

# --- Gemini ---
# Per-request client deadline (seconds). The grounded detection request is heavy
# (whole token list in, one box per token out) and can exceed the short default.
GEMINI_TIMEOUT = int(os.environ.get("OCR_GEMINI_TIMEOUT", "600"))
GEMINI_MODEL = os.environ.get("OCR_GEMINI_MODEL", "gemini-pro-latest")
# Fallback used on a model-not-found (404). 'gemini-pro-latest' is an alias that
# tracks the current pro model; the flash fallback is cheaper/faster.
GEMINI_FALLBACK_MODEL = os.environ.get("OCR_GEMINI_FALLBACK_MODEL", "gemini-3-flash-preview")
