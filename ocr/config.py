"""Default configuration for the OCR pipeline.

Every value can be overridden by an environment variable, and the CLI entry
points expose the commonly-changed ones as flags. Paths default to the sample
document that already lives in ``data/content/``.
"""

import os

# --- Paths ---
# Run the CLIs from the repo root; this default is relative to it.
PDF_PATH = os.environ.get("OCR_PDF_PATH", "data/content/test_document.pdf")

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
CROP_FIT_INK = os.environ.get("OCR_CROP_FIT_INK", "1") not in ("0", "false", "False", "")

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
