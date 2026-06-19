"""Cursive OCR dataset pipeline.

Extracts word-level bounding boxes from scanned handwriting PDFs using Gemini,
then helps turn them into the {points, metadata} JSON format used for training.

Modules:
    config           - default paths, page range, and model names (env-overridable)
    pdf_utils        - load PDF pages; map 0-1000 OCR boxes to pixel crops; fit crops to ink
    gemini_ocr       - Gemini transcription + word/box OCR (free or transcript-grounded)
    reconcile        - align grounded boxes to the transcript; force text + infill gaps
    tool             - render the browser handwriting-capture tool from the HTML template
    extract_boxes    - CLI: transcribe each page + detect word/punctuation boxes
    vectorize        - CLI: trace cropped ink into (x, y, pen) stroke points (continuous, lift-free)
    package_boxes    - CLI: one folder per box (text_recognized.txt, box.jpg, vectorized.jpg)
    qa               - CLI: check the transcript matches the concatenated box texts
    crop             - CLI: crop a single OCR box to a high-DPI image
    trace_strokes    - CLI: trace strokes for one box via Gemini (model-based alt to vectorize)
    verify_dataset   - CLI: overlay a traced stroke dataset onto a PDF page (page/box coords)
    visualize_words  - CLI: draw the bounding boxes of the first N words of a page

Refactored out of the Colab notebook ``cursive_ocr_extraction.ipynb`` and the
standalone ``data/content/*.py`` scripts.
"""
