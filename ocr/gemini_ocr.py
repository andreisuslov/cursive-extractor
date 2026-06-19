"""Gemini-backed word-level OCR with bounding boxes."""

import json
import os
import re
import time

import google.generativeai as genai
from PIL import ImageDraw

from . import config

# Provider errors worth retrying (transient server-side timeouts/overload).
_TRANSIENT = ("504", "503", "Deadline", "Unavailable", "Internal", "overloaded")


def _generate(model, parts, retries=2):
    """generate_content with a generous client deadline and transient retry."""
    last = None
    for attempt in range(retries + 1):
        try:
            return model.generate_content(parts, request_options={"timeout": config.GEMINI_TIMEOUT})
        except Exception as e:
            msg = str(e)
            if any(t in msg for t in _TRANSIENT) and attempt < retries:
                print(f"Transient Gemini error ({msg[:50]}...); retry {attempt + 1}/{retries}")
                last = e
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise last


# Boxes come back as [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale.
OCR_PROMPT = """
Task: Perform Word-Level OCR with Bounding Boxes.

Instructions:
1. Identify every word in the image, in natural reading order (left-to-right,
   then top-to-bottom).
2. Attach any adjacent punctuation to the word it belongs to so each entry is ONE
   token (e.g. "wealth.", "(note)", "don't", "$50", "12/21/00", "self-worth").
   Do NOT emit standalone punctuation entries. Keep hyphenated words as ONE token
   ("self-worth"), never MERGE two separate words ("a month" stays two tokens),
   and copy the capitalization you see EXACTLY ("That" not "that").
3. For each token give its text and bounding box as [ymin, xmin, ymax, xmax].
4. The bounding box MUST fully enclose the entire token and every stroke of every
   letter -- ascenders, descenders, dots on i/j, t crossbars, tails -- with a
   little margin. Never clip any part of a glyph.
5. Normalize coordinates to a scale of 0 to 1000 (where 1000 is image height/width).
6. Return a raw JSON array of objects, nothing else. Use ONLY the keys "text"
   and "box_2d"; no other keys, comments, or prose.

JSON Schema:
[
  {"text": "Word1", "box_2d": [ymin, xmin, ymax, xmax]},
  {"text": "Word2", "box_2d": [ymin, xmin, ymax, xmax]}
]
"""

# Plain-text full-page transcription (no JSON), used for the QA cross-check and to
# ground detection. Tokenization/casing rules MIRROR OCR_PROMPT so the two passes
# agree (these rules are what most QA mismatches come from).
TRANSCRIBE_PROMPT = """
Transcribe ALL handwritten text in this image exactly, in natural reading order
(left-to-right, then top-to-bottom). Keep every word and its punctuation together
(e.g. "wealth.", "don't"). Keep hyphenated words as one token ("self-worth"),
never merge two separate words ("a month" stays two words), and preserve the exact
capitalization you see. Separate tokens with single spaces. Return ONLY the
transcribed text -- no commentary, no quotes, no markdown.
"""


def grounded_ocr_prompt(tokens):
    """Detection prompt that localizes a KNOWN, NUMBERED list of tokens.

    The model returns each box tagged with its token NUMBER, so we map boxes to
    tokens by the returned index -- robust to the model skipping a token or
    drifting out of order (which it does on dense pages). This removes the
    split/merge/case disagreements that dominate QA failures.
    """
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tokens))
    return f"""
Task: Locate each NUMBERED handwritten word on the page and return its box.

The page text, in reading order, as a NUMBERED list (number. token):
{listing}

Instructions:
1. For each numbered token, find that exact word on the page and return its box,
   tagged with the SAME number.
2. Return a raw JSON array of objects
   {{"index": <token number>, "box_2d": [ymin, xmin, ymax, xmax]}}.
3. "index" MUST be the token's number from the list above (1..{len(tokens)}).
   Return one object per token you can locate. If a token genuinely cannot be
   found, omit it rather than guessing.
4. box_2d on a 0-1000 scale (1000 = image height/width). Fully enclose every
   stroke of the word -- ascenders, descenders, dots on i/j, t crossbars, tails --
   with a small margin; never clip a glyph.
5. Output ONLY the JSON array. Use only the keys "index" and "box_2d".
"""


def parse_indexed_boxes(text, n_tokens):
    """Parse a numbered-token detection response into ``{1-based index: box}``."""
    text = _strip_fences(text)
    result = {}
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict) and "index" in d:
                    box = _box_of(d)
                    idx = d.get("index")
                    if isinstance(idx, int) and 1 <= idx <= n_tokens and box:
                        result.setdefault(idx, [int(v) for v in box[:4]])
            if result:
                return result
    except json.JSONDecodeError:
        pass
    for m in re.finditer(
        r'"index"\s*:\s*(\d+)[^{}]*?"\w+_2d"\s*:\s*'
        r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)",
        text,
    ):
        idx = int(m.group(1))
        if 1 <= idx <= n_tokens:
            result.setdefault(
                idx, [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
            )
    return result


# Tolerant extractor: pulls every well-formed {"text": ..., "<k>_2d": [y,x,y,x]}
# object out of a response, so one malformed entry or trailing junk can't sink
# the whole page. The box key is matched loosely because the model sometimes
# emits "html_2d" (or similar) instead of "box_2d". Used when strict JSON fails.
_WORD_BOX_RE = re.compile(
    r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"\w+_2d"\s*:\s*'
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _box_of(entry):
    """Return a 4+ element box list from an entry, under any '*_2d' key."""
    if not isinstance(entry, dict):
        return None
    for key in ("box_2d", "html_2d"):
        v = entry.get(key)
        if isinstance(v, list) and len(v) >= 4:
            return v
    for key, v in entry.items():  # any other "<something>_2d" the model invents
        if key.endswith("_2d") and isinstance(v, list) and len(v) >= 4:
            return v
    return None


def parse_word_boxes(text):
    """Parse a model response into a list of {text, box_2d} dicts, tolerantly.

    Accepts any ``*_2d`` box key (the model sometimes emits ``html_2d``) and
    normalizes it to ``box_2d``. Tries strict JSON first, then regex-extracts.
    """
    text = _strip_fences(text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            good = []
            for d in data:
                box = _box_of(d)
                if isinstance(d, dict) and "text" in d and box is not None:
                    good.append({"text": d["text"], "box_2d": box})
            if good:
                return good
    except json.JSONDecodeError:
        pass

    out = []
    for m in _WORD_BOX_RE.finditer(text):
        out.append(
            {
                "text": json.loads(f'"{m.group(1)}"'),  # unescape JSON string body
                "box_2d": [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))],
            }
        )
    return out


def load_api_key():
    """Return the Google API key from the environment, or raise if unset."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Export it before running, e.g.\n"
            "    export GOOGLE_API_KEY=...your key..."
        )
    return api_key


def build_model(model_name=None, json_output=True):
    """Configure the Gemini client and return a model.

    ``temperature=0`` is essential here: at the default temperature these models
    intermittently return degenerate OCR (e.g. mostly punctuation) on a dense
    page; deterministic decoding returns the full, clean word list. Pass
    ``json_output=False`` for plain-text tasks like transcription.
    """
    genai.configure(api_key=load_api_key())
    model_name = model_name or config.GEMINI_MODEL
    gen_config = {"temperature": 0.0}
    if json_output:
        gen_config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(model_name=model_name, generation_config=gen_config)


def transcribe_page(model, pil_image):
    """Return the full plain-text transcription of a page (reading order).

    ``model`` should be built with ``json_output=False``.
    """
    response = _generate(model, [TRANSCRIBE_PROMPT, pil_image])
    return response.text.strip()


def extract_words_and_boxes(model, pil_image):
    """Send an image to Gemini and return a list of {text, box_2d} dicts."""
    response = _generate(model, [OCR_PROMPT, pil_image])
    return parse_word_boxes(response.text)


def extract_words_and_boxes_grounded(model, pil_image, transcript):
    """Box a KNOWN numbered token list. Returns ``{1-based index: box}`` so the
    caller can map boxes to tokens by index (drift-proof)."""
    tokens = transcript.split()
    response = _generate(model, [grounded_ocr_prompt(tokens), pil_image])
    return parse_indexed_boxes(response.text, len(tokens))


def _stroke_trace_prompt(text):
    return f"""
    Task: Trace the handwriting for the text "{text}" in this image.

    Instructions:
    1. Identify the strokes that make up each character in "{text}".
    2. Return a list of points [x, y, p] for the strokes.
    3. x and y should be normalized coordinates (0 to 1) relative to the image provided.
    4. p is the pen state: 1 for touching paper (drawing), 0 for lifted (moving).
    5. Ensure the strokes follow the path of the handwriting precisely.
    6. Separate characters or strokes with a p=0 point if needed.

    JSON Schema:
    {{
        "text": "{text}",
        "strokes": [[x, y, p], [x, y, p]]
    }}
    """


def trace_strokes(model, pil_image, text):
    """Ask Gemini to trace pen strokes for ``text`` in an image crop.

    Returns ``{"text": ..., "strokes": [[x, y, p], ...]}`` with x/y normalized
    to the crop (0-1). This is an alternative to the OpenCV vectorizer in
    ``ocr.vectorize`` and depends entirely on the model's tracing quality.
    """
    response = _generate(model, [_stroke_trace_prompt(text), pil_image])
    return json.loads(response.text)


def extract_with_fallback(model, pil_image, fallback_model_name=None, transcript=None):
    """Run OCR, falling back to a second model on a 404 (model-not-found).

    If ``transcript`` is given, detection is grounded on its ordered tokens
    (strategy: one box per known word). Returns ``(model, word_data)`` — the model
    may have been swapped for the fallback so the caller can reuse it.
    """
    fallback_model_name = fallback_model_name or config.GEMINI_FALLBACK_MODEL

    def _detect(m):
        if transcript is not None:
            return extract_words_and_boxes_grounded(m, pil_image, transcript)
        return extract_words_and_boxes(m, pil_image)

    try:
        return model, _detect(model)
    except Exception as e:
        if "404" in str(e) and "models/" in str(e):
            print(f"Model not found; falling back to '{fallback_model_name}'...")
            model = build_model(fallback_model_name)
            return model, _detect(model)
        raise


def draw_boxes_on_image(pil_image, word_data):
    """Draw the OCR bounding boxes onto a copy-safe image (in place)."""
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    for item in word_data:
        box = item.get("box_2d")
        if not box:
            continue
        ymin, xmin, ymax, xmax = box
        left = (xmin / 1000) * width
        top = (ymin / 1000) * height
        right = (xmax / 1000) * width
        bottom = (ymax / 1000) * height
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
    return pil_image
