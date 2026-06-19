"""Render the browser-based handwriting-capture tool from its HTML template.

The template (``handwriting_tool.html``) is a self-contained React page with a
single placeholder where the OCR'd word bank is injected as a JS array. Open the
rendered file in a browser to trace handwriting and export training JSON.
"""

import json

from . import config
from .paths import ensure_parent


def render_tool(words, template_path=None, output_path="handwriting_tool.html"):
    """Inject ``words`` into the HTML template and write the capture tool.

    Args:
        words: list of strings used as the prompt word bank.
        template_path: HTML template; defaults to the packaged one.
        output_path: where to write the rendered tool (callers usually pass a
            canonical path from ``ocr.paths.tool_html``).

    Returns the path written.
    """
    template_path = template_path or config.HTML_TEMPLATE_PATH

    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    word_bank_json = json.dumps(words)
    html = template.replace(config.WORD_BANK_PLACEHOLDER, word_bank_json)

    with open(ensure_parent(output_path), "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
