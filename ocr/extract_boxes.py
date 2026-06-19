"""CLI: OCR each PDF page into word-level bounding boxes.

For each page in the requested range it:
  1. asks Gemini for word texts + boxes,
  2. saves an annotated visual and a per-page JSON backup into the output dir,
  3. aggregates all words and renders the browser handwriting-capture tool.

Run from the repo root:
    export GOOGLE_API_KEY=...your key...
    python -m ocr.extract_boxes --pdf data/content/test_document.pdf --start-page 1 --end-page 3
"""

import argparse
import json
import os
import traceback

from pdf2image import convert_from_path

from . import config, paths, reconcile
from .gemini_ocr import build_model, draw_boxes_on_image, extract_with_fallback, transcribe_page
from .tool import render_tool


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OCR PDF pages into word bounding boxes")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Path to the input PDF")
    p.add_argument(
        "--output-root", default=paths.OUTPUT_ROOT, help="Root output folder (default: outputs/)"
    )
    p.add_argument(
        "--start-page",
        type=int,
        default=config.START_PAGE_INDEX + 1,
        help="First page to process (1-based, inclusive)",
    )
    p.add_argument(
        "--end-page",
        type=int,
        default=config.END_PAGE_INDEX,
        help="Last page to process (1-based, inclusive)",
    )
    p.add_argument("--model", default=config.GEMINI_MODEL, help="Gemini model name")
    p.add_argument(
        "--grounded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ground detection on the transcript tokens (one box per word)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Keep only the first N detected words per page"
    )
    p.add_argument(
        "--version",
        type=int,
        default=None,
        help="Force a page version number (default: next free version)",
    )
    p.add_argument("--no-tool", action="store_true", help="Skip rendering the HTML capture tool")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_root

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    print("Reading PDF...")
    images = convert_from_path(args.pdf)
    total_pages = len(images)
    print(f"PDF has {total_pages} pages.")

    # Convert 1-based inclusive CLI args to 0-based [start, end) slice bounds.
    start_index = max(0, args.start_page - 1)
    end_index = min(total_pages, args.end_page)
    if start_index >= end_index:
        raise SystemExit(
            f"Empty page range: start={args.start_page} end={args.end_page} "
            f"(PDF has {total_pages} pages)"
        )

    model = build_model(args.model)  # JSON detection
    text_model = build_model(args.model, json_output=False)  # plain-text transcription
    print(f"Processing pages {start_index + 1}..{end_index} with {model.model_name}")
    print(f"Output -> {paths.run_dir(args.pdf, root)}/")

    master_word_list = []
    for i in range(start_index, end_index):
        page_num = i + 1
        print(f"\n--- Page {page_num} ---")
        page_image = images[i]
        version = (
            args.version
            if args.version is not None
            else paths.next_version(args.pdf, page_num, root)
        )
        try:
            # 1. Full-page transcription (for the QA cross-check) + counts
            transcript = transcribe_page(text_model, page_image)
            tpath = paths.ensure_parent(paths.transcript_txt(args.pdf, page_num, version, root))
            with open(tpath, "w") as f:
                f.write(transcript + "\n")
            tokens = transcript.split()
            symbols = sum(1 for c in transcript if not c.isalnum() and not c.isspace())
            print(f"Transcript: {len(tokens)} words, {symbols} symbols -> {tpath}")

            # 2. Word+punctuation bounding boxes. When grounded, box the known
            # transcript tokens, then reconcile the result back to the transcript
            # (force text = token, infill any the model missed) so every word gets
            # exactly one correctly-labelled box.
            if args.grounded:
                grounded = None
                try:
                    model, grounded = extract_with_fallback(
                        model, page_image, transcript=transcript
                    )
                except Exception as e:
                    print(f"Grounded detection failed ({str(e)[:60]}); using free detection.")
                if grounded is not None and len(grounded) >= 0.5 * len(tokens):
                    word_data, n_infilled = reconcile.reconcile_indexed(grounded, tokens)
                    print(
                        f"Grounded located {len(grounded)}/{len(tokens)} tokens by index "
                        f"-> {len(word_data)} boxes ({n_infilled} infilled)."
                    )
                else:
                    if grounded is not None:
                        print(
                            f"Grounded too sparse ({len(grounded)}/{len(tokens)}); free detection."
                        )
                    model, word_data = extract_with_fallback(model, page_image)
            else:
                model, word_data = extract_with_fallback(model, page_image)
            print(f"Page {page_num}: found {len(word_data)} boxes.")
            if args.limit is not None:
                word_data = word_data[: args.limit]
                print(f"Keeping first {len(word_data)} (--limit {args.limit}).")

            master_word_list.extend(item.get("text", "") for item in word_data)

            json_path = paths.ensure_parent(paths.boxes_json(args.pdf, page_num, version, root))
            with open(json_path, "w") as f:
                json.dump(word_data, f, indent=4)
            print(f"Saved boxes (v{version}): {json_path}")

            visual_path = paths.boxes_overlay(args.pdf, page_num, version, root)
            draw_boxes_on_image(page_image.copy(), word_data).save(visual_path)
            print(f"Saved overlay: {visual_path}")
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            traceback.print_exc()

    print(f"\nDone. Total words collected: {len(master_word_list)}")

    if not args.no_tool and master_word_list:
        out = render_tool(master_word_list, output_path=paths.tool_html(args.pdf, root))
        print(f"Handwriting-capture tool written to: {out}")
    elif not master_word_list:
        print("No words extracted; skipping tool generation.")


if __name__ == "__main__":
    main()
