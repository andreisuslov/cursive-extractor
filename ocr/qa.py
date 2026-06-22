"""CLI: QA check — the page transcript must equal the concatenation of all
per-box ``text_recognized.txt`` files (token for token, in box order).

This catches missing boxes, extra boxes, and word/punctuation mismatches between
the independent transcription pass and the boxed words.

    python -m ocr.qa --pdf datasets/content/test_document.pdf --page 4
"""

import argparse
import collections
import difflib
import os
import re

from . import config, paths


def collect_box_texts(
    pdf_path: str, page: int, version: int | None = None, root: str | None = None
) -> list[str]:
    """Read every box folder's text_recognized.txt, ordered by box index."""
    pdir = paths.page_dir(pdf_path, page, version, root)
    pat = re.compile(re.escape(paths.prefix(pdf_path, page, version)) + r"_box_(\d+)$")
    items = []
    if os.path.isdir(pdir):
        for name in os.listdir(pdir):
            m = pat.fullmatch(name)
            if not (m and os.path.isdir(os.path.join(pdir, name))):
                continue
            tpath = os.path.join(pdir, name, paths.BOX_TEXT_FILE)
            text = ""
            if os.path.exists(tpath):
                with open(tpath) as f:
                    text = f.read().strip()
            items.append((int(m.group(1)), text))
    items.sort()
    return [t for _, t in items]


def run_qa(
    pdf_path: str,
    page: int,
    version: int | None = None,
    root: str | None = None,
    write: bool = True,
) -> tuple[bool, str]:
    """Compare transcript vs concatenated box texts. Returns ``(passed, report)``."""
    tpath = paths.transcript_txt(pdf_path, page, version, root)
    transcript = ""
    if os.path.exists(tpath):
        with open(tpath) as f:
            transcript = f.read()

    t_tokens = transcript.split()
    b_tokens = collect_box_texts(pdf_path, page, version, root)
    passed = bool(t_tokens) and " ".join(t_tokens) == " ".join(b_tokens)

    # Completeness (order-independent): are all words boxed, none missing/extra?
    # More meaningful than exact order when a page has multiple columns/entries
    # that the transcription and detection passes traverse differently.
    tc, bc = collections.Counter(t_tokens), collections.Counter(b_tokens)
    missing, extra = tc - bc, bc - tc  # transcript-only / boxes-only
    complete = bool(t_tokens) and not missing and not extra
    matched = len(t_tokens) - sum(missing.values())
    pct = (100.0 * matched / len(t_tokens)) if t_tokens else 0.0

    lines = [
        f"QA {paths.prefix(pdf_path, page, version)}",
        f"transcript words:  {len(t_tokens)}",
        f"box folders:       {len(b_tokens)}",
        f"exact-order match: {'PASS' if passed else 'FAIL'}",
        f"completeness:      {'PASS' if complete else 'FAIL'}  "
        f"({matched}/{len(t_tokens)} = {pct:.1f}%, "
        f"{sum(missing.values())} missing, {sum(extra.values())} extra)",
    ]
    if not transcript:
        lines.append("(no transcript found — run ocr.extract_boxes first)")
    if missing:
        lines.append(
            "missing (in transcript, not boxed): "
            + ", ".join(f"{w}x{n}" if n > 1 else w for w, n in list(missing.items())[:40])
        )
    if extra:
        lines.append(
            "extra (boxed, not in transcript):   "
            + ", ".join(f"{w}x{n}" if n > 1 else w for w, n in list(extra.items())[:40])
        )
    if not passed and t_tokens:
        lines.append("")
        lines.append("ordered diff (- transcript / + boxes):")
        diff = difflib.unified_diff(
            t_tokens, b_tokens, fromfile="transcript", tofile="boxes", lineterm=""
        )
        lines.extend(list(diff)[:60])
    report = "\n".join(lines)

    if write:
        rpath = paths.ensure_parent(paths.qa_report(pdf_path, page, version, root))
        with open(rpath, "w") as f:
            f.write(report + "\n")
    return passed, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA: transcript vs concatenated box texts")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=2, help="Page number, 1-based")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--version", type=int, default=None, help="Page version (default: latest existing)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, args.output_root)
    )
    if version is None:
        raise SystemExit("No processed version found; run ocr.extract_boxes first.")
    passed, report = run_qa(args.pdf, args.page, version, args.output_root)
    print(report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
