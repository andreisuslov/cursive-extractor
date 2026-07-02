"""Pre-GPU crop screen: Gemini-read each word CROP, keep/relabel/drop by label.

WHY: detection boxes on dense cursive fail three ways -- offset onto the WRONG
word (label/ink mismatch, the dominant poison), spanning neighbour rows, and
truncating the word. Downstream derendering costs ~5 s/word on a paid GPU, so
junk must be filtered BEFORE the GPU, not after (``ocr.verify_labels`` reads
back *strokes*, i.e. post-GPU). This screen reads the raw crop image instead:
a word is KEPT if the read matches its label (case-insensitive exact or
difflib ratio >= ``KEEP_RATIO``, raw or punctuation-stripped -- a stray comma
in the read must not demote a correct pair), RELABELED if the read instead matches some
token of the page transcript (exact or ratio >= ``RELABEL_RATIO`` after
stripping punctuation -- the box landed on a different *real* word, so the
pair is rescued by swapping the label), else DROPPED.

Crops use the exact ``inksight_vectorize --no-clean`` recipe (ink-fit +
box-mask) so the screen sees what the derenderer will see; ``--refined``
boxes (``ocr.refine_boxes``) are already ink-tight, so ``fit_ink`` is off for
them. Crops are packed ~20 to a numbered grid image to keep API calls low.

    envchain gemini python3 -m ocr.screen_crops --pdf <pdf> --page N
        [--version V] [--refined] [--limit K] [--dry-run] [--grid-dir DIR]
        [--output PATH]

Writes ``<page_dir>/<prefix>_boxes_screened.json``: the same box-entry schema,
only kept+relabeled entries, ``text`` updated on relabel, plus a ``"screen"``
dict recording the decision and the Gemini read.
"""

import argparse
import difflib
import json
import os
import string

from PIL import Image, ImageDraw, ImageFont

from . import config, paths
from .gemini_ocr import _generate, build_model
from .pdf_utils import crop_to_box, load_page
from .verify_labels import CELLS_PER_IMAGE, _font, labels_match, parse_readback

# Grid geometry: cells sized to a crop scaled ~150px tall plus a number strip.
GRID_COLS = 4
CELL_W = 420
CELL_H = 200
INK_H = 150
CELL_PAD = 12

KEEP_RATIO = 0.7  # read vs label: keep
RELABEL_RATIO = 0.8  # read vs transcript token: relabel (stricter -- it rewrites the label)

READBACK_PROMPT = """
Each numbered cell in this image contains ONE handwritten word cropped from a
scanned page. The red number at the top-left of each cell is the CELL NUMBER --
it is NOT part of the word. Read the handwriting in each cell and return a raw
JSON object mapping cell number to your transcription, e.g.
{"3": "wealth.", "4": "Uncle"}. Include EVERY cell number shown; if a cell is
unreadable, map it to "". Output ONLY the JSON object.
"""


# --- decision logic (pure) ----------------------------------------------------


def _strip_punct(s: str) -> str:
    return s.strip(string.punctuation + string.whitespace)


def best_transcript_match(read: str, tokens: list[str], ratio: float = RELABEL_RATIO) -> str | None:
    """The transcript token whose punctuation-stripped form best matches ``read``
    (exact case-insensitive, or difflib ratio >= ``ratio``), or None.

    Returns the token VERBATIM (punctuation kept) -- that is the label the
    downstream dataset should carry.
    """
    r = _strip_punct(read).lower()
    if not r:
        return None
    raw = read.strip().lower()
    exact = None
    best, best_score = None, ratio
    for tok in tokens:
        t = _strip_punct(tok).lower()
        if not t:
            continue
        if t == r:
            # Prefer the instance whose punctuation matches the read verbatim
            # ('Mr.' read must relabel to token 'Mr.', not an earlier bare 'Mr' --
            # the label carries punctuation into the training data).
            if tok.strip().lower() == raw:
                return tok
            if exact is None:
                exact = tok
            continue
        score = difflib.SequenceMatcher(None, t, r).ratio()
        if score >= best_score:
            best, best_score = tok, score
    return exact if exact is not None else best


def decide(label: str, read: str, transcript_tokens: list[str]) -> tuple[str, str | None]:
    """Screen one word: ``("keep", None)``, ``("relabel", new_label)`` or
    ``("drop", None)``.

    Keep is judged read-vs-label (the pair is already right); relabel is judged
    read-vs-transcript with a stricter ratio and punctuation stripped, because
    it *rewrites* the label -- the read must clearly be some other real word of
    the page, not noise.
    """
    if not read.strip():
        return "drop", None
    ls, rs = _strip_punct(label).lower(), _strip_punct(read).lower()
    # The keep compare also tries punctuation-stripped forms: a stray comma in the
    # read (', to' for 'to') must not fail keep and then bounce off the transcript
    # guard (which strips punctuation) into a bogus relabel of the same word.
    if labels_match(label, read, KEEP_RATIO) or (ls and rs and labels_match(ls, rs, KEEP_RATIO)):
        return "keep", None
    tok = best_transcript_match(read, transcript_tokens, RELABEL_RATIO)
    if tok is None:
        return "drop", None
    if ls and _strip_punct(tok).lower() == ls:
        return "keep", None  # the matched token IS the label modulo punctuation/case: no-op
    return "relabel", tok


# --- crop grids ---------------------------------------------------------------


def crop_for_entry(page: Image.Image, box_2d: list[int], fit_ink: bool) -> Image.Image:
    """The screen's view of one word: the inksight_vectorize --no-clean recipe."""
    crop, _ = crop_to_box(
        page,
        box_2d,
        padding=config.CROP_PADDING,
        pad_frac=config.CROP_PAD_FRAC,
        fit_ink=fit_ink,
        mask_to_box=True,
        mask_hpad=config.CROP_MASK_HPAD,
        mask_vpad_up=config.CROP_MASK_VPAD_UP,
        mask_vpad_dn=config.CROP_MASK_VPAD_DN,
    )
    return crop


def render_grid(cells: list[tuple[int, Image.Image]]) -> Image.Image:
    """Render ``[(cell_number, crop), ...]`` as one numbered grid image."""
    rows = (len(cells) + GRID_COLS - 1) // GRID_COLS
    img = Image.new("RGB", (GRID_COLS * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(img)
    font = _font(26)
    for k, (number, crop) in enumerate(cells):
        cx = (k % GRID_COLS) * CELL_W
        cy = (k // GRID_COLS) * CELL_H
        draw.rectangle([cx, cy, cx + CELL_W - 1, cy + CELL_H - 1], outline=(200, 200, 200))
        draw.text((cx + 8, cy + 4), str(number), fill=(200, 30, 30), font=font)
        avail_w, avail_h = CELL_W - 2 * CELL_PAD, INK_H
        scale = min(avail_h / max(1, crop.height), avail_w / max(1, crop.width), 1.5)
        scaled = crop.convert("RGB").resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
        )
        ox = cx + CELL_PAD + (avail_w - scaled.width) // 2
        oy = cy + 36 + (avail_h - scaled.height) // 2
        img.paste(scaled, (ox, oy))
    return img


# --- CLI ----------------------------------------------------------------------

SCREENED_SUFFIX = "boxes_screened.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemini-read word crops; keep/relabel/drop pre-GPU")
    p.add_argument("--pdf", default=config.PDF_PATH, help="Source PDF")
    p.add_argument("--page", type=int, default=1, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--output-root", default=paths.OUTPUT_ROOT, help="Root output folder")
    p.add_argument(
        "--refined",
        action="store_true",
        help="Screen <page_dir>/boxes_refined.json (already ink-tight, so fit_ink off)",
    )
    p.add_argument("--dpi", type=int, default=300, help="Page render DPI (crops shrink to grid)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N labeled boxes")
    p.add_argument("--output", default=None, help="Screened boxes json (default: canonical)")
    p.add_argument(
        "--dry-run", action="store_true", help="Render the grid images only; no API call"
    )
    p.add_argument(
        "--grid-dir", default=None, help="Where --dry-run saves grids (default: the page folder)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.output_root
    version = (
        args.version
        if args.version is not None
        else paths.latest_version(args.pdf, args.page, root)
    )
    if version is None:
        raise SystemExit("No processed version found; run ocr.extract_boxes first.")
    page_dir = paths.page_dir(args.pdf, args.page, version, root)
    if args.refined:
        boxes_path = os.path.join(page_dir, "boxes_refined.json")
    else:
        boxes_path = paths.boxes_json(args.pdf, args.page, version, root)
    with open(boxes_path) as f:
        boxes = json.load(f)
    with open(paths.transcript_txt(args.pdf, args.page, version, root)) as f:
        transcript_tokens = f.read().split()

    labeled = [e for e in boxes if e.get("box_2d") and e.get("text")]
    if args.limit is not None:
        labeled = labeled[: args.limit]
    if not labeled:
        raise SystemExit(f"No labeled boxes in {boxes_path}")
    print(f"Screening {len(labeled)}/{len(boxes)} labeled boxes from {boxes_path}")

    page = load_page(args.pdf, args.page - 1, dpi=args.dpi)
    # Refined boxes are already fitted to the ink; re-fitting would re-grow them.
    fit_ink = not args.refined
    crops = [crop_for_entry(page, e["box_2d"], fit_ink) for e in labeled]

    # Global 1-based cell numbers so every cell across all grids is unique.
    numbered = list(enumerate(crops, start=1))
    grids = [
        render_grid(numbered[g : g + CELLS_PER_IMAGE])
        for g in range(0, len(numbered), CELLS_PER_IMAGE)
    ]

    if args.dry_run:
        grid_dir = args.grid_dir or page_dir
        paths.ensure_dir(grid_dir)
        stem = paths.prefix(args.pdf, args.page, version)
        for g, img in enumerate(grids):
            out = os.path.join(grid_dir, f"{stem}_screen_grid_{g:02d}.png")
            img.save(out)
            print(f"[dry-run] wrote {out}")
        return

    model = build_model()
    reads: dict[int, str] = {}
    for g, img in enumerate(grids):
        response = _generate(model, [READBACK_PROMPT, img])
        got = parse_readback(response.text)
        reads.update(got)
        print(f"grid {g + 1}/{len(grids)}: {len(got)} cells read")

    kept, relabeled, dropped = 0, 0, 0
    screened: list[dict] = []
    for number, entry in enumerate(labeled, start=1):
        read = reads.get(number, "")
        decision, new_text = decide(entry.get("text", ""), read, transcript_tokens)
        if decision == "drop":
            dropped += 1
            print(f"  DROP cell {number}: label {entry.get('text', '')!r} read {read!r}")
            continue
        out_entry = dict(entry)
        out_entry["screen"] = {"decision": decision, "read": read}
        if decision == "relabel":
            relabeled += 1
            out_entry["screen"]["orig_text"] = entry.get("text", "")
            out_entry["text"] = new_text
            print(f"  RELABEL cell {number}: {entry.get('text', '')!r} -> {new_text!r} "
                  f"(read {read!r})")
        else:
            kept += 1
        screened.append(out_entry)

    out_path = args.output or os.path.join(
        page_dir, f"{paths.prefix(args.pdf, args.page, version)}_{SCREENED_SUFFIX}"
    )
    paths.ensure_parent(out_path)
    with open(out_path, "w") as f:
        json.dump(screened, f, indent=4)
    n = len(labeled)
    print(f"Wrote {out_path}")
    print(
        f"Screened {n}: kept {kept}, relabeled {relabeled}, dropped {dropped}, "
        f"yield {(kept + relabeled) / n:.3f}"
    )


if __name__ == "__main__":
    main()
