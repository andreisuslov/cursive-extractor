"""EXPERIMENTAL holistic HTR forced-alignment cut for traced cursive words.

The geometry cutter (``ocr._segment_prototype``) and the recognition-guided cut
(``ocr._recognizer``) both place per-letter boundaries from LOCAL ink shape /
per-slice template scores. They top out around ~20% honest per-letter top-1 under
the independent font judge on the page-4 diary words. The remaining lever the owner
wanted tried: a HOLISTIC sequence model that reads the WHOLE word at once and learns
where letters are, then forced-align it to the KNOWN transcription for boundaries.

This module is that model, in the standard CRNN/CTC form:

    word image (de-sheared binary, H x W)  ->  CNN  ->  BiLSTM  ->  CTC over chars

Because every word's transcription is KNOWN, the task is FORCED ALIGNMENT, not free
recognition: we train the CTC model on the real diary word crops + their transcripts
(all extracted pages, with on-the-fly augmentation), then run CTC Viterbi alignment
of each word's per-frame log-probs against its KNOWN char sequence. Each char's frame
span -> an x boundary between consecutive letters (``boundaries_from_states``). The
boundaries live in the same de-sheared x-space the geometry cut uses, so they drop
straight into ``ocr._recognizer.recognition_accuracy``.

MEASUREMENT (honest, non-circular):
  - WHOLE words are held out (``split_records``); the model never trains on a test
    word. Slices from one word share a hand/scan/cut bias, so a per-letter split would
    leak -- only a whole-word split is honest (same rule as ``_bootstrap_recognizer``).
  - The held-out cut is judged by the FONT-TEMPLATE recognizer (``_recognizer``), which
    did NOT guide the cut. Judging the HTR cut with the HTR model would be CIRCULAR
    (alignment maximises the HTR model's own per-frame score), so we do not; the font
    judge is the independent test. Geometry is scored by the same judge as the baseline.

HONEST EXPECTATIONS: ~490 words (~2.5k letters) is SMALL for HTR -- these models are
normally trained on thousands to tens of thousands of lines. So this is plausibly
data-starved and may NOT beat geometry. Whatever the CLI prints (``cut_quality``) is
the result; a negative one is reported as such, with the numbers, not papered over.

CLI::

    python -m ocr._htr_align --pdf test_document --pages 1,2,3,4 [--epochs N] [--save PATH]

Weights (when ``--save``) go to ``runs/htr/`` (git-ignored). torch only; CPU-friendly.
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ocr import paths

from . import _recognizer as R
from . import _segment_prototype as sp

HTR_H = 32  # fixed input height (px) fed to the CNN
WIDTH_DS = 4  # the CNN downsamples width by exactly 4 (two stride-2 width pools)
DEFAULT_WEIGHTS = "runs/htr/htr_ctc.pt"


# --- 1. data: real diary word images + transcripts --------------------------


def word_region(prep: dict) -> np.ndarray | None:
    """The de-sheared word ink as a float mask (ink=1.0), cropped to the body x-span
    ``[x_min, x_max]`` and to its ink rows. This is the CTC model's raw input (before
    height-normalising). Returns ``None`` if degenerate."""
    sbin = prep["sbin"]
    x0, x1 = round(float(prep["x_min"])), round(float(prep["x_max"]))
    region = (sbin[:, x0 : x1 + 1] > 0).astype(np.float32)
    ys = np.nonzero(region.any(axis=1))[0]
    if ys.size == 0 or region.shape[1] < 2:
        return None
    return region[ys.min() : ys.max() + 1]


def region_to_canvas(
    region: np.ndarray,
    height: int = HTR_H,
    min_frames: int = 2,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Height-normalise a word ``region`` to a ``height x W`` float canvas (ink in
    [0, 1]). ``W`` preserves the region's aspect but is floored so the CNN yields at
    least ``min_frames`` time-steps (``W // WIDTH_DS >= min_frames``) -- CTC needs at
    least one frame per char. With ``augment`` a random anisotropic scale + shear +
    speckle is applied first (multiplies the tiny real set; shear re-introduces the
    slant variability de-shearing removed)."""
    m = (region > 0).astype(np.uint8) * 255
    if augment and rng is not None:
        h, w = m.shape
        sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1)
        shear = rng.uniform(-0.3, 0.3)
        out_w = max(2, int(w * sx + abs(shear) * h)) + 2
        out_h = max(2, int(h * sy)) + 2
        affine = np.array([[sx, shear, 1.0], [0.0, sy, 1.0]], np.float32)
        m = cv2.warpAffine(m, affine, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
        if rng.random() < 0.5:  # flip ~2% of pixels (scan/threshold speckle)
            flip = rng.random(m.shape) < 0.02
            m[flip] = 255 - m[flip]
    h, w = m.shape
    target_w = max(round(w * height / h), WIDTH_DS * min_frames + WIDTH_DS)
    canvas = cv2.resize(
        (m > 0).astype(np.float32), (target_w, height), interpolation=cv2.INTER_AREA
    )
    return np.clip(canvas, 0.0, 1.0)


def load_records(
    pdf: str, pages: list[int], dpi: int, versions: dict[int, int | None] | None = None
) -> list[dict]:
    """Build one record per usable word over ``pages``: ``{word, region, x_min, x_max,
    prep, page, index}``. Reuses ``_recognizer.prepare`` (clean -> band -> de-shear) so
    the CTC input and the geometry baseline see the SAME ink. Words that fail to prepare
    or are shorter than 2 chars are skipped."""
    pdf_path = sp._resolve_pdf(pdf)
    versions = versions or {}
    records: list[dict] = []
    for page in pages:
        version = versions.get(page, paths.latest_version(pdf_path, page))
        page_image = sp.render_page(pdf_path, page, dpi=dpi)
        with open(paths.boxes_json(pdf_path, page, version)) as f:
            boxes = json.load(f)
        pitch_px = sp._row_pitch(boxes, page_image.size[1])
        for index, b in enumerate(boxes):
            if not isinstance(b.get("box_2d"), list):
                continue
            word = sp._read_word(
                paths.box_text(pdf_path, page, index, version), b.get("text", "")
            ).strip()
            if len(word) < 2:
                continue
            try:
                prep = R.prepare(page_image, b["box_2d"], word, pitch_px)
            except Exception:  # loader: skip a word that blows up, keep going
                prep = None
            if prep is None:
                continue
            region = word_region(prep)
            if region is None:
                continue
            records.append(
                {
                    "word": word,
                    "region": region,
                    "x_min": prep["x_min"],
                    "x_max": prep["x_max"],
                    "prep": prep,
                    "page": page,
                    "index": index,
                }
            )
    return records


def build_vocab(records: list[dict]) -> list[str]:
    """Sorted list of every char appearing in any word (train AND test, so forced
    alignment of a held-out word never hits an out-of-vocab char). Blank is implicit:
    its class id is ``len(vocab)``."""
    return sorted({c for r in records for c in r["word"]})


def split_records(
    records: list[dict], test_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Hold out WHOLE words for test (no per-letter leakage). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_test = max(1, round(test_frac * len(records)))
    test_ix = set(idx[:n_test].tolist())
    train = [r for i, r in enumerate(records) if i not in test_ix]
    test = [r for i, r in enumerate(records) if i in test_ix]
    return train, test


# --- 2. the CRNN (CNN -> BiLSTM -> CTC) -------------------------------------


class CRNN(nn.Module):
    """Standard CRNN: a 3-conv stack collapses the height and downsamples width by
    ``WIDTH_DS`` (=4), a 2-layer BiLSTM reads the column sequence, a linear head emits
    per-frame class logits (chars + blank). Deliberately small -- a few hundred real
    words is far too little to feed a big net."""

    def __init__(self, n_classes: int, height: int = HTR_H):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2, W/2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/4, W/4
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8, W/4  (width fixed from here)
        )
        self.feat_h = height // 8
        self.lstm = nn.LSTM(
            64 * self.feat_h, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(N, 1, H, W)``; returns per-frame logits ``(N, T=W/WIDTH_DS,
        n_classes)``."""
        f = self.cnn(x)
        n, c, hh, wd = f.shape
        f = f.permute(0, 3, 1, 2).reshape(n, wd, c * hh)  # (N, T, feat)
        out, _ = self.lstm(f)
        return self.fc(out)


# --- 3. training (CTC = forced alignment, since transcripts are known) -------


def _collate(batch_records: list[dict], vocab_ix: dict[str, int], height: int, rng):
    """Augmented canvases for a batch -> padded ``(N,1,H,Wmax)`` tensor + CTC targets.
    ``input_lengths`` are the REAL (pre-pad) frame counts so CTC ignores the padding."""
    canvases = [
        region_to_canvas(r["region"], height, len(r["word"]) + 1, augment=True, rng=rng)
        for r in batch_records
    ]
    w_max = max(c.shape[1] for c in canvases)
    x = np.zeros((len(canvases), 1, height, w_max), np.float32)
    in_lens, targets, tgt_lens = [], [], []
    for i, (c, r) in enumerate(zip(canvases, batch_records, strict=True)):
        x[i, 0, :, : c.shape[1]] = c
        in_lens.append(c.shape[1] // WIDTH_DS)
        ids = [vocab_ix[ch] for ch in r["word"]]
        targets.extend(ids)
        tgt_lens.append(len(ids))
    return (
        torch.from_numpy(x),
        torch.tensor(in_lens),
        torch.tensor(targets),
        torch.tensor(tgt_lens),
    )


def train_htr(
    train_records: list[dict],
    vocab: list[str],
    epochs: int,
    seed: int,
    height: int = HTR_H,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 10,
) -> CRNN:
    """Train the CRNN with CTC loss. Batches are width-bucketed (records sorted by
    region width, contiguous batches) to keep zero-padding small -- a BiLSTM otherwise
    bleeds padding frames into real ones. Augmentation is re-rolled every epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab_ix = {c: i for i, c in enumerate(vocab)}
    blank = len(vocab)
    model = CRNN(len(vocab) + 1, height).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    by_width = sorted(train_records, key=lambda r: r["region"].shape[1])
    batches = [by_width[i : i + batch_size] for i in range(0, len(by_width), batch_size)]
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(batches))
        total = 0.0
        for bi in order:
            x, in_lens, targets, tgt_lens = _collate(batches[bi], vocab_ix, height, rng)
            x = x.to(device)
            logits = model(x)  # (N, T, C)
            logp = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, N, C)
            in_lens = torch.clamp(in_lens, max=logp.shape[0])
            loss = ctc(logp, targets, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3}/{epochs}  ctc_loss {total / len(batches):.3f}")
    return model


# --- 4. CTC forced alignment -> letter boundaries ---------------------------


def ctc_forced_align(logp: np.ndarray, target_ids: list[int], blank: int) -> list[int] | None:
    """Viterbi forced alignment of frame log-probs ``logp`` ``(T, C)`` to the KNOWN
    label ``target_ids`` (length L). Returns the per-frame extended-state index over the
    CTC label ``[blank, c0, blank, c1, ..., c_{L-1}, blank]`` (length ``2L+1``), or
    ``None`` if there are too few frames (``T < L``). The path is monotone and must
    traverse every char state, so every char gets >= 1 frame."""
    t_len, length = logp.shape[0], len(target_ids)
    if length == 0 or t_len < length:
        return None
    ext = [blank]
    for c in target_ids:
        ext += [c, blank]
    s_len = len(ext)
    ext_arr = np.asarray(ext)
    neg = -1e30
    dp = np.full((t_len, s_len), neg)
    bp = np.full((t_len, s_len), -1, dtype=int)
    dp[0, 0] = logp[0, ext[0]]
    if s_len > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, t_len):
        prev = dp[t - 1]
        # candidate predecessors: stay (s), step (s-1), skip (s-2 if it bridges two
        # DIFFERENT chars across a blank). Vectorised over states.
        stay = prev
        step = np.concatenate([[neg], prev[:-1]])
        skip = np.concatenate([[neg, neg], prev[:-2]])
        can_skip = (ext_arr != blank) & np.concatenate(
            [[False, False], ext_arr[:-2] != ext_arr[2:]]
        )
        skip = np.where(can_skip, skip, neg)
        choices = np.stack([stay, step, skip])  # (3, S)
        arg = choices.argmax(axis=0)
        best = choices.max(axis=0)
        dp[t] = best + logp[t, ext_arr]
        bp[t] = np.where(
            arg == 0,
            np.arange(s_len),
            np.where(arg == 1, np.arange(s_len) - 1, np.arange(s_len) - 2),
        )
    last = s_len - 1 if dp[t_len - 1, s_len - 1] >= dp[t_len - 1, s_len - 2] else s_len - 2
    states = [0] * t_len
    s = last
    for t in range(t_len - 1, -1, -1):
        states[t] = int(s)
        s = bp[t, s]
    return states


def boundaries_from_states(
    states: list[int], t_len: int, length: int, x_min: float, x_max: float
) -> list[float]:
    """Map a forced-alignment state path to ``L-1`` internal x boundaries. Char ``j``
    emits on extended state ``2j+1``; the cut between char ``j-1`` and ``j`` is the
    frame midpoint between char ``j-1``'s last emission frame and char ``j``'s first,
    rescaled from frame-space ``[0, T)`` to x-space ``[x_min, x_max]``. Falls back to an
    even split for any char that never emitted (shouldn't happen for a valid path)."""
    first = [None] * length
    last = [None] * length
    for t, s in enumerate(states):
        if s % 2 == 1:
            j = (s - 1) // 2
            if first[j] is None:
                first[j] = t
            last[j] = t
    if any(f is None for f in first):
        return list(np.linspace(x_min, x_max, length + 1)[1:-1])
    span = x_max - x_min
    out = []
    for j in range(1, length):
        bf = (last[j - 1] + first[j] + 1) / 2.0
        out.append(x_min + (bf / t_len) * span)
    return out


def align_word(
    model: CRNN,
    region: np.ndarray,
    word: str,
    x_min: float,
    x_max: float,
    vocab: list[str],
    height: int = HTR_H,
    device: str = "cpu",
) -> list[float]:
    """Forced-align one word image to its KNOWN transcription and return the ``L-1``
    internal cut x-positions (de-sheared space, same as the geometry cut)."""
    vocab_ix = {c: i for i, c in enumerate(vocab)}
    target_ids = [vocab_ix[c] for c in word if c in vocab_ix]
    if len(target_ids) != len(word):  # unknown char -> can't align; even split
        return list(np.linspace(x_min, x_max, len(word) + 1)[1:-1])
    canvas = region_to_canvas(region, height, len(word) + 1)
    x = torch.from_numpy(canvas)[None, None].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)[0]  # (T, C)
    logp = F.log_softmax(logits, dim=1).cpu().numpy()
    states = ctc_forced_align(logp, target_ids, blank=len(vocab))
    if states is None:
        return list(np.linspace(x_min, x_max, len(word) + 1)[1:-1])
    return boundaries_from_states(states, logp.shape[0], len(word), x_min, x_max)


# --- 5. evaluation (independent font judge, held-out words) -----------------


def cut_quality(
    test_records: list[dict],
    model: CRNN,
    vocab: list[str],
    font: R.LetterRecognizer,
    height: int = HTR_H,
    device: str = "cpu",
) -> dict:
    """Score the geometry cut vs the HTR forced-alignment cut on HELD-OUT words, both
    judged by the INDEPENDENT font-template recognizer (it did not guide either cut).
    Returns per-cut ``[top1_sum, letter_total]``. The HTR-judges-HTR cell is omitted on
    purpose: it is circular (alignment maximises the HTR model's own score), so the font
    judge is the only honest comparison here."""
    out = {"geo/font": [0, 0], "htr/font": [0, 0]}
    for rec in test_records:
        prep, word = rec["prep"], rec["word"]
        try:
            geo_bxs, _ = R.geometry_boundaries(prep, word)
            htr_bxs = align_word(
                model, rec["region"], word, rec["x_min"], rec["x_max"], vocab, height, device
            )
        except Exception:  # eval: skip a word that blows up, keep going
            continue
        for name, bxs in (("geo", geo_bxs), ("htr", htr_bxs)):
            acc = R.recognition_accuracy(prep, word, bxs, recognizer=font)
            out[f"{name}/font"][0] += acc["top1"]
            out[f"{name}/font"][1] += acc["n"]
    return out


# --- CLI --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Holistic HTR (CTC) forced-alignment cut eval")
    p.add_argument("--pdf", default="test_document", help="Source PDF (slug or path)")
    p.add_argument("--pages", default="1,2,3,4", help="Comma-separated 1-based pages")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for the source pages")
    p.add_argument("--epochs", type=int, default=60, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--height", type=int, default=HTR_H, help="Input image height (px)")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=0, help="Split + init seed")
    p.add_argument("--test-frac", type=float, default=0.25, help="Fraction of WORDS held out")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--save", default=None, help=f"Save weights here (e.g. {DEFAULT_WEIGHTS})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    torch.manual_seed(args.seed)

    print(f"=== LOADING real diary words (pages {pages}, dpi {args.dpi}) ===")
    records = load_records(args.pdf, pages, args.dpi)
    vocab = build_vocab(records)
    n_letters = sum(len(r["word"]) for r in records)
    widths = [r["region"].shape[1] for r in records]
    print(
        f"words: {len(records)}   letters: {n_letters}   vocab: {len(vocab)} chars   "
        f"region width px: min {min(widths)} / median {int(np.median(widths))} / max {max(widths)}"
    )
    print(f"vocab: {''.join(vocab)}")

    train_records, test_records = split_records(records, args.test_frac, args.seed)
    print(
        f"\nsplit (whole words, seed {args.seed}): "
        f"{len(train_records)} train / {len(test_records)} test words  "
        f"({sum(len(r['word']) for r in test_records)} test letters)"
    )

    print(f"\n=== TRAINING CRNN+CTC ({len(vocab) + 1} classes, {args.epochs} epochs, {device}) ===")
    model = train_htr(
        train_records,
        vocab,
        args.epochs,
        args.seed,
        height=args.height,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    if args.save:
        paths.ensure_parent(args.save)
        torch.save(
            {"state_dict": model.state_dict(), "vocab": vocab, "height": args.height}, args.save
        )
        print(f"saved weights -> {os.path.abspath(args.save)}")

    print("\n=== CUT QUALITY on HELD-OUT words (independent FONT judge) ===")
    font = R.get_recognizer()
    cq = cut_quality(test_records, model, vocab, font, args.height, device)

    def pct(key: str) -> str:
        ok, n = cq[key]
        return f"{ok}/{n}={ok / n:.1%}" if n else "n/a"

    gf, nf = cq["geo/font"]
    hf, nh = cq["htr/font"]
    print(f"{'cut':<16}{'FONT judge (independent)':>26}")
    print(f"{'geometry':<16}{pct('geo/font'):>26}")
    print(f"{'HTR forced-align':<16}{pct('htr/font'):>26}")
    if nf and nf == nh:
        delta = (hf - gf) / nf
        verdict = "HTR wins" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "geometry wins")
        print(
            f"\nHONEST read: on {nf} held-out letters, HTR forced-align "
            f"{hf}/{nf}={hf / nf:.1%} vs geometry {gf}/{nf}={gf / nf:.1%} "
            f"(delta {delta:+.1%}) -> {verdict}."
        )
    print(
        f"\nNOTE: {len(records)} words (~{n_letters} letters) is SMALL for HTR (these "
        "normally want thousands+); treat a non-win as a data-starvation result, not a "
        "dead end -- more transcribed pages are the lever."
    )


if __name__ == "__main__":
    main()
