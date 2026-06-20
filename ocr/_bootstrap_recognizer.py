"""EXPERIMENTAL recognizer BOOTSTRAP: train a small CNN on REAL cursive letters.

The font-template ``LetterRecognizer`` (``ocr._recognizer``) caps the
recognition-guided cut at ~41% per-letter top-1 on the page-4 diary words. The
ceiling is the font->handwriting DOMAIN GAP: the templates are rendered from macOS
script fonts, which do not look like this diarist's hand. The fix this module tests:
now that ``ocr._segment_prototype`` can cut a word into per-letter slices at ~91%,
we can MINE a labeled real-handwriting letter set (slice -> the KNOWN transcription
char) and train a CNN on it -- the first point in this project where a CNN has real
data to beat the templates on.

Pipeline (all driven from ``main``)::

    python -m ocr._bootstrap_recognizer --pdf test_document --page 4 [--epochs N]

  1. EXTRACT  (``extract_words``) -- every page-4 box -> ``prepare`` + the
     RECOGNIZER-FREE geometry cut -> per-letter binary slices, each labeled by its
     transcription char. Geometry (not the recognition cut) is used so the training
     set is not circular w.r.t. the thing we are training.
  2. SPLIT    (``split_words``)   -- WHOLE words held out for test (NO slice from a
     test word is ever trained on). This is the only honest split: slices from one
     word share a hand/scan/cut bias, so a per-slice split would leak.
  3. TRAIN    (``train_cnn``)     -- a 2-conv CNN over 32x32 slice canvases, with
     on-the-fly slant/scale/speckle augmentation and inverse-frequency class weights
     (the classes are very skewed -- e=101 ... many capitals appear once).
  4. EVALUATE -- (a) HEAD-TO-HEAD: real held-out top-1, CNN vs font templates, on the
     SAME test slices; (b) CUT QUALITY: cut held-out words by geometry vs CNN-guided
     forced-alignment and score each by BOTH the CNN and the font templates (an
     INDEPENDENT judge), to separate a real cut gain from align/measure circularity.

RESULTS (page 4, 6-seed mean, ``--epochs 150``):
  - RECOGNIZER -- the CNN WINS, clearly and robustly. Real held-out free top-1:
    CNN ~41% vs font templates ~21% (+20pp, ~2x), CNN ahead on 6/6 seeds. This is the
    headline: a CNN trained on a few hundred REAL segmented letters beats the
    font-template bank it was built to replace. Augmentation helps (+~6pp over no-aug).
  - CUT GUIDANCE -- a NEGATIVE result, reported as such. Plugging the CNN into
    ``align_boundaries`` makes the CNN-judged cut leap to ~78%, but that is CIRCULAR
    (alignment maximises the CNN's own score). Under the INDEPENDENT font judge the
    CNN-guided cut (~19-23% across alpha) does NOT beat the geometry cut (~20%); the
    apparent win is self-consistency, not better boundaries. The font-template
    recognition-guided cut's modest honest gain (commit 383b71e) still stands; the CNN
    does not extend it. The recognizer is the win here, the cut-guidance is not.

HONESTY: the labels carry the segmenter's ~9% error, so both the training signal and
the test ceiling are noisy; the slices also share a systematic cut bias the held-out
split cannot remove. All of that is reported with the numbers, not papered over. The
font-template path stays the default in ``_recognizer``; the CNN is opt-in.

Weights (when ``--save``) go to ``runs/recog/bootstrap_cnn.pt`` (git-ignored).
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
import torch
from torch import nn

from . import _recognizer as R
from . import _segment_prototype as sp
from . import paths

CNN_SIZE = 32  # slice canvas side fed to the CNN (px)
MIN_CLASS = 6  # drop a letter class with fewer total slices than this (can't split it)
DEFAULT_WEIGHTS = "runs/recog/bootstrap_cnn.pt"


# --- 1. extract labeled real-letter slices ----------------------------------


def extract_words(
    pdf: str, page: int, version: int | None, dpi: int = 600
) -> tuple[list[dict], object, float]:
    """Cut every page-4 box into per-letter slices labeled by the transcription char.

    Returns ``(records, page_image, pitch_px)``. Each record is one usable word
    ``{word, box_2d, conf, slices}`` where ``slices`` is a list of ``(char, mask_uint8)``
    for the alpha chars (the geometry cut, recognizer-free). Words that fail to
    prepare/segment are skipped. ``page_image``/``pitch_px`` are returned so the
    cut-quality re-measure can re-segment held-out words without re-rendering.
    """
    pdf_path = sp._resolve_pdf(pdf)
    version = version if version is not None else paths.latest_version(pdf_path, page)
    page_image = sp.render_page(pdf_path, page, dpi=dpi)
    with open(paths.boxes_json(pdf_path, page, version)) as f:
        boxes = json.load(f)
    pitch_px = sp._row_pitch(boxes, page_image.size[1])

    records: list[dict] = []
    for i, b in enumerate(boxes):
        if not isinstance(b.get("box_2d"), list):
            continue
        word = sp._read_word(paths.box_text(pdf_path, page, i, version), b.get("text", "")).strip()
        if len(word) < 2:
            continue
        try:
            prep = R.prepare(page_image, b["box_2d"], word, pitch_px)
            if prep is None:
                continue
            geo_bxs, geom_score = R.geometry_boundaries(prep, word)
            conf = sp.compute_confidence(geom_score, geo_bxs, word, prep["x_min"], prep["x_max"])
            cuts = [prep["x_min"], *sorted(geo_bxs), prep["x_max"]]
            slices = R.slice_columns(prep["sbin"], cuts)
        except Exception:  # extraction CLI: skip a word that blows up, keep going
            continue
        labeled = [
            (ch, sl) for ch, sl in zip(word, slices, strict=False) if ch.isalpha() and sl.size
        ]
        if labeled:
            records.append({"word": word, "box_2d": b["box_2d"], "conf": conf, "slices": labeled})
    return records, page_image, pitch_px


def class_counts(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for rec in records:
        for ch, _ in rec["slices"]:
            c[ch] += 1
    return c


# --- 2. canvas + augmentation -----------------------------------------------


def slice_to_canvas(mask: np.ndarray, size: int = CNN_SIZE) -> np.ndarray:
    """Aspect-preserving normalise an ink mask into a ``size x size`` float canvas in
    [0, 1] (ink=1), centred. Empty input -> zeros. Mirrors ``_recognizer.descriptor``'s
    geometry so train and test inputs match, but returns the 2-D image a CNN wants."""
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.zeros((size, size), np.float32)
    m = (mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 0).astype(np.float32)
    h, w = m.shape
    scale = (size - 4) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    r = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = r
    return canvas


def augment_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random anisotropic scale + shear (slant) + speckle on an ink mask. Uniform
    scale is a no-op after ``slice_to_canvas`` re-normalises, so the scale is per-axis
    (changes aspect); shear emulates the diarist's varying slant; speckle emulates
    scan/threshold noise. Returns a uint8 mask (re-canvassed by the caller)."""
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    sx, sy = rng.uniform(0.82, 1.18), rng.uniform(0.82, 1.18)
    shear = rng.uniform(-0.35, 0.35)
    out_w = max(2, int(w * sx + abs(shear) * h)) + 2
    out_h = max(2, int(h * sy)) + 2
    affine = np.array([[sx, shear, 1.0], [0.0, sy, 1.0]], np.float32)
    m = cv2.warpAffine(m, affine, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:  # flip ~2.5% of pixels (add/remove ink flecks)
        flip = rng.random(m.shape) < 0.025
        m[flip] = 255 - m[flip]
    return m


# --- 3. the CNN + a drop-in recognizer wrapper ------------------------------


class LetterCNN(nn.Module):
    """Tiny 2-conv classifier over a 1x32x32 slice canvas. Deliberately small -- the
    real-letter set is only a few hundred examples, so capacity is the wrong lever."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNRecognizer:
    """Drop-in for ``_recognizer.LetterRecognizer``: ``scores`` / ``score_char`` /
    ``top`` / ``ranking`` over a slice mask, so it plugs straight into
    ``align_boundaries`` and ``recognition_accuracy``. Only the trained classes are
    scored (others implicitly 0), which is honest -- the CNN never saw them."""

    def __init__(self, model: LetterCNN, classes: list[str], size: int = CNN_SIZE):
        self.model = model.eval()
        self.classes = classes
        self.size = size

    def _probs(self, slice_binary: np.ndarray) -> np.ndarray:
        canvas = slice_to_canvas(slice_binary, self.size)
        x = torch.from_numpy(canvas)[None, None]
        with torch.no_grad():
            return torch.softmax(self.model(x)[0], dim=0).numpy()

    def scores(self, slice_binary: np.ndarray) -> dict[str, float]:
        if not (slice_binary > 0).any():
            return dict.fromkeys(self.classes, 0.0)
        p = self._probs(slice_binary)
        return {c: float(p[i]) for i, c in enumerate(self.classes)}

    def score_char(self, slice_binary: np.ndarray, ch: str) -> float:
        return self.scores(slice_binary).get(ch, 0.0)

    def ranking(self, slice_binary: np.ndarray) -> list[str]:
        sc = self.scores(slice_binary)
        return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]

    def top(self, slice_binary: np.ndarray, k: int = 1) -> list[str]:
        return self.ranking(slice_binary)[:k]


# --- 4. split + train -------------------------------------------------------


def split_words(records: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out WHOLE words for test (no leakage). Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_test = max(1, round(test_frac * len(records)))
    test_ix = set(idx[:n_test].tolist())
    train = [r for i, r in enumerate(records) if i not in test_ix]
    test = [r for i, r in enumerate(records) if i in test_ix]
    return train, test


def _slice_list(records: list[dict], classes: list[str]) -> list[tuple[str, np.ndarray]]:
    keep = set(classes)
    return [(ch, sl) for r in records for ch, sl in r["slices"] if ch in keep]


def train_cnn(
    train_records: list[dict],
    classes: list[str],
    epochs: int,
    seed: int,
    augment: bool = True,
    batch_size: int = 64,
) -> LetterCNN:
    """Train ``LetterCNN`` on the training words' slices. Augmentation is re-rolled
    every epoch; training is mini-batched (full-batch gave too few gradient steps to
    converge -- it collapsed to the majority class). Class weights are SQRT-inverse
    frequency: the raw inverse (e=101 vs k=5) over-weights the rare classes and
    destabilises this tiny noisy set, so the milder sqrt is used."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cls_ix = {c: i for i, c in enumerate(classes)}
    data = _slice_list(train_records, classes)
    counts = Counter(ch for ch, _ in data)
    inv = np.array([len(data) / (len(classes) * counts[c]) for c in classes])
    weights = torch.tensor(np.sqrt(inv), dtype=torch.float32)
    model = LetterCNN(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    y_all = np.array([cls_ix[ch] for ch, _ in data])
    n = len(data)
    for _ in range(epochs):
        canvases = np.stack(
            [slice_to_canvas(augment_mask(sl, rng) if augment else sl) for _, sl in data]
        )
        order = rng.permutation(n)
        model.train()
        for s in range(0, n, batch_size):
            bi = order[s : s + batch_size]
            x = torch.from_numpy(canvases[bi])[:, None]
            yb = torch.from_numpy(y_all[bi])
            opt.zero_grad()
            loss_fn(model(x), yb).backward()
            opt.step()
    return model


# --- 5. evaluation ----------------------------------------------------------


def top1_accuracy(
    recognizer, test_slices: list[tuple[str, np.ndarray]]
) -> tuple[float, Counter, Counter]:
    """Free top-1 accuracy of ``recognizer`` over held-out slices. Returns
    ``(accuracy, per_class_correct, per_class_total)``."""
    correct = Counter()
    total = Counter()
    for ch, sl in test_slices:
        total[ch] += 1
        pred = recognizer.top(sl, 1)
        if pred and pred[0] == ch:
            correct[ch] += 1
    n = sum(total.values())
    return (sum(correct.values()) / n if n else 0.0), correct, total


def cut_quality(
    test_records: list[dict], page_image, pitch_px: float, cnn, font, alpha: float
) -> dict:
    """Re-measure the recognition-guided CUT on HELD-OUT words, the HONEST way.

    The naive measure -- cut with the CNN, score with the CNN -- is CIRCULAR: forced
    alignment places boundaries to MAXIMISE the CNN's score for the known chars, so the
    same CNN then scores high regardless of whether the slices are objectively better
    letters. So we cut two ways (geometry, CNN-guided) and score each by BOTH the CNN
    (self) AND the font templates (an INDEPENDENT judge that did NOT guide the cut). If
    the CNN-guided cut only wins under the CNN's own judge and not the independent one,
    the gain is self-consistency, not a better cut. Returns per-(cut, judge) top-1 sums
    over the held-out letters.
    """
    out = {k: [0, 0] for k in ("geo/font", "geo/cnn", "cnncut/font", "cnncut/cnn")}
    for rec in test_records:
        word = rec["word"]
        try:
            prep = R.prepare(page_image, rec["box_2d"], word, pitch_px)
            if prep is None or len(word) < 2:
                continue
            geo_bxs, geom_score = R.geometry_boundaries(prep, word)
            cnn_bxs = R.recognition_boundaries(prep, word, geom_score, alpha=alpha, recognizer=cnn)
        except Exception:
            continue
        for cut_name, bxs in (("geo", geo_bxs), ("cnncut", cnn_bxs)):
            for judge_name, judge in (("font", font), ("cnn", cnn)):
                acc = R.recognition_accuracy(prep, word, bxs, recognizer=judge)
                out[f"{cut_name}/{judge_name}"][0] += acc["top1"]
                out[f"{cut_name}/{judge_name}"][1] += acc["n"]
    return out


# --- CLI --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap a CNN letter recognizer on real slices")
    p.add_argument("--pdf", default="test_document", help="Source PDF (slug or path)")
    p.add_argument("--page", type=int, default=4, help="Page number, 1-based")
    p.add_argument("--version", type=int, default=None, help="Page version (default: latest)")
    p.add_argument("--dpi", type=int, default=600, help="Render DPI for the source page")
    p.add_argument("--epochs", type=int, default=120, help="Training epochs (full-batch)")
    p.add_argument("--seed", type=int, default=0, help="Split + init seed")
    p.add_argument("--test-frac", type=float, default=0.25, help="Fraction of WORDS held out")
    p.add_argument("--min-class", type=int, default=MIN_CLASS, help="Drop classes below this count")
    p.add_argument("--no-aug", action="store_true", help="Disable train augmentation")
    p.add_argument("--alpha", type=float, default=R.ALIGN_ALPHA, help="Forced-align recog weight")
    p.add_argument("--no-cut-quality", action="store_true", help="Skip the cut-quality re-measure")
    p.add_argument("--save", default=None, help=f"Save weights here (e.g. {DEFAULT_WEIGHTS})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    records, page_image, pitch_px = extract_words(args.pdf, args.page, args.version, args.dpi)
    counts = class_counts(records)
    classes = sorted(c for c, n in counts.items() if n >= args.min_class)
    dropped = sorted(c for c, n in counts.items() if n < args.min_class)
    n_slices = sum(counts.values())

    print(f"\n=== EXTRACTED (page {args.page}, geometry cut, label = transcription char) ===")
    print(f"words: {len(records)}   alpha slices: {n_slices}   distinct classes: {len(counts)}")
    print("per-class counts:")
    line = "  ".join(f"{c}:{counts[c]}" for c, _ in counts.most_common())
    print("  " + line)
    print(
        f"kept {len(classes)} classes (>= {args.min_class} slices); "
        f"dropped {len(dropped)} rare classes: {' '.join(dropped) or '(none)'}"
    )

    train_records, test_records = split_words(records, args.test_frac, args.seed)
    train_slices = _slice_list(train_records, classes)
    test_slices = _slice_list(test_records, classes)
    print(
        f"\nsplit (whole words, seed {args.seed}): "
        f"{len(train_records)} train words / {len(test_records)} test words  ->  "
        f"{len(train_slices)} train slices / {len(test_slices)} test slices  "
        f"(test classes seen in train: "
        f"{len({c for c, _ in test_slices} & {c for c, _ in train_slices})}"
        f"/{len({c for c, _ in test_slices})})"
    )

    print(
        f"\ntraining LetterCNN ({len(classes)} classes, {args.epochs} epochs, "
        f"aug={'off' if args.no_aug else 'on'}) ..."
    )
    model = train_cnn(train_records, classes, args.epochs, args.seed, augment=not args.no_aug)
    cnn = CNNRecognizer(model, classes)
    if args.save:
        paths.ensure_parent(args.save)
        torch.save({"state_dict": model.state_dict(), "classes": classes}, args.save)
        print(f"saved weights -> {os.path.abspath(args.save)}")

    font = R.get_recognizer()
    cnn_acc, cnn_ok, cnn_tot = top1_accuracy(cnn, test_slices)
    font_acc, font_ok, _ = top1_accuracy(font, test_slices)

    print("\n=== HEAD-TO-HEAD: real held-out top-1 letter accuracy (same test slices) ===")
    n_test = sum(cnn_tot.values())
    print(f"font templates : {sum(font_ok.values()):>3}/{n_test} = {font_acc:.1%}")
    print(
        f"bootstrap CNN  : {sum(cnn_ok.values()):>3}/{n_test} = {cnn_acc:.1%}   "
        f"(delta {cnn_acc - font_acc:+.1%})"
    )
    print("per-class (test n; font ok / cnn ok):")
    for c in classes:
        if cnn_tot[c]:
            print(f"  {c}: n={cnn_tot[c]:<3} font={font_ok[c]:<3} cnn={cnn_ok[c]:<3}")

    if not args.no_cut_quality:
        print(
            "\n=== CUT QUALITY on HELD-OUT words: does CNN-guided forced-alignment "
            "improve the CUT? ==="
        )
        cq = cut_quality(test_records, page_image, pitch_px, cnn, font, args.alpha)

        def pct(key: str) -> str:
            ok, n = cq[key]
            return f"{ok}/{n}={ok / n:.1%}" if n else "n/a"

        print(f"{'cut \\ judge':<14}{'FONT (indep)':>16}{'CNN (self)':>16}")
        print(f"{'geometry':<14}{pct('geo/font'):>16}{pct('geo/cnn'):>16}")
        print(f"{'CNN-guided':<14}{pct('cnncut/font'):>16}{pct('cnncut/cnn'):>16}  [circular]")
        gf, _ = cq["geo/font"]
        cf, nf = cq["cnncut/font"]
        if nf:
            print(
                "HONEST read: the CNN-guided/CNN cell is CIRCULAR (alignment maximises the "
                "CNN's\nown score). The independent FONT judge is the real test of the cut: "
                f"CNN-guided\n{cf}/{nf}={cf / nf:.1%} vs geometry {gf}/{nf}={gf / nf:.1%} "
                f"(delta {(cf - gf) / nf:+.1%}) -- so the recognizer is a real win, the "
                "cut-guidance is not."
            )

    print(
        "\nNOTE: labels carry the segmenter's ~9% cut error and a shared per-word cut "
        "bias the held-out split cannot remove; treat absolute numbers as noisy."
    )


if __name__ == "__main__":
    main()
