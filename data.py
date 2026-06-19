# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import copy
import functools
import json
import os
import random
import warnings
import zipfile
from math import comb

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


########## LOADING DATA AND COMBINING WORDS ##########


@functools.lru_cache(maxsize=5)
def load_and_parse_data(dataset_name: str) -> list:
    """Load ``data/<dataset_name>.json.zip`` and normalize each item's points (apply
    aspect ratio, shift x to start at 0, recenter y). Cached per dataset name."""
    file_path = f"{CURRENT_DIR}/data/{dataset_name}.json.zip"
    print(f"Trying to load dataset file from {file_path}")

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        json_filename = zip_ref.namelist()[0]
        with zip_ref.open(json_filename) as file:
            data = json.load(file)

    for item in data:
        strokes = np.array(item["points"])
        strokes[:, 0] *= item["metadata"]["aspectRatio"]
        strokes[:, 0] -= strokes[0, 0]
        strokes[:, 1] -= 0.65
        item["points"] = strokes
    print(f"Succeeded in loading the {dataset_name} dataset; contains {len(data)} items.")
    return data


def combine_handwriting_examples(examples: list) -> dict:
    """Merge several single-word examples into one multi-word example (concatenated
    text, summed counts, list of per-word point arrays)."""
    return {
        "metadata": {
            "author": examples[0]["metadata"]["author"],
            "asciiSequence": " ".join(ex["metadata"]["asciiSequence"] for ex in examples),
            "pointCount": sum(ex["metadata"]["pointCount"] for ex in examples),
            "strokeCount": sum(ex["metadata"]["strokeCount"] for ex in examples),
            "aspectRatio": examples[0]["metadata"]["aspectRatio"],
        },
        "points": [ex["points"].copy() for ex in examples],
    }


def generate_word_combos(
    raw_json: list, desired_num_combos: int = 10000, num_words: int = 3
) -> list:
    """Sample ``desired_num_combos`` random ``num_words``-word examples from ``raw_json``."""
    num_combos = comb(len(raw_json), num_words)
    print(
        f"For a dataset of {len(raw_json)} examples we can generate "
        f"{num_combos} combinations of {num_words} examples."
    )
    print(f"Generating {desired_num_combos} random combinations.")
    combo_json = []
    for _ in range(desired_num_combos):
        ixs = np.random.choice(len(raw_json), size=num_words, replace=False)
        examples_to_merge = [raw_json[ix] for ix in ixs]
        combo_json.append(combine_handwriting_examples(examples_to_merge))
    return combo_json


########## TOKENIZATION, AUGMENTATION, AND DATA IO ##########


def decompose_offsets(offsets: np.ndarray) -> np.ndarray:
    """Cartesian ``(dx, dy, pen)`` offsets -> polar ``(r, theta, pen)``."""
    dx, dy = offsets[:, 0], offsets[:, 1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return np.column_stack((r, theta, offsets[:, 2]))


def reconstruct_offsets(polar_data: np.ndarray) -> np.ndarray:
    """Polar ``(r, theta, pen)`` -> Cartesian ``(dx, dy, pen)`` (inverse of decompose_offsets)."""
    r, theta = polar_data[:, 0], polar_data[:, 1]
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack((dx, dy, polar_data[:, 2]))


def strokes_to_offsets(points: np.ndarray, prev_points: np.ndarray | None = None) -> np.ndarray:
    """Absolute ``(x, y, pen)`` points -> polar offsets. ``prev_points`` (the previous
    word) seeds the first offset so consecutive words are spaced correctly."""
    offsets = np.zeros_like(points)
    offsets[1:, 0:2] = np.diff(points[:, 0:2], axis=0)  # Same dx, dy computation

    if prev_points is not None:
        offsets[0, 1] = points[0, 1] - prev_points[-1, 1]
        offsets[0, 0] = (prev_points[:, 0].max() - prev_points[-1, 0]) + (
            points[0, 0] - points[:, 0].min()
        )

    offsets[:, 2] = points[:, 2]
    return decompose_offsets(offsets)


def offsets_to_strokes(offsets_dec: np.ndarray) -> np.ndarray:
    """Polar offsets -> absolute ``(x, y, pen)`` points (cumulative sum of dx, dy)."""
    # Calculate cumulative sums over (dx, dt) to get absolute pen positions
    offsets = reconstruct_offsets(offsets_dec)

    absolute_coords = np.cumsum(offsets[:, :2], axis=0)  # just over (dx, dy) dimensions
    stroke_data = np.hstack((absolute_coords, offsets[:, 2:3]))
    return stroke_data


def random_horizontal_shear(
    stroke: np.ndarray, shear_range: tuple[float, float] = (-0.4, 0.4)
) -> np.ndarray:
    """Shear x by a random factor in ``shear_range`` (x' = x + factor*y); modifies in place."""
    shear_factor = np.random.uniform(*shear_range)
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    stroke[:, :2] = np.dot(stroke[:, :2], shear_matrix.T)
    return stroke


def random_rotate(
    stroke: np.ndarray, angle_range: tuple[float, float] = (-0.08, 0.08)
) -> np.ndarray:
    """Rotate xy by a random angle (degrees) in ``angle_range``; modifies in place."""
    angle = np.random.uniform(*angle_range)
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    stroke[:, :2] = np.dot(stroke[:, :2], rotation_matrix.T)
    return stroke


def downsample(arr: np.ndarray, fraction: float, drop_prob: float = 0.05) -> np.ndarray:
    """Reduce each stroke's point count by ``fraction`` (linspace resample), then randomly
    drop interior points with probability ``drop_prob`` (endpoints always kept). Pen-up
    markers and stroke boundaries are preserved; ``fraction == 1`` returns ``arr`` unchanged."""
    if fraction == 1:
        return arr
    result, stroke = [], []
    for point in arr:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                new_len = max(2, int(len(stroke) * (1 - fraction)))
                indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
                reduced_stroke = np.array(stroke)[indices]
                if drop_prob > 0:
                    reduced_stroke = [
                        p
                        for i, p in enumerate(reduced_stroke)
                        if i == 0 or i == len(reduced_stroke) - 1 or random.random() > drop_prob
                    ]
                result.extend(reduced_stroke)
            result.append(point)
            stroke = []
    if stroke:
        new_len = max(2, int(len(stroke) * (1 - fraction)))
        indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
        reduced_stroke = np.array(stroke)[indices]
        if drop_prob > 0:
            reduced_stroke = [
                p
                for i, p in enumerate(reduced_stroke)
                if i == 0 or i == len(reduced_stroke) - 1 or random.random() > drop_prob
            ]
        result.extend(reduced_stroke)
    return np.array(result)


class StrokeDataset(Dataset):
    def __init__(self, raw_word_strokes, texts, args, max_text_length=50, name=""):
        self.raw_word_strokes = raw_word_strokes  # lists of Nx3 word-stroke arrays per sentence
        self.texts = texts  # List of corresponding text strings
        self.args = args
        self.alphabet = args.alphabet  # String of all possible characters
        self.augment = args.augment
        # When False, augmentation still runs (same RNG draws -> same downsampling) but the
        # geometric transforms are identity, so train-time variety is removed for A/B tests.
        self.augment_geometric = getattr(args, "augment_geometric", True)
        self.max_seq_length = args.max_seq_length
        self.max_text_length = max_text_length
        self.name = name
        self.counter = 0

        self.theta_bins = np.linspace(-np.pi, np.pi, 220)

        r_bins_pen_down = np.concatenate(
            [np.asarray([0]), np.linspace(0.0001, 0.060, 30), np.geomspace(0.06001, 0.90, 120)]
        )  # 100 discrete radii
        r_bins_pen_up = r_bins_pen_down + max(r_bins_pen_down) + 1  # Offset for pen-up states
        self.r_bins = np.concatenate(
            [r_bins_pen_down, r_bins_pen_up]
        )  # 200 bins for: {radii x pen up/down}

        self.feature_sizes = [len(self.r_bins), len(self.theta_bins)]
        self.cumulative_sizes = np.cumsum([0, *self.feature_sizes])

        # Add special tokens for strokes
        self.PAD_TOKEN = sum(self.feature_sizes)
        self.END_TOKEN = sum(self.feature_sizes) + 1
        self.WORD_TOKEN = sum(self.feature_sizes) + 2

        # Character tokenization
        self.char_PAD_TOKEN = 0
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.alphabet)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def split_by_word_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        # Find pairs of WORD_TOKENs
        word_boundaries = np.where(
            (tokens[:-1] == self.WORD_TOKEN) & (tokens[1:] == self.WORD_TOKEN)
        )[0]
        # Split using these boundaries
        splits = np.split(tokens, word_boundaries + 1)
        return [s for s in splits if len(s) > 0]

    def concat_with_word_tokens(self, token_lists):
        word_tokens = np.array([self.WORD_TOKEN, self.WORD_TOKEN])
        return np.concatenate(
            [
                np.concatenate([tokens, word_tokens]) if i < len(token_lists) - 1 else tokens
                for i, tokens in enumerate(token_lists)
            ]
        )

    def augment_stroke(
        self,
        stroke,
        shear_range=(-0.08, 0.08),
        rotate_range=(-1.0, 1.0),
        height_scale_range=(0.93, 1.08),
        width_scale_range=(0.95, 1.08),
        height_jitter=0.01,
    ):
        """Augment one word's strokes at training time. The defaults add a *mild* amount of
        handwriting variety (gentle both-way slant + incline, slight tall/short and
        condensed/spread scaling, small vertical jitter) -- deliberately kept small so the
        model stays legible at laptop scale, where legibility is the hard requirement. Wider
        bands were measured to trade legibility for variety monotonically: best test loss rose
        1.18 (no aug) -> 1.80 (moderate, shear +-0.15) -> 2.08 (wide, shear +-0.3, 0.6-1.6
        height), with renders visibly jaggier at each step (static/local_training/). These mild
        ranges keep the slant variety that actually transfers while staying
        legible. The ranges are optional (so the signature stays compatible); pass an identity
        range -- (1.0, 1.0) for the scales, (0.0, 0.0) for shear/rotate, or 0.0 for jitter --
        to weaken or disable a given augmentation (what ``--no-augment`` routes through).
        """
        stroke[:, 0:1] *= np.random.uniform(*width_scale_range)  # condensed vs spread (x)
        stroke[:, 1:2] *= np.random.uniform(*height_scale_range)  # tall vs short letters (y)
        stroke = random_horizontal_shear(stroke, shear_range=shear_range)  # both-way slant
        stroke = random_rotate(stroke, angle_range=rotate_range)  # line incline (re-enabled)
        stroke[:, 1:2] += np.random.uniform(-height_jitter, height_jitter)  # ride higher/lower
        downsample_percent = self.args.downsample_mean + self.args.downsample_width * (
            np.random.rand() - 0.5
        )
        stroke = downsample(stroke, downsample_percent)
        return stroke

    def __len__(self) -> int:
        return len(self.raw_word_strokes)

    def get_vocab_size(self) -> int:
        return sum(self.feature_sizes) + 3  # +3 for PAD, END, and WORD tokens

    def get_char_vocab_size(self) -> int:
        return len(self.alphabet) + 1  # +1 for PAD token

    def get_stroke_seq_length(self) -> int:
        return self.max_seq_length

    def get_text_seq_length(self) -> int:
        return self.max_text_length

    def count_truncated(self, sample_size: int = 256) -> tuple[int, int]:
        """Count how many of the first ``sample_size`` examples tokenize to more than
        ``max_seq_length - 1`` tokens -- i.e. examples that ``__getitem__`` silently
        truncates (dropping their tail, e.g. a later word).

        Read-only diagnostic: when ``augment`` is on it measures one representative
        augmented draw and restores the global RNG state afterwards, so it changes
        nothing the model sees. Returns ``(n_truncated, n_checked)``.
        """
        limit = self.max_seq_length - 1
        n_checked = min(len(self), sample_size)
        n_truncated = 0
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            for idx in range(n_checked):
                words = self.raw_word_strokes[idx]
                if self.augment:  # mirror __getitem__'s per-example augmentation
                    np.random.seed(self.args.seed + idx)
                    random.seed(self.args.seed + idx)
                    words = [self.augment_stroke(w.copy()) for w in words]
                encoded = [
                    self.encode_stroke(
                        strokes_to_offsets(words[i], words[i - 1] if i > 0 else None)
                    )
                    for i in range(len(words))
                ]
                if len(self.concat_with_word_tokens(encoded)) > limit:
                    n_truncated += 1
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
        return n_truncated, n_checked

    def encode_stroke(self, stroke):
        # Encode magnitude and pen state together
        r_idx = np.digitize(stroke[:, 0], self.r_bins[: len(self.r_bins) // 2]) - 1
        r_idx[stroke[:, 2] == 0] += len(self.r_bins) // 2  # Offset for pen-up states

        theta_idx = np.digitize(stroke[:, 1], self.theta_bins) - 1

        encoded = np.column_stack(
            [
                theta_idx + self.cumulative_sizes[1],
                r_idx + self.cumulative_sizes[0],
            ]
        )
        return encoded.flatten()

    def decode_stroke(self, ix):
        ix_list = self.split_by_word_tokens(ix)
        return [self.decode_word_strokes(ix) for ix in ix_list]

    def decode_word_strokes(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()

        # Remove PAD, END, and WORD tokens
        ix = ix[(ix != self.PAD_TOKEN) & (ix != self.END_TOKEN) & (ix != self.WORD_TOKEN)]

        # Reshape the flattened array back to Nx2
        ix = ix[: (len(ix) // 2) * 2]
        ix = ix.reshape(-1, 2)

        r_idx = ix[:, 1] - self.cumulative_sizes[0]
        pen = (r_idx < len(self.r_bins) // 2).astype(int)
        r_idx[pen == 0] -= len(self.r_bins) // 2
        r = self.r_bins[: len(self.r_bins) // 2][r_idx.clip(0, len(self.r_bins) // 2 - 1)]
        theta = self.theta_bins[
            (ix[:, 0] - self.cumulative_sizes[1]).clip(0, len(self.theta_bins) - 1)
        ]

        return np.column_stack([r, theta, pen])

    def encode_text(self, text, do_padding=True):
        encoded_text = torch.tensor(
            [self.stoi.get(ch, self.char_PAD_TOKEN) for ch in text], dtype=torch.long
        )
        if do_padding:
            c = torch.full((self.max_text_length,), self.char_PAD_TOKEN, dtype=torch.long)
            text_len = min(len(encoded_text), self.max_text_length)
            c[:text_len] = encoded_text[:text_len]
        else:
            c = encoded_text
        return c

    def decode_text(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()

        first_pad = np.where(ix == self.char_PAD_TOKEN)[0]
        end_idx = first_pad[0] if len(first_pad) > 0 else len(ix)
        return "".join(self.itos.get(i, "") for i in ix[:end_idx])

    def __getitem__(self, idx):
        word_strokes = self.raw_word_strokes[idx]
        text = self.texts[idx]

        # Apply augmentation per word if enabled
        if self.augment:
            np.random.seed(
                self.args.seed + idx + self.counter
            )  # use the same augmentation across all words in sample
            if self.augment_geometric:
                word_strokes = [self.augment_stroke(word.copy()) for word in word_strokes]
            else:  # identity geometry: consumes the same RNG, so downsampling is unchanged
                word_strokes = [
                    self.augment_stroke(
                        word.copy(),
                        shear_range=(0.0, 0.0),
                        rotate_range=(0.0, 0.0),
                        height_scale_range=(1.0, 1.0),
                        width_scale_range=(1.0, 1.0),
                        height_jitter=0.0,
                    )
                    for word in word_strokes
                ]
        self.counter = (self.counter + 1) % 100000

        # Encode each word separately and combine with WORD_TOKENs
        encoded_words = [
            self.encode_stroke(
                strokes_to_offsets(
                    word_strokes[i], prev_points=word_strokes[i - 1] if i > 0 else None
                )
            )
            for i in range(len(word_strokes))
        ]
        encoded_stroke = self.concat_with_word_tokens(encoded_words)

        # Create input and target sequences
        x = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)
        y = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)

        seq_len = min(
            len(encoded_stroke), self.max_seq_length - 1
        )  # -1 to leave room for END token
        x[:seq_len] = torch.tensor(encoded_stroke[:seq_len], dtype=torch.long)
        x[seq_len] = self.END_TOKEN

        y[:seq_len] = x[1 : seq_len + 1]
        y[seq_len] = self.END_TOKEN

        c = self.encode_text(text)
        return x, c, y


def create_datasets(args) -> tuple[StrokeDataset, StrokeDataset]:
    """Load the dataset, split train/test, combinatorially expand into multi-word
    examples, and wrap them as ``(train_dataset, test_dataset)`` StrokeDatasets."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_and_parse_data(args.dataset_name)

    # partition the input data into a training and the test set
    test_set_size = min(
        1000, max(10, int(len(data) * 0.05))
    )  # between 10 and 1000 examples: ideally 5% of dataset
    rp = torch.randperm(len(data)).tolist()

    train_examples = generate_word_combos(
        [data[i] for i in rp[:-test_set_size]],
        desired_num_combos=args.train_size,
        num_words=args.num_words,
    )
    train_examples = [train_examples[i] for i in torch.randperm(len(train_examples)).tolist()]

    test_examples = generate_word_combos(
        [data[i] for i in rp[-test_set_size:]],
        desired_num_combos=args.test_size,
        num_words=args.num_words,
    )
    test_examples = [test_examples[i] for i in torch.randperm(len(test_examples)).tolist()]

    train_word_strokes = [[copy.deepcopy(stroke) for stroke in v["points"]] for v in train_examples]
    train_texts = [copy.deepcopy(v["metadata"]["asciiSequence"]) for v in train_examples]

    test_word_strokes = [[copy.deepcopy(stroke) for stroke in v["points"]] for v in test_examples]
    test_texts = [copy.deepcopy(v["metadata"]["asciiSequence"]) for v in test_examples]

    print(f"Number of examples in the train dataset: {len(train_examples)}")
    print(f"Number of examples in the test dataset: {len(test_examples)}")
    print(
        f"Average number of words per example: "
        f"{np.mean([len(strokes) for strokes in train_word_strokes]):.1f}"
    )
    print(f"Max token sequence length: {args.max_seq_length}")
    print(f"Number of unique characters in the ascii vocabulary: {len(args.alphabet)}")
    print("Ascii vocabulary:")
    print(f'\t"{args.alphabet}"')
    print(
        f"Split up the dataset into {len(train_examples)} training examples "
        f"and {len(test_examples)} test examples"
    )

    # wrap in dataset objects
    train_dataset = StrokeDataset(train_word_strokes, train_texts, args, name="train")
    test_dataset = StrokeDataset(test_word_strokes, test_texts, args, name="test")

    # Surface silent truncation: examples longer than max_seq_length lose their tail
    # (e.g. a later word) -- the cause of multi-word prompts dropping their last word.
    n_trunc, n_checked = train_dataset.count_truncated()
    if n_checked:
        pct = 100.0 * n_trunc / n_checked
        print(
            f"Examples exceeding max_seq_length={args.max_seq_length} "
            f"(silently truncated): {n_trunc}/{n_checked} sampled ({pct:.0f}%)"
        )
        if n_trunc:
            warnings.warn(
                f"{n_trunc}/{n_checked} sampled training examples exceed "
                f"max_seq_length={args.max_seq_length} and are silently truncated "
                f"(tail tokens -- e.g. later words -- are dropped); raise max_seq_length "
                f"to keep full examples.",
                stacklevel=2,
            )
    return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    From Andrej Karpathy: this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except (
            StopIteration
        ):  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
