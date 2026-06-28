"""InkSight Small-p probe on diary crops (offline->online derendering). 2026-06-28.

Decisive positive result: InkSight derenders faded diary ink into clean ordered cursive
strokes off-the-shelf (no fine-tuning) -- far better than ocr/vectorize.py's tracer.
See WORKLOG.md (2026-06-28 entry). Evidence: inksight_probe_6.png.

SETUP (not committed -- ~0.5GB weights + a TF venv):
  git clone --depth 1 https://github.com/google-research/inksight.git
  uv venv --python 3.11 .venv-inksight
  uv pip install --python .venv-inksight tensorflow==2.17.0 tensorflow-text==2.17.0 \
      huggingface_hub pillow numpy matplotlib
  curl -L -o small-p-cpu.zip https://storage.googleapis.com/derendering_model/small-p-cpu.zip
  unzip small-p-cpu.zip
  # expects: ./inksight/ (repo, for utils.ink/utils.visualize), ./small-p-cpu/ (weights),
  #          ./probe_crops/NN_label.jpg (word crops)
  .venv-inksight/bin/python _inksight_probe.py 6
"""

import glob
import os
import re
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.pyplot as plt

if not hasattr(matplotlib.cm, "get_cmap"):  # matplotlib>=3.9 removed it; utils/visualize uses it
    matplotlib.cm.get_cmap = lambda name=None, lut=None: (
        matplotlib.colormaps[name].resampled(lut) if lut else matplotlib.colormaps[name]
    )

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "inksight"))
from utils.ink import Stroke, Ink  # noqa
from utils.visualize import plot_ink  # noqa


def text_to_tokens(text):
    return [int(t) for t in re.findall(r"<ink_token_(\d+)>", text)]


def detokenize(tokens):
    L = 224
    npd = L + 1
    start = npd * 2
    res, cur, idx = [], [], 0
    while idx < len(tokens):
        t = tokens[idx]
        if t == start:
            if cur:
                res.append(cur)
            cur = []
            idx += 1
        elif idx + 1 < len(tokens) and tokens[idx + 1] != start:
            x = tokens[idx]
            y = tokens[idx + 1] - npd
            if 0 <= x <= L and 0 <= y <= L:
                cur.append((x, y))
            idx += 2
        else:
            idx += 1
    if cur:
        res.append(cur)
    return Ink([Stroke(s) for s in res])


def scale_and_pad(orig, pad_black=True):
    ratio = min(224 / orig.width, 224 / orig.height)
    nc = orig.resize((int(orig.width * ratio), int(orig.height * ratio)))
    color = (0, 0, 0) if pad_black else (255, 255, 255)
    img = Image.new("RGB", (224, 224), color)
    img.paste(nc, ((224 - nc.width) // 2, (224 - nc.height) // 2))
    return img


def load_and_pad_img(image):  # for display background
    w, h = image.size
    r = min(224 / w, 224 / h)
    image = image.resize((int(w * r), int(h * r)))
    w, h = image.size
    if h < 224:
        pad = Image.new("RGB", (w, 224), (255, 255, 255))
        pad.paste(image, (0, (224 - h) // 2))
    else:
        pad = Image.new("RGB", (224, h), (255, 255, 255))
        pad.paste(image, ((224 - w) // 2, 0))
    return pad


n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
crops = sorted(glob.glob(os.path.join(HERE, "probe_crops", "*.jpg")))[:n]
print(f"loading model… ({len(crops)} crops)")
t0 = time.time()
model = tf.saved_model.load(os.path.join(HERE, "small-p-cpu"))
cf = model.signatures["serving_default"]
print(f"model loaded in {time.time() - t0:.0f}s")

cols = 2
rows = len(crops)
fig, axes = plt.subplots(rows, cols, figsize=(8, 3.0 * rows))
if rows == 1:
    axes = np.array([axes])
saved = []
for i, path in enumerate(crops):
    label = os.path.basename(path).split("_", 1)[1].rsplit(".", 1)[0]
    orig = Image.open(path).convert("RGB")
    img = scale_and_pad(orig)
    enc = tf.reshape(tf.io.encode_jpeg(np.array(img)[:, :, :3]), (1, 1))
    t1 = time.time()
    out = cf(input_text=tf.constant(["Recognize and derender."]), **{"image/encoded": enc})
    txt = out["output_0"].numpy()[0][0].decode()
    ink = detokenize(text_to_tokens(txt))
    if len(ink) == 0:  # fallback prompt
        out = cf(input_text=tf.constant(["Derender the ink."]), **{"image/encoded": enc})
        txt = out["output_0"].numpy()[0][0].decode()
        ink = detokenize(text_to_tokens(txt))
    recog = re.sub(r"<[^>]*>", "", txt).strip()[:40]
    dt = time.time() - t1
    print(
        f"  [{i + 1}/{len(crops)}] '{label}' -> {len(ink)} strokes, recog='{recog}', {dt:.0f}s",
        flush=True,
    )
    saved.append(
        {
            "label": label,
            "recog": recog,
            "strokes": [[[float(p[0]), float(p[1])] for p in s] for s in ink.strokes],
        }
    )
    with open(os.path.join(HERE, "inksight_inks.json"), "w") as _f:  # incremental: never lose work
        json.dump(saved, _f)
    axes[i, 0].imshow(orig)
    axes[i, 0].set_title(f'crop: "{label}"', fontsize=9)
    axes[i, 0].axis("off")
    plot_ink(ink, axes[i, 1], input_image=load_and_pad_img(orig))
    axes[i, 1].set_title(f"InkSight ({len(ink)} strokes)", fontsize=9)
    axes[i, 1].axis("off")
plt.tight_layout()
out_png = os.path.join(HERE, f"inksight_probe_{len(crops)}.png")
plt.savefig(out_png, dpi=100)
print("saved", out_png)
