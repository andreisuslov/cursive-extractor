"""Word-box renderer for the auto-labelling workflow (see CHANGELOG M16). Run from repo root.

  render page                                  # cache the page image
  render crop I [--box X0,Y0,X1,Y1] [--pad 0.4]
  render slices I --box X0,Y0,X1,Y1 --cuts x1,x2
  render ctx I [--box X0,Y0,X1,Y1]
(prefix each with `python3 -m ocr.experiments.auto_label.`; renders land in the page's _render/ dir)

crop: padded context around the box, green lines = box edges, red ruler in BOX-LOCAL px
      (0 = box left edge; ink left of 0 is outside the box).
slices: top = box crop with red cut lines; bottom = each slice tile captioned with the
      letter it would be labelled as (from the word's text).
"""

import argparse
import base64
import io
import json
import os

from PIL import Image, ImageDraw

PG = os.environ.get("WORDTOOL_PAGE", "1")  # 1 = portal export; else boxes_refined.json
SLUG = os.environ.get("WORDTOOL_SLUG", "0-02-1977-yellow-spiral-bound-notebook_pages_8-14")
ROOT = f"outputs/{SLUG}/page_{int(PG):03d}"
HERE = os.path.join(ROOT, "_render")  # scratch renders; outputs/ is git-ignored
os.makedirs(HERE, exist_ok=True)
PAGE_PNG = os.path.join(HERE, f"page_{PG}.png")
S = 6  # upscale

if PG == "1":
    with open(os.path.join(ROOT, "labels_corrected.json")) as fh:
        LABELS = json.load(fh)
else:  # box_2d is [ymin,xmin,ymax,xmax] on a 0-1000 scale -> page px
    _pw, _ph = Image.open(os.path.join(ROOT, f"{SLUG}_p{int(PG):03d}_page.png")).size
    with open(os.path.join(ROOT, "boxes_refined.json")) as fh:
        LABELS = [
            {
                "text": t["text"],
                "box": [
                    round(t["box_2d"][1] / 1000 * _pw),
                    round(t["box_2d"][0] / 1000 * _ph),
                    round(t["box_2d"][3] / 1000 * _pw),
                    round(t["box_2d"][2] / 1000 * _ph),
                ],
            }
            for t in json.load(fh)
        ]


def page_img():
    if not os.path.exists(PAGE_PNG):
        if PG == "1":
            with open(os.path.join(ROOT, "label_review.json")) as fh:
                lr = json.load(fh)
            img = Image.open(io.BytesIO(base64.b64decode(lr["page"].split(",", 1)[1])))
        else:
            img = Image.open(os.path.join(ROOT, f"{SLUG}_p{int(PG):03d}_page.png"))
        img.save(PAGE_PNG)
    return Image.open(PAGE_PNG).convert("RGB")


def cmd_crop(i, box, pad):
    page = page_img()
    w = LABELS[i]
    x0, y0, x1, y1 = box or w["box"]
    cw, ch = x1 - x0, y1 - y0
    px, py = int(cw * pad), int(ch * pad)
    gx0, gy0 = max(0, x0 - px), max(0, y0 - py)
    crop = page.crop((gx0, gy0, min(page.width, x1 + px), min(page.height, y1 + py)))
    W, H = crop.size
    crop = crop.resize((W * S, H * S), Image.LANCZOS)
    canvas = Image.new("RGB", (W * S, H * S + 44), "white")
    canvas.paste(crop, (0, 0))
    d = ImageDraw.Draw(canvas)
    for bx in (x0 - gx0, x1 - gx0):  # box edges, green
        d.line([(bx * S, 0), (bx * S, H * S)], fill=(0, 160, 0), width=2)
    lo = -((x0 - gx0) // 10) * 10
    for x in range(lo, W - (x0 - gx0) + 1, 10):  # ruler in box-local px
        big = x % 50 == 0
        rx = (x0 - gx0 + x) * S
        d.line(
            [(rx, H * S), (rx, H * S + (18 if big else 9))], fill="red" if big else "gray", width=2
        )
        if big:
            d.text((rx + 3, H * S + 22), str(x), fill="red")
    out = os.path.join(HERE, f"crop_{i:03d}.png")
    canvas.save(out)
    print(
        json.dumps(
            {
                "file": out,
                "text": w["text"],
                "box": [x0, y0, x1, y1],
                "orig_box": w["box"],
                "box_w": cw,
                "box_h": ch,
            }
        )
    )


def cmd_slices(i, box, cuts):
    page = page_img()
    w = LABELS[i]
    x0, y0, x1, y1 = box or w["box"]
    cw, ch = x1 - x0, y1 - y0
    crop = page.crop((x0, y0, x1, y1)).resize((cw * S, ch * S), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    for c in cuts:
        d.line([(c * S, 0), (c * S, ch * S)], fill=(255, 0, 0), width=3)
    text = w["text"]
    bounds = [0.0, *sorted(cuts), float(cw)]
    tiles, tw = [], 0
    for k in range(len(bounds) - 1):
        a, b = int(bounds[k] * S), max(int(bounds[k] * S) + S, int(bounds[k + 1] * S))
        t = crop.crop((a, 0, min(b, cw * S), ch * S))
        tiles.append(t)
        tw += t.width + 14
    H2 = ch * S
    canvas = Image.new("RGB", (max(crop.width, tw), H2 + 30 + H2 + 44), "white")
    canvas.paste(crop, (0, 0))
    d = ImageDraw.Draw(canvas)
    xoff = 0
    for k, t in enumerate(tiles):
        canvas.paste(t, (xoff, H2 + 30))
        letter = text[k] if k < len(text) else "?"
        d.rectangle([xoff, H2 + 30, xoff + t.width - 1, H2 + 30 + H2 - 1], outline=(0, 0, 255))
        d.text((xoff + t.width // 2 - 4, H2 + 34 + H2), letter, fill=(0, 0, 255))
        xoff += t.width + 14
    out = os.path.join(HERE, f"slices_{i:03d}.png")
    canvas.save(out)
    print(json.dumps({"file": out, "text": text, "n_slices": len(tiles), "n_expected": len(text)}))


def cmd_ctx(i, box):
    """The box drawn as a full rectangle amid its neighbours — shows if it eats other lines."""
    page = page_img()
    auto = os.path.join(ROOT, "labels_auto.json")
    if os.path.exists(auto):
        with open(auto) as fh:
            others = json.load(fh)
    else:
        others = LABELS
    w = LABELS[i]
    x0, y0, x1, y1 = box or others[i]["box"]
    cw, ch = x1 - x0, y1 - y0
    px, py = int(cw * 0.5), int(ch * 1.5)  # tall margin: neighbouring lines must be visible
    gx0, gy0 = max(0, x0 - px), max(0, y0 - py)
    gx1, gy1 = min(page.width, x1 + px), min(page.height, y1 + py)
    crop = page.crop((gx0, gy0, gx1, gy1))
    W, H = crop.size
    s = 4
    canvas = crop.resize((W * s, H * s), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(canvas)
    for j, o in enumerate(others):  # neighbours in thin blue
        b = o["box"]
        if j != i and b[2] > gx0 and b[0] < gx1 and b[3] > gy0 and b[1] < gy1:
            d.rectangle(
                [(b[0] - gx0) * s, (b[1] - gy0) * s, (b[2] - gx0) * s, (b[3] - gy0) * s],
                outline=(40, 90, 255),
                width=2,
            )
    d.rectangle(
        [(x0 - gx0) * s, (y0 - gy0) * s, (x1 - gx0) * s, (y1 - gy0) * s],
        outline=(0, 200, 0),
        width=4,
    )  # the box under judgement, thick green
    out = os.path.join(HERE, f"ctx_{i:03d}.png")
    canvas.save(out)
    med = sorted(o["box"][3] - o["box"][1] for o in others)[len(others) // 2]
    print(
        json.dumps(
            {
                "file": out,
                "text": w["text"],
                "box": [x0, y0, x1, y1],
                "box_w": cw,
                "box_h": ch,
                "median_box_h": med,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["page", "crop", "slices", "ctx"])
    p.add_argument("i", nargs="?", type=int)
    p.add_argument("--box", type=str, default=None)
    p.add_argument("--pad", type=float, default=0.4)
    p.add_argument("--cuts", type=str, default="")
    a = p.parse_args()
    box = [round(float(v)) for v in a.box.split(",")] if a.box else None
    if a.cmd == "page":
        page_img()
        print(PAGE_PNG)
    elif a.cmd == "crop":
        cmd_crop(a.i, box, a.pad)
    elif a.cmd == "ctx":
        cmd_ctx(a.i, box)
    else:
        cmd_slices(a.i, box, [float(v) for v in a.cuts.split(",") if v.strip()])
