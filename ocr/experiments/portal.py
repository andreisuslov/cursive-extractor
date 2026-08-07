"""Portal ops: push auto-labelled pages to andreisuslov.com/cursive, pull worker cuts back.

The loop this closes:
  seed    labels_auto.json -> a portal page a worker can open (optionally assigned to them)
  ingest  worker's saved page -> only the cuts the WORKER actually placed -> label_boxes/

Why ingest filters: agent-proposed cuts land a few px off and measurably DEGRADE the
cut-predictor (0.150 -> 0.166 MAE/bh, CHANGELOG M16). Once a worker saves, auto and human
cuts look identical in the file, so provenance is *derived* by diffing the save against the
seed we sent: a word whose cuts moved is human-placed, one whose cuts are untouched is still
the agent's and is excluded from training. Nothing to track in the labeller, nothing to go stale.

    export CURSIVE_ADMIN_TOKEN=...        # envchain cursive
    python -m ocr.experiments.portal seed --page-dir outputs/<slug>/page_002 \
        --id yellow_p002 --assign worker1
    python -m ocr.experiments.portal ingest --page-dir outputs/<slug>/page_002 --id yellow_p002
    python -m ocr.experiments.portal list
    python -m ocr.experiments.portal --selftest
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("CURSIVE_BASE", "https://andreisuslov.com")
# Ingested cuts are irreplaceable hand labels; keep them version-controlled in the repo rather
# than in ~/Downloads, which is routinely emptied. Override with LABEL_BOXES_DIR.
REBAKE_DIR = Path(os.environ.get("LABEL_BOXES_DIR")
                  or Path(__file__).resolve().parents[2] / "label_boxes")
MOVED_PX = 1.0  # a cut this far from the seed's counts as human-placed
NEAR_PX = 40.0  # box-centre distance within which a saved word matches a seed word


# --- provenance -------------------------------------------------------------
def cut_xs(word: dict) -> list[float]:
    """Cut x-positions (box-local), however the labeller stored the polylines."""
    return sorted(sum(p[0] for p in c) / len(c) for c in word.get("cuts") or [])


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def match_seed(word: dict, seed: list[dict]) -> dict | None:
    """The seed word this saved word came from: same text, nearest box centre."""
    sx, sy = _centre(word["box"])
    best, bestd = None, NEAR_PX
    for s in seed:
        if s.get("text") != word.get("text"):
            continue
        cx, cy = _centre(s["box"])
        d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        if d <= bestd:
            best, bestd = s, d
    return best


def is_human_cut(word: dict, seed: list[dict]) -> bool:
    """True if the worker placed/moved these cuts (vs leaving the agent's untouched)."""
    xs = cut_xs(word)
    if not xs:
        return False
    src = match_seed(word, seed)
    if src is None:
        return True  # word the worker added or moved far — theirs
    ref = cut_xs(src)
    if len(ref) != len(xs):
        return True
    return any(abs(a - b) > MOVED_PX for a, b in zip(ref, xs, strict=True))


def harvest(doc: dict, seed: list[dict]) -> tuple[list[dict], dict]:
    """Worker-cut words fit for training, plus a breakdown of everything else."""
    keep, stats = (
        [],
        {
            "saved": len(doc.get("words", [])),
            "human": 0,
            "untouched": 0,
            "no_cuts": 0,
            "wrong_count": 0,
        },
    )
    for w in doc.get("words", []):
        if w.get("skip"):
            continue
        if not cut_xs(w):
            stats["no_cuts"] += 1
            continue
        if not is_human_cut(w, seed):
            stats["untouched"] += 1
            continue
        stats["human"] += 1
        # cut_predictor needs exactly len(text)-1 interior cuts to use the word
        if len(cut_xs(w)) != max(0, len(w.get("text", "")) - 1):
            stats["wrong_count"] += 1
            continue
        keep.append(w)
    return keep, stats


# --- page assets ------------------------------------------------------------
def page_data_url(page_dir: str) -> str:
    """The page image as a data-url: the page PNG if present, else a portal export's copy."""
    pngs = glob.glob(os.path.join(page_dir, "*_page.png"))
    if pngs:
        from PIL import Image

        buf = io.BytesIO()
        Image.open(pngs[0]).convert("RGB").save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    for name in ("label_review.json", "labels_corrected.json"):
        p = os.path.join(page_dir, name)
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("page"):
                return d["page"]
    raise SystemExit(f"no page image found in {page_dir}")


def load_words(page_dir: str) -> list[dict]:
    p = os.path.join(page_dir, "labels_auto.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found — run the auto_label workflow for this page first")
    with open(p) as fh:
        return json.load(fh)


# --- api --------------------------------------------------------------------
def api(path: str, payload: dict | None = None) -> dict:
    token = os.environ.get("CURSIVE_ADMIN_TOKEN")
    if not token:
        raise SystemExit("set CURSIVE_ADMIN_TOKEN (envchain cursive)")
    # token goes in a header, never the query string: URLs are logged by proxies and the
    # cloudflared tunnel, which the server's own log redaction cannot reach.
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE}/cursive/api/{path}",
        data=data,
        # Cloudflare 403s the default Python-urllib UA, so send our own
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "cursive-portal/1.0",
        },
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# --- commands ---------------------------------------------------------------
def cmd_seed(a):
    words = load_words(a.page_dir)
    doc = {"page": page_data_url(a.page_dir), "words": words}
    done = sum(1 for w in words if w.get("done"))
    print(f"seeding {a.id}: {len(words)} words, {done} pre-cut ({done / len(words) * 100:.0f}%)")
    if a.dry_run:
        return
    api(f"seed/{a.id}", doc)
    if a.assign:
        api("admin/assign", {"user": a.assign, "page_ids": [a.id]})
        print(f"assigned to {a.assign}")
    print(f"seeded -> {BASE}/cursive")


def cmd_ingest(a):
    seed = load_words(a.page_dir)
    doc = api(f"download/{a.id}")
    keep, stats = harvest(doc, seed)
    print(
        f"{a.id}: {stats['saved']} words saved — {stats['human']} human-cut "
        f"({stats['wrong_count']} dropped on cut-count), {stats['untouched']} still agent cuts, "
        f"{stats['no_cuts']} uncut"
    )
    if not keep:
        print("nothing worker-cut yet; nothing written")
        return
    out = REBAKE_DIR / f"portal_{a.id}.json"
    if a.dry_run:
        print(f"[dry-run] would write {len(keep)} words -> {out}")
        return
    REBAKE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"page": doc.get("page") or page_data_url(a.page_dir), "words": keep}, fh)
    print(f"wrote {len(keep)} human-cut words -> {out}")
    print("now: python -m ocr.experiments.cut_predictor        # check it still improves")
    print("     python -m ocr.experiments.cut_predictor --rebake")


def cmd_list(_a):
    for p in api("pages")["pages"]:
        print(
            f"  {p['id']:24} {p.get('done', 0):>4}/{p.get('words', 0):<5} words"
            f"  completed={p.get('completed')}  -> {p.get('assigned_to')}"
        )


def _selftest():
    def _w(text, box, xs):
        return {"text": text, "box": box, "cuts": [[[x, 0], [x, box[3] - box[1]]] for x in xs]}

    seed = [_w("cat", [0, 0, 90, 50], [30, 60]), _w("dog", [0, 100, 90, 150], [30, 60])]
    same = json.loads(json.dumps(seed[0]))  # untouched -> agent's
    moved = json.loads(json.dumps(seed[1]))
    moved["cuts"][0] = [[41, 0], [41, 50]]  # dragged -> human
    assert not is_human_cut(same, seed)
    assert is_human_cut(moved, seed)
    nudged = json.loads(json.dumps(seed[0]))  # sub-pixel wobble is not an edit
    nudged["cuts"][0] = [[30.4, 0], [30.4, 50]]
    assert not is_human_cut(nudged, seed)
    added = {"text": "cat", "box": [500, 500, 590, 550], "cuts": [[[30, 0], [30, 50]]]}
    assert is_human_cut(added, seed)  # far from any seed word -> theirs
    keep, stats = harvest({"words": [same, moved]}, seed)
    assert [w["text"] for w in keep] == ["dog"], keep
    assert stats["human"] == 1 and stats["untouched"] == 1, stats
    wrong = json.loads(json.dumps(seed[1]))  # human, but cut count != len-1
    wrong["cuts"] = [[[10, 0], [10, 50]]]
    _, s2 = harvest({"words": [wrong]}, seed)
    assert s2["human"] == 1 and s2["wrong_count"] == 1, s2
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, needs in (
        ("seed", cmd_seed, True),
        ("ingest", cmd_ingest, True),
        ("list", cmd_list, False),
    ):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if needs:
            s.add_argument("--page-dir", required=True, help="outputs/<slug>/page_<NNN>")
            s.add_argument("--id", required=True, help="portal page id")
            s.add_argument("--dry-run", action="store_true")
        if name == "seed":
            s.add_argument("--assign", help="username to assign the page to")
    a = ap.parse_args()
    a.fn(a)
