# Local training samples

Cursive generated entirely on a laptop (Apple Silicon, MPS) with Weights & Biases
disabled, via `scripts/train_local.py` on the `easybank` dataset. These demonstrate
that the pipeline trains and samples locally with no cloud/GPU cluster.

| run | steps | wall-clock (MPS) | test loss |
|-----|-------|------------------|-----------|
| 3k  | 3000  | ~4 min           | 2.13      |
| 12k | 12000 | ~16 min          | 1.82      |

Files are `easybank_<steps>_<words>.png`, where `<words>` are the two prompt words.

Honest read: 12k steps improves quality on already-legible words ("laundry" goes
jagged -> smooth, "zoom" becomes readable) and reaches real legibility on the easier
words; hard cases ("onzn", "page") stay rough and "loan" never renders (the model
ends the sequence early). Gains mostly land by ~7-8k steps then plateau at constant
lr. The published model trains ~125k steps with a larger config/dataset; this is a
laptop proof-of-life, not production quality.

Regenerate: `python3 scripts/train_local.py --dataset easybank --steps 12000`
(checkpoint + fresh PNGs land in the gitignored `runs/`).
