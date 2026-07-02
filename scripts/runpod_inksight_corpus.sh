#!/usr/bin/env bash
# Corpus derender runner: InkSight-derender every worklist page on a rented RunPod GPU.
#
# WHY this shape (operational lessons from the first GPU runs, baked in):
#   * runpodctl create stdout is NOT trusted -- a probe once left 6 pods running because
#     usage text was mixed into error output. Reconcile against 'runpodctl get pod'
#     before and after create, tear down on EXIT/INT, and finish with a loud reconcile.
#   * scp is REJECTED on these hosts: upload = tar stream into ssh 'cat > f',
#     download = ssh 'cat f'. ssh -o options are written inline in one place (zsh does
#     not word-split unquoted vars, so never stash them in a variable).
#   * TF hangs on process exit after writing its output, so page completion is detected
#     by POLLING the remote output file, never by waiting on the process; each finished
#     page is pulled immediately (never lose paid-for work), then the hung process is
#     pkill'd before the next page starts.
#   * The derender runs --no-clean with an EXPLICIT output name: contamination is
#     removed later in stroke space (ocr.ink_qa; clean_word's erosion damages target
#     ink), and the default output path would CLOBBER the skeleton strokes.json.
#
# Usage:
#   scripts/runpod_inksight_corpus.sh --print-worklist > worklist.tsv
#   scripts/runpod_inksight_corpus.sh --worklist worklist.tsv [--shard 2/3] --yes
#
# The worklist is "pdf_rel_path<TAB>page" lines ('#' comments and blank lines ok) --
# edit or split it freely. --shard I/N keeps lines I-1 mod N (1-based) for parallel
# pods. Already-derendered pages (local *_strokes_inksight.json exists) are skipped,
# so re-running after a failure resumes where it left off.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU_TYPE="NVIDIA L40S"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DISK_GB=60
RATE="0.99"     # USD/hr, for the pre-flight estimate only
SEC_PER_WORD=5  # measured on an L40S (~58x Mac CPU)
WEIGHTS_URL="https://storage.googleapis.com/derendering_model/small-p-cpu.zip"
PRINT_WORKLIST=0
WORKLIST=""
SHARD="1/1"
YES=0

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod_inksight_corpus.sh --print-worklist > worklist.tsv
  scripts/runpod_inksight_corpus.sh --worklist worklist.tsv [options] --yes

Options:
  --print-worklist   Emit "pdf_rel_path<TAB>page" for every latest-version page
                     under outputs/ that has a boxes json, then exit.
  --worklist FILE    Work list to run ('#' comments / blank lines ignored).
  --shard I/N        Round-robin shard: keep lines where (lineno-1) % N == I-1.
  --gpu-type STR     RunPod GPU type (default: "NVIDIA L40S").
  --image STR        Container image (default: runpod/pytorch py3.11 cuda12.4).
  --disk-gb N        Container disk GB (default: 60).
  --rate USD         USD/hr for the cost estimate (default: 0.99).
  --yes              Actually create a pod and run (without it: estimate + exit).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --print-worklist) PRINT_WORKLIST=1 ;;
    --worklist) WORKLIST="$2"; shift ;;
    --shard) SHARD="$2"; shift ;;
    --gpu-type) GPU_TYPE="$2"; shift ;;
    --image) IMAGE="$2"; shift ;;
    --disk-gb) DISK_GB="$2"; shift ;;
    --rate) RATE="$2"; shift ;;
    --yes) YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- worklist mode
if [ "$PRINT_WORKLIST" -eq 1 ]; then
  python3 - <<'EOF'
import glob, os, re

latest = {}
for run in sorted(glob.glob(os.path.join("outputs", "*"))):
    slug = os.path.basename(run)
    pdf = os.path.join(run, slug + ".pdf")
    if not os.path.isdir(run) or not os.path.isfile(pdf):
        continue
    for p in glob.glob(os.path.join(run, "page_*")):
        m = re.fullmatch(r"page_(\d+)(?:_(\d+))?", os.path.basename(p))
        if not m or not os.path.isdir(p):
            continue
        if not glob.glob(os.path.join(p, "*_boxes.json")):
            continue
        key, ver = (slug, int(m.group(1))), int(m.group(2) or 0)
        if key not in latest or ver > latest[key][0]:
            latest[key] = (ver, pdf)
for (slug, page), (ver, pdf) in sorted(latest.items()):
    print(f"{pdf}\t{page}")
EOF
  exit 0
fi

# ---------------------------------------------------------------- run mode
[ -n "$WORKLIST" ] || { echo "need --worklist FILE (or --print-worklist)" >&2; usage >&2; exit 2; }
[ -f "$WORKLIST" ] || { echo "worklist not found: $WORKLIST" >&2; exit 2; }

SCRATCH="$(mktemp -d)"

# Resolve every shard item against the local outputs/ layout (ocr.paths is the one
# source of truth for page dirs / prefixes) and count its words from the boxes json
# the derenderer will actually use -- the SAME preference order as
# ocr.inksight_vectorize: *_boxes_screened.json > boxes_refined.json > canonical.
python3 - "$WORKLIST" "$SHARD" "$SCRATCH/items.tsv" "$SCRATCH/upload.lst" <<'EOF'
import json, os, sys

sys.path.insert(0, os.getcwd())
from ocr import paths

worklist, shard, items_tsv, upload_lst = sys.argv[1:5]
si, sn = (int(x) for x in shard.split("/"))
if not (1 <= si <= sn):
    sys.exit(f"bad --shard {shard!r} (want I/N with 1 <= I <= N)")

lines = []
with open(worklist) as f:
    for raw in f:
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)

items, uploads = [], []
for k, line in enumerate(lines):
    if k % sn != si - 1:
        continue
    parts = line.split("\t")
    if len(parts) < 2:
        sys.exit(f"bad worklist line (want pdf<TAB>page): {line!r}")
    pdf, page = parts[0], int(parts[1])
    if not os.path.isfile(pdf):
        sys.exit(f"worklist pdf not found: {pdf}")
    ver = paths.latest_version(pdf, page)
    if ver is None:
        sys.exit(f"no processed page dir for {pdf} page {page} (run ocr.extract_boxes first)")
    pdir = paths.page_dir(pdf, page, ver)
    prefix = paths.prefix(pdf, page, ver)
    boxes = paths.boxes_json(pdf, page, ver)
    screened = os.path.join(pdir, f"{prefix}_boxes_screened.json")
    refined = os.path.join(pdir, "boxes_refined.json")
    use = next((p for p in (screened, refined, boxes) if os.path.isfile(p)), boxes)
    if not os.path.isfile(use):
        sys.exit(f"boxes json missing: {use}")
    with open(use) as f:
        nwords = sum(1 for b in json.load(f) if "box_2d" in b)
    items.append((pdf, page, pdir, prefix, nwords))
    uploads.append(pdf)
    for extra in (boxes, screened, refined):
        if os.path.isfile(extra):
            uploads.append(extra)

with open(items_tsv, "w") as f:
    for it in items:
        f.write("\t".join(str(x) for x in it) + "\n")
with open(upload_lst, "w") as f:
    for u in dict.fromkeys(uploads):  # dedup, keep order
        f.write(u + "\n")
print(f"resolved {len(items)} pages / {sum(i[4] for i in items)} words (shard {shard})")
EOF

# Load items; skip pages whose derender already exists locally (resume support).
ITEM_PDF=(); ITEM_PAGE=(); ITEM_DIR=(); ITEM_PREFIX=(); ITEM_WORDS=()
SKIPPED=0
PENDING_WORDS=0
while IFS=$'\t' read -r pdf page pdir prefix nwords; do
  if [ -s "$pdir/${prefix}_strokes_inksight.json" ]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  ITEM_PDF+=("$pdf"); ITEM_PAGE+=("$page"); ITEM_DIR+=("$pdir")
  ITEM_PREFIX+=("$prefix"); ITEM_WORDS+=("$nwords")
  PENDING_WORDS=$((PENDING_WORDS + nwords))
done < "$SCRATCH/items.tsv"
N_ITEMS=${#ITEM_PDF[@]}

# ------------------------------------------------- cost guard (BEFORE any pod)
EST=$(python3 -c "
w = int('$PENDING_WORDS'); r = float('$RATE'); h = w * $SEC_PER_WORD / 3600.0
print('%d words -> ~%.1f h -> ~USD %.2f (at USD %.2f/hr, %ds/word)' % (w, h, h * r, r, $SEC_PER_WORD))
")
echo ">>> plan: $N_ITEMS pages to derender ($SKIPPED already done locally, skipped)"
echo ">>> estimate: $EST on $GPU_TYPE (secure cloud)"
if [ "$N_ITEMS" -eq 0 ]; then
  echo ">>> nothing to do."
  exit 0
fi
if [ "$YES" -ne 1 ]; then
  echo "!!! dry run only -- no pod created. Re-run with --yes to proceed."
  exit 1
fi
command -v runpodctl >/dev/null || { echo "runpodctl not found on PATH" >&2; exit 2; }

# ------------------------------------------------- pod lifecycle
POD_ID=""
POD_NAME="inksight-$(date +%Y%m%d-%H%M%S)-$$"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  echo ""
  if [ -n "$POD_ID" ]; then
    echo ">>> tearing down pod $POD_ID ($POD_NAME)"
    runpodctl remove pod "$POD_ID" \
      || echo "!!! FAILED to remove pod $POD_ID -- remove it manually: runpodctl remove pod $POD_ID"
  fi
  echo ">>> final reconcile ('runpodctl get pod'):"
  pods=$(runpodctl get pod 2>&1 || true)
  echo "$pods"
  if echo "$pods" | awk 'NR > 1 && $1 ~ /^[A-Za-z0-9]{10,}$/ {found = 1} END {exit !found}'; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! PODS STILL RUNNING (listed above) -- these are BILLING. Verify"
    echo "!!! each one and remove any you own: runpodctl remove pod <id>"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  else
    echo ">>> no pods appear to be running."
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

echo ""
echo ">>> pods BEFORE create (reconcile -- never trust create stdout):"
runpodctl get pod || true

echo ""
echo ">>> creating pod '$POD_NAME' ($GPU_TYPE, secure cloud, ${DISK_GB}GB disk)"
runpodctl create pod --name "$POD_NAME" --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
  --containerDiskSize "$DISK_GB" --secureCloud --ports "22/tcp" \
  || echo "!!! create exited nonzero -- reconciling against 'get pod' anyway"

for _try in 1 2 3 4 5 6; do
  sleep 5
  POD_ID=$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2 == n {print $1}' | head -n1) || POD_ID=""
  if [ -n "$POD_ID" ]; then break; fi
done
if [ -z "$POD_ID" ]; then
  echo "!!! pod '$POD_NAME' never appeared in 'runpodctl get pod' -- create failed."
  exit 1
fi
echo ">>> POD ID: $POD_ID  (name: $POD_NAME)"
echo ">>> teardown is automatic on exit; manual: runpodctl remove pod $POD_ID"

# Public ssh endpoint (IP:PORT->22 in 'get pod -a'); pods take a minute to expose it.
SSH_HOST=""; SSH_PORT=""
echo ">>> waiting for the pod's public ssh port (up to 15 min)"
deadline=$(( $(date +%s) + 900 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  row=$(runpodctl get pod -a 2>/dev/null | awk -v id="$POD_ID" '$1 == id') || row=""
  hp=$(printf '%s' "$row" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+->22' | head -n1) || hp=""
  if [ -n "$hp" ]; then
    SSH_HOST="${hp%%:*}"
    p="${hp#*:}"; SSH_PORT="${p%%->*}"
    break
  fi
  sleep 10
done
if [ -z "$SSH_HOST" ]; then
  echo "!!! no public ssh port after 15 min ('runpodctl get pod -a' row: $row)"
  exit 1
fi
echo ">>> ssh endpoint: root@$SSH_HOST:$SSH_PORT"

# All remote access goes through this. Options are inline on the ssh command itself
# (a $SSH_OPTS variable breaks under zsh, which does not word-split unquoted vars).
rssh() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 -o ServerAliveInterval=15 -o LogLevel=ERROR \
    -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

echo ">>> waiting for sshd (up to 10 min)"
deadline=$(( $(date +%s) + 600 ))
until rssh "echo ok" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "!!! sshd never came up"
    exit 1
  fi
  sleep 10
done

# ------------------------------------------------- upload + remote setup
echo ">>> uploading ocr/ + $(wc -l < "$SCRATCH/upload.lst" | tr -d ' ') data files (tar over ssh 'cat'; scp is rejected on these hosts)"
COPYFILE_DISABLE=1 tar -czf "$SCRATCH/payload.tgz" \
  --exclude '__pycache__' --exclude '.DS_Store' \
  -T "$SCRATCH/upload.lst" ocr  # -T must precede the operand (BSD tar reads it as a filename after one)
rssh "cat > /root/payload.tgz" < "$SCRATCH/payload.tgz"
rssh "mkdir -p /root/ct && tar -xzf /root/payload.tgz -C /root/ct"

cat > "$SCRATCH/setup.sh" <<EOF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq poppler-utils unzip curl >/dev/null
command -v python3.11 >/dev/null || apt-get install -y -qq python3.11 python3.11-venv >/dev/null
[ -d /root/venv ] || python3.11 -m venv /root/venv
/root/venv/bin/pip install -q --upgrade pip
/root/venv/bin/pip install -q tensorflow==2.17.0 tensorflow-text==2.17.0 \
    pillow numpy opencv-python-headless pdf2image
if [ ! -d /root/small-p-cpu ]; then
  curl -sSL -o /root/small-p-cpu.zip "$WEIGHTS_URL"
  unzip -q -o /root/small-p-cpu.zip -d /root
fi
echo SETUP_OK
EOF
echo ">>> remote setup (python3.11 venv + TF 2.17 + InkSight weights ~0.5GB; a few minutes)"
rssh "cat > /root/setup.sh" < "$SCRATCH/setup.sh"
rssh "bash /root/setup.sh"

# ------------------------------------------------- derender loop
DONE_PAGES=0
FAILED=()
i=0
while [ "$i" -lt "$N_ITEMS" ]; do
  pdf="${ITEM_PDF[$i]}"; page="${ITEM_PAGE[$i]}"; pdir="${ITEM_DIR[$i]}"
  prefix="${ITEM_PREFIX[$i]}"; nwords="${ITEM_WORDS[$i]}"
  i=$((i + 1))
  out_rel="$pdir/${prefix}_strokes_inksight.json"
  echo ""
  echo ">>> [page $i/$N_ITEMS] $pdf p$page ($nwords words) -> $out_rel"

  # Launch detached; completion is detected by the output file landing, NOT by exit
  # (TF reliably hangs on exit after writing the json).
  cmd="cd /root/ct && PYTHONUNBUFFERED=1 OCR_INKSIGHT_MODEL=/root/small-p-cpu"
  cmd="$cmd nohup /root/venv/bin/python -m ocr.inksight_vectorize"
  cmd="$cmd --pdf '$pdf' --page $page --rich --no-clean --output '$out_rel'"
  cmd="$cmd >> /root/derender.log 2>&1 < /dev/null & echo launched"
  rssh "$cmd"

  timeout=$(( nwords * 20 + 900 ))  # generous: 4x the 5s/word budget + model load
  start=$(date +%s)
  ssh_fails=0
  gone=0
  page_done=0
  while :; do
    elapsed=$(( $(date +%s) - start ))
    if [ "$elapsed" -gt "$timeout" ]; then
      echo "!!! [page $i/$N_ITEMS] TIMEOUT after ${elapsed}s -- marking failed, moving on"
      FAILED+=("$pdf p$page (timeout)")
      break
    fi
    status=$(rssh "if [ -s '/root/ct/$out_rel' ]; then echo DONE; elif pgrep -f ocr.inksight_vectorize >/dev/null 2>&1; then echo WAIT; else echo GONE; fi; grep -c 'Derendered [0-9]*/' /root/derender.log 2>/dev/null || true" 2>/dev/null) || status=""
    if [ -z "$status" ]; then
      ssh_fails=$((ssh_fails + 1))
      if [ "$ssh_fails" -ge 20 ]; then
        echo "!!! 20 consecutive ssh failures -- aborting run (pod may be dead)"
        exit 1
      fi
      sleep 15
      continue
    fi
    ssh_fails=0
    state=$(printf '%s\n' "$status" | sed -n 1p)
    # Process died without producing output (crash/OOM): don't burn the whole page
    # timeout on a billing pod. Two consecutive sightings, to dodge launch races.
    if [ "$state" = "GONE" ]; then
      gone=$((gone + 1))
      if [ "$gone" -ge 2 ]; then
        echo "!!! [page $i/$N_ITEMS] derender process died with no output -- marking failed"
        rssh "tail -5 /root/derender.log" 2>/dev/null || true
        FAILED+=("$pdf p$page (process died)")
        break
      fi
    else
      gone=0
    fi
    done_words=$(printf '%s\n' "$status" | sed -n 2p)
    done_words=${done_words:-0}
    remaining=$(( PENDING_WORDS - done_words ))
    if [ "$remaining" -lt 0 ]; then remaining=0; fi
    printf '    %s  words done %s/%s  ETA ~%s min\n' \
      "$(date +%H:%M:%S)" "$done_words" "$PENDING_WORDS" "$(( remaining * SEC_PER_WORD / 60 ))"
    if [ "$state" = "DONE" ]; then
      page_done=1
      break
    fi
    sleep 20
  done

  if [ "$page_done" -eq 1 ]; then
    # Pull IMMEDIATELY (download = ssh 'cat'); validate before accepting.
    pulled=0
    for _attempt in 1 2; do
      if rssh "cat '/root/ct/$out_rel'" > "$out_rel.tmp" 2>/dev/null \
         && python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$out_rel.tmp" 2>/dev/null; then
        mv "$out_rel.tmp" "$out_rel"
        pulled=1
        break
      fi
      sleep 10
    done
    if [ "$pulled" -eq 1 ]; then
      DONE_PAGES=$((DONE_PAGES + 1))
      echo "    pulled -> $out_rel"
    else
      rm -f "$out_rel.tmp"
      echo "!!! [page $i/$N_ITEMS] pull failed or invalid json -- marking failed"
      FAILED+=("$pdf p$page (invalid pull)")
    fi
  fi
  # Clear the (possibly hung) derender process before the next page.
  rssh "pkill -f ocr.inksight_vectorize >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
done

echo ""
echo ">>> derendered $DONE_PAGES/$N_ITEMS pages ($SKIPPED skipped as already done, ${#FAILED[@]} failed)"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "!!! failed items (re-run the same command to retry; finished pages auto-skip):"
  for f in "${FAILED[@]}"; do echo "!!!   $f"; done
  exit 1
fi
