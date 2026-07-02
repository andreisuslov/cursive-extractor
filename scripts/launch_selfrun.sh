#!/usr/bin/env bash
# Launch the autonomous, self-terminating diarybank pipeline (scripts/pod_selfrun.sh) on a
# fresh RunPod GPU, then WALK AWAY. Your machine is only needed for the ~15 min of pod
# creation + dependency install; after handoff the pod owns the whole 16-18 h job AND its
# own teardown, so Claude Code / this laptop can go offline.
#
# What "safe" means here:
#   * Cost is estimated and printed BEFORE any pod is created; nothing runs without --yes.
#   * Until the detached job is confirmed running, a failure tears the pod down. AFTER
#     handoff the trap deliberately LEAVES the pod running (killing it would abort the job)
#     -- the pod self-terminates when done (see pod_selfrun.sh).
#   * Refuses to derender un-screened (contaminated) pages unless --allow-unscreened:
#     screening is the near-free local gate that keeps junk off the paid GPU.
#
# Keys are read from envchain (wandb, gemini, runpod) at launch and written only into the
# pod's /root/selfrun.env. runpodctl must already be configured (it is: ~/.runpod).
#
# Usage:
#   scripts/launch_selfrun.sh --print-worklist > worklist.tsv     # then screen those pages
#   scripts/launch_selfrun.sh --worklist worklist.tsv --yes
# Options: --worklist F | --shard I/N | --max-hours N (22) | --min-qa F (0.85)
#          --gpu-type S | --wandb-entity S | --wandb-project S | --dataset-name S
#          --train-args "..." | --no-verify (skip Gemini label read-back on-pod)
#          --allow-unscreened | --yes

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

GPU_TYPE="NVIDIA L40S"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DISK_GB=60
RATE="0.99"; SEC_PER_WORD=5
WEIGHTS_URL="https://storage.googleapis.com/derendering_model/small-p-cpu.zip"
WORKLIST=""; SHARD="1/1"; YES=0; PRINT_WORKLIST=0; ALLOW_UNSCREENED=0; NO_VERIFY=0
MAX_HOURS=22; MIN_QA="0.85"
WANDB_ENTITY="suslov-harrisburg-university-of-science-and-technology"
WANDB_PROJECT="diarybank_gated"; DATASET_NAME="diarybank"
TRAIN_ARGS="--wandb_run_name selfrun --max_steps 20000 --n_layer 5 --num_words 4 --batch_size 32 --max_seq_length 1500 --learning_rate 1e-2 --downsample_mean 0.65 --device cuda"

while [ $# -gt 0 ]; do
  case "$1" in
    --print-worklist) PRINT_WORKLIST=1 ;;
    --worklist) WORKLIST="$2"; shift ;;
    --shard) SHARD="$2"; shift ;;
    --max-hours) MAX_HOURS="$2"; shift ;;
    --min-qa) MIN_QA="$2"; shift ;;
    --gpu-type) GPU_TYPE="$2"; shift ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift ;;
    --dataset-name) DATASET_NAME="$2"; shift ;;
    --train-args) TRAIN_ARGS="$2"; shift ;;
    --no-verify) NO_VERIFY=1 ;;
    --allow-unscreened) ALLOW_UNSCREENED=1 ;;
    --yes) YES=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# ---- worklist print mode (delegate to the existing resolver) -----------------
if [ "$PRINT_WORKLIST" -eq 1 ]; then exec "$REPO/scripts/runpod_inksight_corpus.sh" --print-worklist; fi
[ -n "$WORKLIST" ] && [ -f "$WORKLIST" ] || { echo "need --worklist FILE (or --print-worklist)" >&2; exit 2; }

SCRATCH="$(mktemp -d)"; trap 'rm -rf "$SCRATCH"' EXIT

# ---- resolve pages, REQUIRE screened boxes, build items + upload list ---------
python3 - "$WORKLIST" "$SHARD" "$ALLOW_UNSCREENED" "$SCRATCH/items.tsv" "$SCRATCH/upload.lst" <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from ocr import paths

worklist, shard, allow_unscreened, items_tsv, upload_lst = sys.argv[1:6]
allow_unscreened = allow_unscreened == "1"
si, sn = (int(x) for x in shard.split("/"))
assert 1 <= si <= sn, f"bad --shard {shard!r}"

lines = [s.strip() for s in open(worklist) if s.strip() and not s.startswith("#")]
items, uploads, unscreened, total = [], [], [], 0
for k, line in enumerate(lines):
    if k % sn != si - 1:
        continue
    pdf, page = line.split("\t")[0], int(line.split("\t")[1])
    assert os.path.isfile(pdf), f"pdf not found: {pdf}"
    ver = paths.latest_version(pdf, page)
    assert ver is not None, f"no page dir for {pdf} p{page}"
    pdir, prefix = paths.page_dir(pdf, page, ver), paths.prefix(pdf, page, ver)
    canonical = paths.boxes_json(pdf, page, ver)
    screened = os.path.join(pdir, f"{prefix}_boxes_screened.json")
    refined = os.path.join(pdir, "boxes_refined.json")
    transcript = paths.transcript_txt(pdf, page, ver)
    if not os.path.isfile(screened):
        unscreened.append(f"{pdf} p{page}")
    use = screened if os.path.isfile(screened) else (refined if os.path.isfile(refined) else canonical)
    with open(use) as f:
        total += sum(1 for b in json.load(f) if "box_2d" in b)
    items.append((pdf, page))
    for extra in (pdf, canonical, screened, refined, transcript):
        if os.path.isfile(extra):
            uploads.append(extra)

if unscreened and not allow_unscreened:
    sys.stderr.write("!!! these pages have NO screened boxes (run ocr.screen_crops first, or pass --allow-unscreened):\n")
    for u in unscreened:
        sys.stderr.write(f"!!!   {u}\n")
    sys.exit(3)

with open(items_tsv, "w") as f:
    for pdf, page in items:
        f.write(f"{pdf}\t{page}\n")
with open(upload_lst, "w") as f:
    for u in dict.fromkeys(uploads):
        f.write(u + "\n")
print(f"resolved {len(items)} pages / {total} words to derender (shard {shard}); {len(unscreened)} unscreened")
PY
[ $? -eq 0 ] || exit $?

N_PAGES=$(wc -l < "$SCRATCH/items.tsv" | tr -d ' ')
WORDS=$(python3 - "$SCRATCH/items.tsv" <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd()); from ocr import paths
tot = 0
for line in open(sys.argv[1]):
    pdf, page = line.split("\t")[0], int(line.split("\t")[1])
    ver = paths.latest_version(pdf, page); pdir = paths.page_dir(pdf, page, ver); prefix = paths.prefix(pdf, page, ver)
    for c in (os.path.join(pdir, f"{prefix}_boxes_screened.json"), os.path.join(pdir, "boxes_refined.json"), paths.boxes_json(pdf, page, ver)):
        if os.path.isfile(c):
            tot += sum(1 for b in json.load(open(c)) if "box_2d" in b); break
print(tot)
PY
)

# ---- cost/time estimate + gate (BEFORE any pod) ------------------------------
python3 -c "
w=$WORDS; r=$RATE; s=$SEC_PER_WORD
dh=w*s/3600; th=1.7; setup=0.25
tot=dh+th+setup
print(f'>>> plan: $N_PAGES pages, {w} words to derender + build + train, self-terminating')
print(f'>>> derender ~{dh:.1f} h  train ~{th:.1f} h  setup ~{setup:.2f} h  =>  ~{tot:.1f} h  ~USD {tot*r:.2f} (L40S secure)')
print(f'>>> hard ceiling: MAX_HOURS=$MAX_HOURS (~USD {$MAX_HOURS*r:.2f} worst case)')
"
if [ "$YES" -ne 1 ]; then echo "!!! dry run -- re-run with --yes to launch."; exit 1; fi
command -v runpodctl >/dev/null || { echo "runpodctl not found" >&2; exit 2; }

# ---- fetch keys from envchain (only ever written to the pod's env file) -------
echo ">>> reading keys from envchain (wandb, gemini, runpod)"
WANDB_KEY=$(envchain wandb sh -c 'printf %s "$WANDB_API_KEY"') || { echo "no WANDB_API_KEY" >&2; exit 2; }
RUNPOD_KEY=$(envchain runpod sh -c 'printf %s "$RUNPOD_API_KEY"') || { echo "no RUNPOD_API_KEY" >&2; exit 2; }
GOOGLE_KEY=""
[ "$NO_VERIFY" -eq 1 ] || GOOGLE_KEY=$(envchain gemini sh -c 'printf %s "$GOOGLE_API_KEY"') || true
[ -n "$WANDB_KEY" ] && [ -n "$RUNPOD_KEY" ] || { echo "missing keys" >&2; exit 2; }

# ---- pod lifecycle (teardown ONLY until handoff) -----------------------------
POD_ID=""; HANDOFF=0
POD_NAME="ct-selfrun-$(date +%Y%m%d-%H%M%S)-$$"
cleanup() {
  st=$?; trap - EXIT INT TERM
  if [ "$HANDOFF" -eq 1 ]; then
    echo ">>> handoff complete -- LEAVING pod $POD_ID running; it self-terminates when done."
    echo ">>> monitor: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}   (kill early: runpodctl remove pod $POD_ID)"
  elif [ -n "$POD_ID" ]; then
    echo ">>> launch aborted before handoff -- tearing down pod $POD_ID"
    runpodctl remove pod "$POD_ID" || echo "!!! remove FAILED -- do it manually: runpodctl remove pod $POD_ID"
  fi
  rm -rf "$SCRATCH"; exit "$st"
}
trap cleanup EXIT INT TERM

echo ">>> pods before create (never trust create stdout):"; runpodctl get pod || true
echo ">>> creating '$POD_NAME' ($GPU_TYPE, secure, ${DISK_GB}GB)"
runpodctl create pod --name "$POD_NAME" --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
  --containerDiskSize "$DISK_GB" --secureCloud --ports "22/tcp" \
  || echo "!!! create nonzero -- reconciling anyway"
for _ in 1 2 3 4 5 6; do
  sleep 5
  POD_ID=$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2==n{print $1}' | head -1) || POD_ID=""
  [ -n "$POD_ID" ] && break
done
[ -n "$POD_ID" ] || { echo "!!! pod never appeared -- create failed"; exit 1; }
echo ">>> POD ID: $POD_ID"

# public ssh endpoint
SSH_HOST=""; SSH_PORT=""; deadline=$(( $(date +%s) + 900 ))
echo ">>> waiting for public ssh port (up to 15 min)"
while [ "$(date +%s)" -lt "$deadline" ]; do
  hp=$(runpodctl get pod -a 2>/dev/null | awk -v id="$POD_ID" '$1==id' | grep -oE '[0-9.]+:[0-9]+->22' | head -1) || hp=""
  if [ -n "$hp" ]; then SSH_HOST="${hp%%:*}"; p="${hp#*:}"; SSH_PORT="${p%%->*}"; break; fi
  sleep 10
done
[ -n "$SSH_HOST" ] || { echo "!!! no ssh port after 15 min"; exit 1; }
rssh() { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
  -o ServerAliveInterval=15 -o LogLevel=ERROR -p "$SSH_PORT" "root@$SSH_HOST" "$@"; }
echo ">>> ssh root@$SSH_HOST:$SSH_PORT ; waiting for sshd (up to 10 min)"
deadline=$(( $(date +%s) + 600 ))
until rssh "echo ok" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "!!! sshd never came up"; exit 1; }; sleep 10
done

# ---- upload code + data + items + env ----------------------------------------
echo ">>> uploading repo + $(wc -l < "$SCRATCH/upload.lst" | tr -d ' ') data files"
COPYFILE_DISABLE=1 tar -czf "$SCRATCH/payload.tgz" --exclude '__pycache__' --exclude '.DS_Store' \
  -T "$SCRATCH/upload.lst" \
  ocr datasets/build_diarybank.py train.py model.py data.py sample.py requirements.txt scripts/pod_selfrun.sh
rssh "cat > /root/payload.tgz" < "$SCRATCH/payload.tgz"
rssh "mkdir -p /root/ct && tar -xzf /root/payload.tgz -C /root/ct"
rssh "cat > /root/items.tsv" < "$SCRATCH/items.tsv"

{
  echo "export RUNPOD_API_KEY='$RUNPOD_KEY'"
  echo "export WANDB_API_KEY='$WANDB_KEY'"
  [ -n "$GOOGLE_KEY" ] && echo "export GOOGLE_API_KEY='$GOOGLE_KEY'"
  echo "export WANDB_ENTITY='$WANDB_ENTITY'"
  echo "export WANDB_PROJECT='$WANDB_PROJECT'"
  echo "export DATASET_NAME='$DATASET_NAME'"
  echo "export MIN_QA='$MIN_QA'"
  echo "export MAX_HOURS='$MAX_HOURS'"
  echo "export TRAIN_ARGS=\"--wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT $TRAIN_ARGS\""
} > "$SCRATCH/selfrun.env"
rssh "cat > /root/selfrun.env" < "$SCRATCH/selfrun.env"

# ---- synchronous setup (watch it; better to catch install failure now) -------
cat > "$SCRATCH/setup.sh" <<EOF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq poppler-utils unzip curl python3.11-venv >/dev/null
[ -d /root/venv-tf ] || python3.11 -m venv /root/venv-tf
/root/venv-tf/bin/pip install -q --upgrade pip
/root/venv-tf/bin/pip install -q tensorflow==2.17.0 tensorflow-text==2.17.0 \
    pillow numpy opencv-python-headless pdf2image google-generativeai
python3 -m pip install -q -r /root/ct/requirements.txt
if [ ! -d /root/small-p-cpu ]; then
  curl -sSL -o /root/small-p-cpu.zip "$WEIGHTS_URL" && unzip -q -o /root/small-p-cpu.zip -d /root
fi
echo SETUP_OK
EOF
echo ">>> remote setup (TF venv + torch deps + 0.5GB weights; ~10-15 min)"
rssh "cat > /root/setup.sh" < "$SCRATCH/setup.sh"
rssh "bash /root/setup.sh" | tail -3
rssh "test -d /root/venv-tf && test -d /root/small-p-cpu" || { echo "!!! setup incomplete"; exit 1; }

# ---- kick off the detached, self-terminating run + confirm handoff -----------
echo ">>> launching pod_selfrun.sh (detached, self-terminating)"
rssh "cd /root/ct && setsid nohup bash scripts/pod_selfrun.sh >/root/boot.log 2>&1 < /dev/null & echo started"
ok=0
for _ in $(seq 1 12); do
  sleep 5
  if rssh "grep -q 'starting:' /root/run.log 2>/dev/null"; then ok=1; break; fi
done
if [ "$ok" -eq 1 ]; then
  echo ">>> confirmed: pipeline is running on the pod."
  rssh "sed -n '1,3p' /root/run.log" 2>/dev/null || true
  HANDOFF=1
else
  echo "!!! pipeline did not report 'starting:' within 60s -- check /root/boot.log:"
  rssh "tail -20 /root/boot.log" 2>/dev/null || true
  echo "!!! NOT handing off; tearing the pod down to avoid a silent billing run."
fi
