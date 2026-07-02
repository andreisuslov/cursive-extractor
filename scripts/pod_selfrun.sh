#!/usr/bin/env bash
# Autonomous, SELF-TERMINATING pipeline that runs ENTIRELY ON the RunPod GPU pod:
# InkSight-derender every worklist page -> stroke-QA gate -> (optional) label read-back
# -> build the diarybank dataset -> train the CursiveTransformer. Then the pod KILLS
# ITSELF. Nothing on the launching machine needs to stay online -- once launch_selfrun.sh
# hands off, this script owns the whole job and its teardown.
#
# WHY this shape:
#   * The pod terminates itself via RunPod's GraphQL podTerminate on ANY exit (success,
#     error, or the MAX_HOURS watchdog) -- money can't leak if the laptop is offline.
#   * A MAX_HOURS watchdog is the hard ceiling: a hung derender can't bleed for days.
#   * Results are durable WITHOUT the pod: train.py logs checkpoints as W&B artifacts,
#     and on exit we stash /root/run.log to W&B so even a silent failure is diagnosable.
#   * One page failing (crash/OOM) is logged and skipped, never aborts the 16 h run.
#
# Inputs (env, set by launch_selfrun.sh into /root/selfrun.env):
#   RUNPOD_API_KEY   - to self-terminate (RUNPOD_POD_ID is injected by RunPod itself)
#   WANDB_API_KEY    - training + durable log stash
#   GOOGLE_API_KEY   - OPTIONAL; if set, verify_labels read-back gate runs on-pod
#   WANDB_ENTITY, WANDB_PROJECT, DATASET_NAME, TRAIN_ARGS, MIN_QA, MAX_HOURS
# Expects the repo unpacked at /root/ct and an items file /root/items.tsv ("pdf<TAB>page").

set -uo pipefail   # deliberately NOT -e: the terminate trap must ALWAYS run; errors handled inline.

CT=/root/ct
LOG=/root/run.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

[ -f /root/selfrun.env ] && . /root/selfrun.env
MAX_HOURS="${MAX_HOURS:-22}"
MIN_QA="${MIN_QA:-0.85}"
DATASET_NAME="${DATASET_NAME:-diarybank}"
VENV=/root/venv-tf/bin/python          # TF stack: derender + ink_qa + verify_labels
SYS=python3                            # image's torch, +requirements.txt: train.py

log()  { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
warn() { echo "!!! [$(date -u +%H:%M:%S)] $*"; }

# ---- THE self-terminate (bulletproof; runs on every exit) --------------------
terminate_self() {
  code=$?
  trap - EXIT TERM INT
  log "pipeline exiting (code $code) -- stashing log + terminating pod"
  # Durable failure/success record to W&B (best-effort; never blocks teardown).
  if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_SILENT=true "$SYS" - "$LOG" "$code" <<'PY' 2>/dev/null || true
import os, sys, wandb
log, code = sys.argv[1], sys.argv[2]
r = wandb.init(project=os.environ.get("WANDB_PROJECT", "diarybank_gated"),
               entity=os.environ.get("WANDB_ENTITY") or None,
               name="selfrun-status", job_type="pipeline", reinit=True)
wandb.summary["exit_code"] = int(code)
wandb.save(log)
r.finish()
PY
  fi
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" \
      >/dev/null 2>&1 || true
  fi
  command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "${RUNPOD_POD_ID:-}" >/dev/null 2>&1 || true
  log "terminate requested. bye."
}
trap terminate_self EXIT
trap 'exit 124' TERM INT

# ---- hard time ceiling: watchdog kills us; the EXIT trap then terminates ------
( sleep "$(( MAX_HOURS * 3600 ))"; warn "MAX_HOURS=${MAX_HOURS} reached -- killing run"; kill -TERM $$ ) &
WATCHDOG=$!

# ============================ pipeline =======================================
cd "$CT"
export OCR_INKSIGHT_MODEL=/root/small-p-cpu PYTHONUNBUFFERED=1

N=$(grep -cve '^[[:space:]]*$' /root/items.tsv || echo 0)
log "starting: $N pages, MAX_HOURS=$MAX_HOURS, MIN_QA=$MIN_QA, verify_labels=$([ -n "${GOOGLE_API_KEY:-}" ] && echo on || echo off)"

i=0
while IFS=$'\t' read -r pdf page; do
  [ -n "${pdf:-}" ] || continue
  case "$pdf" in \#*) continue ;; esac
  i=$((i + 1))
  prefix=$("$VENV" -c "import sys; from ocr import paths; p=paths; v=p.latest_version(sys.argv[1],int(sys.argv[2])); print(p.prefix(sys.argv[1],int(sys.argv[2]),v))" "$pdf" "$page" 2>/dev/null) || prefix=""
  pdir=$("$VENV" -c "import sys; from ocr import paths; p=paths; v=p.latest_version(sys.argv[1],int(sys.argv[2])); print(p.page_dir(sys.argv[1],int(sys.argv[2]),v))" "$pdf" "$page" 2>/dev/null) || pdir=""
  if [ -z "$prefix" ] || [ -z "$pdir" ]; then warn "[$i/$N] cannot resolve $pdf p$page -- skip"; continue; fi
  out="$pdir/${prefix}_strokes_inksight.json"

  if [ -s "$out" ]; then
    log "[$i/$N] $prefix already derendered -- skip"
  else
    log "[$i/$N] derender $pdf p$page"
    # inksight_vectorize picks boxes source itself: screened > refined > canonical.
    if ! "$VENV" -m ocr.inksight_vectorize --pdf "$pdf" --page "$page" --rich --no-clean --output "$out"; then
      warn "[$i/$N] derender FAILED -- skip page"; continue
    fi
  fi
  # Stroke-space QA gate (clip to box + AIoU); writes <prefix>_strokes_qa.json.
  if ! "$VENV" -m ocr.ink_qa --pdf "$pdf" --page "$page"; then
    warn "[$i/$N] ink_qa FAILED -- skip page"; continue
  fi
  # Optional label read-back on the rendered strokes -> metadata.label_ok.
  if [ -n "${GOOGLE_API_KEY:-}" ]; then
    "$VENV" -m ocr.verify_labels --pdf "$pdf" --page "$page" --min-qa "$MIN_QA" \
      || warn "[$i/$N] verify_labels failed (continuing without label_ok for this page)"
  fi
done < /root/items.tsv

log "derender+gate done. building dataset '$DATASET_NAME' (min-qa $MIN_QA)"
# ink_qa already clipped strokes to the box + dropped ruled lines, so --clean is skipped
# (its dominant-band filter could clip a legitimate descender on an already-clean word).
BUILD_ARGS=(--name "$DATASET_NAME" --strokes-suffix strokes_qa.json --min-qa "$MIN_QA")
[ -n "${GOOGLE_API_KEY:-}" ] && BUILD_ARGS+=(--require-label-ok)
if ! "$SYS" datasets/build_diarybank.py "${BUILD_ARGS[@]}"; then
  warn "build_diarybank FAILED -- nothing to train on. exiting."; exit 1
fi
[ -s "datasets/${DATASET_NAME}.json.zip" ] || { warn "dataset zip missing -- exiting"; exit 1; }

log "training CursiveTransformer on '$DATASET_NAME'"
# TRAIN_ARGS carries all hyperparameters + wandb entity/project/key (set by launcher).
# shellcheck disable=SC2086
if ! "$SYS" train.py --dataset_name "$DATASET_NAME" $TRAIN_ARGS; then
  warn "train.py exited nonzero (checkpoints up to the failure are in W&B)"; exit 1
fi

kill "$WATCHDOG" 2>/dev/null || true
log "ALL DONE -- model in W&B ($WANDB_PROJECT). pod will now self-terminate."
# EXIT trap -> terminate_self -> pod gone.
