#!/usr/bin/env bash
set -euo pipefail

# Recommended two-phase Optuna tuning pipeline for CPU-only runs.
# You can override defaults via environment variables.
# Example:
#   BROAD_LIMIT=20000 SCNN_BROAD_TRIALS=60 bash scripts/run_tuning.sh
# This script writes all search artifacts into OUTPUT_DIR.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAIN_NORMALIZATION="${STAIN_NORMALIZATION:-macenko}"
STAIN_REFERENCE_IMAGE="${STAIN_REFERENCE_IMAGE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/optuna}"
STUDY_PREFIX="${STUDY_PREFIX:-}"
BASE_SEED="${BASE_SEED:-42}"
N_JOBS="${N_JOBS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
SAVE_TOP_K="${SAVE_TOP_K:-10}"
DEVICE="${DEVICE:-cpu}"
TRACKER_LOG="${TRACKER_LOG:-}"
RUN_PHASES="${RUN_PHASES:-all}"

# Optuna behavior
TPE_STARTUP_TRIALS="${TPE_STARTUP_TRIALS:-12}"
PRUNER_STARTUP_TRIALS="${PRUNER_STARTUP_TRIALS:-6}"
PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-1}"

# Phase 1 (broad)
BROAD_LIMIT="${BROAD_LIMIT:-15000}"
SCNN_BROAD_TRIALS="${SCNN_BROAD_TRIALS:-40}"
SCNN_BROAD_EPOCHS="${SCNN_BROAD_EPOCHS:-3}"
RESNET_FROZEN_BROAD_TRIALS="${RESNET_FROZEN_BROAD_TRIALS:-20}"
RESNET_PARTIAL_BROAD_TRIALS="${RESNET_PARTIAL_BROAD_TRIALS:-20}"
RESNET_BROAD_EPOCHS="${RESNET_BROAD_EPOCHS:-3}"

# Phase 2 (narrow)
# Runs on full dataset by default; set NARROW_LIMIT to use a subset.
NARROW_LIMIT="${NARROW_LIMIT:-}"
SCNN_NARROW_TRIALS="${SCNN_NARROW_TRIALS:-12}"
SCNN_NARROW_EPOCHS="${SCNN_NARROW_EPOCHS:-5}"
RESNET_FROZEN_NARROW_TRIALS="${RESNET_FROZEN_NARROW_TRIALS:-8}"
RESNET_PARTIAL_NARROW_TRIALS="${RESNET_PARTIAL_NARROW_TRIALS:-8}"
RESNET_NARROW_EPOCHS="${RESNET_NARROW_EPOCHS:-5}"

run() {
  echo "[run] $*"
  "$@"
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

TOTAL_STEPS=0
CURRENT_STEP=0

log() {
  local msg="$1"
  echo "$msg"
  echo "$msg" >> "$RUN_LOG"
}

run_step() {
  local label="$1"
  shift
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local start_epoch
  start_epoch="$(date +%s)"
  log "[$(timestamp)] [${CURRENT_STEP}/${TOTAL_STEPS}] START ${label}"
  run "$@"
  local end_epoch
  end_epoch="$(date +%s)"
  local duration=$((end_epoch - start_epoch))
  log "[$(timestamp)] [${CURRENT_STEP}/${TOTAL_STEPS}] DONE  ${label} (${duration}s)"
}

study_name() {
  local base="$1"
  if [ -n "$STUDY_PREFIX" ]; then
    echo "${STUDY_PREFIX}_${base}"
  else
    echo "$base"
  fi
}

case "$RUN_PHASES" in
  all)
    TOTAL_STEPS=6
    ;;
  broad|narrow)
    TOTAL_STEPS=3
    ;;
  *)
    echo "Invalid RUN_PHASES='$RUN_PHASES' (expected: all|broad|narrow)" >&2
    exit 1
    ;;
esac

COMMON_OPTS=(
  --data-root "$DATA_ROOT"
  --num-workers "$NUM_WORKERS"
  --stain-normalization "$STAIN_NORMALIZATION"
  --output-dir "$OUTPUT_DIR"
  --base-seed "$BASE_SEED"
  --n-jobs "$N_JOBS"
  --tpe-startup-trials "$TPE_STARTUP_TRIALS"
  --pruner-startup-trials "$PRUNER_STARTUP_TRIALS"
  --pruner-warmup-steps "$PRUNER_WARMUP_STEPS"
  --save-top-k "$SAVE_TOP_K"
  --device "$DEVICE"
)

mkdir -p "$OUTPUT_DIR"
if [ -n "$TRACKER_LOG" ]; then
  RUN_LOG="$TRACKER_LOG"
else
  RUN_LOG="${OUTPUT_DIR}/run_tuning_$(date '+%Y%m%d_%H%M%S').log"
fi

log "== run_tuning started at $(timestamp) =="
log "config: study_prefix=${STUDY_PREFIX:-<none>} device=${DEVICE} output_dir=${OUTPUT_DIR}"
log "config: run_phases=${RUN_PHASES}"

# Optional explicit reference image path (otherwise training can auto-resolve).
if [ -n "$STAIN_REFERENCE_IMAGE" ]; then
  COMMON_OPTS+=(--stain-reference-image "$STAIN_REFERENCE_IMAGE")
fi

# Timeout disabled when set to 0.
if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
  COMMON_OPTS+=(--timeout-seconds "$TIMEOUT_SECONDS")
fi

NARROW_LIMIT_OPTS=()
if [ -n "$NARROW_LIMIT" ]; then
  NARROW_LIMIT_OPTS=(--limit-per-split "$NARROW_LIMIT")
  log "== Phase 2: narrow refinement on subset (limit_per_split=${NARROW_LIMIT}) =="
else
  log "== Phase 2: narrow refinement on full data =="
fi

if [ "$RUN_PHASES" = "all" ] || [ "$RUN_PHASES" = "broad" ]; then
  log "== Phase 1: broad search on subset (limit_per_split=${BROAD_LIMIT}) =="
  run_step "$(study_name small_cnn_broad)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model small_cnn \
    --search-mode broad \
    --num-trials "$SCNN_BROAD_TRIALS" \
    --num-epochs "$SCNN_BROAD_EPOCHS" \
    --limit-per-split "$BROAD_LIMIT" \
    --study-name "$(study_name small_cnn_broad)" \
    "${COMMON_OPTS[@]}"

  run_step "$(study_name resnet_frozen_broad)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model resnet \
    --tl-mode frozen \
    --search-mode broad \
    --num-trials "$RESNET_FROZEN_BROAD_TRIALS" \
    --num-epochs "$RESNET_BROAD_EPOCHS" \
    --limit-per-split "$BROAD_LIMIT" \
    --study-name "$(study_name resnet_frozen_broad)" \
    "${COMMON_OPTS[@]}"

  run_step "$(study_name resnet_partial_broad)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model resnet \
    --tl-mode partial \
    --search-mode broad \
    --num-trials "$RESNET_PARTIAL_BROAD_TRIALS" \
    --num-epochs "$RESNET_BROAD_EPOCHS" \
    --limit-per-split "$BROAD_LIMIT" \
    --study-name "$(study_name resnet_partial_broad)" \
    "${COMMON_OPTS[@]}"
fi

if [ "$RUN_PHASES" = "all" ] || [ "$RUN_PHASES" = "narrow" ]; then
  run_step "$(study_name small_cnn_narrow)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model small_cnn \
    --search-mode narrow \
    --num-trials "$SCNN_NARROW_TRIALS" \
    --num-epochs "$SCNN_NARROW_EPOCHS" \
    ${NARROW_LIMIT_OPTS[@]+"${NARROW_LIMIT_OPTS[@]}"} \
    --study-name "$(study_name small_cnn_narrow)" \
    "${COMMON_OPTS[@]}"

  run_step "$(study_name resnet_frozen_narrow)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model resnet \
    --tl-mode frozen \
    --search-mode narrow \
    --num-trials "$RESNET_FROZEN_NARROW_TRIALS" \
    --num-epochs "$RESNET_NARROW_EPOCHS" \
    ${NARROW_LIMIT_OPTS[@]+"${NARROW_LIMIT_OPTS[@]}"} \
    --study-name "$(study_name resnet_frozen_narrow)" \
    "${COMMON_OPTS[@]}"

  run_step "$(study_name resnet_partial_narrow)" "$PYTHON_BIN" -m pcam.tuning.optuna_search \
    --model resnet \
    --tl-mode partial \
    --search-mode narrow \
    --num-trials "$RESNET_PARTIAL_NARROW_TRIALS" \
    --num-epochs "$RESNET_NARROW_EPOCHS" \
    ${NARROW_LIMIT_OPTS[@]+"${NARROW_LIMIT_OPTS[@]}"} \
    --study-name "$(study_name resnet_partial_narrow)" \
    "${COMMON_OPTS[@]}"
fi

log "== Done. Results written to ${OUTPUT_DIR} =="
log "== tracker log: ${RUN_LOG} =="
