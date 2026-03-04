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
BASE_SEED="${BASE_SEED:-42}"
N_JOBS="${N_JOBS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
SAVE_TOP_K="${SAVE_TOP_K:-10}"

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
SCNN_NARROW_TRIALS="${SCNN_NARROW_TRIALS:-20}"
SCNN_NARROW_EPOCHS="${SCNN_NARROW_EPOCHS:-8}"
RESNET_FROZEN_NARROW_TRIALS="${RESNET_FROZEN_NARROW_TRIALS:-12}"
RESNET_PARTIAL_NARROW_TRIALS="${RESNET_PARTIAL_NARROW_TRIALS:-12}"
RESNET_NARROW_EPOCHS="${RESNET_NARROW_EPOCHS:-8}"

run() {
  echo "[run] $*"
  "$@"
}

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
)

# Optional explicit reference image path (otherwise training can auto-resolve).
if [ -n "$STAIN_REFERENCE_IMAGE" ]; then
  COMMON_OPTS+=(--stain-reference-image "$STAIN_REFERENCE_IMAGE")
fi

# Timeout disabled when set to 0.
if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
  COMMON_OPTS+=(--timeout-seconds "$TIMEOUT_SECONDS")
fi

echo "== Phase 1: broad search on subset (limit_per_split=${BROAD_LIMIT}) =="
run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model small_cnn \
  --search-mode broad \
  --num-trials "$SCNN_BROAD_TRIALS" \
  --num-epochs "$SCNN_BROAD_EPOCHS" \
  --limit-per-split "$BROAD_LIMIT" \
  --study-name small_cnn_broad \
  "${COMMON_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model resnet \
  --tl-mode frozen \
  --search-mode broad \
  --num-trials "$RESNET_FROZEN_BROAD_TRIALS" \
  --num-epochs "$RESNET_BROAD_EPOCHS" \
  --limit-per-split "$BROAD_LIMIT" \
  --study-name resnet_frozen_broad \
  "${COMMON_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model resnet \
  --tl-mode partial \
  --search-mode broad \
  --num-trials "$RESNET_PARTIAL_BROAD_TRIALS" \
  --num-epochs "$RESNET_BROAD_EPOCHS" \
  --limit-per-split "$BROAD_LIMIT" \
  --study-name resnet_partial_broad \
  "${COMMON_OPTS[@]}"

echo "== Phase 2: narrow refinement on full data =="
run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model small_cnn \
  --search-mode narrow \
  --num-trials "$SCNN_NARROW_TRIALS" \
  --num-epochs "$SCNN_NARROW_EPOCHS" \
  --study-name small_cnn_narrow \
  "${COMMON_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model resnet \
  --tl-mode frozen \
  --search-mode narrow \
  --num-trials "$RESNET_FROZEN_NARROW_TRIALS" \
  --num-epochs "$RESNET_NARROW_EPOCHS" \
  --study-name resnet_frozen_narrow \
  "${COMMON_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.tuning.optuna_search \
  --model resnet \
  --tl-mode partial \
  --search-mode narrow \
  --num-trials "$RESNET_PARTIAL_NARROW_TRIALS" \
  --num-epochs "$RESNET_NARROW_EPOCHS" \
  --study-name resnet_partial_narrow \
  "${COMMON_OPTS[@]}"

echo "== Done. Results written to ${OUTPUT_DIR} =="
