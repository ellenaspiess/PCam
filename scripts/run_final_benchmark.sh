#!/usr/bin/env bash
set -euo pipefail

# Multi-seed final benchmark for reproducible reporting.
# Runs SmallCNN + ResNet(frozen/partial), then summarizes metrics.
# The script is intentionally parameterized via environment variables to keep
# experiment tracking reproducible without editing source files.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE_SMALL_CNN="${BATCH_SIZE_SMALL_CNN:-64}"
BATCH_SIZE_RESNET="${BATCH_SIZE_RESNET:-32}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SEEDS="${SEEDS:-42 52 62 72 82}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-}"
OUT_BASE="${OUT_BASE:-experiments/final_benchmark}"

STAIN_NORMALIZATION="${STAIN_NORMALIZATION:-macenko}"
STAIN_REFERENCE_JSON="${STAIN_REFERENCE_JSON:-experiments/stain_refs/references.json}"
STAIN_REFERENCE_IMAGE="${STAIN_REFERENCE_IMAGE:-}"

SCNN_OPTUNA_JSON="${SCNN_OPTUNA_JSON:-experiments/optuna/small_cnn_narrow_best_params.json}"
RESNET_FROZEN_OPTUNA_JSON="${RESNET_FROZEN_OPTUNA_JSON:-experiments/optuna/resnet_frozen_narrow_best_params.json}"
RESNET_PARTIAL_OPTUNA_JSON="${RESNET_PARTIAL_OPTUNA_JSON:-experiments/optuna/resnet_partial_narrow_best_params.json}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run() {
  echo "[run] $*"
  "$@"
}

resolve_reference_image() {
  # Priority: explicit image path > reference manifest.
  if [ -n "$STAIN_REFERENCE_IMAGE" ]; then
    echo "$STAIN_REFERENCE_IMAGE"
    return
  fi
  if [ ! -f "$STAIN_REFERENCE_JSON" ]; then
    return
  fi
  "$PYTHON_BIN" - "$STAIN_REFERENCE_JSON" <<'PY'
import json
import sys

if len(sys.argv) < 2:
    sys.exit(0)
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(0)
ref = payload.get("reference_image_path")
if isinstance(ref, str) and ref:
    print(ref)
PY
}

REF_IMAGE="$(resolve_reference_image || true)"
REF_OPTS=()
if [ -n "$REF_IMAGE" ]; then
  REF_OPTS+=(--stain-reference-image "$REF_IMAGE")
fi

COMMON_OPTS=(
  --data-root "$DATA_ROOT"
  --num-workers "$NUM_WORKERS"
  --num-epochs "$NUM_EPOCHS"
  --stain-normalization "$STAIN_NORMALIZATION"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  "${REF_OPTS[@]}"
)

if [ -n "$LIMIT_PER_SPLIT" ]; then
  COMMON_OPTS+=(--limit-per-split "$LIMIT_PER_SPLIT")
fi

for seed in $SEEDS; do
  echo "== Seed ${seed} =="

  SCNN_OPTS=(--batch-size "$BATCH_SIZE_SMALL_CNN")
  if [ -f "$SCNN_OPTUNA_JSON" ]; then
    SCNN_OPTS+=(--optuna-best-json "$SCNN_OPTUNA_JSON")
  fi
  run "$PYTHON_BIN" -m pcam.training.train_small_cnn \
    "${COMMON_OPTS[@]}" \
    "${SCNN_OPTS[@]}" \
    --seed "$seed" \
    --ckpt-dir "$OUT_BASE/small_cnn/seed_${seed}"

  RESNET_FROZEN_OPTS=(--batch-size "$BATCH_SIZE_RESNET" --tl-mode frozen)
  if [ -f "$RESNET_FROZEN_OPTUNA_JSON" ]; then
    RESNET_FROZEN_OPTS+=(--optuna-best-json "$RESNET_FROZEN_OPTUNA_JSON")
  fi
  run "$PYTHON_BIN" -m pcam.training.train_resnet \
    "${COMMON_OPTS[@]}" \
    "${RESNET_FROZEN_OPTS[@]}" \
    --seed "$seed" \
    --output-dir "$OUT_BASE/resnet_frozen/seed_${seed}"

  RESNET_PARTIAL_OPTS=(--batch-size "$BATCH_SIZE_RESNET" --tl-mode partial)
  if [ -f "$RESNET_PARTIAL_OPTUNA_JSON" ]; then
    RESNET_PARTIAL_OPTS+=(--optuna-best-json "$RESNET_PARTIAL_OPTUNA_JSON")
  fi
  run "$PYTHON_BIN" -m pcam.training.train_resnet \
    "${COMMON_OPTS[@]}" \
    "${RESNET_PARTIAL_OPTS[@]}" \
    --seed "$seed" \
    --output-dir "$OUT_BASE/resnet_partial/seed_${seed}"
done

# Build consolidated result tables for reporting (per-seed + aggregated).
run "$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_benchmark.py" \
  --input-dir "$OUT_BASE" \
  --output-prefix "$OUT_BASE/summary" \
  --expected-seeds "$SEEDS" \
  --strict

echo "== Done. Benchmark outputs in: $OUT_BASE =="
