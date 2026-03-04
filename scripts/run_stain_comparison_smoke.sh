#!/usr/bin/env bash
set -euo pipefail

# Quick smoke comparison for stain normalization methods on a tiny subset.
# Runs short training jobs for macenko vs reinhard.
# Intended for pipeline validation, not final model comparison.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-512}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"

OUT_BASE="${OUT_BASE:-experiments/smoke_stain_compare}"
STAIN_REFERENCE_IMAGE="${STAIN_REFERENCE_IMAGE:-}"
STAIN_REFERENCE_JSON="${STAIN_REFERENCE_JSON:-}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

REF_OPTS=()
# If a manifest is provided, derive reference image path from it.
if [ -n "$STAIN_REFERENCE_JSON" ]; then
  STAIN_REFERENCE_IMAGE="$("$PYTHON_BIN" - "$STAIN_REFERENCE_JSON" <<'PY'
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(0)
manifest = sys.argv[1]

with open(manifest, "r", encoding="utf-8") as f:
    data = json.load(f)

ref_path = data.get("reference_image_path")
if not ref_path:
    raise SystemExit("No 'reference_image_path' in reference json")

print(str(Path(ref_path)))
PY
)"
fi
# Forward explicit reference image if available.
if [ -n "$STAIN_REFERENCE_IMAGE" ]; then
  REF_OPTS+=(--stain-reference-image "$STAIN_REFERENCE_IMAGE")
fi

run() {
  echo "[run] $*"
  "$@"
}

echo "== SmallCNN smoke (macenko vs reinhard) =="
run "$PYTHON_BIN" -m pcam.training.train_small_cnn \
  --data-root "$DATA_ROOT" \
  --stain-normalization macenko \
  --limit-per-split "$LIMIT_PER_SPLIT" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --ckpt-dir "$OUT_BASE/small_cnn/macenko" \
  "${REF_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.training.train_small_cnn \
  --data-root "$DATA_ROOT" \
  --stain-normalization reinhard \
  --limit-per-split "$LIMIT_PER_SPLIT" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --ckpt-dir "$OUT_BASE/small_cnn/reinhard" \
  "${REF_OPTS[@]}"

echo "== ResNet(partial) smoke (macenko vs reinhard) =="
run "$PYTHON_BIN" -m pcam.training.train_resnet \
  --data-root "$DATA_ROOT" \
  --tl-mode partial \
  --stain-normalization macenko \
  --limit-per-split "$LIMIT_PER_SPLIT" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$OUT_BASE/resnet_partial/macenko" \
  "${REF_OPTS[@]}"

run "$PYTHON_BIN" -m pcam.training.train_resnet \
  --data-root "$DATA_ROOT" \
  --tl-mode partial \
  --stain-normalization reinhard \
  --limit-per-split "$LIMIT_PER_SPLIT" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$OUT_BASE/resnet_partial/reinhard" \
  "${REF_OPTS[@]}"

echo "== Done. Outputs in: $OUT_BASE =="
