# PCam

CPU-only PCam training pipeline for PatchCamelyon metastasis detection.

## Setup (macOS, CPU only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Repository Layout

```text
src/pcam/
  data/       # dataset + dataloader utilities
  models/     # SmallCNN and ResNet18 model definitions
  training/   # train/eval entry points and helpers
```

## Train

```bash
python -m pcam.training.train_small_cnn
python -m pcam.training.train_resnet
python -m pcam.training.train_small_cnn --optuna-best-json experiments/optuna/small_cnn_narrow_best_params.json
python -m pcam.training.train_resnet --optuna-best-json experiments/optuna/resnet_partial_narrow_best_params.json --tl-mode partial
```

Both training entry points are configured for CPU-only execution and support
longer runs with early stopping plus optional Optuna-parameter injection.
Default stain normalization is `macenko` (via `torchstain`), configurable with
`--stain-normalization macenko|reinhard|none`.

Optional: select a fixed reference image from train split for stain normalization:

```bash
python scripts/select_stain_references.py --data-root data/raw --output-dir experiments/stain_refs
```

Then pass it to training/tuning (example):

```bash
python -m pcam.training.train_small_cnn --stain-normalization macenko --stain-reference-image experiments/stain_refs/reference_image.npy
```

If no `--stain-reference-image` is provided, training now auto-resolves
`experiments/stain_refs/references.json` and uses its `reference_image_path`
for Macenko normalization.

## Tune (Optuna)

```bash
python -m pcam.tuning.optuna_search --model small_cnn --search-mode broad --num-trials 30 --num-epochs 3
python -m pcam.tuning.optuna_search --model small_cnn --search-mode narrow --num-trials 20 --num-epochs 8
python -m pcam.tuning.optuna_search --model resnet --tl-mode frozen --search-mode broad --num-trials 20 --num-epochs 3
python -m pcam.tuning.optuna_search --model resnet --tl-mode frozen --search-mode narrow --num-trials 12 --num-epochs 8
python -m pcam.tuning.optuna_search --model resnet --tl-mode partial --search-mode broad --num-trials 20 --num-epochs 3
python -m pcam.tuning.optuna_search --model resnet --tl-mode partial --search-mode narrow --num-trials 12 --num-epochs 8
```

Search artifacts are saved under `experiments/optuna/`:
- `<study_name>.db` (Optuna SQLite storage)
- `<study_name>_best_params.json`
- `<study_name>_trials.json`
- `<study_name>_topK.json` (best completed trials, configurable via `--save-top-k`)
- `<study_name>_config.json` (complete search configuration for reproducibility)

Current Optuna search space includes learning rate, weight decay, dropout, batch size,
scheduler (`none|cosine|plateau`) and early-stopping patience.
For ResNet, run separate studies for each transfer-learning mode (`--tl-mode frozen|partial`).
Use `--search-mode broad` for phase 1 and `--search-mode narrow` for phase 2 refinement.
If `--num-epochs` is omitted, defaults are `3` (broad) and `8` (narrow).

Run the full recommended two-phase pipeline:

```bash
bash scripts/run_tuning.sh
```

Useful overrides for the script:

```bash
TIMEOUT_SECONDS=21600 N_JOBS=1 SAVE_TOP_K=15 bash scripts/run_tuning.sh
```

## Final Benchmark (Multi-Seed)

Run a multi-seed benchmark (SmallCNN + ResNet frozen + ResNet partial)
with test-set evaluation and automatic summary tables:

```bash
bash scripts/run_final_benchmark.sh
```

Default seeds are: `42 52 62 72 82` (5 seeds).

Useful overrides:

```bash
SEEDS="42 52 62 72 82" NUM_EPOCHS=100 OUT_BASE=experiments/final_benchmark_v2 bash scripts/run_final_benchmark.sh
```

Outputs:
- Per-run artifacts under `experiments/final_benchmark/<model>/seed_<seed>/`
- `test_metrics.json` for each run
- ROC/PR curves for each run (`*_test_curves.json`, `*_test_curves.png`)
- Aggregates at:
  - `experiments/final_benchmark/summary_per_seed.csv`
  - `experiments/final_benchmark/summary_per_model.csv`
  - `experiments/final_benchmark/summary.json`
