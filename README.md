# PCam

PCam benchmark pipeline for PatchCamelyon metastasis detection with shared
preprocessing, stain normalization, Optuna tuning, model training, and
multi-seed evaluation.

## Scope

This repository contains a student research pipeline for binary classification
on PatchCamelyon (PCam). It includes:

- shared preprocessing and dataloading utilities
- optional stain normalization with fixed reference support
- baseline training for SmallCNN and ResNet18 transfer-learning variants
- Optuna-based hyperparameter search
- multi-seed benchmarking and summary export

This project was initially developed as part of the university course
Advanced Machine Learning at Ulm University of Applied Sciences (THU)
and was later extended and optimized in the context of work at the 
Institute of Neuropathology, Ulm University Medical Center, Faculty of Medicine,
Ulm University.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional for development checks:

```bash
pip install -r requirements-dev.txt
```

## Data

The raw PCam dataset is not stored in this repository. On first use, the
training and preprocessing entry points automatically download the required
PatchCamelyon split files to `data/raw/pcam/` if they are missing.

This requires:

- internet access
- `gdown` and `h5py` installed via `requirements.txt`

If you prefer a different location, pass `--data-root /path/to/data`.

## Repository Layout

```text
src/pcam/
  preprocessing/  # dataset, transforms, dataloaders
  models/     # SmallCNN and ResNet18 model definitions
  training/   # train/eval entry points and helpers
  tuning/     # Optuna search entry points
scripts/      # benchmark, tuning, and reporting helpers
experiments/  # saved runs, summaries, and reference artifacts
```

## Runtime Notes

- The current `train_small_cnn` and `train_resnet` entry points use Apple
  Silicon `mps`.
- The Optuna search entry point supports `cpu`, `mps`, and `auto`
  device selection.
- Existing benchmark outputs under `experiments/` are included as result
  artifacts for the accompanying student paper.
- If you only want to inspect or summarize existing results, you do not need to
  rerun training.

## Train

```bash
python -m pcam.training.train_small_cnn
python -m pcam.training.train_resnet
python -m pcam.training.train_small_cnn --optuna-best-json experiments/optuna/small_cnn_narrow_best_params.json
python -m pcam.training.train_resnet --optuna-best-json experiments/optuna/resnet_partial_narrow_best_params.json --tl-mode partial
```

Both training entry points support longer runs with early stopping plus
optional Optuna-parameter injection.
Default stain normalization is `macenko` (via `torchstain`), configurable with
`--stain-normalization macenko|reinhard|none`.
The training commands above currently assume an Apple Silicon environment with
`mps` available.

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
DEVICE=mps STUDY_PREFIX=cpu_fast_v1 bash scripts/run_tuning.sh
```

Process tracking: the script prints `[step/6] START/DONE` lines and writes a
run log to `experiments/optuna/run_tuning_YYYYmmdd_HHMMSS.log` (override with
`TRACKER_LOG=/path/to/file.log`).

Use a fresh study namespace without deleting old runs:

```bash
STUDY_PREFIX=cpu_fast_v1 bash scripts/run_tuning.sh
```

Run only narrow (resume-friendly, skips broad):

```bash
RUN_PHASES=narrow STUDY_PREFIX=cpu_fast_v1 bash scripts/run_tuning.sh
```

By default, narrow refinement runs on the full dataset. Use `NARROW_LIMIT` only
if you intentionally want a subset for faster iteration.

CPU-fast overrides for narrow phase (optional):

```bash
NARROW_LIMIT=20000 SCNN_NARROW_TRIALS=10 SCNN_NARROW_EPOCHS=4 \
RESNET_FROZEN_NARROW_TRIALS=6 RESNET_PARTIAL_NARROW_TRIALS=6 RESNET_NARROW_EPOCHS=4 \
bash scripts/run_tuning.sh
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

## Existing Artifacts

The repository already contains experiment outputs for tuning and final
benchmark runs. For a student paper, the most relevant files are usually:

- `experiments/optuna/*_best_params.json` for selected hyperparameters
- `experiments/final_benchmark_*/summary_per_seed.csv` for per-seed reporting
- `experiments/final_benchmark_*/summary_per_model.csv` for aggregate reporting
- per-run `test_metrics.json` and curve files for qualitative inspection

## Reproducing Results

If you want to reproduce the main pipeline from scratch, the recommended order
is:

1. Install dependencies and the local package.
2. Optionally generate a fixed stain reference image.
3. Run Optuna tuning with `bash scripts/run_tuning.sh`.
4. Run the final multi-seed benchmark with `bash scripts/run_final_benchmark.sh`.
5. Inspect `summary_per_seed.csv`, `summary_per_model.csv`, and the saved test
   metrics under `experiments/`.

If you only want to inspect the results used for the student paper, use the
existing artifacts under `experiments/` instead of rerunning the full pipeline.

## Reproducibility Notes

- Fixed seeds are used for tuning and benchmark runs.
- Final checkpoints are selected by validation AUPRC.
- Benchmark summaries are generated from saved JSON artifacts rather than from
  ad hoc manual reporting.
- Stain normalization can use a fixed reference image via
  `experiments/stain_refs/references.json`.

## Optional Linting

Linting is not required to reproduce the paper results, but it is useful for
keeping the repository tidy.

```bash
pip install -r requirements-dev.txt
ruff check .
```
