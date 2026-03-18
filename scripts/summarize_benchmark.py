#!/usr/bin/env python3
"""Aggregate multi-seed benchmark runs into per-seed and per-model summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

MODEL_NAMES = ("small_cnn", "resnet_frozen", "resnet_partial")


def _safe_float(value) -> float:
    """Parse values to float, returning NaN for missing/invalid entries."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: list[float]) -> float:
    """Mean that ignores NaNs; returns NaN if all values are NaN."""
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return float("nan")
    return float(mean(filtered))


def _nanstd(values: list[float]) -> float:
    """Sample std that ignores NaNs and handles small sample sizes safely."""
    filtered = [v for v in values if not math.isnan(v)]
    if len(filtered) < 2:
        return 0.0 if len(filtered) == 1 else float("nan")
    return float(stdev(filtered))


def _load_json(path: Path) -> dict:
    """Load json payload from path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _required_artifacts(model: str, seed_dir: Path) -> tuple[Path, Path, Path] | None:
    """Return required artifact paths for one model/seed run."""
    if model == "small_cnn":
        return (
            seed_dir / "hist.json",
            seed_dir / "train_config.json",
            seed_dir / "test_metrics.json",
        )
    elif model == "resnet_frozen":
        return (
            seed_dir / "resnet18_frozen_hist.json",
            seed_dir / "resnet18_frozen_train_config.json",
            seed_dir / "resnet18_frozen_test_metrics.json",
        )
    elif model == "resnet_partial":
        return (
            seed_dir / "resnet18_partial_hist.json",
            seed_dir / "resnet18_partial_train_config.json",
            seed_dir / "resnet18_partial_test_metrics.json",
        )
    return None


def _summarize_single_run(model: str, seed_dir: Path) -> dict:
    """Extract one run's key validation/test metrics from artifact files."""
    required = _required_artifacts(model, seed_dir)
    if required is None:
        raise ValueError(f"Unsupported model: {model}")

    hist_path, cfg_path, test_path = required

    hist = _load_json(hist_path)
    cfg = _load_json(cfg_path)
    test = _load_json(test_path)
    if not isinstance(hist, list) or not hist:
        raise ValueError(f"Invalid or empty history in {hist_path}")

    best_by_auprc = max(hist, key=lambda x: float(x.get("val_auprc", float("-inf"))))
    final = hist[-1]

    return {
        "model": model,
        "seed": int(cfg.get("seed", -1)),
        "epochs_ran": len(hist),
        "best_epoch_by_val_auprc": int(best_by_auprc.get("epoch", -1)),
        "best_val_loss": _safe_float(best_by_auprc.get("val_loss")),
        "best_val_auroc": _safe_float(best_by_auprc.get("val_auroc")),
        "best_val_auprc": _safe_float(best_by_auprc.get("val_auprc")),
        "best_val_f1": _safe_float(best_by_auprc.get("val_f1")),
        "final_val_loss": _safe_float(final.get("val_loss")),
        "final_val_auroc": _safe_float(final.get("val_auroc")),
        "final_val_auprc": _safe_float(final.get("val_auprc")),
        "final_val_f1": _safe_float(final.get("val_f1")),
        "test_loss": _safe_float(test.get("test_loss")),
        "test_auroc": _safe_float(test.get("test_auroc")),
        "test_auprc": _safe_float(test.get("test_auprc")),
        "test_f1": _safe_float(test.get("test_f1")),
        "stain_normalization": cfg.get("stain_normalization"),
        "stain_reference_image": cfg.get("stain_reference_image"),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    """Compute per-model mean/std statistics across seeds."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    metrics = [
        "best_val_auroc",
        "best_val_auprc",
        "best_val_f1",
        "test_auroc",
        "test_auprc",
        "test_f1",
    ]

    out: list[dict] = []
    for model, items in sorted(grouped.items()):
        agg: dict[str, float | str | int] = {"model": model, "num_seeds": len(items)}
        for metric in metrics:
            values = [_safe_float(it.get(metric)) for it in items]
            agg[f"{metric}_mean"] = _nanmean(values)
            agg[f"{metric}_std"] = _nanstd(values)
        out.append(agg)
    return out


def main() -> None:
    """CLI entry point for benchmark aggregation."""
    parser = argparse.ArgumentParser(
        description="Summarize final benchmark runs into CSV/JSON."
    )
    parser.add_argument("--input-dir", default="experiments/final_benchmark")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--expected-seeds",
        default=None,
        help='Optional explicit seed list, e.g. "42 52 62 72 82".',
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if expected run directories or required artifacts are missing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_dir / "summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    expected_seeds: list[int] | None = None
    if args.expected_seeds:
        expected_seeds = [int(token) for token in args.expected_seeds.split() if token.strip()]

    rows: list[dict] = []
    missing_messages: list[str] = []
    for model in MODEL_NAMES:
        model_dir = input_dir / model
        seed_dirs: list[Path]
        if expected_seeds is not None:
            seed_dirs = [model_dir / f"seed_{seed}" for seed in expected_seeds]
        else:
            if not model_dir.exists():
                if args.strict:
                    missing_messages.append(f"Missing model directory: {model_dir}")
                continue
            seed_dirs = sorted(model_dir.glob("seed_*"))

        for seed_dir in seed_dirs:
            required = _required_artifacts(model, seed_dir)
            if required is None:
                continue

            if not seed_dir.exists():
                missing_messages.append(f"Missing run directory: {seed_dir}")
                continue

            missing_files = [path for path in required if not path.exists()]
            if missing_files:
                missing_names = ", ".join(str(path.name) for path in missing_files)
                missing_messages.append(
                    f"Incomplete run at {seed_dir}; missing: {missing_names}"
                )
                continue

            try:
                row = _summarize_single_run(model, seed_dir)
            except Exception as exc:
                missing_messages.append(f"Failed to parse run at {seed_dir}: {exc}")
                continue
            else:
                rows.append(row)

    if missing_messages and args.strict:
        lines = "\n".join(f"- {msg}" for msg in missing_messages[:50])
        suffix = (
            ""
            if len(missing_messages) <= 50
            else f"\n... and {len(missing_messages) - 50} more"
        )
        raise SystemExit(
            "Benchmark summary aborted due to missing/incomplete runs:\n"
            f"{lines}{suffix}"
        )
    if missing_messages and not args.strict:
        print("WARNING: Some runs were missing/incomplete and skipped:")
        for msg in missing_messages[:50]:
            print(f"- {msg}")
        if len(missing_messages) > 50:
            print(f"... and {len(missing_messages) - 50} more")

    per_seed_csv = output_prefix.with_name(f"{output_prefix.name}_per_seed.csv")
    per_model_csv = output_prefix.with_name(f"{output_prefix.name}_per_model.csv")
    summary_json = output_prefix.with_suffix(".json")

    if rows:
        fieldnames = list(rows[0].keys())
        with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    aggregates = _aggregate(rows)
    if aggregates:
        agg_fields = list(aggregates[0].keys())
        with per_model_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields)
            writer.writeheader()
            writer.writerows(aggregates)

    payload = {
        "input_dir": str(input_dir),
        "num_runs": len(rows),
        "per_seed": rows,
        "per_model": aggregates,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {summary_json}")
    if rows:
        print(f"Wrote: {per_seed_csv}")
    if aggregates:
        print(f"Wrote: {per_model_csv}")


if __name__ == "__main__":
    main()
