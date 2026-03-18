"""Optuna-based hyperparameter search for PCam models (CPU-oriented)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from pcam.models.resnet import ResNetConfig, ResNetPCam
from pcam.models.small_cnn import SmallCNN
from pcam.preprocessing.dataloaders import get_pcam_dataloaders
from pcam.training.evaluation import evaluate_binary_classifier

SearchMode = str


def _set_global_seed(seed: int) -> None:
    """Set random seeds used inside each trial."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device from user selection."""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def _train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss for the split."""
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def _build_scheduler(optimizer: optim.Optimizer, scheduler_name: str, num_epochs: int):
    """Create a scheduler for Optuna trial training."""
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    if scheduler_name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _sample_small_cnn_params(trial: optuna.Trial, search_mode: SearchMode) -> dict[str, Any]:
    """Sample SmallCNN hyperparameters for broad or narrow search stages."""
    if search_mode == "narrow":
        return {
            "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-6, 3e-4, log=True),
            "dropout_p": trial.suggest_float("dropout_p", 0.05, 0.35),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "scheduler_name": trial.suggest_categorical("scheduler", ["cosine", "plateau"]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 4),
        }

    if search_mode == "broad":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.6),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "scheduler_name": trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 6),
        }

    raise ValueError(f"Unsupported search_mode: {search_mode}")


def _sample_resnet_params(trial: optuna.Trial, search_mode: SearchMode) -> dict[str, Any]:
    """Sample ResNet hyperparameters for broad or narrow search stages."""
    if search_mode == "narrow":
        return {
            "lr": trial.suggest_float("lr", 2e-5, 8e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-6, 3e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.3),
            "scheduler_name": trial.suggest_categorical("scheduler", ["cosine", "plateau"]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 4),
        }

    if search_mode == "broad":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.5),
            "scheduler_name": trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 6),
        }

    raise ValueError(f"Unsupported search_mode: {search_mode}")


def _small_cnn_objective(
    trial: optuna.Trial,
    data_root: str,
    num_epochs: int,
    num_workers: int,
    stain_normalization: str,
    stain_reference_image: str | Path | None,
    limit_per_split: int | None,
    search_mode: SearchMode,
    base_seed: int,
    device: torch.device,
) -> float:
    """Optuna objective for SmallCNN maximizing validation AUPRC."""
    _set_global_seed(base_seed + trial.number)
    params = _sample_small_cnn_params(trial, search_mode)

    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=params["batch_size"],
        center_crop_size=64,
        num_workers=num_workers,
        stain_normalization=stain_normalization,
        stain_reference_image=stain_reference_image,
        limit_per_split=limit_per_split,
    )

    model = SmallCNN(dropout_p=params["dropout_p"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = _build_scheduler(optimizer, params["scheduler_name"], num_epochs)

    best_val_auprc = float("-inf")
    best_epoch = 0
    no_improve_epochs = 0
    min_delta = 1e-4

    for epoch in range(1, num_epochs + 1):
        _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        _, val_auroc, val_auprc, val_f1 = evaluate_binary_classifier(
            model,
            loaders["val"],
            criterion,
            device,
        )

        trial.report(val_auprc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_auprc > best_val_auprc + min_delta:
            best_val_auprc = val_auprc
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if scheduler is not None:
            if params["scheduler_name"] == "plateau":
                scheduler.step(val_auprc)
            else:
                scheduler.step()

        trial.set_user_attr("latest_val_auroc", float(val_auroc))
        trial.set_user_attr("latest_val_f1", float(val_f1))
        trial.set_user_attr("best_epoch", int(best_epoch))

        if no_improve_epochs >= params["early_stopping_patience"]:
            break

    return float(best_val_auprc)


def _resnet_objective(
    trial: optuna.Trial,
    data_root: str,
    num_epochs: int,
    num_workers: int,
    stain_normalization: str,
    stain_reference_image: str | Path | None,
    limit_per_split: int | None,
    search_mode: SearchMode,
    base_seed: int,
    tl_mode: str,
    device: torch.device,
) -> float:
    """Optuna objective for ResNet transfer-learning variants."""
    _set_global_seed(base_seed + trial.number)
    params = _sample_resnet_params(trial, search_mode)

    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=params["batch_size"],
        center_crop_size=64,
        num_workers=num_workers,
        stain_normalization=stain_normalization,
        stain_reference_image=stain_reference_image,
        limit_per_split=limit_per_split,
    )

    config = ResNetConfig(tl_mode=tl_mode, pretrained=True, dropout_p=params["dropout_p"])
    model = ResNetPCam(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = _build_scheduler(optimizer, params["scheduler_name"], num_epochs)

    best_val_auprc = float("-inf")
    best_epoch = 0
    no_improve_epochs = 0
    min_delta = 1e-4

    for epoch in range(1, num_epochs + 1):
        _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        _, val_auroc, val_auprc, val_f1 = evaluate_binary_classifier(
            model,
            loaders["val"],
            criterion,
            device,
        )

        trial.report(val_auprc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_auprc > best_val_auprc + min_delta:
            best_val_auprc = val_auprc
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if scheduler is not None:
            if params["scheduler_name"] == "plateau":
                scheduler.step(val_auprc)
            else:
                scheduler.step()

        trial.set_user_attr("latest_val_auroc", float(val_auroc))
        trial.set_user_attr("latest_val_f1", float(val_f1))
        trial.set_user_attr("best_epoch", int(best_epoch))

        if no_improve_epochs >= params["early_stopping_patience"]:
            break

    return float(best_val_auprc)


def run_optuna_search(
    model_name: str,
    search_mode: SearchMode = "broad",
    tl_mode: str | None = None,
    data_root: str = "data/raw",
    num_trials: int = 25,
    num_epochs: int = 3,
    num_workers: int = 0,
    stain_normalization: str = "macenko",
    stain_reference_image: str | Path | None = None,
    limit_per_split: int | None = None,
    output_dir: str | Path = "experiments/optuna",
    study_name: str | None = None,
    base_seed: int = 42,
    n_jobs: int = 1,
    timeout_seconds: int | None = None,
    tpe_startup_trials: int = 10,
    pruner_startup_trials: int = 5,
    pruner_warmup_steps: int = 1,
    save_top_k: int = 10,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run Optuna search for ``small_cnn`` or ``resnet`` and persist artifacts.

    Saved outputs include:
    - best-params summary JSON
    - full trials JSON
    - top-K completed trials JSON
    - full config JSON
    - Optuna SQLite study database
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "resnet" and tl_mode not in {"frozen", "partial"}:
        raise ValueError("For model_name='resnet', --tl-mode must be one of: frozen, partial")
    if model_name != "resnet":
        tl_mode = None

    if study_name is None:
        study_name = (
            f"{model_name}_{search_mode}_search"
            if tl_mode is None
            else f"{model_name}_{tl_mode}_{search_mode}_search"
        )

    device = _resolve_device(device_name)
    print(f"Using device: {device}")

    storage = f"sqlite:///{(output_dir / f'{study_name}.db').resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=base_seed,
        n_startup_trials=tpe_startup_trials,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_startup_trials,
        n_warmup_steps=pruner_warmup_steps,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    if model_name == "small_cnn":
        def objective(trial: optuna.Trial) -> float:
            return _small_cnn_objective(
                trial=trial,
                data_root=data_root,
                num_epochs=num_epochs,
                num_workers=num_workers,
                stain_normalization=stain_normalization,
                stain_reference_image=stain_reference_image,
                limit_per_split=limit_per_split,
                search_mode=search_mode,
                base_seed=base_seed,
                device=device,
            )

    else:
        def objective(trial: optuna.Trial) -> float:
            return _resnet_objective(
                trial=trial,
                data_root=data_root,
                num_epochs=num_epochs,
                num_workers=num_workers,
                stain_normalization=stain_normalization,
                stain_reference_image=stain_reference_image,
                limit_per_split=limit_per_split,
                search_mode=search_mode,
                base_seed=base_seed,
                tl_mode=tl_mode,
                device=device,
            )

    study.optimize(
        objective,
        n_trials=num_trials,
        timeout=timeout_seconds,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )

    result = {
        "study_name": study.study_name,
        "model_name": model_name,
        "tl_mode": tl_mode,
        "search_mode": search_mode,
        "best_value_val_auprc": float(study.best_value),
        "best_params": study.best_trial.params,
        "best_trial_number": int(study.best_trial.number),
        "n_trials_total": len(study.trials),
    }

    with (output_dir / f"{study_name}_best_params.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    trials_summary = [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    with (output_dir / f"{study_name}_trials.json").open("w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(complete_trials, key=lambda t: float(t.value), reverse=True)[:save_top_k]
    top_k_summary = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        }
        for t in top_trials
    ]
    with (output_dir / f"{study_name}_top{save_top_k}.json").open("w", encoding="utf-8") as f:
        json.dump(top_k_summary, f, indent=2)

    search_config = {
        "model_name": model_name,
        "search_mode": search_mode,
        "tl_mode": tl_mode,
        "data_root": data_root,
        "num_trials": num_trials,
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "stain_normalization": stain_normalization,
        "stain_reference_image": str(stain_reference_image) if stain_reference_image else None,
        "limit_per_split": limit_per_split,
        "base_seed": base_seed,
        "n_jobs": n_jobs,
        "timeout_seconds": timeout_seconds,
        "tpe_startup_trials": tpe_startup_trials,
        "pruner_startup_trials": pruner_startup_trials,
        "pruner_warmup_steps": pruner_warmup_steps,
        "save_top_k": save_top_k,
        "device": str(device),
        "storage": storage,
        "study_name": study_name,
    }
    with (output_dir / f"{study_name}_config.json").open("w", encoding="utf-8") as f:
        json.dump(search_config, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Optuna search configuration."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for PCam models."
    )
    parser.add_argument("--model", choices=["small_cnn", "resnet"], required=True)
    parser.add_argument("--tl-mode", choices=["frozen", "partial"], default=None)
    parser.add_argument("--search-mode", choices=["broad", "narrow"], default="broad")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="If omitted, defaults to 3 for broad and 8 for narrow.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--stain-normalization",
        choices=["macenko", "reinhard", "none"],
        default="macenko",
    )
    parser.add_argument("--stain-reference-image", default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments/optuna")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--tpe-startup-trials", type=int, default=10)
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=int, default=1)
    parser.add_argument("--save-top-k", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    default_epochs = 8 if args.search_mode == "narrow" else 3
    run_optuna_search(
        model_name=args.model,
        search_mode=args.search_mode,
        tl_mode=args.tl_mode,
        data_root=args.data_root,
        num_trials=args.num_trials,
        num_epochs=args.num_epochs if args.num_epochs is not None else default_epochs,
        num_workers=args.num_workers,
        stain_normalization=args.stain_normalization,
        stain_reference_image=args.stain_reference_image,
        limit_per_split=args.limit_per_split,
        output_dir=args.output_dir,
        study_name=args.study_name,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
        timeout_seconds=args.timeout_seconds,
        tpe_startup_trials=args.tpe_startup_trials,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        save_top_k=args.save_top_k,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
