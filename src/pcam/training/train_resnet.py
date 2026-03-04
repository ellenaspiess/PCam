from __future__ import annotations
"""Training entry point for ResNet18 transfer-learning variants on PCam."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from pcam.preprocessing.dataloaders import get_pcam_dataloaders
from pcam.preprocessing.datasets import resolve_reference_image_path
from pcam.models.resnet import ResNetConfig, ResNetPCam
from pcam.training.evaluation import (
    collect_labels_and_scores,
    compute_binary_curve_data,
    compute_binary_metrics,
    count_trainable_parameters,
    evaluate_binary_classifier,
    save_binary_curve_plot,
)
from pcam.training.train_utils import (
    build_scheduler,
    load_optuna_best_params,
    set_global_seed,
    should_stop_early,
)


def train_resnet(
    data_root: str | Path = "data/raw",
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler: str = "cosine",
    dropout_p: float = 0.1,
    tl_mode: str = "partial",
    num_workers: int = 0,
    stain_normalization: str = "macenko",
    stain_reference_image: str | Path | None = None,
    output_dir: str | Path = "experiments/runs",
    save_every: int = 5,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 1e-4,
    limit_per_split: int | None = None,
    seed: int = 42,
) -> Path:
    """Train ResNet18 transfer-learning variant and persist all run artifacts.

    The final exported model is the checkpoint with best validation AUPRC.
    """
    device = torch.device("cpu")
    print("Using device:", device)
    set_global_seed(seed)
    resolved_reference_image = resolve_reference_image_path(stain_normalization, stain_reference_image)

    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64,
        num_workers=num_workers,
        stain_normalization=stain_normalization,
        stain_reference_image=resolved_reference_image,
        limit_per_split=limit_per_split,
    )

    config = ResNetConfig(tl_mode=tl_mode, pretrained=True, dropout_p=dropout_p)
    model = ResNetPCam(config).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"TL mode: {tl_mode} | trainable params: {count_trainable_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    lr_scheduler = build_scheduler(optimizer, scheduler, num_epochs)

    best_val_auprc = float("-inf")
    early_stop_best = float("-inf")
    best_state_auprc = None
    no_improve_epochs = 0
    history: list[dict[str, Any]] = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_labels = []
        train_probs = []

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_labels.append(labels.detach().cpu())
            train_probs.append(torch.sigmoid(logits.detach()).cpu())

        train_loss /= len(loaders["train"].dataset)
        train_y_true = torch.cat(train_labels).numpy()
        train_y_score = torch.cat(train_probs).numpy()
        train_auroc, train_auprc, train_f1 = compute_binary_metrics(train_y_true, train_y_score)

        val_loss, val_auroc, val_auprc, val_f1 = evaluate_binary_classifier(
            model=model,
            dataloader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        if lr_scheduler is not None:
            if scheduler == "plateau":
                lr_scheduler.step(val_auprc)
            else:
                lr_scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auroc": train_auroc,
                "train_auprc": train_auprc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state_auprc = model.state_dict()
            torch.save(best_state_auprc, output_dir / f"resnet18_{tl_mode}_best_by_auprc.pt")

        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "val_auroc": float(val_auroc),
                    "val_auprc": float(val_auprc),
                    "val_f1": float(val_f1),
                },
                output_dir / f"resnet18_{tl_mode}_ckpt_epoch{epoch}.pt",
            )

        print(
            f"[ResNet-{tl_mode}] Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"train_AUROC={train_auroc:.3f} | train_AUPRC={train_auprc:.3f} | "
            f"train_F1={train_f1:.3f} | val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.3f} | val_AUPRC={val_auprc:.3f} | "
            f"val_F1={val_f1:.3f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        stop, early_stop_best, no_improve_epochs = should_stop_early(
            current_value=val_auprc,
            best_value=early_stop_best,
            no_improve_epochs=no_improve_epochs,
            mode="max",
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
        )
        if stop:
            print(f"Early stopping at epoch {epoch} (patience={early_stopping_patience}).")
            break

    out_path = output_dir / f"resnet18_{tl_mode}_final.pt"
    if best_state_auprc is not None:
        # Keep final checkpoint aligned with the model-selection criterion.
        model.load_state_dict(best_state_auprc)
        torch.save(best_state_auprc, out_path)
    else:
        torch.save(model.state_dict(), out_path)

    test_loss, test_auroc, test_auprc, test_f1 = evaluate_binary_classifier(model, loaders["test"], criterion, device)
    test_metrics = {
        "test_loss": float(test_loss),
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "test_f1": float(test_f1),
    }
    with (output_dir / f"resnet18_{tl_mode}_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    test_y_true, test_y_score = collect_labels_and_scores(model=model, dataloader=loaders["test"], device=device)
    curve_data = compute_binary_curve_data(test_y_true, test_y_score)
    with (output_dir / f"resnet18_{tl_mode}_test_curves.json").open("w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)
    np.savez_compressed(
        output_dir / f"resnet18_{tl_mode}_test_predictions.npz",
        y_true=test_y_true,
        y_score=test_y_score,
    )
    save_binary_curve_plot(curve_data, output_dir / f"resnet18_{tl_mode}_test_curves.png", title_prefix=f"ResNet-{tl_mode} Test")

    print(
        f"[ResNet-{tl_mode}] Test metrics | "
        f"loss={test_loss:.4f} | AUROC={test_auroc:.3f} | AUPRC={test_auprc:.3f} | F1={test_f1:.3f}"
    )

    train_config = {
        "data_root": str(data_root),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "dropout_p": dropout_p,
        "tl_mode": tl_mode,
        "num_workers": num_workers,
        "stain_normalization": stain_normalization,
        "stain_reference_image": str(resolved_reference_image) if resolved_reference_image else None,
        "save_every": save_every,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "limit_per_split": limit_per_split,
        "seed": seed,
    }

    with (output_dir / f"resnet18_{tl_mode}_hist.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / f"resnet18_{tl_mode}_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    print("Saved model to", out_path)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for ResNet training and runtime options."""
    parser = argparse.ArgumentParser(description="Train ResNet18 on PCam (CPU-only).")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--optuna-best-json", default=None)

    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default=None)
    parser.add_argument("--dropout-p", type=float, default=None)
    parser.add_argument("--tl-mode", choices=["frozen", "partial"], default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--stain-normalization",
        choices=["macenko", "reinhard", "none"],
        default="macenko",
    )
    parser.add_argument("--stain-reference-image", default=None)
    parser.add_argument("--output-dir", default="experiments/runs")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_train_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge defaults, optional Optuna params, and explicit CLI overrides."""
    config: dict[str, Any] = {
        "num_epochs": 100,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "dropout_p": 0.1,
        "tl_mode": "partial",
        "save_every": 5,
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 1e-4,
        "seed": 42,
    }

    if args.optuna_best_json:
        config.update(load_optuna_best_params(args.optuna_best_json))

    for key in (
        "num_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "scheduler",
        "dropout_p",
        "tl_mode",
        "save_every",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    return config


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    cfg = _resolve_train_config(args)

    train_resnet(
        data_root=args.data_root,
        num_epochs=int(cfg["num_epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        scheduler=str(cfg["scheduler"]),
        dropout_p=float(cfg["dropout_p"]),
        tl_mode=str(cfg["tl_mode"]),
        num_workers=args.num_workers,
        stain_normalization=args.stain_normalization,
        stain_reference_image=args.stain_reference_image,
        output_dir=args.output_dir,
        save_every=int(cfg["save_every"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
        limit_per_split=args.limit_per_split,
        seed=int(cfg["seed"]),
    )


if __name__ == "__main__":
    main()
