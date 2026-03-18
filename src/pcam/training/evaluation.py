"""Evaluation helpers for binary PCam classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute AUROC, AUPRC, and F1 from labels and probabilities.

    For edge cases where a metric is undefined (e.g. single-class labels), the
    function returns ``NaN`` for that metric instead of raising.
    """
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_score))
    except ValueError:
        auprc = float("nan")

    y_pred = (y_score >= threshold).astype(np.int64)
    try:
        f1 = float(f1_score(y_true, y_pred))
    except ValueError:
        f1 = float("nan")

    return auroc, auprc, f1


@torch.no_grad()
def evaluate_binary_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate binary classifier and return loss, AUROC, AUPRC, and F1."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()

    auroc, auprc, f1 = compute_binary_metrics(y_true, y_score)
    return total_loss, auroc, auprc, f1


def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def collect_labels_and_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ground-truth labels and prediction scores for a full split."""
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_labels.append(labels.float().cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    return y_true, y_score


def compute_binary_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """Compute ROC and PR curve points with safe fallbacks."""
    curve_data: dict[str, list[float]] = {
        "roc_fpr": [],
        "roc_tpr": [],
        "roc_thresholds": [],
        "pr_precision": [],
        "pr_recall": [],
        "pr_thresholds": [],
    }

    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        curve_data["roc_fpr"] = fpr.tolist()
        curve_data["roc_tpr"] = tpr.tolist()
        curve_data["roc_thresholds"] = roc_thresholds.tolist()
    except ValueError:
        pass

    try:
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        curve_data["pr_precision"] = precision.tolist()
        curve_data["pr_recall"] = recall.tolist()
        curve_data["pr_thresholds"] = pr_thresholds.tolist()
    except ValueError:
        pass

    return curve_data


def save_binary_curve_plot(
    curve_data: dict[str, list[float]],
    output_path: str | Path,
    title_prefix: str = "Model",
) -> None:
    """Save side-by-side ROC and PR curves as a PNG image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    roc_fpr = curve_data.get("roc_fpr", [])
    roc_tpr = curve_data.get("roc_tpr", [])
    if roc_fpr and roc_tpr:
        axes[0].plot(roc_fpr, roc_tpr, label="ROC")
        axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Chance")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"{title_prefix} ROC")
        axes[0].legend()
    else:
        axes[0].set_title(f"{title_prefix} ROC")
        axes[0].text(0.5, 0.5, "ROC unavailable", ha="center", va="center")

    pr_recall = curve_data.get("pr_recall", [])
    pr_precision = curve_data.get("pr_precision", [])
    if pr_recall and pr_precision:
        axes[1].plot(pr_recall, pr_precision, label="PR")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"{title_prefix} PR")
        axes[1].legend()
    else:
        axes[1].set_title(f"{title_prefix} PR")
        axes[1].text(0.5, 0.5, "PR unavailable", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
