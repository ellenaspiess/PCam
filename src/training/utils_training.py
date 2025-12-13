# src/training/utils_training.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_binary_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Eval für binäre Klassifikation:
    - Loss
    - AUROC
    - AUPRC (Average Precision)
    """
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

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return total_loss, auroc, auprc


def count_trainable_parameters(model: nn.Module) -> int:
    """Hilfsfunktion, um die Anzahl trainierbarer Parameter zu inspizieren."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
