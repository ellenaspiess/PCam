# src/training/train_small_cnn.py
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn, optim

from src.datasets.dataloaders import get_pcam_dataloaders
from src.models.small_cnn import SmallCNN


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
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


def train(
    data_root: str | Path = "data/raw",
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    limit_per_split: int | None = None,
):
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    dataloaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64,
        num_workers=0,   # <--- hier explizit
        limit_per_split=limit_per_split,
    )

    model = SmallCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in dataloaders["train"]:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(dataloaders["train"].dataset)
        val_loss, val_auroc, val_auprc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.3f} | "
            f"val_AUPRC={val_auprc:.3f}"
        )

    Path("experiments/runs").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "experiments/runs/small_cnn_first_run.pt")
    print("Model saved to experiments/runs/small_cnn_first_run.pt")


if __name__ == "__main__":
    train()
