# src/training/train_resnet_pcam.py
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim

from src.datasets.dataloaders import get_pcam_dataloaders
from src.models.resnet_pcam import ResNetPCam, ResNetConfig
from src.training.utils_training import evaluate_binary_classifier, count_trainable_parameters


def train_resnet(
    data_root: str | Path = "data/raw",
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    tl_mode: str = "partial",  # "frozen", "partial", "full"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataloader – das wird erst funktionieren, wenn PCam geladen werden kann
    dataloaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64,
    )

    config = ResNetConfig(tl_mode=tl_mode, pretrained=True)
    model = ResNetPCam(config).to(device)

    print(f"Transfer Learning mode: {tl_mode}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )

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
        val_loss, val_auroc, val_auprc = evaluate_binary_classifier(
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
    out_path = Path("experiments/runs") / f"resnet18_{tl_mode}_first_run.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    train_resnet()
