# src/training/train_resnet_pcam.py
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim

from src.datasets.dataloaders import get_pcam_dataloaders
from src.models.resnet_pcam import ResNetPCam, ResNetConfig
from src.training.utils_training import evaluate_binary_classifier


def _get_device() -> torch.device:
    """Select the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_resnet(
    data_root: str | Path = "data/raw",
    num_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    tl_mode: str = "partial",  # "frozen", "partial", "full"
) -> None:
    device = _get_device()
    print("Using device:", device)

    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64,
        num_workers=0,  # safer for macOS
    )

    config = ResNetConfig(tl_mode=tl_mode, pretrained=True)
    model = ResNetPCam(config).to(device)

    # Only pass trainable parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"TL mode: {tl_mode} | "
        f"trainable params: {sum(p.numel() for p in trainable_params):,}"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(trainable_params, lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(loaders["train"].dataset)

        val_loss, val_auroc, val_auprc = evaluate_binary_classifier(
            model, loaders["val"], criterion, device
        )

        print(
            f"[ResNet-{tl_mode}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"AUROC={val_auroc:.3f} | "
            f"AUPRC={val_auprc:.3f}"
        )

    # Save model checkpoint
    Path("experiments/runs").mkdir(parents=True, exist_ok=True)
    out_path = Path("experiments/runs") / f"resnet18_{tl_mode}_local.pt"
    torch.save(model.state_dict(), out_path)
    print("Saved model to", out_path)


if __name__ == "__main__":
    train_resnet()
