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
    dropout_p: float = 0.1,
    num_workers: int = 4,
    use_amp: bool = True,
    ckpt_dir: str | Path = "experiments/runs",
    save_every: int = 1,
):
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    # GPU-specific performance hints
    if device.type == "cuda":
        # Use cuDNN autotuner for best performance on fixed-size inputs
        torch.backends.cudnn.benchmark = True

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64,
        num_workers=num_workers,
        limit_per_split=limit_per_split,
    )

    model = SmallCNN(dropout_p=dropout_p).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # track best model by AUROC (preferred default) and by val-loss as a secondary artifact
    best_val_loss = float("inf")
    best_state_val = None
    best_val_auroc = float("-inf")
    best_state_auroc = None

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in dataloaders["train"]:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(dataloaders["train"].dataset)
        val_loss, val_auroc, val_auprc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_auroc=val_auroc,
                val_auprc=val_auprc,
                lr=optimizer.param_groups[0]["lr"],
            )
        )

        # update best by val-loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_val = model.state_dict()
            torch.save(best_state_val, ckpt_dir / "small_cnn_best_by_val_loss.pt")

        # update best by AUROC
        if val_auroc is not None and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state_auroc = model.state_dict()
            torch.save(best_state_auroc, ckpt_dir / "small_cnn_best_by_auroc.pt")

        # periodic checkpoint
        if epoch % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "val_loss": float(val_loss),
                "val_auroc": float(val_auroc),
                "val_auprc": float(val_auprc),
            }
            torch.save(ckpt, ckpt_dir / f"small_cnn_ckpt_epoch{epoch}.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AUROC={val_auroc:.3f} | "
            f"val_AUPRC={val_auprc:.3f}"
        )

    # Save preferred model: best-by-AUROC as the default final model
    if best_state_auroc is not None:
        final_path = ckpt_dir / "small_cnn_final.pt"
        torch.save(best_state_auroc, final_path)
        print(f"Saved best-by-AUROC model to {final_path} (preferred default)")
    else:
        # fallback: save current model state if no AUROC tracked
        torch.save(model.state_dict(), ckpt_dir / "small_cnn_final.pt")
        print("Saved current model state as final (no AUROC-best found)")

    # Also save best-by-val-loss state for reproducibility/inspection
    if best_state_val is not None:
        val_path = ckpt_dir / "small_cnn_best_by_val_loss.pt"
        torch.save(best_state_val, val_path)
        print(f"Saved best-by-val-loss model to {val_path}")

    # Save training history
    with open(ckpt_dir / "hist.json", "w") as f:
        import json

        json.dump(history, f, indent=2)

    return history, best_state_val, best_state_auroc


if __name__ == "__main__":
    train()
