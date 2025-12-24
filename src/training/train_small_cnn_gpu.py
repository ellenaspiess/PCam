# src/training/train_small_cnn.py
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score, average_precision_score

# Deine Imports
from src.datasets.dataloaders import get_pcam_dataloaders
from src.models.small_cnn_gpu import SmallCNN

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate(model, dataloader, criterion, device):
    """Evaluiert das Modell auf der GPU und berechnet Metriken auf CPU."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Ab auf die GPU
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            # Sigmoid für Wahrscheinlichkeiten
            probs = torch.sigmoid(logits)
            
            # Für Scikit-Learn zurück auf CPU holen
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss /= len(dataloader.dataset)
    
    # Listen zusammenfügen
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)

    # Metriken berechnen
    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        # Fallback, falls z.B. nur eine Klasse im Batch wäre (unwahrscheinlich bei Val)
        auroc = 0.5
        auprc = 0.0

    return total_loss, auroc, auprc

def train_small_cnn(
    data_root: str | Path = "data/raw",
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    run_name: str = "small_cnn_v1"
) -> None:
    
    # 1. Setup
    device = _get_device()
    print(f"🚀 Training SmallCNN auf: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 2. Daten laden (GPU-Optimiert)
    workers = 4 if device.type == 'cuda' else 0
    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64, # Wichtig laut Feedback!
        num_workers=workers,
        pin_memory=(device.type == 'cuda')
    )

    # 3. Modell
    model = SmallCNN(dropout_p=0.5)
    model = model.to(device) # Modell auf GPU

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = 0.0
    
    # 4. Loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for images, labels in loaders["train"]:
            # Daten auf GPU
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Auswertung
        train_loss = running_loss / len(loaders["train"].dataset)
        val_loss, val_auroc, val_auprc = evaluate(model, loaders["val"], criterion, device)
        
        duration = time.time() - start_time

        print(
            f"[Epoch {epoch:02d}/{num_epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | "
            f"AUPRC: {val_auprc:.4f} | "
            f"Time: {duration:.0f}s"
        )

        # Speichern (Bestes Modell nach AUPRC - wichtig bei Imbalance!)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            save_path = Path("experiments/models")
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / f"{run_name}_best.pth")

    print(f"✅ Training fertig. Bestes AUPRC: {best_val_auprc:.4f}")