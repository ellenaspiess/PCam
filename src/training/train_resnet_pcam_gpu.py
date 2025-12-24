# src/training/train_resnet_pcam.py
from __future__ import annotations
from pathlib import Path
import time

import torch
from torch import nn, optim

# Stelle sicher, dass diese Imports funktionieren (Dateipfade müssen stimmen)
from src.datasets.dataloaders import get_pcam_dataloaders
from src.models.resnet_pcam_gpu import ResNetPCam, ResNetConfig
from src.training.utils_training import evaluate_binary_classifier

def _get_device() -> torch.device:
    """Wählt automatisch die GPU (CUDA/MPS) oder CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Für Mac M1/M2/M3 Support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_resnet(
    data_root: str | Path = "data/raw",
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    tl_mode: str = "partial", # "frozen", "partial", "full"
    run_name: str = "resnet_experiment"
) -> None:
    
    # 1. Setup Device & Performance Tuning
    device = _get_device()
    print(f"🚀 Training startet auf: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Beschleunigt bei fixer Input-Size

    # 2. Dataloader holen (GPU-Optimiert!)
    # Falls du auf Windows bist und Fehler kriegst, setze num_workers=0
    workers = 4 if device.type == 'cuda' else 0 
    
    loaders = get_pcam_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        center_crop_size=64, # Center Crop Trap vermeiden!
        num_workers=workers,
        pin_memory=(device.type == 'cuda') # Wichtig für GPU Speed
    )

    # 3. Modell & Config initialisieren
    config = ResNetConfig(tl_mode=tl_mode, pretrained=True)
    model = ResNetPCam(config)
    model = model.to(device) # Ab auf die GPU

    # Nur die parameter dem Optimizer geben, die requires_grad=True haben
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Modell: ResNet18 ({tl_mode}) | Trainable Params: {len(trainable_params)}")

    # 4. Training Loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Iterate über Training Batches
        for images, labels in loaders["train"]:
            # Daten auf GPU schieben
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward & Backward
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Epoch-Ende Berechnungen
        epoch_loss = running_loss / len(loaders["train"].dataset)
        
        # Validierung (Auch auf GPU)
        val_loss, val_auroc, val_auprc = evaluate_binary_classifier(
            model, loaders["val"], criterion, device
        )

        duration = time.time() - start_time
        print(
            f"[Epoch {epoch:02d}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | "
            f"AUPRC: {val_auprc:.4f} | "
            f"Zeit: {duration:.0f}s"
        )

    # 5. Speichern
    save_path = Path("experiments/models")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"{run_name}_{tl_mode}.pth")
    print(f"✅ Modell gespeichert unter {save_path}")