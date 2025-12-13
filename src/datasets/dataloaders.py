# src/datasets/dataloaders.py
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from .pcam_dataset import get_pcam_datasets


def get_pcam_dataloaders(
    data_root: str | Path = "data/raw",
    batch_size: int = 64,
    num_workers: int = 0,      # <--- HIER: 0 statt 4
    center_crop_size: int = 64,
) -> Dict[str, DataLoader]:
    """Return train/val/test dataloaders for PCam."""
    datasets = get_pcam_datasets(data_root, center_crop_size=center_crop_size)

    loaders: Dict[str, DataLoader] = {}

    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,   # <--- bleibt, wird jetzt 0 übergeben
            pin_memory=True,
        )

    return loaders
