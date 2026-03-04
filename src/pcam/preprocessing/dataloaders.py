from __future__ import annotations
"""DataLoader construction utilities for PCam experiments."""

import importlib
from pathlib import Path
from typing import Dict

import h5py
from torch.utils.data import DataLoader, Subset

from .datasets import get_pcam_datasets


def get_pcam_dataloaders(
    data_root: str | Path = "data/raw",
    batch_size: int = 64,
    num_workers: int = 0,
    center_crop_size: int = 64,
    stain_normalization: str = "macenko",
    stain_reference_image: str | Path | None = None,
    limit_per_split: int | None = None,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders for PCam.

    Args:
        data_root: Root folder for torchvision PCam dataset files.
        batch_size: Batch size for each split loader.
        num_workers: Number of loader worker processes.
        center_crop_size: Spatial crop size (typically 64 for PCam).
        stain_normalization: One of ``macenko``, ``reinhard``, ``none``.
        stain_reference_image: Optional explicit reference image path.
        limit_per_split: Optional cap on examples per split for fast runs.

    Returns:
        Mapping ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``.
    """
    datasets = get_pcam_datasets(
        data_root=data_root,
        center_crop_size=center_crop_size,
        stain_normalization=stain_normalization,
        stain_reference_image=stain_reference_image,
    )
    loaders: Dict[str, DataLoader] = {}

    for split, ds in datasets.items():
        if hasattr(ds, "_pickling_module_attrs"):
            # Torchvision PCam can carry lazy module references that must be
            # re-bound in subprocess contexts.
            for attr_name, module_name in getattr(ds, "_pickling_module_attrs").items():
                try:
                    mod = importlib.import_module(module_name)
                    setattr(ds, attr_name, getattr(mod, attr_name, mod))
                except Exception:
                    # Best effort only; continue with default dataset state.
                    pass

        if not hasattr(ds, "h5py"):
            # Compatibility for torchvision versions expecting this attribute.
            setattr(ds, "h5py", h5py)

        if limit_per_split is not None:
            ds = Subset(ds, range(min(len(ds), limit_per_split)))

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    return loaders
