# src/datasets/dataloaders.py
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader, Subset
import importlib
import h5py

from .pcam_dataset import get_pcam_datasets


def get_pcam_dataloaders(
    data_root: str | Path = "data/raw",
    batch_size: int = 64,
    num_workers: int = 0,
    center_crop_size: int = 64,
    limit_per_split: int | None = None,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Return train/val/test dataloaders for PCam.

    This is a simple and stable implementation that avoids custom pickling
    logic. Use `num_workers=0` in IDE/debug sessions on Windows for
    reliability; increase `num_workers` when running from a terminal.
    """

    datasets = get_pcam_datasets(data_root, center_crop_size=center_crop_size)
    loaders: Dict[str, DataLoader] = {}

    for split, ds in datasets.items():
        # Some torchvision implementations of PCAM include an attribute
        # `_pickling_module_attrs` that maps attribute names to module
        # paths. When creating dataloaders (especially with workers) PyTorch
        # may attempt to access these attributes which can raise
        # AttributeError if they aren't present. Try to materialize them
        # here for robustness.
        if hasattr(ds, "_pickling_module_attrs"):
            try:
                for attr_name, module_name in getattr(ds, "_pickling_module_attrs").items():
                    try:
                        mod = importlib.import_module(module_name)
                        # prefer module attribute if available, else the module
                        value = getattr(mod, attr_name, mod)
                        setattr(ds, attr_name, value)
                    except Exception:
                        # best-effort: ignore failures to import/assign
                        pass
            except Exception:
                pass

        # Ensure h5py module is available on dataset instances that
        # expect it (some torchvision PCAM internals use `self.h5py`).
        if not hasattr(ds, "h5py"):
            setattr(ds, "h5py", h5py)
        if limit_per_split:
            ds = Subset(ds, range(min(len(ds), limit_per_split)))

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders
