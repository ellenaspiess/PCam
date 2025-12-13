# src/datasets/pcam_dataset.py
from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_pcam_transforms(center_crop_size: int = 64, train: bool = True):
    """Build transforms for PCam patches."""
    tfms = [
        transforms.CenterCrop(center_crop_size),
    ]

    if train:
        tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),
        ]

    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(tfms)


def get_pcam_datasets(
    data_root: str | Path, center_crop_size: int = 64
) -> Dict[str, Dataset]:
    """
    Load PCam train/val/test splits using torchvision.datasets.PCAM.
    Downloads data into data_root if not present.
    """
    data_root = Path(data_root)

    train_set = datasets.PCAM(
        root=data_root,
        split="train",
        transform=get_pcam_transforms(center_crop_size, train=True),
        download=True,
    )
    val_set = datasets.PCAM(
        root=data_root,
        split="val",
        transform=get_pcam_transforms(center_crop_size, train=False),
        download=True,
    )
    test_set = datasets.PCAM(
        root=data_root,
        split="test",
        transform=get_pcam_transforms(center_crop_size, train=False),
        download=True,
    )

    return {"train": train_set, "val": val_set, "test": test_set}
