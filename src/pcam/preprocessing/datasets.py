from __future__ import annotations
"""Dataset and transform utilities for PatchCamelyon (PCam).

This module centralizes augmentation, stain normalization, and dataset loading
so all training/tuning entry points share the exact same preprocessing logic.
"""

import json
import random
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


PCAM_SPLITS = ("train", "val", "test")
DEFAULT_STAIN_REFERENCE_MANIFEST = Path("experiments/stain_refs/references.json")


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees."""

    def __call__(self, img):
        angle = random.choice((0, 90, 180, 270))
        return TF.rotate(img, angle)


class TorchstainNormalization:
    """Stain normalization via ``torchstain`` (Macenko or Reinhard).

    The transform expects and returns torch tensors in CHW format with values in
    ``[0, 1]``. Internally, conversion to uint8 HWC is done because the numpy
    backends in ``torchstain`` operate on image-like arrays.
    """

    def __init__(self, method: str = "macenko", reference_image_path: str | Path | None = None) -> None:
        if method not in {"macenko", "reinhard"}:
            raise ValueError(f"Unsupported torchstain method: {method}")
        self.method = method
        self.reference_image_path = Path(reference_image_path) if reference_image_path else None
        self._normalizer = None
        self._is_fit = False

    def _ensure_normalizer(self):
        """Initialize and optionally pre-fit the selected torchstain normalizer."""
        if self._normalizer is not None:
            return
        try:
            import torchstain  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Stain normalization requires 'torchstain'. Install it with: pip install torchstain"
            ) from exc
        if self.method == "macenko":
            self._normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        else:
            self._normalizer = torchstain.normalizers.ReinhardNormalizer(backend="numpy")

        # Prefer a fixed, pre-selected reference image if provided.
        if self.reference_image_path and self.reference_image_path.exists():
            ref = np.load(self.reference_image_path)
            if ref.dtype != np.uint8:
                ref = np.clip(ref, 0, 255).astype(np.uint8)
            self._fit(ref)

    def _fit(self, img_hwc_u8: np.ndarray) -> None:
        """Fit stain statistics on a reference image."""
        assert self._normalizer is not None
        try:
            self._normalizer.fit(img_hwc_u8)
        except TypeError:
            self._normalizer.fit(I=img_hwc_u8)
        except Exception:
            self._is_fit = False
            return
        self._is_fit = True

    def _normalize(self, img_hwc_u8: np.ndarray):
        """Apply stain normalization and return normalized image array."""
        assert self._normalizer is not None
        try:
            out = self._normalizer.normalize(img_hwc_u8)
        except TypeError:
            out = self._normalizer.normalize(I=img_hwc_u8)
        except Exception:
            # Numerical edge cases (e.g. low-variance background patches) can
            # break Macenko eigen decomposition. Fall back to identity.
            return img_hwc_u8
        return out[0] if isinstance(out, tuple) else out

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Input tensor is [C,H,W] in [0,1].
        self._ensure_normalizer()
        img_hwc_u8 = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if not self._is_fit:
            # Use first observed image as reference target.
            self._fit(img_hwc_u8)
            if not self._is_fit:
                return img

        norm_hwc = self._normalize(img_hwc_u8)
        if not isinstance(norm_hwc, np.ndarray):
            norm_hwc = np.asarray(norm_hwc)

        if norm_hwc.dtype != np.float32 and norm_hwc.dtype != np.float64:
            norm_hwc = norm_hwc.astype(np.float32) / 255.0
        else:
            norm_hwc = np.clip(norm_hwc.astype(np.float32), 0.0, 1.0)

        return torch.from_numpy(norm_hwc).permute(2, 0, 1).contiguous()


def resolve_reference_image_path(
    stain_normalization: str,
    stain_reference_image: str | Path | None,
) -> str | Path | None:
    """Resolve reference image path for stain normalization.

    Resolution order:
    1. Explicit ``stain_reference_image`` argument.
    2. ``experiments/stain_refs/references.json`` for Macenko runs.
    3. ``None`` (normalizer will self-fit on first observed patch).
    """
    if stain_reference_image is not None:
        return stain_reference_image

    if stain_normalization != "macenko":
        return None

    manifest_path = DEFAULT_STAIN_REFERENCE_MANIFEST
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None

    reference_path = manifest.get("reference_image_path")
    if not isinstance(reference_path, str) or not reference_path:
        return None

    ref = Path(reference_path)
    if ref.is_absolute():
        return ref

    cwd_candidate = (Path.cwd() / ref).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    manifest_relative = (manifest_path.parent / ref).resolve()
    if manifest_relative.exists():
        return manifest_relative

    return ref


def get_pcam_transforms(
    center_crop_size: int = 64,
    train: bool = True,
    stain_normalization: str = "macenko",
    stain_reference_image: str | Path | None = None,
) -> transforms.Compose:
    """Build the transform pipeline used across PCam experiments.

    Pipeline order:
    1. Spatial crop / train-time augmentation
    2. ``ToTensor``
    3. Optional stain normalization
    4. ImageNet normalization
    """
    tfms = [transforms.CenterCrop(center_crop_size)]

    if train:
        tfms.extend(
            [
                RandomRotate90(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                ),
            ]
        )

    tfms.extend(
        [
            transforms.ToTensor(),
            (
                TorchstainNormalization(stain_normalization, reference_image_path=stain_reference_image)
                if stain_normalization == "macenko"
                else (
                    TorchstainNormalization(stain_normalization, reference_image_path=stain_reference_image)
                    if stain_normalization == "reinhard"
                    else transforms.Lambda(lambda x: x)
                )
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transforms.Compose(tfms)


def get_pcam_datasets(
    data_root: str | Path,
    center_crop_size: int = 64,
    stain_normalization: str = "macenko",
    stain_reference_image: str | Path | None = None,
) -> Dict[str, Dataset]:
    """Load train/val/test PCam datasets with consistent preprocessing."""
    root = Path(data_root)
    resolved_reference_image = resolve_reference_image_path(
        stain_normalization=stain_normalization,
        stain_reference_image=stain_reference_image,
    )
    datasets_by_split: Dict[str, Dataset] = {}

    for split in PCAM_SPLITS:
        ds = datasets.PCAM(
            root=root,
            split=split,
            transform=get_pcam_transforms(
                center_crop_size=center_crop_size,
                train=(split == "train"),
                stain_normalization=stain_normalization,
                stain_reference_image=resolved_reference_image,
            ),
            download=True,
        )
        # Some torchvision PCAM versions expect this on the dataset object.
        if not hasattr(ds, "h5py"):
            setattr(ds, "h5py", h5py)
        datasets_by_split[split] = ds

    return datasets_by_split
