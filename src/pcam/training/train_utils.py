"""Shared training utilities for configuration, scheduling, and early stopping."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer, lr_scheduler


def load_optuna_best_params(path: str | Path) -> dict[str, Any]:
    """Load ``best_params`` from an Optuna result JSON file.

    Also normalizes naming differences (e.g. ``scheduler_name`` -> ``scheduler``).
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError(f"No valid 'best_params' found in {path}")

    result: dict[str, Any] = dict(best_params)
    if "scheduler_name" in result and "scheduler" not in result:
        result["scheduler"] = result.pop("scheduler_name")
    if "tl_mode" in payload and payload["tl_mode"] is not None:
        result["tl_mode"] = payload["tl_mode"]
    return result


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible CPU training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    num_epochs: int,
):
    """Create scheduler by name (``none``, ``cosine``, ``plateau``)."""
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    if scheduler_name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def should_stop_early(
    current_value: float,
    best_value: float,
    no_improve_epochs: int,
    mode: str,
    min_delta: float,
    patience: int,
) -> tuple[bool, float, int]:
    """Update early-stopping state and return ``(stop, best_value, no_improve)``."""
    improved = (
        current_value > best_value + min_delta
        if mode == "max"
        else current_value < best_value - min_delta
    )

    if improved:
        return False, current_value, 0

    no_improve_epochs += 1
    return no_improve_epochs >= patience, best_value, no_improve_epochs
