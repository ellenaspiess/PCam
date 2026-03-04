from __future__ import annotations
"""Small CNN baseline architecture for PCam patch classification."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Compact CNN baseline for binary PCam classification."""

    def __init__(self, num_classes: int = 1, dropout_p: float = 0.1) -> None:
        """Initialize convolutional feature extractor and linear head."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample logits with shape ``[N]`` for BCEWithLogitsLoss."""
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits.squeeze(-1)
