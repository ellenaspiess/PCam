"""ResNet18 model wrapper for PCam binary classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models

TLMode = Literal["frozen", "partial"]


@dataclass
class ResNetConfig:
    """Configuration for ResNet18 transfer learning on PCam.

    Attributes:
        num_classes: Number of output classes (1 for binary logits).
        tl_mode: Transfer-learning strategy (``frozen`` or ``partial``).
        pretrained: Whether to initialize with ImageNet weights.
        dropout_p: Dropout probability in the replaced classification head.
    """

    num_classes: int = 1
    tl_mode: TLMode = "partial"
    pretrained: bool = True
    dropout_p: float = 0.0


class ResNetPCam(nn.Module):
    """ResNet18 adapted for binary PCam classification."""

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResNetConfig()

        if self.config.pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(self.config.dropout_p),
            nn.Linear(in_features, self.config.num_classes),
        )

        self.backbone = backbone
        self._apply_tl_mode(self.config.tl_mode)

    def _apply_tl_mode(self, mode: TLMode) -> None:
        """Freeze/unfreeze layers according to the selected transfer mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if mode == "frozen":
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif mode == "partial":
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1D logits for BCEWithLogitsLoss."""
        logits = self.backbone(x)
        return logits.squeeze(-1)
