# src/models/resnet_pcam.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models


TLMode = Literal["frozen", "partial", "full"]


@dataclass
class ResNetConfig:
    """Config für das ResNet-Transfer-Learning auf PCam."""
    num_classes: int = 1
    tl_mode: TLMode = "partial"  # "frozen", "partial", "full"
    pretrained: bool = True


class ResNetPCam(nn.Module):
    """
    ResNet18 für PCam (Patch-Klassifikation).

    tl_mode:
        - "frozen": Feature-Extractor ist eingefroren, nur letzter Linear-Layer trainierbar
        - "partial": frühe Layer eingefroren, letzte(n) Block/Blöcke + Kopf trainierbar
        - "full": alle Layer trainierbar
    """

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()

        if config is None:
            config = ResNetConfig()

        self.config = config

        # ResNet18 laden (mit oder ohne ImageNet-Pretraining)
        if config.pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
            except Exception:
                # Falls kein Internet für Weights: Fallback auf random init
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        # Letztes FC-Layer durch ein neues ersetzen (für Binary-Output)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, config.num_classes)

        self.backbone = backbone

        # Transfer-Learning-Strategie anwenden
        self._apply_tl_mode(config.tl_mode)

    def _apply_tl_mode(self, mode: TLMode) -> None:
        # Erstmal alles einfrieren
        if mode in ("frozen", "partial"):
            for param in self.backbone.parameters():
                param.requires_grad = False

        if mode == "frozen":
            # Nur das letzte FC trainierbar
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

        elif mode == "partial":
            # Letzten ResNet-Block + FC wieder freigeben
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

        elif mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        # Shape [B, 1] -> für BCEWithLogitsLoss später zu [B] sqeezen
        return logits.squeeze(-1)
