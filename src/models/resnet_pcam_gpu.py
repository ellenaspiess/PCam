# src/models/resnet_pcam_gpu.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models

# Definiert die Modi für deine Ablation Study
TLMode = Literal["frozen", "partial", "full"]

@dataclass
class ResNetConfig:
    """Konfiguration für das ResNet Modell."""
    num_classes: int = 1
    tl_mode: TLMode = "partial"  # Standard-Modus
    pretrained: bool = True
    dropout_p: float = 0.0       # Optional: Dropout vor dem letzten Layer

class ResNetPCam(nn.Module):
    """
    ResNet18 Wrapper für PCam.
    Erfüllt die Anforderung aus dem Feedback (Transfer Learning Strategie).
    """

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ResNetConfig()
        self.config = config

        # 1. Backbone laden
        if config.pretrained:
            # Nutzt die modernen Weights, falls verfügbar, sonst Fallback
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)

        # 2. Den FC-Layer ersetzen (Binary Classification)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=config.dropout_p),
            nn.Linear(in_features, config.num_classes)
        )

        self.backbone = backbone
        
        # 3. Layer einfrieren je nach Modus
        self._apply_tl_mode(config.tl_mode)

    def _apply_tl_mode(self, mode: TLMode) -> None:
        # Erstmal alles einfrieren für 'frozen' und 'partial'
        if mode in ("frozen", "partial"):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Dann spezifische Teile auftauen
        if mode == "frozen":
            # Nur der neue Head (fc) wird trainiert
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
                
        elif mode == "partial":
            # Letzter ResNet-Block (Layer4) + Head
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
                
        elif mode == "full":
            # Alles trainierbar
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward Pass durchs Backbone
        logits = self.backbone(x)
        # Wichtig: Output Shape [Batch, 1] -> Squeeze zu [Batch] für BCEWithLogitsLoss
        return logits.squeeze(-1)