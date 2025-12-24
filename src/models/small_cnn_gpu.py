# src/models/small_cnn.py
from torch import nn
import torch

class SmallCNN(nn.Module):
    """
    Ein kompaktes CNN für PCam (binäre Klassifikation).
    Nutzt Global Average Pooling statt riesiger FC-Layer (Feedback-optimiert).
    """

    def __init__(self, num_classes: int = 1, dropout_p: float = 0.5) -> None:
        super().__init__()

        # Feature Extractor: 3 Blöcke (Conv -> BN -> ReLU -> Pool)
        self.features = nn.Sequential(
            # Block 1: 96x96 -> 48x48
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 48x48 -> 24x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling: Macht aus (Batch, 128, 12, 12) -> (Batch, 128, 1, 1)
        # Das spart Parameter und verhindert Overfitting (Feedback Punkt 2).
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1) # Flatten für den Linear Layer
        logits = self.classifier(x)
        return logits.squeeze(-1) # Shape [Batch] für Loss-Funktion