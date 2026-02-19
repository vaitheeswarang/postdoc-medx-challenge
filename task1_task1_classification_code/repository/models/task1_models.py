\
"""
Model zoo for Task 1.

Two options are provided:
1) simple_cnn: lightweight CNN tuned for 28x28 grayscale.
2) resnet18_small: ResNet-18 with 1-channel input and smaller first conv for 28x28 images.
"""

from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18_small(num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    """
    ResNet18 adapted for 1-channel 28x28:
    - 1-channel input conv
    - smaller kernel/stride for first conv
    - remove initial maxpool to keep spatial detail
    """
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    if dropout > 0:
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    else:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


ModelName = Literal["simple_cnn", "resnet18_small"]


def build_model(name: ModelName, num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    if name == "resnet18_small":
        return build_resnet18_small(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")
