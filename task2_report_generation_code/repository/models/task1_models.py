from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def load_task1_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", ckpt.get("cfg", {}))
    dropout = float(config.get("dropout", 0.1))
    model = SimpleCNN(dropout=dropout).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config
