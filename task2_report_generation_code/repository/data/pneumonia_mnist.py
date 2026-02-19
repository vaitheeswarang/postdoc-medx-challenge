from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import medmnist
from medmnist import INFO


@dataclass
class DataConfig:
    dataset_name: str = "pneumoniamnist"
    batch_size: int = 64
    num_workers: int = 2
    download: bool = True
    image_size: int = 224
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


def _to_rgb_pil(x: np.ndarray) -> Image.Image:
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return Image.fromarray(x, mode="L").convert("RGB")


def build_transforms(cfg: DataConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(_to_rgb_pil),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[cfg.normalize_mean]*3, std=[cfg.normalize_std]*3),
    ])


def build_dataset(split: str, cfg: DataConfig):
    info = INFO[cfg.dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    tfm = build_transforms(cfg)
    return DataClass(split=split, transform=tfm, download=cfg.download)


def build_dataloader(split: str, cfg: DataConfig, shuffle: bool = False) -> DataLoader:
    ds = build_dataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
