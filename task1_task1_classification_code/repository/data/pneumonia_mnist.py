\
"""
Dataset utilities for MedMNIST v2: PneumoniaMNIST
- Builds train/val/test PyTorch DataLoaders
- Applies normalization and conservative augmentations suitable for chest X-rays
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO


@dataclass
class DataConfig:
    data_flag: str = "pneumoniamnist"
    download: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    # Augmentation controls
    augment: bool = True
    # Normalization statistics (computed approximately for PneumoniaMNIST)
    # If desired, compute exact mean/std and update.
    mean: float = 0.5
    std: float = 0.25


def _build_transforms(cfg: DataConfig) -> Dict[str, transforms.Compose]:
    """
    Medical-image friendly augmentations:
    - small affine transform (rotation/translation)
    - no horizontal flip by default (can be anatomically questionable)
    """
    to_tensor = transforms.ToTensor()

    normalize = transforms.Normalize(mean=[cfg.mean], std=[cfg.std])

    if cfg.augment:
        train_tf = transforms.Compose(
            [
                # PIL image comes from medmnist dataset
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=10,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05),
                            shear=None,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=0,
                        )
                    ],
                    p=0.7,
                ),
                to_tensor,
                normalize,
            ]
        )
    else:
        train_tf = transforms.Compose([to_tensor, normalize])

    eval_tf = transforms.Compose([to_tensor, normalize])

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_dataloaders(
    cfg: DataConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns: train_loader, val_loader, test_loader, dataset_info
    """
    info = INFO[cfg.data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    tfs = _build_transforms(cfg)

    train_ds = DataClass(split="train", transform=tfs["train"], download=cfg.download)
    val_ds = DataClass(split="val", transform=tfs["val"], download=cfg.download)
    test_ds = DataClass(split="test", transform=tfs["test"], download=cfg.download)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    dataset_info = {
        "task": info["task"],
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "labels": info["label"],
        "description": info.get("description", ""),
    }
    return train_loader, val_loader, test_loader, dataset_info


def estimate_mean_std(cfg: DataConfig, max_batches: int = 50, batch_size: int = 256) -> Tuple[float, float]:
    """
    Quick approximate mean/std estimation over train split.
    Uses raw ToTensor without normalization.
    """
    info = INFO[cfg.data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", transform=transforms.ToTensor(), download=cfg.download)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)

    n = 0
    mean = 0.0
    m2 = 0.0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        # x: [B,1,28,28]
        b = x.shape[0]
        x = x.view(b, -1)
        batch_mean = x.mean(dim=1)  # per-sample mean
        batch_var = x.var(dim=1, unbiased=False)  # per-sample var

        for bm, bv in zip(batch_mean, batch_var):
            n += 1
            delta = float(bm) - mean
            mean += delta / n
            delta2 = float(bm) - mean
            m2 += delta * delta2 + float(bv)

    var = m2 / max(n, 1)
    std = float(np.sqrt(var))
    return float(mean), float(std)
