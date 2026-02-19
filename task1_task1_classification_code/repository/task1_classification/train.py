\
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.pneumonia_mnist import DataConfig, build_dataloaders
from models.task1_models import build_model
from task1_classification.utils.metrics import compute_metrics
from task1_classification.utils.plotting import save_training_curves
from task1_classification.utils.io import save_json


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    n = 0

    crit = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)  # medmnist gives shape [B,1]
        logits = model(x)
        loss = crit(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred)
        y_prob.append(prob)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    avg_loss = total_loss / max(n, 1)
    return avg_loss, metrics.accuracy


def train_one_epoch(model: nn.Module, loader, device: torch.device, opt, crit) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.squeeze().long().to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return total_loss / max(n, 1), correct / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet18_small"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--out_dir", type=str, default="reports/task1")
    ap.add_argument("--ckpt_dir", type=str, default="models/task1")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")

    cfg = DataConfig(augment=not args.no_augment)
    train_loader, val_loader, test_loader, ds_info = build_dataloaders(cfg, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=2, dropout=args.dropout).to(device)

    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, opt, crit)
        val_loss, val_acc = run_eval(model, val_loader, device)

        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        torch.save({"model_state": model.state_dict(), "args": vars(args), "ds_info": ds_info}, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args), "ds_info": ds_info}, best_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    save_training_curves(history, os.path.join(args.out_dir, "training_curves.png"))
    save_json(
        {
            "dataset": ds_info,
            "train_config": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "device": str(device),
        },
        os.path.join(args.out_dir, "train_summary.json"),
    )

    print(f"Saved best checkpoint: {best_path}")
    print("Run evaluation:")
    print(f"python task1_classification/eval.py --checkpoint {best_path}")


if __name__ == "__main__":
    main()
