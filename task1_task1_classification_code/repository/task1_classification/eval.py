\
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.pneumonia_mnist import DataConfig, build_dataloaders
from models.task1_models import build_model
from task1_classification.utils.metrics import compute_metrics, compute_confusion, compute_roc
from task1_classification.utils.plotting import save_confusion_matrix, save_roc_curve
from task1_classification.utils.io import save_json


@torch.no_grad()
def predict_all(model: nn.Module, loader, device: torch.device):
    model.eval()
    y_true, y_pred, y_prob, xs = [], [], [], []
    for x, y in tqdm(loader, desc="infer", leave=False):
        x = x.to(device)
        y = y.squeeze().long().to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

        xs.append(x.detach().cpu())
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_prob.append(prob.detach().cpu().numpy())
    xs = torch.cat(xs, dim=0)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return xs, y_true, y_pred, y_prob


def save_failure_cases(xs: torch.Tensor, y_true: np.ndarray, y_pred: np.ndarray, out_path: str, max_items: int = 32) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mis_idx = np.where(y_true != y_pred)[0]
    n = int(min(len(mis_idx), max_items))
    if n == 0:
        return 0

    sel = mis_idx[:n]
    cols = 8
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for i, idx in enumerate(sel, start=1):
        img = xs[idx].squeeze(0).numpy()
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return int(len(mis_idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--out_dir", type=str, default="reports/task1")
    ap.add_argument("--no_augment", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    ds_info = ckpt.get("ds_info", {"labels": {"0": "normal", "1": "pneumonia"}})

    cfg = DataConfig(augment=not args.no_augment)
    _, _, test_loader, _ = build_dataloaders(cfg, batch_size=args.batch_size)

    model_name = train_args.get("model", "simple_cnn")
    dropout = float(train_args.get("dropout", 0.2))

    model = build_model(model_name, num_classes=2, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    xs, y_true, y_pred, y_prob = predict_all(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = compute_confusion(y_true, y_pred)
    fpr, tpr, thr = compute_roc(y_true, y_prob)

    os.makedirs(args.out_dir, exist_ok=True)
    save_confusion_matrix(cm, os.path.join(args.out_dir, "confusion_matrix.png"))
    save_roc_curve(fpr, tpr, os.path.join(args.out_dir, "roc_curve.png"))

    mis_total = save_failure_cases(xs, y_true, y_pred, os.path.join(args.out_dir, "failure_cases.png"))

    # Precision/Recall/F1 per-class (optional detail)
    # Here kept simple since binary average already reported.

    save_json(
        {
            "checkpoint": args.checkpoint,
            "model": model_name,
            "device": str(device),
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "auc": metrics.auc,
            },
            "confusion_matrix": cm.tolist(),
            "misclassified_count": mis_total,
            "test_size": int(len(y_true)),
            "labels": ds_info.get("labels", {}),
        },
        os.path.join(args.out_dir, "test_metrics.json"),
    )

    print("Test metrics:")
    print(f"  Accuracy : {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall   : {metrics.recall:.4f}")
    print(f"  F1-score : {metrics.f1:.4f}")
    print(f"  ROC-AUC  : {metrics.auc:.4f}")
    print(f"Misclassified: {mis_total} / {len(y_true)}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
