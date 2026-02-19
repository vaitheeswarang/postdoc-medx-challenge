\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return ClassificationMetrics(
        accuracy=acc,
        precision=float(pr),
        recall=float(rc),
        f1=float(f1),
        auc=auc,
    )


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr
