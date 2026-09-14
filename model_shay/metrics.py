"""Metric definitions shared by every model in this repo.

Deliberately torch-free: model_tree imports these too, and a gradient-boosted
tree has no business pulling in a 2.5GB deep-learning stack. Keeping one copy
is what makes the neural and tree numbers directly comparable.
"""

from __future__ import annotations

import math

import numpy as np

from model_shay.config import Config


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE/MAE/RMSE/R2 in return units, plus MAE in bps for readability."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse, mae = float(np.mean(err ** 2)), float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae,
            "mae_bps": mae * 1e4, "r2": (1.0 - mse / var) if var > 0 else float("nan")}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1, and the ordinal-friendly within-one-class rate."""
    from sklearn.metrics import f1_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"acc": float((y_true == y_pred).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "within_1": float((np.abs(y_true - y_pred) <= 1).mean())}


def compute_metrics(cfg: Config, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return (classification_metrics(y_true, y_pred) if cfg.task == "classification"
            else regression_metrics(y_true, y_pred))




# Metrics where BIGGER is better, so a tracker compares their negation.
HIGHER_BETTER = {"val_acc", "val_f1"}
