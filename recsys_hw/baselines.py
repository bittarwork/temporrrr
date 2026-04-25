"""Simple baselines to compare against GA-optimized blending (assignment step 4.3)."""

from __future__ import annotations

import numpy as np


def mse_constant_mean(y_train: np.ndarray, y_val: np.ndarray) -> float:
    """Predict the training mean for every validation row."""
    mu = float(np.mean(y_train))
    d = y_val - mu
    return float(np.mean(d * d))


def mse_dt_column_only(X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Use only the last column (DT output) as prediction — ablation-style baseline."""
    pred = np.clip(X_val[:, -1], 0.0, 1.0)
    d = pred - y_val
    return float(np.mean(d * d))


def mse_uniform_blend(X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Equal weights on all feature dimensions."""
    d = X_val.shape[1]
    w = np.ones(d, dtype=np.float64) / d
    pred = X_val @ w
    diff = pred - y_val
    return float(np.mean(diff * diff))
