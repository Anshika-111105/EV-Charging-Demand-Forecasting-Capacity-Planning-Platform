"""Evaluation metrics for time-series forecasting with zero-demand safety."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Zero-safe Mean Absolute Percentage Error (percentage 0-100%).
    Uses threshold epsilon for near-zero values to avoid infinite percentages.
    """
    y_true_safe = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / y_true_safe) * 100.0)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Symmetric Mean Absolute Percentage Error (percentage 0-100%)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y) * 100%)."""
    sum_true = float(np.sum(np.abs(y_true)))
    if sum_true < 1e-6:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0)


def calculate_peak_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute error in peak demand estimation."""
    max_true = float(np.max(y_true)) if len(y_true) > 0 else 0.0
    max_pred = float(np.max(y_pred)) if len(y_pred) > 0 else 0.0
    return float(abs(max_true - max_pred))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all standard and robust forecasting metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0 or len(y_p) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "smape": 0.0,
            "wape": 0.0,
            "r2": 0.0,
            "peak_error": 0.0,
        }

    # Non-zero actual demand mask for classic MAPE
    nonzero_mask = y_t > 0.1
    classic_mape = (
        float(np.mean(np.abs((y_t[nonzero_mask] - y_p[nonzero_mask]) / y_t[nonzero_mask])) * 100.0)
        if np.sum(nonzero_mask) > 0
        else 0.0
    )

    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-6 else 0.0

    return {
        "mae": calculate_mae(y_t, y_p),
        "rmse": calculate_rmse(y_t, y_p),
        "mape": calculate_mape(y_t, y_p),
        "classic_mape_nonzero": classic_mape,
        "smape": calculate_smape(y_t, y_p),
        "wape": calculate_wape(y_t, y_p),
        "r2": r2,
        "peak_error": calculate_peak_error(y_t, y_p),
    }
