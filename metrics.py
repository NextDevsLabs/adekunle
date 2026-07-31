"""Evaluation metrics and Diebold-Mariano test."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal forecast accuracy."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = np.mean(d)
    n = len(d)
    gamma = [np.sum((d[h:] - d_mean) * (d[:-h] - d_mean)) / n for h in range(1, h + 1)]
    var_d = np.var(d, ddof=1) + 2 * sum(gamma)
    if var_d <= 0:
        return float("nan"), float("nan")
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def pairwise_dm_tests(
    actual: np.ndarray,
    forecasts: Dict[str, np.ndarray],
) -> pd.DataFrame:
    models = list(forecasts.keys())
    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            e1 = actual - forecasts[m1]
            e2 = actual - forecasts[m2]
            stat, p = diebold_mariano(e1, e2)
            rows.append(
                {
                    "model_1": m1,
                    "model_2": m2,
                    "dm_statistic": stat,
                    "p_value": p,
                    "significant_5pct": p < 0.05 if not np.isnan(p) else False,
                }
            )
    return pd.DataFrame(rows)


def metrics_table(
    actual: np.ndarray,
    forecasts: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for model, preds in forecasts.items():
        row = {"model": model, **compute_metrics(actual, preds)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE")
