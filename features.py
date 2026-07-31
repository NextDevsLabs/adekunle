"""Feature engineering and chronological train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET = "pct_within_4h_all"

TRAIN_END = "2023-03-01"
VAL_END = "2025-03-01"


@dataclass
class SplitConfig:
    train_end: str = TRAIN_END
    val_end: str = VAL_END
    target: str = TARGET


def add_period_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["period"].dt.year
    out["month"] = out["period"].dt.month
    out["quarter"] = out["period"].dt.quarter
    out["covid_period"] = (
        (out["period"] >= "2020-03-01") & (out["period"] <= "2021-06-01")
    ).astype(int)
    out["crs_period"] = (
        (out["period"] >= "2019-05-01") & (out["period"] <= "2023-05-01")
    ).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    out = df.copy().sort_values("period").reset_index(drop=True)
    lag_cols = {
        f"{target}_lag_1": 1,
        f"{target}_lag_3": 3,
        f"{target}_lag_12": 12,
        "total_attendances_lag_1": 1,
        "total_attendances_lag_12": 12,
        "emergency_admissions_lag_1": 1,
    }
    for name, lag in lag_cols.items():
        base = name.replace(f"_lag_{lag}", "")
        if base in out.columns:
            out[name] = out[base].shift(lag)
    out["target_roll_mean_3"] = out[target].rolling(3, min_periods=1).mean()
    out["target_roll_mean_12"] = out[target].rolling(12, min_periods=1).mean()
    return out


def interpolate_missing(df: pd.DataFrame, exclude_cols: List[str] | None = None) -> pd.DataFrame:
    """Linear interpolation for true missing numeric values only."""
    out = df.copy()
    exclude_cols = exclude_cols or ["covid_period", "crs_period", "month", "quarter", "year"]
    numeric_cols = [
        c for c in out.columns
        if c not in exclude_cols and c != "period" and pd.api.types.is_numeric_dtype(out[c])
    ]
    out[numeric_cols] = out[numeric_cols].interpolate(method="linear", limit_direction="both")
    return out


def assign_split(df: pd.DataFrame, config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "train"
    out.loc[out["period"] > config.train_end, "split"] = "validation"
    out.loc[out["period"] > config.val_end, "split"] = "test"
    return out


def get_rf_features(df: pd.DataFrame) -> List[str]:
    candidates = [
        "month", "quarter", "year", "covid_period", "crs_period",
        "attendances_type1", "attendances_type2", "attendances_type3",
        "total_attendances", "emergency_admissions", "emergency_admissions_type1",
        "attendances_over_4h", "booked_attendances",
        f"{TARGET}_lag_1", f"{TARGET}_lag_3", f"{TARGET}_lag_12",
        "total_attendances_lag_1", "total_attendances_lag_12",
        "emergency_admissions_lag_1", "target_roll_mean_3", "target_roll_mean_12",
    ]
    return [c for c in candidates if c in df.columns]


def prepare_modeling_frame(df: pd.DataFrame, config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    out = add_period_flags(df)
    out = add_lag_features(out, target=config.target)
    out = interpolate_missing(out)
    out = assign_split(out, config)
    out = out.dropna(subset=[config.target]).reset_index(drop=True)
    return out


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    train_s = train.copy()
    val_s = val.copy()
    test_s = test.copy()
    scaler.fit(train[feature_cols])
    train_s[feature_cols] = scaler.transform(train[feature_cols])
    val_s[feature_cols] = scaler.transform(val[feature_cols])
    test_s[feature_cols] = scaler.transform(test[feature_cols])
    return train_s, val_s, test_s, scaler


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str = TARGET,
    seq_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    values = df[feature_cols + [target]].values
    periods = df["period"].values
    X, y, idx_periods = [], [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len : i, :-1])
        y.append(values[i, -1])
        idx_periods.append(periods[i])
    return np.array(X), np.array(y), pd.Series(idx_periods)
