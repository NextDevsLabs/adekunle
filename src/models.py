"""Forecasting model trainers: ARIMA, Prophet, Random Forest, LSTM."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

TARGET = "pct_within_4h_all"


def fit_arima(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = TARGET,
) -> Tuple[object, np.ndarray, Dict]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_train = train[target].astype(float)
    orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 0, 1),
    ]
    seasonal_orders = [(0, 0, 0, 0), (1, 0, 1, 12), (1, 1, 1, 12)]

    best_model = None
    best_rmse = np.inf
    best_cfg = {}

    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
                val_pred = fitted.forecast(len(val))
                score = np.sqrt(np.mean((val[target].values - val_pred) ** 2))
                if score < best_rmse:
                    best_rmse = score
                    best_model = fitted
                    best_cfg = {"order": order, "seasonal_order": seasonal_order}
            except Exception:
                continue

    if best_model is None:
        model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
        best_model = model.fit(disp=False)
        best_cfg = {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 12)}

    val_pred = best_model.forecast(len(val))
    return best_model, np.asarray(val_pred), best_cfg


def forecast_arima(model, steps: int) -> np.ndarray:
    return np.asarray(model.forecast(steps))


def refit_arima_on_train_val(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: Dict,
    target: str = TARGET,
):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = pd.concat([train[target], val[target]]).astype(float)
    model = SARIMAX(
        y,
        order=cfg["order"],
        seasonal_order=cfg["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def fit_prophet(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = TARGET,
) -> Tuple[object, np.ndarray, Dict]:
    from prophet import Prophet

    regressors = [
        c for c in ["total_attendances", "emergency_admissions", "covid_period", "crs_period"]
        if c in train.columns
    ]

    train_df = train.rename(columns={"period": "ds", target: "y"})[["ds", "y"] + regressors].copy()
    val_df = val.rename(columns={"period": "ds", target: "y"})[["ds", "y"] + regressors].copy()

    best_model = None
    best_rmse = np.inf
    best_cfg = {}

    grid = {
        "changepoint_prior_scale": [0.05, 0.1, 0.5],
        "seasonality_prior_scale": [1.0, 5.0, 10.0],
    }

    for params in ParameterGrid(grid):
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **params,
        )
        for reg in regressors:
            m.add_regressor(reg)
        m.fit(train_df)
        future = m.predict(val_df.drop(columns=["y"]))
        val_pred = future["yhat"].values
        score = np.sqrt(np.mean((val_df["y"].values - val_pred) ** 2))
        if score < best_rmse:
            best_rmse = score
            best_model = m
            best_cfg = params

    future = best_model.predict(val_df.drop(columns=["y"]))
    return best_model, future["yhat"].values, best_cfg


def forecast_prophet(
    model,
    frame: pd.DataFrame,
    target: str = TARGET,
) -> np.ndarray:
    regressors = [
        c for c in ["total_attendances", "emergency_admissions", "covid_period", "crs_period"]
        if c in frame.columns
    ]
    df = frame.rename(columns={"period": "ds", target: "y"})[["ds"] + regressors].copy()
    return model.predict(df)["yhat"].values


def refit_prophet(train_val: pd.DataFrame, cfg: Dict, target: str = TARGET):
    from prophet import Prophet

    regressors = [
        c for c in ["total_attendances", "emergency_admissions", "covid_period", "crs_period"]
        if c in train_val.columns
    ]
    df = train_val.rename(columns={"period": "ds", target: "y"})[["ds", "y"] + regressors].copy()
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        **cfg,
    )
    for reg in regressors:
        m.add_regressor(reg)
    m.fit(df)
    return m


def fit_random_forest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    target: str = TARGET,
) -> Tuple[RandomForestRegressor, np.ndarray, Dict]:
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 2, 4],
    }

    best_model = None
    best_rmse = np.inf
    best_cfg = {}

    X_train = train[feature_cols].values
    y_train = train[target].values
    X_val = val[feature_cols].values
    y_val = val[target].values

    for params in ParameterGrid(grid):
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = np.sqrt(np.mean((y_val - val_pred) ** 2))
        if score < best_rmse:
            best_rmse = score
            best_model = model
            best_cfg = params

    val_pred = best_model.predict(X_val)
    return best_model, val_pred, best_cfg


def fit_lstm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    target: str = TARGET,
    seq_len: int = 12,
    epochs: int = 100,
) -> Tuple[object, np.ndarray, Dict, object]:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler

    tf.random.set_seed(42)

    combined = pd.concat([train, val], ignore_index=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(combined[feature_cols + [target]])

    scaled_df = combined.copy()
    scaled_df[feature_cols + [target]] = scaled

    train_len = len(train)
    X_all, y_all, periods = [], [], []
    for i in range(seq_len, len(scaled_df)):
        X_all.append(scaled_df.iloc[i - seq_len : i][feature_cols].values)
        y_all.append(scaled_df.iloc[i][target])
        periods.append(scaled_df.iloc[i]["period"])

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    periods = pd.Series(periods)

    train_mask = periods <= train["period"].max()
    val_mask = (periods > train["period"].max()) & (periods <= val["period"].max())

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    model = keras.Sequential([
        keras.layers.LSTM(32, return_sequences=True, input_shape=(seq_len, len(feature_cols))),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(16),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    early_stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )

    val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    target_idx = len(feature_cols)
    dummy = np.zeros((len(val_pred_scaled), len(feature_cols) + 1))
    dummy[:, target_idx] = val_pred_scaled
    val_pred = scaler.inverse_transform(dummy)[:, target_idx]

    cfg = {"seq_len": seq_len, "epochs": epochs, "feature_cols": feature_cols}
    return model, val_pred, cfg, scaler


def forecast_lstm(
    model,
    history: pd.DataFrame,
    future: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    scaler,
    seq_len: int = 12,
) -> np.ndarray:
    combined = pd.concat([history, future], ignore_index=True)
    scaled = scaler.transform(combined[feature_cols + [target]])
    preds = []

    for i in range(len(history), len(combined)):
        seq = scaled[i - seq_len : i, : len(feature_cols)]
        pred_scaled = model.predict(seq.reshape(1, seq_len, len(feature_cols)), verbose=0)[0, 0]
        preds.append(pred_scaled)
        scaled[i, len(feature_cols)] = pred_scaled

    dummy = np.zeros((len(preds), len(feature_cols) + 1))
    dummy[:, len(feature_cols)] = preds
    return scaler.inverse_transform(dummy)[:, len(feature_cols)]
