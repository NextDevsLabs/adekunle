"""End-to-end analysis pipeline for NHS A&E forecasting dissertation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import save_master_dataset
from src.features import (
    TARGET,
    get_rf_features,
    prepare_modeling_frame,
    scale_features,
    split_dataframe,
)
from src.metrics import compute_metrics, metrics_table, pairwise_dm_tests
from src.models import (
    fit_arima,
    fit_lstm,
    fit_prophet,
    fit_random_forest,
    forecast_arima,
    forecast_lstm,
    forecast_prophet,
    refit_arima_on_train_val,
    refit_prophet,
)

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["savefig.dpi"] = 150


def run_eda(df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    # 1. Performance trend
    fig, ax = plt.subplots()
    ax.plot(df["period"], df[TARGET], linewidth=2, color="#1f4e79")
    ax.axvspan(pd.Timestamp("2019-05-01"), pd.Timestamp("2023-05-01"), alpha=0.15, color="orange", label="CRS period")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15, color="red", label="COVID period")
    ax.set_title("Monthly A&E Four-Hour Performance (England)")
    ax.set_xlabel("Period")
    ax.set_ylabel("% seen within 4 hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_performance_trend.png")
    plt.close(fig)

    # 2. Seasonal boxplot
    tmp = df.dropna(subset=[TARGET]).copy()
    tmp["month_name"] = tmp["period"].dt.month_name()
    month_order = list(pd.date_range("2020-01-01", periods=12, freq="MS").month_name())
    fig, ax = plt.subplots()
    sns.boxplot(data=tmp, x="month_name", y=TARGET, order=month_order, ax=ax, color="#4c9ed9")
    ax.set_title("Seasonal Distribution of Four-Hour Performance by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("% seen within 4 hours")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_seasonal_boxplot.png")
    plt.close(fig)

    # 3. Attendance vs performance
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="total_attendances", y=TARGET, hue="crs_period" if "crs_period" in df else None, ax=ax)
    ax.set_title("Total Attendances vs Four-Hour Performance")
    ax.set_xlabel("Total attendances")
    ax.set_ylabel("% seen within 4 hours")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_attendance_vs_performance.png")
    plt.close(fig)

    # 4. Structural breaks annotated
    fig, ax = plt.subplots()
    ax.plot(df["period"], df["total_attendances"], label="Total attendances", color="#2e75b6")
    ax2 = ax.twinx()
    ax2.plot(df["period"], df[TARGET], label="4h performance %", color="#c00000")
    ax.axvspan(pd.Timestamp("2019-05-01"), pd.Timestamp("2023-05-01"), alpha=0.1, color="orange")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.1, color="red")
    ax.set_title("Attendances and Performance with Structural Breaks")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_structural_breaks.png")
    plt.close(fig)

    # 5. Missingness / booking availability
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots()
        miss.plot(kind="barh", ax=ax, color="#7f7f7f")
        ax.set_title("Proportion of Missing Values by Column")
        ax.set_xlabel("Missing proportion")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_missingness.png")
        plt.close(fig)

    # Summary stats
    summary = df[[TARGET, "total_attendances", "emergency_admissions"]].describe().round(2)
    summary.to_csv(TABLE_DIR / "eda_summary_statistics.csv")

    split_counts = model_df["split"].value_counts().rename_axis("split").reset_index(name="count")
    split_counts.to_csv(TABLE_DIR / "split_counts.csv", index=False)


def run_models(model_df: pd.DataFrame) -> dict:
    train, val, test = split_dataframe(model_df)
    feature_cols = get_rf_features(model_df)
    train_s, val_s, test_s, _ = scale_features(train, val, test, feature_cols, method="standard")

    results = {}

    # ARIMA
    print("Training ARIMA...")
    arima_model, arima_val_pred, arima_cfg = fit_arima(train, val)
    arima_final = refit_arima_on_train_val(train, val, arima_cfg)
    arima_test_pred = forecast_arima(arima_final, len(test))
    results["ARIMA"] = {
        "val_pred": arima_val_pred,
        "test_pred": arima_test_pred,
        "cfg": arima_cfg,
    }

    # Prophet
    print("Training Prophet...")
    prophet_model, prophet_val_pred, prophet_cfg = fit_prophet(train, val)
    train_val = pd.concat([train, val], ignore_index=True)
    prophet_final = refit_prophet(train_val, prophet_cfg)
    prophet_test_pred = forecast_prophet(prophet_final, test)
    results["Prophet"] = {
        "val_pred": prophet_val_pred,
        "test_pred": prophet_test_pred,
        "cfg": prophet_cfg,
    }

    # Random Forest
    print("Training Random Forest...")
    rf_model, rf_val_pred, rf_cfg = fit_random_forest(train_s, val_s, feature_cols)
    rf_model.fit(
        pd.concat([train_s, val_s])[feature_cols].values,
        pd.concat([train_s, val_s])[TARGET].values,
    )
    rf_test_pred = rf_model.predict(test_s[feature_cols].values)
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    results["RandomForest"] = {
        "val_pred": rf_val_pred,
        "test_pred": rf_test_pred,
        "cfg": rf_cfg,
        "importances": importances,
    }

    # LSTM
    print("Training LSTM...")
    try:
        lstm_model, lstm_val_pred, lstm_cfg, lstm_scaler = fit_lstm(train, val, feature_cols)
        history = pd.concat([train, val], ignore_index=True)
        lstm_test_pred = forecast_lstm(
            lstm_model, history, test, feature_cols, TARGET, lstm_scaler, lstm_cfg["seq_len"]
        )
        results["LSTM"] = {
            "val_pred": lstm_val_pred,
            "test_pred": lstm_test_pred,
            "cfg": lstm_cfg,
        }
    except Exception as exc:
        print(f"LSTM failed: {exc}")
        results["LSTM"] = None

    return results, train, val, test, feature_cols


def run_evaluation(results: dict, test: pd.DataFrame) -> None:
    y_true = test[TARGET].values
    forecasts = {k: v["test_pred"] for k, v in results.items() if v is not None}

    comparison = metrics_table(y_true, forecasts)
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    dm = pairwise_dm_tests(y_true, forecasts)
    dm.to_csv(TABLE_DIR / "diebold_mariano_tests.csv", index=False)

    # Forecast vs actual plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test["period"], y_true, marker="o", linewidth=2, label="Actual", color="black")
    colors = {"ARIMA": "#1f77b4", "Prophet": "#ff7f0e", "RandomForest": "#2ca02c", "LSTM": "#d62728"}
    for name, preds in forecasts.items():
        ax.plot(test["period"], preds, marker="s", linestyle="--", label=name, color=colors.get(name))
    ax.set_title("Test Set Forecasts vs Actual (Apr 2025 – Mar 2026)")
    ax.set_xlabel("Period")
    ax.set_ylabel("% seen within 4 hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_test_forecasts.png")
    plt.close(fig)

    # Residuals for best model
    best = comparison.iloc[0]["model"]
    resid = y_true - forecasts[best]
    fig, ax = plt.subplots()
    ax.bar(test["period"].dt.strftime("%Y-%m"), resid, color="#5b9bd5")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Residuals for Best Model ({best})")
    ax.set_xlabel("Period")
    ax.set_ylabel("Actual - Predicted")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_residuals_best_model.png")
    plt.close(fig)

    # Feature importance chart
    fi_path = TABLE_DIR / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(12)
        fig, ax = plt.subplots()
        sns.barplot(data=fi, y="feature", x="importance", ax=ax, color="#70ad47")
        ax.set_title("Random Forest Feature Importance (Top 12)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "08_feature_importance.png")
        plt.close(fig)

    with open(TABLE_DIR / "pipeline_summary.json", "w") as f:
        json.dump({
            "best_model": best,
            "metrics": comparison.to_dict(orient="records"),
        }, f, indent=2)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))


def run_sensitivity(model_df: pd.DataFrame) -> None:
    """Re-run best statistical and ML models excluding CRS months."""
    no_crs = model_df[model_df["crs_period"] == 0].copy()
    train, val, test = split_dataframe(no_crs)
    if len(test) < 3:
        return

    feature_cols = get_rf_features(no_crs)
    train_s, val_s, test_s, _ = scale_features(train, val, test, feature_cols)

    _, _, arima_cfg = fit_arima(train, val)
    arima_final = refit_arima_on_train_val(train, val, arima_cfg)
    arima_pred = forecast_arima(arima_final, len(test))

    rf_model, _, _ = fit_random_forest(train_s, val_s, feature_cols)
    rf_model.fit(pd.concat([train_s, val_s])[feature_cols].values, pd.concat([train_s, val_s])[TARGET].values)
    rf_pred = rf_model.predict(test_s[feature_cols].values)

    y_true = test[TARGET].values
    sens = metrics_table(y_true, {"ARIMA_noCRS": arima_pred, "RandomForest_noCRS": rf_pred})
    sens.to_csv(TABLE_DIR / "sensitivity_no_crs.csv", index=False)


def main():
    print("Loading data...")
    raw = save_master_dataset()
    model_df = prepare_modeling_frame(raw)
    model_df.to_csv(TABLE_DIR / "modeling_dataset.csv", index=False)

    # Add flags to raw for EDA plots
    from src.features import add_period_flags
    eda_df = add_period_flags(raw)

    print("Running EDA...")
    run_eda(eda_df, model_df)

    print("Training models...")
    results, train, val, test, _ = run_models(model_df)

    print("Evaluating...")
    run_evaluation(results, test)

    print("Sensitivity analysis...")
    run_sensitivity(model_df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
