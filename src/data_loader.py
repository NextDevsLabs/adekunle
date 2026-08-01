"""Load and merge NHS England Monthly A&E Time Series workbook."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Repo root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Monthly-AE-Time-Series-March-2026-F5ldj2 (1).xls"
DEFAULT_MASTER_CSV = PROJECT_ROOT / "outputs" / "tables" / "master_dataset.csv"


def _find_header_row(df: pd.DataFrame, marker: str = "Period") -> int:
    for idx, row in df.iterrows():
        values = [str(v).strip() for v in row.values]
        if marker in values and not any(v.startswith(f"{marker}:") for v in values):
            return int(idx)
    raise ValueError(f"Header row with '{marker}' not found")


def _read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_period(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        alt = pd.to_datetime(series, format="%b-%y", errors="coerce")
        parsed = parsed.fillna(alt)
    return parsed.dt.to_period("M").dt.to_timestamp()


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols:
            return cols[key]
    for name in candidates:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def load_activity(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = _read_sheet(path, "Activity")
    period_col = _pick_column(df, ["Period"])
    rename_map = {
        period_col: "period",
        _pick_column(df, ["Type 1 Departments - Major A&E", "A&E attendances type 1"]): "attendances_type1",
        _pick_column(df, ["Type 2 Departments - Single Specialty"]): "attendances_type2",
        _pick_column(df, ["Type 3 Departments - Other A&E/Minor Injury Unit"]): "attendances_type3",
        _pick_column(df, ["Total Attendances", "All A&E attendances"]): "total_attendances",
        _pick_column(df, ["Total Emergency Admissions via A&E", "Emergency Admissions, all types"]): "emergency_admissions",
        _pick_column(df, ["Emergency Admissions via Type 1 A&E"]): "emergency_admissions_type1",
    }
    out = df[[k for k in rename_map if k is not None]].copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k is not None})
    out["period"] = _parse_period(out["period"])
    out = out.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    for col in out.columns:
        if col != "period":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_performance(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = _read_sheet(path, "Performance")
    period_col = _pick_column(df, ["Period"])
    rename_map = {
        period_col: "period",
        _pick_column(df, ["Percentage in 4 hours or less (all)", "Percentage of attendances within 4 hours"]): "pct_within_4h_all",
        _pick_column(df, ["Percentage in 4 hours or less (type 1)"]): "pct_within_4h_type1",
        _pick_column(df, ["Percentage in 4 hours or less (type 2)"]): "pct_within_4h_type2",
        _pick_column(df, ["Percentage in 4 hours or less (type 3)"]): "pct_within_4h_type3",
        _pick_column(df, ["Total Attendances > 4 hours", "A&E attendances greater than 4 hours"]): "attendances_over_4h",
        _pick_column(df, ["Total Attendances < 4 hours"]): "attendances_under_4h",
    }
    out = df[[k for k in rename_map if k is not None]].copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k is not None})
    out["period"] = _parse_period(out["period"])
    out = out.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    for col in out.columns:
        if col != "period":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_booking(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = _read_sheet(path, "Booking")
    period_col = _pick_column(df, ["Period"])
    booking_col = _pick_column(
        df,
        ["A&E Booked Appointment attendances", "Total attendances", "Booked Appointment Attendances"],
    )
    rename_map = {period_col: "period"}
    if booking_col:
        rename_map[booking_col] = "booked_attendances"
    out = df[[k for k in rename_map if k is not None]].copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k is not None})
    out["period"] = _parse_period(out["period"])
    out = out.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
    if "booked_attendances" in out.columns:
        out["booked_attendances"] = pd.to_numeric(out["booked_attendances"], errors="coerce")
    return out


def load_master_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load merged dataset from the NHS Excel workbook, or fall back to master_dataset.csv."""
    if path.exists():
        activity = load_activity(path)
        performance = load_performance(path)
        booking = load_booking(path)

        df = activity.merge(performance, on="period", how="inner")
        df = df.merge(booking, on="period", how="left")

        numeric_cols = [c for c in df.columns if c != "period"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("period").reset_index(drop=True)
        return df

    # Fallback: pre-merged CSV (useful if the large .xls is not present)
    csv_path = DEFAULT_MASTER_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["period"])
        return df.sort_values("period").reset_index(drop=True)

    raise FileNotFoundError(
        f"Data file not found at {path}. "
        f"Place the NHS Monthly A&E Time Series .xls in data/, "
        f"or ensure {csv_path} exists. See data/README.md."
    )


def save_master_dataset(
    output_path: Optional[Path] = None,
    data_path: Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "tables" / "master_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_master_dataset(data_path)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    data = save_master_dataset()
    print(f"Rows: {len(data)}")
    print(f"Period range: {data['period'].min()} to {data['period'].max()}")
    print(data.columns.tolist())
