from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for loading and cleaning the smart-meter dataset."""

    csv_path: Path
    datetime_col: str = "datetime"
    target_col: str = "global_active_power"
    optional_numeric_cols: tuple[str, ...] = ("voltage", "global_intensity")


def _read_csv_robust(csv_path: Path) -> pd.DataFrame:
    """Read a CSV that may use different separators/decimal marks.

    The classic UCI "Household Power Consumption" dataset uses `;` as separator and
    commas as decimal separators in some locales.

    We try a few common variants to be resilient.
    """

    read_attempts: list[dict] = [
        {"sep": ";"},
        {"sep": ","},
        # Fallback: python engine can sometimes handle irregular quoting better
        {"engine": "python", "sep": ","},
    ]

    last_err: Optional[Exception] = None
    for kwargs in read_attempts:
        try:
            common = {"na_values": ["?"], "encoding_errors": "ignore"}
            if kwargs.get("engine") == "python":
                df = pd.read_csv(csv_path, **common, **kwargs)
            else:
                df = pd.read_csv(csv_path, low_memory=False, **common, **kwargs)
            # If the separator guess was wrong, pandas may parse the entire header as a single column.
            if df.shape[1] <= 1:
                raise ValueError("Parsed only 1 column; likely wrong separator")
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    raise RuntimeError(f"Failed to read CSV at {csv_path}: {last_err}")


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse datetime from common column conventions.

    Supported formats:
    - A single column named 'datetime' (or any column containing 'date' and 'time')
    - Two columns 'Date' and 'Time' (UCI dataset)
    """

    cols_lower = {c.lower(): c for c in df.columns}

    if "datetime" in cols_lower:
        raw = df[cols_lower["datetime"]]
        return pd.to_datetime(raw, errors="coerce")

    # UCI format: Date + Time
    if "date" in cols_lower and "time" in cols_lower:
        dt_str = df[cols_lower["date"]].astype(str).str.strip() + " " + df[cols_lower["time"]].astype(str).str.strip()
        # UCI uses dd/mm/yyyy
        dt = pd.to_datetime(dt_str, errors="coerce", format="%d/%m/%Y %H:%M:%S")
        if dt.isna().all():
            dt = pd.to_datetime(dt_str, errors="coerce")
        return dt

    # Fallback: try to find a column with 'date' in name
    date_like = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_like:
        dt = pd.to_datetime(df[date_like[0]], errors="coerce")
        return dt

    raise ValueError(
        "Could not find a datetime column. Expected 'datetime' or 'Date'+'Time'. "
        f"Available columns: {list(df.columns)}"
    )


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        # Replace common missing value markers
        s = df[col].replace(["?", "", "NA", "N/A", None], pd.NA)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df


def load_and_clean_data(config: DataLoaderConfig) -> pd.DataFrame:
    """Load and clean electricity smart-meter time-series data.

    Cleaning steps:
    - Robust CSV reading (supports ';' and ',')
    - Parse datetime
    - Convert key columns to numeric
    - Handle missing values via time interpolation (after sorting)

    Returns
    -------
    pd.DataFrame
        Clean dataframe with at least:
        - datetime
        - global_active_power
        - voltage (if present)
        - global_intensity (if present)
    """

    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df_raw = _read_csv_robust(csv_path)

    # Normalize column names to snake_case-like (minimal)
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse datetime
    dt = _parse_datetime(df_raw)
    df[config.datetime_col] = dt

    # Map expected columns from either normalized or original names
    # If dataset already has snake_case, keep those; otherwise try typical UCI column names.
    col_aliases = {
        "global_active_power": ["global_active_power", "global_active_power"],
        "voltage": ["voltage"],
        "global_intensity": ["global_intensity"],
    }

    # If normalized copy doesn't contain the needed columns, try to bring them from raw.
    for canonical, options in col_aliases.items():
        if canonical in df.columns:
            continue
        for opt in options:
            # try raw with spaces
            raw_name = opt.replace("_", " ").title()
            if raw_name in df_raw.columns:
                df[canonical] = df_raw[raw_name]
                break

    # Coerce numerics
    numeric_cols = (config.target_col,) + config.optional_numeric_cols
    df = _coerce_numeric(df, numeric_cols)

    # Keep only relevant columns if present
    keep_cols = [config.datetime_col, config.target_col]
    for col in config.optional_numeric_cols:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].dropna(subset=[config.datetime_col]).copy()

    # Sort, de-duplicate, set index
    df = df.sort_values(config.datetime_col)
    df = df.drop_duplicates(subset=[config.datetime_col])

    df = df.set_index(config.datetime_col)

    # Handle missing values: time-based interpolation then forward/back fill.
    cols_to_interp = [c for c in numeric_cols if c in df.columns]
    if cols_to_interp:
        df[cols_to_interp] = df[cols_to_interp].interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()

    return df.reset_index()


def load_hourly_data(csv_path: Path, target_col: str = "global_active_power") -> pd.DataFrame:
    """Convenience helper: load and resample to hourly frequency.

    Prophet and many analytics tasks benefit from a fixed, lower frequency series.
    """

    cfg = DataLoaderConfig(csv_path=csv_path, target_col=target_col)
    df = load_and_clean_data(cfg)

    df = df.set_index("datetime").sort_index()
    hourly = df.resample("h").mean(numeric_only=True)
    hourly[target_col] = hourly[target_col].interpolate(method="time", limit_direction="both")
    hourly = hourly.reset_index()
    return hourly
