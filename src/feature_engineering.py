from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add basic calendar/time features useful for time-series modeling."""

    out = df.copy()
    dt = pd.to_datetime(out[datetime_col], errors="coerce")

    out["hour"] = dt.dt.hour.astype("int16")
    out["day_of_week"] = dt.dt.dayofweek.astype("int16")  # Monday=0
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["month"] = dt.dt.month.astype("int16")

    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "global_active_power",
    windows: tuple[int, ...] = (3, 6, 24, 168),
    min_periods: int = 2,
) -> pd.DataFrame:
    """Add rolling statistics to capture local consumption patterns.

    Parameters
    ----------
    windows:
        Rolling window sizes in number of rows (recommended: hourly data).
    """

    out = df.copy()

    # Ensure sorted by time if datetime exists
    if "datetime" in out.columns:
        out = out.sort_values("datetime")

    s = out[target_col].astype(float)

    for w in windows:
        out[f"rolling_mean_{w}"] = s.rolling(window=w, min_periods=min_periods).mean()
        out[f"rolling_std_{w}"] = s.rolling(window=w, min_periods=min_periods).std()

    # Short-term load variability: std / mean over 24h window
    if f"rolling_mean_24" in out.columns and f"rolling_std_24" in out.columns:
        out["load_variability_24"] = out["rolling_std_24"] / (out["rolling_mean_24"].abs() + 1e-9)

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline (time + rolling features)."""

    out = add_time_features(df)
    out = add_rolling_features(out)
    return out
