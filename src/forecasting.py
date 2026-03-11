from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProphetConfig:
    """Configuration for short-term forecasting with Prophet."""

    horizon_hours: int = 24
    history_days: int = 180
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False
    changepoint_prior_scale: float = 0.05


def _naive_seasonal_forecast(prophet_df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Baseline forecast used when Prophet is unavailable.

    Uses a simple seasonal-naive approach for hourly data:
    - If at least 24 hours exist: repeat the last 24 values for the next 24 hours
    - Else: repeat the last observed value

    Also provides a simple uncertainty band (±10%).
    """

    if prophet_df.empty:
        raise ValueError("Cannot forecast: empty time series")

    prophet_df = prophet_df.sort_values("ds")
    last_ts = prophet_df["ds"].max()
    future_ds = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

    y_hist = prophet_df["y"].astype(float)
    if len(y_hist) >= 24:
        pattern = y_hist.tail(24).to_list()
        yhat = (pattern * ((horizon_hours // 24) + 1))[:horizon_hours]
    else:
        yhat = [float(y_hist.iloc[-1])] * horizon_hours

    yhat = pd.Series(yhat, dtype=float)
    out = pd.DataFrame(
        {
            "ds": future_ds,
            "yhat": yhat,
            "yhat_lower": yhat * 0.90,
            "yhat_upper": yhat * 1.10,
        }
    )
    return out


def forecast_next_24h_prophet(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    target_col: str = "global_active_power",
    config: ProphetConfig | None = None,
) -> pd.DataFrame:
    """Train a Prophet model and forecast the next 24 hours.

    Prophet requires a dataframe with:
    - ds: datetime
    - y: target value

    Returns a forecast dataframe with at least:
    - ds
    - yhat
    - yhat_lower
    - yhat_upper
    """

    if config is None:
        config = ProphetConfig()

    # We keep Prophet as the primary model, but gracefully fall back to a baseline
    # if the Stan backend isn't available (common on Windows without build tools).
    Prophet = None
    try:
        from prophet import Prophet as _Prophet  # type: ignore

        Prophet = _Prophet
    except Exception:
        Prophet = None

    history = df[[datetime_col, target_col]].copy()
    history[datetime_col] = pd.to_datetime(history[datetime_col], errors="coerce")
    history = history.dropna(subset=[datetime_col])
    history = history.sort_values(datetime_col)

    prophet_df = history.rename(columns={datetime_col: "ds", target_col: "y"})

    # Prophet behaves better with a regular frequency. If data isn't regular, resample hourly.
    prophet_df = prophet_df.set_index("ds").resample("h").mean(numeric_only=True)
    prophet_df["y"] = prophet_df["y"].interpolate(method="time", limit_direction="both")
    prophet_df = prophet_df.reset_index()

    # Use a recent history window for short-term forecasting (faster + typically more relevant)
    if config.history_days and len(prophet_df) > 0:
        cutoff = prophet_df["ds"].max() - pd.Timedelta(days=int(config.history_days))
        prophet_df = prophet_df[prophet_df["ds"] >= cutoff].copy()

    if Prophet is None:
        return _naive_seasonal_forecast(prophet_df, horizon_hours=config.horizon_hours)

    try:
        model = Prophet(
            daily_seasonality=config.daily_seasonality,
            weekly_seasonality=config.weekly_seasonality,
            yearly_seasonality=config.yearly_seasonality,
            changepoint_prior_scale=config.changepoint_prior_scale,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=config.horizon_hours, freq="h")
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        # Fall back if Stan backend is missing or optimization fails.
        return _naive_seasonal_forecast(prophet_df, horizon_hours=config.horizon_hours)
