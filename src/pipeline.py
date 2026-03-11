from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.anomaly_detection import IsolationForestConfig, detect_anomalies_isolation_forest
from src.data_loader import load_hourly_data
from src.feature_engineering import engineer_features
from src.forecasting import ProphetConfig, forecast_next_24h_prophet


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end pipeline configuration."""

    csv_path: Path
    target_col: str = "global_active_power"
    anomaly: IsolationForestConfig = IsolationForestConfig()
    forecast: ProphetConfig = ProphetConfig()


@dataclass(frozen=True)
class PipelineResult:
    processed: pd.DataFrame
    forecast: pd.DataFrame


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full analytics pipeline.

    Steps:
    1) Load and clean data
    2) Resample to hourly
    3) Feature engineering
    4) Anomaly detection
    5) Forecast next 24 hours
    """

    hourly = load_hourly_data(config.csv_path, target_col=config.target_col)
    feats = engineer_features(hourly)
    with_anoms = detect_anomalies_isolation_forest(feats, target_col=config.target_col, config=config.anomaly)

    forecast_df = forecast_next_24h_prophet(
        hourly,
        datetime_col="datetime",
        target_col=config.target_col,
        config=config.forecast,
    )

    return PipelineResult(processed=with_anoms, forecast=forecast_df)
