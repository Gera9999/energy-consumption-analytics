from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestConfig:
    """Configuration for Isolation Forest anomaly detection."""

    contamination: float = 0.01
    random_state: int = 42
    n_estimators: int = 300


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    target_col: str = "global_active_power",
    config: IsolationForestConfig | None = None,
) -> pd.DataFrame:
    """Detect anomalies in electricity consumption using Isolation Forest.

    Isolation Forest is an unsupervised algorithm that isolates observations by
    randomly partitioning the feature space. Points that require fewer splits to
    isolate are more likely to be anomalies.

    Output column
    -------------
    anomaly:
        -1 for anomaly
         1 for normal observation
    """

    if config is None:
        config = IsolationForestConfig()

    out = df.copy()
    x = out[[target_col]].astype(float).to_numpy()

    # Basic imputation safeguard (should already be clean)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).any():
        col_mean = np.nanmean(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    model.fit(x)

    out["anomaly"] = model.predict(x)
    out["anomaly_score"] = model.decision_function(x)

    return out
