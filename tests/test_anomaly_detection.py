import pandas as pd

from src.anomaly_detection import detect_anomalies_isolation_forest


def test_anomaly_detection_outputs_label_column() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=200, freq="h"),
            "global_active_power": [1.0] * 199 + [50.0],
        }
    )

    out = detect_anomalies_isolation_forest(df, target_col="global_active_power")

    assert "anomaly" in out.columns
    assert set(out["anomaly"].unique()).issubset({-1, 1})
    assert len(out) == len(df)
