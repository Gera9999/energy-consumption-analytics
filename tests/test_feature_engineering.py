import pandas as pd

from src.feature_engineering import engineer_features


def test_engineer_features_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=50, freq="h"),
            "global_active_power": range(50),
            "voltage": [230.0] * 50,
            "global_intensity": [10.0] * 50,
        }
    )

    out = engineer_features(df)

    for col in ["hour", "day_of_week", "is_weekend", "rolling_mean_24", "rolling_std_24"]:
        assert col in out.columns

    assert len(out) == len(df)
