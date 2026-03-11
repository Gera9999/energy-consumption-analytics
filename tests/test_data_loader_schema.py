from pathlib import Path

import pandas as pd

from src.data_loader import DataLoaderConfig, load_and_clean_data


def test_load_and_clean_data_on_tiny_csv(tmp_path: Path) -> None:
    # Minimal CSV with UCI-like columns using comma separator.
    content = (
        "Date,Time,Global_active_power,Voltage,Global_intensity\n"
        "16/12/2006,17:24:00,4.216,234.84,18.4\n"
        "16/12/2006,17:25:00,5.360,233.63,23.0\n"
    )
    p = tmp_path / "tiny.csv"
    p.write_text(content, encoding="utf-8")

    cfg = DataLoaderConfig(csv_path=p)
    df = load_and_clean_data(cfg)

    assert isinstance(df, pd.DataFrame)
    assert set(["datetime", "global_active_power"]).issubset(df.columns)
    assert df["datetime"].notna().all()
