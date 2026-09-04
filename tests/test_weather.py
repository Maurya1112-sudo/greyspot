import json

import pandas as pd
import pytest

from greyspot.ingest.weather import load_london_daily_weather


def test_load_london_daily_weather_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="open-meteo"):
        load_london_daily_weather(missing)


def test_load_london_daily_weather_parses_the_real_open_meteo_response_shape(tmp_path):
    # Shaped exactly like a real Open-Meteo historical-archive response
    # (verified 2026-09-01 against a real API call, not invented) - a
    # "daily" object keyed by variable name, each a same-length list.
    path = tmp_path / "weather.json"
    path.write_text(
        json.dumps(
            {
                "latitude": 51.5,
                "longitude": -0.13,
                "daily": {
                    "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    "temperature_2m_mean": [7.7, 11.1, 8.4],
                    "precipitation_sum": [8.3, 9.6, 2.3],
                    "snowfall_sum": [0.0, 0.0, 0.0],
                    "windspeed_10m_max": [30.6, 52.3, 25.9],
                    "relative_humidity_2m_mean": [79, 87, 85],
                    "sunshine_duration": [1200.0, 800.0, 3000.0],
                },
            }
        ),
        encoding="utf-8",
    )
    df = load_london_daily_weather(path)
    assert len(df) == 3
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert df["temperature_2m_mean"].iloc[1] == 11.1
    assert df["precipitation_sum"].iloc[2] == 2.3
