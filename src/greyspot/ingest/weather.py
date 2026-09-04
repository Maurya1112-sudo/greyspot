"""Daily London weather - a genuinely new data source, added 2026-09-01
after re-reading Gao et al.'s full PhD thesis (Table 7.2, "Data
Characteristics by Region"): their own STZITD-GNN feature set explicitly
includes "Meteorological Characteristics" (source: The Met Office, 8
classes, one row per day) for every borough studied. This project's daily
pipeline had zero weather signal before this - a real, disclosed gap
relative to the paper's own inputs, not an assumption that weather
doesn't matter.

Source: Open-Meteo's historical weather archive
(https://open-meteo.com/en/docs/historical-weather-api) - free, no API
key, no registration, matching this project's established "free,
key-less" data-source pattern (OpenFreeMap, OSMnx, STATS19, OS Open
Roads). Not the Met Office's own API (which needs a registered API key)
- a disclosed, deliberate substitution for a source with the same
underlying UK Met Office-derived data but no signup friction.

One London-wide daily series is used for every borough (a single lat/lon
near central London, 51.5074N -0.1278W) rather than a per-borough series
- weather genuinely does not vary meaningfully across London boroughs at
daily grain (a few km apart), unlike genuinely local features (AADF
traffic counts, road network structure) where per-location data matters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LONDON_LAT, LONDON_LON = 51.5074, -0.1278

WEATHER_DAILY_VARS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "sunshine_duration",
]


def download_london_daily_weather(start_date: str, end_date: str, dest: Path, timeout: int = 60) -> Path:
    """Downloads and caches one JSON file of daily London weather for
    [start_date, end_date] (inclusive, "YYYY-MM-DD") from Open-Meteo's
    free historical archive. Overwrites `dest` if it already exists -
    callers that want to avoid re-downloading should check
    `dest.exists()` first (matching this project's other ingest modules'
    caching convention, e.g. `ingest.network.build_borough_graph`)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LONDON_LAT}&longitude={LONDON_LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={','.join(WEATHER_DAILY_VARS)}"
        "&timezone=Europe%2FLondon"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading London daily weather %s to %s -> %s", start_date, end_date, dest)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return dest


def load_london_daily_weather(path: Path) -> pd.DataFrame:
    """Loads a cached Open-Meteo JSON file (see `download_london_daily_weather`)
    into a `date` + one column per `WEATHER_DAILY_VARS` entry DataFrame -
    one row per calendar day, no segment_id (weather is day-level, not
    road-level; broadcast onto every segment via
    `daily_features.attach_daily_weather_features`).

    Raises `FileNotFoundError` with a clear "how to get this" message
    rather than a bare file-not-found, matching this project's convention
    for optional-but-important enrichment sources (e.g.
    `ingest.exposure.load_local_authority_aadf`).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with "
            "greyspot.ingest.weather.download_london_daily_weather(start_date, end_date, path), "
            "or directly from https://open-meteo.com/en/docs/historical-weather-api (free, no API key)."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    daily = raw["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for col in WEATHER_DAILY_VARS:
        if col in daily:
            df[col] = daily[col]
    return df
