"""Download and load STATS19 collision data, filtered to a chosen local authority.

STATS19 is DfT's road casualty statistics database. Raw CSVs are downloaded
directly from data.dft.gov.uk (confirmed working, no API key required) rather
than through the beta `pystats19` package.

Field note (checked against the live 2025 file on 2026-08-31): the legacy
`local_authority_district` column is no longer populated (always -1) in
recent releases. Use `local_authority_ons_district`, which carries the ONS
GSS code (Westminster = "E09000033"), for geographic filtering instead.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data"
WESTMINSTER_ONS_CODE = "E09000033"

# STATS19 severity codes -> human-readable labels (per DfT data guide).
SEVERITY_LABELS = {1: "fatal", 2: "serious", 3: "slight"}


def collision_csv_url(year: int) -> str:
    return f"{BASE_URL}/dft-road-casualty-statistics-collision-{year}.csv"


def download_collision_year(year: int, raw_dir: Path, timeout: int = 120) -> Path:
    """Download one year's collision CSV into raw_dir if not already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"collision-{year}.csv"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    url = collision_csv_url(year)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dest


def load_collision_years(
    years: list[int],
    raw_dir: Path,
    download_missing: bool = True,
) -> pd.DataFrame:
    """Load and concatenate STATS19 collision CSVs for the given years.

    Returns the raw (unfiltered) national collision table with a parsed
    `datetime` column added.
    """
    frames = []
    for year in years:
        path = raw_dir / f"collision-{year}.csv"
        if not path.exists():
            if not download_missing:
                logger.warning("Missing %s and download_missing=False; skipping", path)
                continue
            download_collision_year(year, raw_dir)
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No STATS19 collision files found for years {years} in {raw_dir}. "
            "If sandboxed network access is unavailable, download these files "
            "manually from https://www.gov.uk/government/statistics/road-safety-data "
            f"into {raw_dir}."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(
        combined["date"] + " " + combined["time"].fillna("00:00"),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    return combined


def filter_to_local_authority(
    df: pd.DataFrame, ons_code: str = WESTMINSTER_ONS_CODE
) -> pd.DataFrame:
    """Filter a national STATS19 collision table to one ONS local authority."""
    mask = df["local_authority_ons_district"] == ons_code
    out = df.loc[mask].copy()
    out["severity_label"] = out["collision_severity"].map(SEVERITY_LABELS)
    logger.info(
        "Filtered %d/%d collisions to local authority %s",
        len(out), len(df), ons_code,
    )
    return out


def load_local_authority_collisions(
    ons_code: str, years: list[int], raw_dir: Path, download_missing: bool = True
) -> pd.DataFrame:
    """Load + filter STATS19 collisions to any one local authority (by ONS
    code). See `ingest.boroughs` for a registry of known boroughs/codes."""
    national = load_collision_years(years, raw_dir, download_missing=download_missing)
    return filter_to_local_authority(national, ons_code)


def load_westminster_collisions(
    years: list[int], raw_dir: Path, download_missing: bool = True
) -> pd.DataFrame:
    """Convenience wrapper: load + filter STATS19 collisions to Westminster."""
    return load_local_authority_collisions(WESTMINSTER_ONS_CODE, years, raw_dir, download_missing)


# --- Casualty / vehicle tables (richer per-collision context) -------------
# STATS19 casualty_type codes 0 (pedestrian) and 1 (cyclist) have been
# stable across specification versions; used here only to build simple,
# clearly-labelled vulnerable-user counts, not to infer anything about the
# people involved.
CASUALTY_TYPE_PEDESTRIAN = 0
CASUALTY_TYPE_CYCLIST = 1


def _generic_csv_url(table: str, year: int) -> str:
    return f"{BASE_URL}/dft-road-casualty-statistics-{table}-{year}.csv"


def _load_table_years(table: str, years: list[int], raw_dir: Path, download_missing: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        path = raw_dir / f"{table}-{year}.csv"
        if not path.exists():
            if not download_missing:
                logger.warning("Missing %s and download_missing=False; skipping", path)
                continue
            raw_dir.mkdir(parents=True, exist_ok=True)
            url = _generic_csv_url(table, year)
            logger.info("Downloading %s -> %s", url, path)
            with requests.get(url, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No {table} files found for years {years} in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def load_casualty_years(years: list[int], raw_dir: Path, download_missing: bool = True) -> pd.DataFrame:
    return _load_table_years("casualty", years, raw_dir, download_missing)


def load_vehicle_years(years: list[int], raw_dir: Path, download_missing: bool = True) -> pd.DataFrame:
    return _load_table_years("vehicle", years, raw_dir, download_missing)


def collision_severity_and_vulnerable_user_features(casualties: pd.DataFrame) -> pd.DataFrame:
    """Per-collision counts: fatal/serious/slight casualties, pedestrian/cyclist
    casualties. Joins onto the collision table via `collision_index`."""
    grouped = casualties.groupby("collision_index")
    out = pd.DataFrame(
        {
            "n_casualties": grouped.size(),
            "n_fatal_casualties": grouped["casualty_severity"].apply(lambda s: (s == 1).sum()),
            "n_serious_casualties": grouped["casualty_severity"].apply(lambda s: (s == 2).sum()),
            "n_slight_casualties": grouped["casualty_severity"].apply(lambda s: (s == 3).sum()),
            "n_pedestrian_casualties": grouped["casualty_type"].apply(
                lambda s: (s == CASUALTY_TYPE_PEDESTRIAN).sum()
            ),
            "n_cyclist_casualties": grouped["casualty_type"].apply(
                lambda s: (s == CASUALTY_TYPE_CYCLIST).sum()
            ),
        }
    ).reset_index()
    return out


def vehicle_mix_features(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Per-collision vehicle count and distinct vehicle-type count."""
    grouped = vehicles.groupby("collision_index")
    out = pd.DataFrame(
        {
            "n_vehicles": grouped.size(),
            "n_distinct_vehicle_types": grouped["vehicle_type"].nunique(),
        }
    ).reset_index()
    return out
