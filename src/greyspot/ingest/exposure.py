"""DfT AADF (Annual Average Daily Flow) traffic-count data — the exposure
data the project dossier flags as important (Section 5.2/5B: "raw collision
counts are interpreted alongside the amount of travel where suitable
exposure data can be obtained") but which was missing until now.

Unlike the STATS19 casualty/vehicle enrichment features, AADF is **not**
derived from collision outcomes — it is an independent measurement of
traffic volume — so same-year AADF is safe to use directly as a model
feature with no lagging required to avoid leakage. This mirrors the
dossier's own "count / rate / relative risk" display hierarchy (Section
5B): AADF is the denominator that would let a future phase convert a raw
count into a rate, not a proxy for risk itself.

Source: https://roadtraffic.dft.gov.uk/downloads (DfT Road Traffic
Statistics), confirmed free, no registration, bulk CSV download.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

AADF_COLUMNS = [
    "count_point_id", "year", "local_authority_name", "road_name", "road_category",
    "latitude", "longitude", "pedal_cycles", "all_motor_vehicles",
]


def load_local_authority_aadf(csv_path: Path, local_authority_name: str = "Westminster") -> pd.DataFrame:
    """Load DfT AADF count-point-year records for one local authority.

    The upstream file is ~600k rows/150MB for all of Great Britain, so this
    always filters on load rather than caching an unfiltered copy.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download it from "
            "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip "
            "and extract into data/raw/aadf_raw/."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    out = df.loc[df["local_authority_name"] == local_authority_name, AADF_COLUMNS].copy()
    logger.info(
        "Loaded %d AADF count-point-year records for %s (years %s-%s)",
        len(out), local_authority_name,
        out["year"].min() if len(out) else "n/a", out["year"].max() if len(out) else "n/a",
    )
    return out
