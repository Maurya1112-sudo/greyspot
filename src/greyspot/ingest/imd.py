"""English Indices of Deprivation 2019 (London, LSOA level) — equity context.

Downloaded manually from the London Datastore (no account required, unlike
OS Open Roads): a direct file link, not an API, so this loader just reads
the cached XLSX rather than fetching it itself. STATS19's
`lsoa_of_accident_location` uses 2011 LSOA codes, which line up directly
with this file's `LSOA code (2011)` column.

This is intentionally a simple, honestly-labelled proxy: it reports the
IMD decile of the LSOA each collision happened in, then averages that
across collisions in a segment. It is NOT a measurement of who lives on
that road, and it says nothing about deprivation where a segment has no
collision history. Per the dossier's equity chapter (Section 15), this
must stay a visible context signal, not a hidden feature.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LSOA_COL = "LSOA code (2011)"
DECILE_COL = "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)"
SCORE_COL = "Index of Multiple Deprivation (IMD) Score"


def load_imd_lookup(xlsx_path: Path) -> pd.DataFrame:
    """Load the LSOA -> IMD decile/score lookup from the London Datastore file."""
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"{xlsx_path} not found. Download it from "
            "https://data.london.gov.uk/dataset/indices-of-deprivation-2l15g "
            "('ID 2019 for London.xlsx') into data/raw/."
        )
    df = pd.read_excel(xlsx_path, sheet_name="IMD 2019")
    lookup = df[[LSOA_COL, DECILE_COL, SCORE_COL]].rename(
        columns={
            LSOA_COL: "lsoa_code",
            DECILE_COL: "imd_decile",
            SCORE_COL: "imd_score",
        }
    )
    return lookup


def attach_imd_to_collisions(collisions: pd.DataFrame, imd_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join each collision to its LSOA's IMD decile/score. Unmatched LSOAs (e.g.
    outside London, or a code-vintage mismatch) get NaN, logged rather than
    silently dropped or imputed."""
    merged = collisions.merge(
        imd_lookup, left_on="lsoa_of_accident_location", right_on="lsoa_code", how="left"
    )
    unmatched = merged["imd_decile"].isna().sum()
    if unmatched:
        logger.warning(
            "%d/%d collisions did not match an LSOA in the IMD 2019 lookup",
            unmatched, len(merged),
        )
    return merged
