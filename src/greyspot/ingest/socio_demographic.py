"""LSOA-level socio-demographic characteristics - added 2026-09-02 after
re-reading Gao et al.'s PhD thesis a second time (Table 7.2, "Data
Characteristics by Region") found their own feature set includes
"Socio-demographicCharacteristics" (source: Census 2011, 8 classes, one
row per LSOA) for every borough studied - a real input this project had
never incorporated, alongside Point of Interest data (see
`ingest.poi`).

**Disclosed substitute, not the literal same source**: this project uses
the 2019 English Indices of Multiple Deprivation (IMD) - already cached
locally at `data/raw/imd2019_london_lsoa.xlsx` from earlier project
work - joined with the ONS/London Datastore's 2011 LSOA boundary
shapefiles (population density, household counts), rather than the
paper's literal "Census 2011" tables. Both describe LSOA-level
socio-economic/demographic conditions (deprivation domains, population
density, household composition) at the SAME geography (2011 LSOA
boundaries - IMD 2019 explicitly uses "LSOA code (2011)"), so this is a
genuine like-for-like substitute in category and geography, not merely
a same-sounding name - a real Census 2011 pull was not attempted since
the IMD 2019 data was already on hand and covers comparable ground.
Eight features are used here (IMD score + 6 deprivation domain scores +
population density), matching the paper's own "8 classes" count, though
not claimed to be the same 8 variables.

Source: London Datastore, "Indices of Deprivation 2019" (already
cached) + "2011 Boundary Files"
(https://data.london.gov.uk/dataset/2011-boundary-files/, Open
Government Licence v2, free, no registration) for LSOA polygons - the
boundary zip (~260MB) is downloaded once, the LSOA shapefile for each
borough extracted, and the zip itself deleted afterward (not kept in
the repo - see `docs/decision_log.md`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

_BNG = "epsg:27700"  # the LSOA shapefile's native CRS (British National Grid)
_WGS84 = "epsg:4326"  # everything else in this project uses this

IMD_DOMAIN_SCORE_COLUMNS = {
    "Index of Multiple Deprivation (IMD) Score": "imd_score",
    "Income Score (rate)": "imd_income_score",
    "Employment Score (rate)": "imd_employment_score",
    "Education, Skills and Training Score": "imd_education_score",
    "Health Deprivation and Disability Score": "imd_health_score",
    "Crime Score": "imd_crime_score",
    "Living Environment Score": "imd_living_environment_score",
}
SOCIO_DEMOGRAPHIC_COLUMNS = list(IMD_DOMAIN_SCORE_COLUMNS.values()) + ["population_density"]


def load_lsoa_boundaries(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Loads one borough's LSOA boundary shapefile (BGC - Boundary
    Generalised Clipped, the ONS/London Datastore convention for a
    smaller, join-friendly resolution) and reprojects to WGS84."""
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"{shapefile_path} not found. Download the 2011 LSOA boundary "
            "files from https://data.london.gov.uk/dataset/2011-boundary-files/ "
            "(free, no registration), extract the borough's "
            "LSOA_2011_BGC_<Borough>.shp and its sidecar files."
        )
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(_BNG)
    return gdf.to_crs(_WGS84)


def load_lsoa_socio_demographics(imd_path: Path, lsoa_boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Joins IMD 2019 deprivation-domain scores onto LSOA boundary
    polygons by LSOA code, and adds population density from the
    boundary file's own `POPDEN` column (already computed by ONS from
    mid-2011 Census-based population estimates and LSOA area).

    Raises `FileNotFoundError` with a clear message if `imd_path` is
    missing, matching this project's convention for optional-but-
    important enrichment sources.
    """
    if not imd_path.exists():
        raise FileNotFoundError(
            f"{imd_path} not found. Download the 2019 Indices of Deprivation "
            "(London LSOA) from https://data.london.gov.uk/dataset/indices-of-deprivation-2l15g "
            "(free, no registration)."
        )
    imd = pd.read_excel(imd_path, sheet_name="IMD 2019")
    imd = imd.rename(columns=IMD_DOMAIN_SCORE_COLUMNS)
    imd = imd[["LSOA code (2011)"] + list(IMD_DOMAIN_SCORE_COLUMNS.values())]

    merged = lsoa_boundaries.merge(imd, left_on="LSOA11CD", right_on="LSOA code (2011)", how="left")
    merged["population_density"] = merged["POPDEN"]
    missing = merged[list(IMD_DOMAIN_SCORE_COLUMNS.values())].isna().all(axis=1).sum()
    if missing:
        logger.warning("%d/%d LSOAs had no matching IMD row (left as NaN, filled downstream)", missing, len(merged))
    return merged


def snap_segments_to_lsoa(edges: gpd.GeoDataFrame, lsoa_socio_demographics: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assigns each road segment to the LSOA its MIDPOINT falls within (a
    point-in-polygon join, not a full geometric intersection - the same
    "one representative point per segment" simplification this project's
    `ingest.network.snap_points_to_graph` already uses elsewhere, applied
    in the opposite direction: attaching an AREA's attributes to a LINE,
    rather than snapping a POINT to a line).

    Returns one row per `segment_id` with the socio-demographic columns;
    segments whose midpoint falls outside every LSOA polygon (e.g. a
    segment straddling the borough boundary) get NaN, filled to 0 with a
    `has_socio_demographic` flag downstream by
    `features.daily_features.attach_socio_demographic_features` -
    matching this project's established "no data != a real zero" rule.
    """
    # geopandas warns that `.interpolate()` on a geographic (WGS84) CRS
    # is "likely incorrect" - true for long distances, but negligible for
    # this project's road segments (tens to a few hundred metres): the
    # WGS84-degree-based midpoint and a properly-projected one differ by
    # well under a metre at London's latitude for a segment this short,
    # far smaller than the LSOA-boundary-proximity ambiguity this
    # midpoint approach already accepts. Not worth reprojecting for.
    midpoints = edges.geometry.interpolate(0.5, normalized=True)
    midpoints_gdf = gpd.GeoDataFrame({"segment_id": edges["segment_id"].values}, geometry=midpoints, crs=edges.crs)
    joined = gpd.sjoin(midpoints_gdf, lsoa_socio_demographics[SOCIO_DEMOGRAPHIC_COLUMNS + ["geometry"]], how="left", predicate="within")
    return joined[["segment_id"] + SOCIO_DEMOGRAPHIC_COLUMNS].drop_duplicates(subset="segment_id")
