"""Tests for LSOA-level socio-demographic features (`ingest/socio_demographic.py`)
- synthetic fixtures throughout, no real shapefile/Excel downloads.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from greyspot.ingest.socio_demographic import (
    SOCIO_DEMOGRAPHIC_COLUMNS,
    load_lsoa_boundaries,
    load_lsoa_socio_demographics,
    snap_segments_to_lsoa,
)


def test_load_lsoa_boundaries_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.shp"
    with pytest.raises(FileNotFoundError, match="2011-boundary-files"):
        load_lsoa_boundaries(missing)


def test_load_lsoa_socio_demographics_missing_file_raises_helpful_error(tmp_path):
    lsoa = gpd.GeoDataFrame({"LSOA11CD": ["E01000001"]}, geometry=[box(0, 0, 1, 1)], crs="epsg:4326")
    missing = tmp_path / "does_not_exist.xlsx"
    with pytest.raises(FileNotFoundError, match="indices-of-deprivation"):
        load_lsoa_socio_demographics(missing, lsoa)


def test_load_lsoa_socio_demographics_joins_by_lsoa_code_and_flags_missing(tmp_path):
    lsoa = gpd.GeoDataFrame(
        {"LSOA11CD": ["E01000001", "E01000002"], "POPDEN": [100.0, 200.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="epsg:4326",
    )
    imd_path = tmp_path / "imd2019_london_lsoa.xlsx"
    imd_sheet = pd.DataFrame({
        "LSOA code (2011)": ["E01000001"],  # E01000002 deliberately absent
        "Index of Multiple Deprivation (IMD) Score": [25.0],
        "Income Score (rate)": [0.2],
        "Employment Score (rate)": [0.15],
        "Education, Skills and Training Score": [0.3],
        "Health Deprivation and Disability Score": [0.5],
        "Crime Score": [0.4],
        "Living Environment Score": [0.6],
    })
    with pd.ExcelWriter(imd_path) as writer:
        imd_sheet.to_excel(writer, sheet_name="IMD 2019", index=False)

    out = load_lsoa_socio_demographics(imd_path, lsoa)

    row1 = out[out["LSOA11CD"] == "E01000001"].iloc[0]
    assert row1["imd_score"] == 25.0
    assert row1["population_density"] == 100.0

    row2 = out[out["LSOA11CD"] == "E01000002"].iloc[0]
    assert pd.isna(row2["imd_score"])  # no matching IMD row - left NaN here, filled downstream
    assert row2["population_density"] == 200.0  # POPDEN comes from the boundary file itself, always present


def test_snap_segments_to_lsoa_assigns_by_segment_midpoint():
    # Two LSOAs side by side (x in [0,1] and [1,2]); segment A's midpoint
    # falls in the first, segment B's in the second.
    lsoa = gpd.GeoDataFrame(
        {
            "LSOA11CD": ["E01000001", "E01000002"],
            **{col: [1.0, 2.0] for col in SOCIO_DEMOGRAPHIC_COLUMNS},
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="epsg:4326",
    )
    edges = gpd.GeoDataFrame(
        {"segment_id": ["A", "B"]},
        geometry=[LineString([(0.1, 0.5), (0.3, 0.5)]), LineString([(1.5, 0.5), (1.7, 0.5)])],
        crs="epsg:4326",
    )
    out = snap_segments_to_lsoa(edges, lsoa)

    row_a = out[out["segment_id"] == "A"].iloc[0]
    row_b = out[out["segment_id"] == "B"].iloc[0]
    for col in SOCIO_DEMOGRAPHIC_COLUMNS:
        assert row_a[col] == 1.0
        assert row_b[col] == 2.0


def test_snap_segments_to_lsoa_segment_outside_every_lsoa_is_excluded():
    lsoa = gpd.GeoDataFrame(
        {"LSOA11CD": ["E01000001"], **{col: [1.0] for col in SOCIO_DEMOGRAPHIC_COLUMNS}},
        geometry=[box(0, 0, 1, 1)],
        crs="epsg:4326",
    )
    edges = gpd.GeoDataFrame(
        {"segment_id": ["outside"]},
        geometry=[LineString([(10, 10), (10.2, 10.2)])],  # nowhere near the LSOA
        crs="epsg:4326",
    )
    out = snap_segments_to_lsoa(edges, lsoa)
    # left join with predicate="within" - a segment matching no LSOA is
    # either absent or present with NaN scores, never silently attributed
    # to the wrong LSOA.
    if len(out):
        assert pd.isna(out.iloc[0][SOCIO_DEMOGRAPHIC_COLUMNS[0]])
