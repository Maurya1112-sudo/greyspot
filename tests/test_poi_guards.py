"""Tests for the POI download guards (rule R14, codified 2026-09-05).

The failure these exist to prevent is silent: a truncated Overpass response
produces a plausible-looking POI file, gets cached, and every later run
reuses it. It happened twice - Wandsworth cached 1,637 adjacencies against
an expected ~10,000, Brent cached 0 - and both borough results had to be
quarantined.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from greyspot.ingest import poi as poi_mod
from greyspot.ingest.poi import (
    POI_COUNT_COLUMNS,
    PoiDownloadError,
    assert_poi_counts_plausible,
    bbox_area_km2,
    download_borough_pois,
)

# Westminster's real geocoded bbox, (west, south, east, north).
WESTMINSTER_BBOX = (-0.2015, 51.4827, -0.1114, 51.5350)


def _counts(total: int, n_rows: int = 100) -> pd.DataFrame:
    """A POI-count table whose columns sum to `total`."""
    per_row = total / (n_rows * len(POI_COUNT_COLUMNS))
    return pd.DataFrame({c: [per_row] * n_rows for c in POI_COUNT_COLUMNS})


def test_bbox_area_matches_known_borough_scale():
    """Westminster's bbox is larger than the ~21.5 km2 borough polygon but
    the same order of magnitude - it must not be off by 100x from a
    lat/lon mix-up."""
    area = bbox_area_km2(WESTMINSTER_BBOX)
    assert 25 < area < 60, area


def test_bbox_area_is_unpacked_positionally_not_by_range():
    """Both of London's coordinates sit inside +/-90, so any attempt to
    identify latitudes by numeric range picks wrong. Swapping the pairs
    must therefore change the answer - proving the order is being used."""
    swapped = (51.4827, -0.2015, 51.5350, -0.1114)
    assert bbox_area_km2(swapped) != pytest.approx(bbox_area_km2(WESTMINSTER_BBOX))


def test_density_guard_accepts_outer_london_density():
    """Brent measured 62.3/km2 and is COMPLETE (two byte-identical
    downloads). An inner-London-calibrated floor wrongly rejected it."""
    counts = _counts(int(62.3 * bbox_area_km2(WESTMINSTER_BBOX)))
    assert assert_poi_counts_plausible(counts, "Brent", WESTMINSTER_BBOX) > 40


def test_density_guard_accepts_a_verified_borough_level_density():
    # Westminster measured 357.5 adjacencies/km2 when verified by hand.
    counts = _counts(int(357.5 * bbox_area_km2(WESTMINSTER_BBOX)))
    assert assert_poi_counts_plausible(counts, "irrelevant", WESTMINSTER_BBOX) > 300


def test_density_guard_refuses_a_truncated_download():
    """The real Wandsworth failure measured 23.6/km2 against a floor of 40."""
    counts = _counts(int(23.6 * bbox_area_km2(WESTMINSTER_BBOX)))
    with pytest.raises(PoiDownloadError, match="below the calibrated floor"):
        assert_poi_counts_plausible(counts, "Wandsworth", WESTMINSTER_BBOX)


def test_density_guard_refuses_an_empty_download():
    with pytest.raises(PoiDownloadError):
        assert_poi_counts_plausible(_counts(0), "Brent", WESTMINSTER_BBOX)


def _fake_features(monkeypatch, behaviour):
    """Patch the Overpass call. `behaviour` maps category -> result or exc."""
    calls = []

    def fake(bbox, tags):
        category = next(iter(tags))
        calls.append(category)
        outcome = behaviour[category]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", fake)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)  # no real backoff in tests
    return calls


def _gdf(n: int, category: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {category: ["x"] * n, "geometry": [Point(-0.15, 51.5)] * n}, crs="epsg:4326"
    )


def test_one_failing_category_is_refused_not_silently_dropped(monkeypatch):
    """The exact Wandsworth failure: amenity truncates, everything else
    succeeds, so the result looks valid while missing a feature class."""
    _fake_features(monkeypatch, {
        "shop": _gdf(500, "shop"),
        "amenity": ConnectionError("Response ended prematurely"),
        "leisure": _gdf(500, "leisure"),
        "tourism": _gdf(500, "tourism"),
    })
    with pytest.raises(PoiDownloadError, match=r"amenity"):
        download_borough_pois("Wandsworth")


def test_all_categories_empty_is_refused(monkeypatch):
    """Brent's failure: everything returns zero features, nothing errors."""
    _fake_features(monkeypatch, {c: gpd.GeoDataFrame({"geometry": []}, geometry="geometry",
                                                     crs="epsg:4326")
                                 for c in poi_mod.POI_TAG_CATEGORIES})
    with pytest.raises(PoiDownloadError, match="zero features"):
        download_borough_pois("Brent")


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    """A truncation that clears on retry must NOT fail the whole download."""
    state = {"n": 0}

    def flaky(bbox, tags):
        category = next(iter(tags))
        if category == "amenity":
            state["n"] += 1
            if state["n"] < 3:
                raise ConnectionError("Response ended prematurely")
        return _gdf(500, category)

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", flaky)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    out = download_borough_pois("Wandsworth")
    assert state["n"] == 3, "amenity should have been retried until it succeeded"
    assert len(out) == 2000
    assert set(out.poi_category) == set(poi_mod.POI_TAG_CATEGORIES)


def test_strict_false_preserves_the_old_permissive_behaviour(monkeypatch):
    """Kept so exploratory scripts can still inspect a partial download -
    but it is opt-in, and no run script uses it."""
    _fake_features(monkeypatch, {
        "shop": _gdf(10, "shop"),
        "amenity": ConnectionError("boom"),
        "leisure": _gdf(10, "leisure"),
        "tourism": _gdf(10, "tourism"),
    })
    out = download_borough_pois("Wandsworth", strict=False)
    assert len(out) == 30


def test_timeouts_are_not_retried(monkeypatch):
    """A timeout means the query is too heavy, not that it was unlucky.
    Retrying it just waits the full timeout again - which stalled the City
    of London run for 18 minutes on 2026-09-05."""
    import requests
    attempts = {"n": 0}

    def always_timeout(bbox, tags):
        if next(iter(tags)) == "amenity":
            attempts["n"] += 1
            raise requests.exceptions.Timeout("read timed out")
        return _gdf(500, next(iter(tags)))

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", always_timeout)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    with pytest.raises(PoiDownloadError, match="amenity"):
        download_borough_pois("City of London")
    assert attempts["n"] == 1, "a timeout must be attempted exactly once, got %d" % attempts["n"]


def test_truncations_are_still_retried(monkeypatch):
    """The complementary case: a truncation DOES clear on retry."""
    attempts = {"n": 0}

    def flaky(bbox, tags):
        if next(iter(tags)) == "amenity":
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ConnectionError("Response ended prematurely")
        return _gdf(500, next(iter(tags)))

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", flaky)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    assert len(download_borough_pois("Wandsworth")) == 2000
    assert attempts["n"] == 2


def test_tiled_fallback_recovers_an_oversized_category(monkeypatch):
    """Wandsworth's amenity query failed four times across two days - always
    the same category, never the others. That is one response too large to
    transfer, not bad luck, so retrying identically cannot fix it. Splitting
    the bbox into tiles asks for the same data in deliverable pieces."""
    calls = {"full": 0, "tiles": 0}

    def by_size(bbox, tags):
        west, south, east, north = bbox
        is_full = abs(east - west) > 0.05
        if next(iter(tags)) == "amenity" and is_full:
            calls["full"] += 1
            raise ConnectionError("Response ended prematurely")
        if next(iter(tags)) == "amenity":
            calls["tiles"] += 1
        return _gdf(100, next(iter(tags)))

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", by_size)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    out = download_borough_pois("Wandsworth")
    assert calls["full"] == 3, "should exhaust retries first, got %d" % calls["full"]
    assert calls["tiles"] == 9, "should then fetch a 3x3 grid, got %d" % calls["tiles"]
    assert "amenity" in set(out.poi_category)


def test_tiled_fallback_deduplicates_across_tile_boundaries(monkeypatch):
    """A feature straddling a boundary is returned by several tiles. Left
    in, the duplicates inflate POI density and defeat the density guard."""
    import geopandas as gpd
    from shapely.geometry import Point
    shared = gpd.GeoDataFrame(
        {"amenity": ["pub", "cafe"], "geometry": [Point(-0.15, 51.5), Point(-0.16, 51.5)]},
        index=pd.MultiIndex.from_tuples([("node", 1), ("node", 2)],
                                        names=["element_type", "osmid"]),
        crs="epsg:4326")

    def by_size(bbox, tags):
        west, south, east, north = bbox
        if next(iter(tags)) == "amenity":
            if abs(east - west) > 0.05:
                raise ConnectionError("Response ended prematurely")
            return shared  # every tile returns the SAME two features
        return _gdf(10, next(iter(tags)))

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", by_size)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    out = download_borough_pois("Wandsworth")
    n_amenity = int((out.poi_category == "amenity").sum())
    assert n_amenity == 2, "9 tiles x 2 features must dedup to 2, got %d" % n_amenity


def test_partial_tiling_is_refused(monkeypatch):
    """If one tile fails, returning the other eight would be exactly the
    silent partial download this module exists to prevent."""
    def one_bad_tile(bbox, tags):
        west, south, east, north = bbox
        if next(iter(tags)) == "amenity":
            if abs(east - west) > 0.05:
                raise ConnectionError("Response ended prematurely")
            if west < -0.19:  # one specific tile always fails
                raise ConnectionError("Response ended prematurely")
        return _gdf(50, next(iter(tags)))

    monkeypatch.setattr(poi_mod.ox, "features_from_bbox", one_bad_tile)
    monkeypatch.setattr(poi_mod, "borough_bbox_wgs84", lambda place: WESTMINSTER_BBOX)
    monkeypatch.setattr(poi_mod.time, "sleep", lambda s: None)
    with pytest.raises(PoiDownloadError, match="amenity"):
        download_borough_pois("Wandsworth")
