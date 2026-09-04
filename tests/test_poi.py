"""Tests for POI density features (`ingest/poi.py`) - synthetic fixtures
throughout, no real Overpass/network calls, matching this project's
established `test_network.py` pattern of monkeypatching
`ox.distance.nearest_edges` rather than needing a real graph.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from greyspot.ingest import network as network_mod
from greyspot.ingest.poi import POI_COUNT_COLUMNS, POI_TAG_CATEGORIES, count_pois_near_segments


def test_poi_count_columns_matches_one_per_category():
    assert POI_COUNT_COLUMNS == [f"poi_{c}_count" for c in POI_TAG_CATEGORIES]


def test_count_pois_near_segments_counts_by_category(monkeypatch):
    # 3 POIs: two 'shop' snapping to segment "1_2_0", one 'amenity'
    # snapping to segment "2_3_0" - the count table must reflect exactly
    # that split, with every OTHER category column present and 0, not
    # just the categories that happened to have a match.
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        # count_pois_near_segments' default max_distance_m=50.0 means
        # snap_points_to_graph always requests distances.
        assert return_dist is True
        edges = np.array([(1, 2, 0), (1, 2, 0), (2, 3, 0)], dtype=object)
        distances = np.array([0.0001, 0.0001, 0.0001])  # all well within 50m
        return edges, distances

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    pois = gpd.GeoDataFrame(
        {"poi_category": ["shop", "shop", "amenity"]},
        geometry=[Point(-0.1, 51.5), Point(-0.1001, 51.5001), Point(-0.2, 51.51)],
        crs="epsg:4326",
    )
    counts = count_pois_near_segments(pois, graph=object())

    row_12 = counts[counts["segment_id"] == "1_2_0"].iloc[0]
    assert row_12["poi_shop_count"] == 2
    assert row_12["poi_amenity_count"] == 0  # present, zero - not missing entirely
    assert row_12["poi_leisure_count"] == 0
    assert row_12["poi_tourism_count"] == 0

    row_23 = counts[counts["segment_id"] == "2_3_0"].iloc[0]
    assert row_23["poi_amenity_count"] == 1
    assert row_23["poi_shop_count"] == 0


def test_count_pois_near_segments_drops_points_beyond_max_distance(monkeypatch):
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        assert return_dist is True
        edges = np.array([(1, 2, 0), (2, 3, 0)], dtype=object)
        distances = np.array([0.0001, 0.01])  # ~11m, ~1113m
        return edges, distances

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    pois = gpd.GeoDataFrame(
        {"poi_category": ["shop", "shop"]},
        geometry=[Point(-0.1, 51.5), Point(-0.2, 51.51)],
        crs="epsg:4326",
    )
    counts = count_pois_near_segments(pois, graph=object(), max_distance_m=50.0)

    # only the close POI (segment "1_2_0") should count - the far one
    # (rejected by max_distance_m) must not silently attach to "2_3_0".
    assert list(counts["segment_id"]) == ["1_2_0"]
    assert counts.iloc[0]["poi_shop_count"] == 1


def test_poi_fine_taxonomy_has_exactly_twenty_classes():
    # The paper's Table 7.2 states "Point of Interest | Ordnance Survey |
    # 20" classes; this taxonomy exists to match that granularity.
    from greyspot.ingest.poi import POI_FINE_CLASSES, POI_FINE_COLUMNS
    assert len(POI_FINE_CLASSES) == 20
    assert len(POI_FINE_COLUMNS) == 20
    assert len(set(POI_FINE_COLUMNS)) == 20  # no duplicate column names


def test_classify_poi_fine_assigns_by_tag_value():
    from greyspot.ingest.poi import classify_poi_fine
    pois = gpd.GeoDataFrame({
        "amenity": ["school", "pub", "hospital", None],
        "shop": [None, None, None, "supermarket"],
        "geometry": [Point(0, 0)] * 4,
    }, crs="epsg:4326")
    out = classify_poi_fine(pois)
    assert out.tolist() == ["education", "nightlife", "healthcare", "retail_food"]


def test_classify_poi_fine_assigns_each_poi_at_most_one_class():
    # A feature tagged both amenity=pub AND tourism=attraction must count
    # ONCE (first match wins), not be double-counted into two classes and
    # inflate total POI density.
    from greyspot.ingest.poi import classify_poi_fine
    pois = gpd.GeoDataFrame({
        "amenity": ["pub"], "tourism": ["attraction"], "geometry": [Point(0, 0)],
    }, crs="epsg:4326")
    out = classify_poi_fine(pois)
    assert out.tolist() == ["nightlife"]  # earlier in POI_FINE_CLASSES order


def test_classify_poi_fine_returns_na_for_unmatched_values():
    from greyspot.ingest.poi import classify_poi_fine
    pois = gpd.GeoDataFrame({
        "amenity": ["bench"], "geometry": [Point(0, 0)],
    }, crs="epsg:4326")
    assert classify_poi_fine(pois).isna().all()


def test_count_pois_near_segments_fine_produces_all_twenty_columns(monkeypatch):
    # Fixed width regardless of which classes appear in a given borough -
    # same reasoning as the road-class one-hot vocabulary.
    from greyspot.ingest import poi as poi_mod
    from greyspot.ingest.poi import POI_FINE_COLUMNS
    pois = gpd.GeoDataFrame({
        "amenity": ["school", "pub"], "geometry": [Point(-0.1, 51.5), Point(-0.1, 51.5)],
    }, crs="epsg:4326")

    def fake_snap(points, graph, max_distance_m=None):
        out = points.copy()
        out["segment_id"] = ["a_b_0"] * len(out)
        return out

    monkeypatch.setattr(poi_mod, "snap_points_to_graph", fake_snap)
    counts = poi_mod.count_pois_near_segments_fine(pois, graph=object())
    assert list(counts.columns) == ["segment_id"] + POI_FINE_COLUMNS
    assert counts.loc[0, "poi_education_count"] == 1
    assert counts.loc[0, "poi_nightlife_count"] == 1


def test_classify_poi_fine_works_on_the_REAL_download_borough_pois_shape():
    """Regression test for a silent, total failure found 2026-09-03.

    `download_borough_pois` emits only `geometry` + `poi_category` (+ now
    `poi_tag_value`) - it does NOT pass through per-tag columns like
    `amenity`. The fine taxonomy was written and unit-tested against a
    RAW OSMnx frame (which does have those columns), so every test passed
    while the function matched literally nothing on real pipeline data:
    "20040/20040 POIs matched none of the 20 fine classes".

    The other tests in this file construct their own tidy fixtures and
    would not have caught it; this one asserts against the shape the
    pipeline actually produces.
    """
    from greyspot.ingest.poi import classify_poi_fine
    pipeline_shape = gpd.GeoDataFrame({
        "poi_category": ["amenity", "amenity", "shop", "leisure", "tourism", "amenity"],
        "poi_tag_value": ["school", "pub", "supermarket", "park", "hotel", "bench"],
        "geometry": [Point(0, 0)] * 6,
    }, crs="epsg:4326")
    out = classify_poi_fine(pipeline_shape)
    assert out.tolist() == ["education", "nightlife", "retail_food", "park", "accommodation", None] or \
           out.fillna("NA").tolist() == ["education", "nightlife", "retail_food", "park", "accommodation", "NA"]
    # the crucial assertion: NOT everything unmatched
    assert out.notna().sum() == 5


def test_classify_poi_fine_respects_tag_family_not_just_value():
    # `poi_tag_value="park"` under the WRONG family must not match the
    # leisure=park class - otherwise a shop called "park" would count as
    # green space.
    from greyspot.ingest.poi import classify_poi_fine
    wrong_family = gpd.GeoDataFrame({
        "poi_category": ["shop"], "poi_tag_value": ["park"], "geometry": [Point(0, 0)],
    }, crs="epsg:4326")
    assert classify_poi_fine(wrong_family).isna().all()
