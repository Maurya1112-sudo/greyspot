"""Tests for OS Open Roads ingestion (`ingest/os_open_roads.py`).

Uses synthetic fixtures throughout, never the real ~2GB national
GeoPackage (`oproad_gpkg_gb/Data/oproad_gb.gpkg`, downloaded manually by
the user from OS Data Hub, 2026-09-01) - matching this project's existing
policy of small synthetic fixtures for ingestion tests, not live
downloads or large local files, to keep CI fast and independent of any
one machine's disk contents.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from greyspot.ingest import os_open_roads as oor
from greyspot.ingest.network import graph_to_edges_gdf


def _row(**kwargs):
    defaults = dict(
        road_classification="Unclassified", road_function="Local Road", trunk_road=False,
    )
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestMapHighway:
    def test_motorway(self):
        assert oor._map_highway(_row(road_classification="Motorway")) == "motorway"

    def test_trunk_road_overrides_a_road(self):
        # a trunk A-road must map to "trunk", not "primary" - trunk_road is
        # a more specific, policy-relevant fact (Highways England-managed)
        # than the bare classification.
        assert oor._map_highway(_row(road_classification="A Road", trunk_road=True)) == "trunk"

    def test_a_road_without_trunk_flag(self):
        assert oor._map_highway(_row(road_classification="A Road", trunk_road=False)) == "primary"

    def test_b_road(self):
        assert oor._map_highway(_row(road_classification="B Road")) == "secondary"

    def test_classified_unnumbered(self):
        assert oor._map_highway(_row(road_classification="Classified Unnumbered")) == "tertiary"

    def test_local_road_function(self):
        assert oor._map_highway(_row(road_classification="Unclassified", road_function="Local Road")) == "residential"

    @pytest.mark.parametrize(
        "function",
        ["Restricted Local Access Road", "Local Access Road", "Secondary Access Road"],
    )
    def test_access_roads_map_to_service(self, function):
        assert oor._map_highway(_row(road_classification="Unclassified", road_function=function)) == "service"

    def test_unknown_classification_falls_back_to_unclassified(self):
        assert oor._map_highway(_row(road_classification="Unknown", road_function="Minor Road")) == "unclassified"


class TestBuildBoroughGraphOsOpenRoads:
    def _fake_links(self) -> gpd.GeoDataFrame:
        # A tiny synthetic network: three links forming a path A-B-C, plus
        # one fictitious link that must be dropped.
        return gpd.GeoDataFrame(
            {
                "id": ["link-1", "link-2", "link-fictitious"],
                "fictitious": [False, False, True],
                "road_classification": ["A Road", "B Road", "A Road"],
                "road_function": ["A Road", "B Road", "A Road"],
                "trunk_road": [False, False, False],
                "name_1": ["High Street", "Low Street", "Ghost Street"],
                "length": [100.0, 150.0, 50.0],
                "start_node": ["node-A", "node-B", "node-A"],
                "end_node": ["node-B", "node-C", "node-C"],
            },
            geometry=[
                LineString([(529000, 180000), (529100, 180000)]),
                LineString([(529100, 180000), (529200, 180100)]),
                LineString([(529000, 180000), (529200, 180100)]),
            ],
            crs="epsg:27700",
        )

    def test_builds_osmnx_compatible_graph(self, monkeypatch):
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: self._fake_links())

        graph = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-0.2, 51.4, -0.1, 51.5), gpkg_path="fake.gpkg",
        )

        # fictitious link dropped -> only 2 real links, each added in both
        # directions (OS Open Roads has no oneway attribute - see module
        # docstring) -> 4 edges, 3 nodes.
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 4
        assert graph.graph["crs"] == "epsg:4326"

        edges = graph_to_edges_gdf(graph)
        assert set(edges["highway"]) == {"primary", "secondary"}
        assert edges.crs is not None
        assert str(edges.crs).lower().replace(":", "").endswith("4326")
        # segment_id must be present and unique - the shared contract every
        # downstream function (feature building, the GAT line graph, the
        # FastAPI /roads endpoint) depends on.
        assert edges["segment_id"].is_unique

    def test_drops_fictitious_links(self, monkeypatch):
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: self._fake_links())
        graph = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-0.2, 51.4, -0.1, 51.5), gpkg_path="fake.gpkg",
        )
        edges = graph_to_edges_gdf(graph)
        assert "Ghost Street" not in edges["name"].to_numpy()

    def test_bidirectional_edges_share_attributes(self, monkeypatch):
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: self._fake_links())
        graph = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-0.2, 51.4, -0.1, 51.5), gpkg_path="fake.gpkg",
        )
        assert graph.has_edge("node-A", "node-B")
        assert graph.has_edge("node-B", "node-A")
        forward = graph.get_edge_data("node-A", "node-B")[0]
        backward = graph.get_edge_data("node-B", "node-A")[0]
        assert forward["highway"] == backward["highway"] == "primary"

    def _fake_links_with_one_far_away(self) -> gpd.GeoDataFrame:
        # Two links genuinely inside central London (real BNG<->WGS84
        # correspondence, not made-up numbers - both endpoints land
        # around (-0.14, 51.50)), plus one link far away in Essex
        # (around (0.89, 51.66)) that a bbox-only clip would still
        # happily include if the bbox were drawn loosely, but a real
        # borough polygon must exclude.
        return gpd.GeoDataFrame(
            {
                "id": ["link-near-1", "link-near-2", "link-far-away"],
                "fictitious": [False, False, False],
                "road_classification": ["A Road", "B Road", "A Road"],
                "road_function": ["A Road", "B Road", "A Road"],
                "trunk_road": [False, False, False],
                "name_1": ["High Street", "Low Street", "Far Road"],
                "length": [100.0, 150.0, 140.0],
                "start_node": ["node-A", "node-B", "node-FAR1"],
                "end_node": ["node-B", "node-C", "node-FAR2"],
            },
            geometry=[
                LineString([(529000, 180000), (529100, 180000)]),
                LineString([(529100, 180000), (529200, 180100)]),
                LineString([(600000, 200000), (600100, 200100)]),
            ],
            crs="epsg:27700",
        )

    def test_polygon_wgs84_excludes_links_outside_the_real_borough_shape(self, monkeypatch):
        # A real gap found 2026-09-02: without a polygon, a bbox-only
        # clip keeps every link inside the (much larger) bounding box,
        # including ones genuinely outside the borough - see
        # `borough_bbox_wgs84`'s own docstring for the real Westminster
        # numbers this caused (11,347 vs ~5,500 links).
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: self._fake_links_with_one_far_away())
        tight_polygon = box(-0.145, 51.503, -0.139, 51.506)  # covers only the two near links

        graph = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-1.0, 51.0, 1.0, 52.0),  # deliberately loose bbox, would keep all 3 without the polygon filter
            gpkg_path="fake.gpkg",
            polygon_wgs84=tight_polygon,
        )
        edges = graph_to_edges_gdf(graph)
        assert "Far Road" not in edges["name"].to_numpy()
        assert set(edges["name"].to_numpy()) == {"High Street", "Low Street"}

    def test_without_polygon_wgs84_keeps_everything_in_the_bbox(self, monkeypatch):
        # Backwards compatibility: omitting polygon_wgs84 (the default)
        # must reproduce the old bbox-only behaviour exactly.
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: self._fake_links_with_one_far_away())

        graph = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-1.0, 51.0, 1.0, 52.0), gpkg_path="fake.gpkg",
        )
        edges = graph_to_edges_gdf(graph)
        assert "Far Road" in edges["name"].to_numpy()

    def test_cached_graph_round_trips_uuid_node_and_edge_ids(self, monkeypatch, tmp_path):
        # Real bug found 2026-09-02, the first time this cache path was
        # ever actually re-loaded rather than freshly built: OS Open
        # Roads node AND edge ids are UUID-style TOIDs, not OSMnx's own
        # integers - ox.load_graphml's hardcoded default (osmid: int)
        # crashed on both, one at a time, until node_dtypes AND
        # edge_dtypes were both overridden to str.
        links = self._fake_links()
        links["id"] = ["5216C3AC-92A7-486A-A3B4-20B29C6F5982", "00E82119-D153-4C94-8699-EB0CDCB894D1", "ignored"]
        links["start_node"] = ["AAAAAAAA-0000-0000-0000-000000000001", "BBBBBBBB-0000-0000-0000-000000000002", "x"]
        links["end_node"] = ["BBBBBBBB-0000-0000-0000-000000000002", "CCCCCCCC-0000-0000-0000-000000000003", "y"]
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: links)

        cache_path = tmp_path / "graph.graphml"
        built = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-0.2, 51.4, -0.1, 51.5), gpkg_path="fake.gpkg", cache_path=cache_path,
        )
        assert cache_path.exists()

        # Second call loads from the cache instead of re-reading the
        # GeoPackage - this is the call that used to crash.
        monkeypatch.setattr(oor.pyogrio, "read_dataframe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-read the source when cached")))
        loaded = oor.build_borough_graph_os_open_roads(
            bbox_wgs84=(-0.2, 51.4, -0.1, 51.5), gpkg_path="fake.gpkg", cache_path=cache_path,
        )
        assert loaded.number_of_nodes() == built.number_of_nodes()
        assert loaded.number_of_edges() == built.number_of_edges()
        assert "AAAAAAAA-0000-0000-0000-000000000001" in loaded.nodes()
