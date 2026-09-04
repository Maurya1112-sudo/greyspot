"""Regression tests for snap_points_to_graph's handling of osmnx's
nearest_edges return shape, which turned out (on osmnx 2.1.1) to be a 1-D
object array of (u, v, key) tuples rather than three parallel arrays.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from greyspot.ingest import network as network_mod
from greyspot.ingest.network import consolidate_borough_graph


def test_snap_collisions_handles_object_array_of_tuples(monkeypatch):
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        # mimics the real osmnx>=2.0 return shape for multiple points
        return np.array([(1, 2, 0), (2, 3, 0)], dtype=object)

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    collisions = pd.DataFrame({"longitude": [-0.1, -0.2], "latitude": [51.5, 51.51]})
    out = network_mod.snap_collisions_to_graph(collisions, graph=object())

    assert list(out["segment_id"]) == ["1_2_0", "2_3_0"]


def test_snap_collisions_drops_missing_coordinates(monkeypatch):
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        assert len(X) == 1  # the NaN row must already be dropped before this call
        return np.array([(1, 2, 0)], dtype=object)

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    collisions = pd.DataFrame({"longitude": [-0.1, None], "latitude": [51.5, 51.51]})
    out = network_mod.snap_collisions_to_graph(collisions, graph=object())

    assert len(out) == 1
    assert out["segment_id"].iloc[0] == "1_2_0"


def test_snap_points_to_graph_rejects_matches_beyond_max_distance(monkeypatch):
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        assert return_dist is True
        edges = np.array([(1, 2, 0), (2, 3, 0)], dtype=object)
        # first point is very close (0.0001 deg ~ 11m), second is far (0.01 deg ~ 1113m)
        distances = np.array([0.0001, 0.01])
        return edges, distances

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    points = pd.DataFrame({"longitude": [-0.1, -0.2], "latitude": [51.5, 51.51]})
    out = network_mod.snap_points_to_graph(points, graph=object(), max_distance_m=200)

    assert out["segment_id"].iloc[0] == "1_2_0"  # close enough - kept
    assert pd.isna(out["segment_id"].iloc[1])  # too far - rejected, not silently attached


def test_snap_points_to_graph_no_cutoff_keeps_all_matches(monkeypatch):
    def fake_nearest_edges(graph, X, Y, return_dist=False):
        assert return_dist is False
        return np.array([(1, 2, 0)], dtype=object)

    monkeypatch.setattr(network_mod.ox.distance, "nearest_edges", fake_nearest_edges)

    points = pd.DataFrame({"longitude": [-0.1], "latitude": [51.5]})
    out = network_mod.snap_points_to_graph(points, graph=object(), max_distance_m=None)
    assert out["segment_id"].iloc[0] == "1_2_0"


def _tiny_geographic_graph() -> nx.MultiDiGraph:
    """Two nodes ~1m apart (an offset-crossing-style pair OSM might map
    as separate nodes) plus a third node ~35m away - a real, small,
    properly-CRS-tagged graph `consolidate_intersections` can actually
    operate on (unlike the `graph=object()` placeholders used above,
    which only work because those tests monkeypatch away every call that
    would need real graph structure)."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(1, x=-0.1, y=51.5)
    g.add_node(2, x=-0.1, y=51.500009)  # ~1m north of node 1
    g.add_node(3, x=-0.1005, y=51.5)  # ~35m west of node 1
    g.add_edge(1, 2, key=0, length=1.0, geometry=LineString([(-0.1, 51.5), (-0.1, 51.500009)]), highway="residential")
    g.add_edge(2, 1, key=0, length=1.0, geometry=LineString([(-0.1, 51.500009), (-0.1, 51.5)]), highway="residential")
    g.add_edge(1, 3, key=0, length=35.0, geometry=LineString([(-0.1, 51.5), (-0.1005, 51.5)]), highway="residential")
    g.add_edge(3, 1, key=0, length=35.0, geometry=LineString([(-0.1005, 51.5), (-0.1, 51.5)]), highway="residential")
    return g


def test_consolidate_borough_graph_merges_only_genuinely_close_nodes():
    # tolerance=5m: nodes 1 and 2 (~1m apart) must merge into one node;
    # node 3 (~35m away) must stay separate - a real, hand-verified
    # distance-based check, not just "the node count went down."
    result = consolidate_borough_graph(_tiny_geographic_graph(), tolerance_m=5.0)
    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1  # the merged pair's self-edge collapses; only the 1<->3 link remains
    assert result.graph["crs"] == "epsg:4326"  # reprojected back, not left in a projected CRS


def test_consolidate_borough_graph_negligible_tolerance_leaves_distinct_nodes_separate():
    # A negligible (not literally 0.0 - osmnx's own consolidate_intersections
    # degenerates on an exact zero-radius buffer and raises internally,
    # a real quirk of that library, not this project's code) tolerance:
    # nothing is close enough to merge, not even the ~1m pair - confirms
    # this isn't merging nodes unconditionally regardless of tolerance.
    result = consolidate_borough_graph(_tiny_geographic_graph(), tolerance_m=0.001)
    assert result.number_of_nodes() == 3


def test_consolidate_borough_graph_output_is_pipeline_compatible():
    # The consolidated graph must work with graph_to_edges_gdf exactly
    # like a raw OSMnx graph - the whole point of returning a plain
    # WGS84 MultiDiGraph rather than something callers need to know is
    # "special."
    result = consolidate_borough_graph(_tiny_geographic_graph(), tolerance_m=5.0)
    edges = network_mod.graph_to_edges_gdf(result)
    assert len(edges) == 1
    assert "segment_id" in edges.columns


def test_consolidate_borough_graph_uses_cache_when_present(tmp_path):
    cache_path = tmp_path / "consolidated.graphml"
    first = consolidate_borough_graph(_tiny_geographic_graph(), tolerance_m=5.0, cache_path=cache_path)
    assert cache_path.exists()
    # A DIFFERENT input graph, but the cache must be returned unchanged -
    # proves the cache is actually consulted, not silently ignored on
    # the second call.
    different_graph = nx.MultiDiGraph(crs="epsg:4326")
    different_graph.add_node(99, x=0.0, y=0.0)
    second = consolidate_borough_graph(different_graph, tolerance_m=5.0, cache_path=cache_path)
    assert second.number_of_nodes() == first.number_of_nodes() == 2


def _junction_graph() -> nx.MultiDiGraph:
    """A cross-roads: node 0 at the centre with four arms (nodes 1-4),
    every arm bidirectional - so the centre node has 8 incident directed
    edges, matching how both OSMnx and OS Open Roads represent a two-way
    four-arm junction."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    g.add_node(0, x=-0.10, y=51.50)
    coords = {1: (-0.10, 51.501), 2: (-0.10, 51.499), 3: (-0.101, 51.50), 4: (-0.099, 51.50)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
        g.add_edge(0, n, key=0, length=100.0, highway="residential",
                   geometry=LineString([(-0.10, 51.50), (x, y)]))
        g.add_edge(n, 0, key=0, length=100.0, highway="residential",
                   geometry=LineString([(x, y), (-0.10, 51.50)]))
    return g


def test_redistribute_junction_crashes_splits_weight_across_connected_segments():
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"],
        "longitude": [-0.10], "latitude": [51.50],  # right on the junction
        "junction_detail": [13],  # >0 = at a junction
        "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph)

    # 8 incident directed edges -> 8 rows, each 1/8, summing to exactly 1.0
    assert len(out) == 8
    assert out["crash_weight"].sum() == pytest.approx(1.0)
    assert out["crash_weight"].to_numpy() == pytest.approx(0.125)
    # every emitted segment_id must be a real edge of the DIRECTED graph -
    # taking incident edges from an undirected copy would emit reversed
    # pairs that silently match nothing downstream.
    real_ids = {f"{u}_{v}_{k}" for u, v, k in graph.edges(keys=True)}
    assert set(out["segment_id"]) <= real_ids


def test_redistribute_junction_crashes_leaves_non_junction_crashes_alone():
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"],
        "longitude": [-0.10], "latitude": [51.50],
        "junction_detail": [0],  # 0 = "not at or within 20 metres of a junction"
        "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph)
    assert len(out) == 1
    assert out["crash_weight"].iloc[0] == 1.0
    assert out["segment_id"].iloc[0] == "0_1_0"  # unchanged


def test_redistribute_junction_crashes_treats_missing_junction_detail_as_non_junction():
    # -1 in STATS19 means missing/unknown, not "at a junction" - guessing
    # would fabricate redistribution for a third of Lambeth's data.
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"],
        "longitude": [-0.10], "latitude": [51.50],
        "junction_detail": [-1],
        "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph)
    assert len(out) == 1
    assert out["crash_weight"].iloc[0] == 1.0


def test_redistribute_junction_crashes_without_the_column_is_a_no_op():
    # Backwards compatibility: collision tables predating the field must
    # degrade to exactly the old behaviour, not raise.
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"], "longitude": [-0.10], "latitude": [51.50],
        "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph)
    assert len(out) == 1
    assert out["crash_weight"].iloc[0] == 1.0


def test_redistribute_junction_crashes_conserves_total_weight_over_mixed_input():
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["at-junction", "not-at-junction", "unknown"],
        "longitude": [-0.10, -0.10, -0.10], "latitude": [51.50, 51.50, 51.50],
        "junction_detail": [13, 0, -1],
        "segment_id": ["0_1_0", "0_2_0", "0_3_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph)
    # 3 crashes in, total weight 3.0 out - redistribution moves risk
    # between segments, it must never create or destroy any.
    assert out["crash_weight"].sum() == pytest.approx(3.0)
    assert out["collision_index"].nunique() == 3


def test_per_physical_road_is_mathematically_identical_to_per_edge_when_bidirectional():
    # Recorded as a REGRESSION TEST FOR A REASONING ERROR, not a feature:
    # `per_physical_road=True` was added believing it would dilute half as
    # hard, but 1/n_roads shared between a road's two directions is exactly
    # 1/(2*n_roads) per directed edge - identical to what per-edge
    # splitting already does. An experiment was launched on this false
    # premise and produced a bit-identical first window before being
    # stopped. The original test only asserted per-ROAD totals, which both
    # variants satisfy, so it could not distinguish them; this one
    # compares the actual per-edge weight vectors.
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"], "longitude": [-0.10], "latitude": [51.50],
        "junction_detail": [13], "segment_id": ["0_1_0"],
    })
    per_edge = network_mod.redistribute_junction_crashes(crashes, graph, per_physical_road=False)
    per_road = network_mod.redistribute_junction_crashes(crashes, graph, per_physical_road=True)
    assert sorted(per_edge["crash_weight"].round(9)) == sorted(per_road["crash_weight"].round(9))


def test_home_share_keeps_most_weight_on_the_snapped_segment():
    # The genuinely different variant: concentrate rather than dilute.
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"], "longitude": [-0.10], "latitude": [51.50],
        "junction_detail": [13], "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph, home_share=0.5)
    assert out["crash_weight"].sum() == pytest.approx(1.0)  # still conserved
    home = out.loc[out["segment_id"] == "0_1_0", "crash_weight"]
    assert home.iloc[0] == pytest.approx(0.5)
    others = out.loc[out["segment_id"] != "0_1_0", "crash_weight"]
    assert len(others) == 7  # the junction's other seven incident edges
    assert others.sum() == pytest.approx(0.5)


def test_home_share_one_is_equivalent_to_no_redistribution():
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["c1"], "longitude": [-0.10], "latitude": [51.50],
        "junction_detail": [13], "segment_id": ["0_1_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph, home_share=1.0)
    assert len(out) == 8  # rows still emitted for every arm...
    home = out.loc[out["segment_id"] == "0_1_0", "crash_weight"].iloc[0]
    assert home == pytest.approx(1.0)  # ...but all the weight stays home
    assert out.loc[out["segment_id"] != "0_1_0", "crash_weight"].sum() == pytest.approx(0.0)


def test_redistribute_junction_crashes_per_physical_road_still_conserves_over_mixed_input():
    graph = _junction_graph()
    crashes = pd.DataFrame({
        "collision_index": ["at-junction", "not-at-junction"],
        "longitude": [-0.10, -0.10], "latitude": [51.50, 51.50],
        "junction_detail": [13, 0],
        "segment_id": ["0_1_0", "0_2_0"],
    })
    out = network_mod.redistribute_junction_crashes(crashes, graph, per_physical_road=True)
    assert out["crash_weight"].sum() == pytest.approx(2.0)
    assert out["collision_index"].nunique() == 2
