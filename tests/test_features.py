import networkx as nx
import pandas as pd

from greyspot.features.build_features import (
    build_segment_year_table,
    collision_counts_by_segment_year,
    node_degrees,
)


def make_toy_graph():
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, key=0)
    g.add_edge(2, 3, key=0)
    return g


def make_toy_edges():
    return pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "u": [1, 2],
            "v": [2, 3],
            "highway": ["residential", "primary"],
            "length": [100.0, 250.0],
            "maxspeed": ["30", "40"],
            "oneway": [False, True],
        }
    )


def test_collision_counts_by_segment_year_aggregates_correctly():
    snapped = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "1_2_0", "2_3_0"],
            "collision_year": [2022, 2022, 2023],
        }
    )
    counts = collision_counts_by_segment_year(snapped)
    row = counts[(counts.segment_id == "1_2_0") & (counts.year == 2022)]
    assert row["collision_count"].iloc[0] == 2


def test_node_degrees_undirected():
    g = make_toy_graph()
    degrees = node_degrees(g)
    # node 2 touches both edges -> degree 2 in the undirected view
    assert degrees[2] == 2


def test_build_segment_year_table_fills_zero_and_computes_lag():
    graph = make_toy_graph()
    edges = make_toy_edges()
    counts = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "1_2_0"],
            "year": [2022, 2023],
            "collision_count": [2, 0],
        }
    )
    table = build_segment_year_table(edges, counts, graph, years=[2022, 2023])

    # every segment x year combination should exist, including the one
    # with zero observed collisions (2_3_0 has no rows in `counts` at all)
    assert len(table) == 4
    unseen = table[(table.segment_id == "2_3_0") & (table.year == 2022)]
    assert unseen["collision_count"].iloc[0] == 0

    # prior_year_count for segment 1_2_0 in 2023 should equal its 2022 count (2)
    row_2023 = table[(table.segment_id == "1_2_0") & (table.year == 2023)]
    assert row_2023["prior_year_count"].iloc[0] == 2
