import numpy as np
import pandas as pd

from greyspot.features.graph_temporal import (
    build_segment_adjacency,
    build_target_vector,
    build_temporal_feature_tensor,
    edge_index_from_line_graph,
)


def _toy_edges():
    # A junction at node 2 where segments "1_2_0" and "2_3_0" meet - the
    # line graph should connect those two segments, but not a disconnected
    # third segment "9_10_0" sharing no endpoint with either.
    return pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0", "9_10_0"],
            "u": [1, 2, 9],
            "v": [2, 3, 10],
        }
    )


def test_build_segment_adjacency_connects_shared_intersection_only():
    line_graph = build_segment_adjacency(_toy_edges())
    assert line_graph.has_edge("1_2_0", "2_3_0")
    assert not line_graph.has_edge("1_2_0", "9_10_0")
    assert set(line_graph.nodes) == {"1_2_0", "2_3_0", "9_10_0"}


def test_edge_index_from_line_graph_is_bidirectional():
    line_graph = build_segment_adjacency(_toy_edges())
    order = ["1_2_0", "2_3_0", "9_10_0"]
    edge_index = edge_index_from_line_graph(line_graph, order)

    assert edge_index.shape == (2, 2)  # one undirected edge -> two directed entries
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs


def test_edge_index_from_line_graph_handles_no_edges():
    import networkx as nx

    empty_graph = nx.Graph()
    empty_graph.add_nodes_from(["a", "b"])
    edge_index = edge_index_from_line_graph(empty_graph, ["a", "b"])
    assert edge_index.shape == (2, 0)


def test_build_temporal_feature_tensor_places_values_correctly():
    table = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "year": [2022, 2022],
            "feat_a": [10.0, 20.0],
        }
    )
    tensor = build_temporal_feature_tensor(table, ["1_2_0", "2_3_0"], ["feat_a"], years=[2021, 2022])

    assert tensor.shape == (2, 2, 1)
    assert tensor[0].sum() == 0  # 2021 has no rows -> all zero, not garbage
    assert tensor[1, 0, 0] == 10.0
    assert tensor[1, 1, 0] == 20.0


def test_build_target_vector_places_values_correctly():
    table = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "year": [2024, 2024],
            "collision_count": [3, 0],
        }
    )
    y = build_target_vector(table, ["1_2_0", "2_3_0"], year=2024)
    assert np.array_equal(y, np.array([3.0, 0.0], dtype=np.float32))
