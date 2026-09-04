"""Segment-level line graph + temporal tensor construction for the GAT+GRU model.

The OSMnx road graph has *intersections* as nodes and *segments* as edges.
A GAT needs segments themselves to be the graph nodes, so that attention
aggregates information across neighbouring segments - this is exactly the
"network ripple" neighbourhood the dossier's evidence panel visualises
(Section 5C). Converting edges-of-a-road-graph into nodes-of-a-new-graph is
the classic "line graph" construction: two segments become connected here
iff they share an intersection.

This is implemented directly (not via `networkx.line_graph`) so that the
line graph's node labels are exactly our own `segment_id` strings, with no
risk of edge-orientation relabelling silently breaking the mapping back to
the feature table.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_segment_adjacency(edges: pd.DataFrame) -> nx.Graph:
    """Line graph over road segments: node = segment_id, edge = shares an
    intersection with another segment."""
    line_graph = nx.Graph()
    line_graph.add_nodes_from(edges["segment_id"])

    node_to_segments: dict[int, list[str]] = {}
    for _, row in edges.iterrows():
        for endpoint in (row["u"], row["v"]):
            node_to_segments.setdefault(endpoint, []).append(row["segment_id"])

    for segments_at_node in node_to_segments.values():
        for i in range(len(segments_at_node)):
            for j in range(i + 1, len(segments_at_node)):
                a, b = segments_at_node[i], segments_at_node[j]
                if a != b:
                    line_graph.add_edge(a, b)

    return line_graph


def edge_index_from_line_graph(line_graph: nx.Graph, segment_order: list[str]) -> np.ndarray:
    """Build a PyG-style `edge_index` array (2, E) of integer node indices,
    both directions included (undirected -> two directed entries per edge,
    the convention PyG's message passing expects), ordered per `segment_order`.
    """
    index_of = {seg: i for i, seg in enumerate(segment_order)}
    src, dst = [], []
    for a, b in line_graph.edges():
        ia, ib = index_of[a], index_of[b]
        src += [ia, ib]
        dst += [ib, ia]
    if not src:
        # a graph with no edges still needs a valid (2, 0) shape, not an
        # error, so an isolated/tiny test graph doesn't crash downstream.
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def build_temporal_feature_tensor(
    segment_year_table: pd.DataFrame,
    segment_order: list[str],
    feature_columns: list[str],
    years: list[int],
) -> np.ndarray:
    """Build a [T, N, F] array: one node-feature matrix per year, ordered by
    `years` and `segment_order`. Missing values are filled with 0 (same
    simplification used by the XGBoost pipeline) so a segment with no
    history contributes a defined, non-NaN vector rather than crashing
    training.
    """
    n_years, n_segments, n_features = len(years), len(segment_order), len(feature_columns)
    tensor = np.zeros((n_years, n_segments, n_features), dtype=np.float32)

    indexed = segment_year_table.set_index(["year", "segment_id"])
    seg_index = {seg: i for i, seg in enumerate(segment_order)}

    for t, year in enumerate(years):
        if year not in indexed.index.get_level_values(0):
            continue
        year_rows = indexed.loc[year]
        for seg_id, row in year_rows.iterrows():
            i = seg_index.get(seg_id)
            if i is None:
                continue
            values = pd.to_numeric(row.reindex(feature_columns), errors="coerce").fillna(0.0).to_numpy()
            tensor[t, i, :] = values

    return tensor


def build_walkforward_instances(
    segment_year_table: pd.DataFrame,
    segment_order: list[str],
    feature_columns: list[str],
    years: list[int],
    window: int,
    target_col: str = "collision_count",
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Build a sliding-window sequence of genuinely distinct
    (x_seq, y_target, target_year) transitions - e.g. for years
    [2021,2022,2023,2024] and window=2: ([2021,2022]->2023),
    ([2022,2023]->2024). Each is a real (input years, target year) pair
    with no overlap in *target* year, so a training instance can never be
    identical to the evaluation instance.

    This exists to fix a real bug found 2026-08-31 (see
    docs/decision_log.md): training a GAT directly against the exact same
    year it is then evaluated on is an in-sample fit, not a held-out
    temporal test, no matter how the training loop is written. Training on
    earlier transitions (e.g. 2022->2023) and evaluating only on the final,
    unseen transition (2023->2024) is the sequence-model equivalent of
    XGBoost's already-correct temporal split (train on early-year rows,
    test on the final year's rows).
    """
    sorted_years = sorted(years)
    if len(sorted_years) <= window:
        raise ValueError(
            f"Need more than {window} years to build even one walk-forward instance; got {sorted_years}."
        )
    instances = []
    for i in range(window, len(sorted_years)):
        window_years = sorted_years[i - window : i]
        target_year = sorted_years[i]
        x_seq = build_temporal_feature_tensor(segment_year_table, segment_order, feature_columns, years=window_years)
        y_target = build_target_vector(segment_year_table, segment_order, year=target_year, target_col=target_col)
        instances.append((x_seq, y_target, target_year))
    return instances


def build_target_vector(
    segment_year_table: pd.DataFrame, segment_order: list[str], year: int, target_col: str = "collision_count"
) -> np.ndarray:
    """Target vector [N] for one year, in the same segment order as the tensor."""
    seg_index = {seg: i for i, seg in enumerate(segment_order)}
    y = np.zeros(len(segment_order), dtype=np.float32)
    year_rows = segment_year_table[segment_year_table["year"] == year]
    for _, row in year_rows.iterrows():
        i = seg_index.get(row["segment_id"])
        if i is not None:
            y[i] = row[target_col]
    return y
