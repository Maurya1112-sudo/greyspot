"""Validates the best config found by experiment_gat_architecture.py
(1 attention head + residual connection, PR-AUC 0.424 on the temporal
split - the first time any GAT variant has beaten gat_no_graph all
session) on the harder spatiotemporal split, and re-runs the graph/temporal
ablations *with* the new config to check the improvement is real and not
an artefact of the easier evaluation regime.

Run from the project root, after scripts/run_pipeline.py has built the
Westminster feature table: `python scripts/experiment_gat_best_config_validation.py`
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.metrics import evaluate_predictions  # noqa: E402
from greyspot.eval.splits import spatial_split  # noqa: E402
from greyspot.features.graph_temporal import (  # noqa: E402
    build_segment_adjacency,
    build_target_vector,
    build_temporal_feature_tensor,
    edge_index_from_line_graph,
)
from greyspot.ingest.network import build_westminster_graph, graph_to_edges_gdf  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal  # noqa: E402
from greyspot.models.xgboost_model import FEATURE_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("validate_best_gat")

YEARS = [2021, 2022, 2023, 2024, 2025]
TEST_YEAR = 2024
TABLE_PATH = ROOT / "data" / "processed" / "westminster" / "segment_year_table.parquet"
GRAPH_CACHE = ROOT / "data" / "interim" / "westminster_graph.graphml"

BEST_CONFIG = dict(heads=1, use_residual=True)


def main() -> None:
    table = pd.read_parquet(TABLE_PATH)
    graph = build_westminster_graph(cache_path=GRAPH_CACHE)
    edges = graph_to_edges_gdf(graph)

    segment_order = table["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)
    sequence_years = [y for y in YEARS if y < TEST_YEAR]
    x_seq = build_temporal_feature_tensor(table, segment_order, FEATURE_COLUMNS, years=sequence_years)
    y_test_year = build_target_vector(table, segment_order, year=TEST_YEAR)

    spatial_train_segs, _ = spatial_split(table, holdout_frac=0.2, seed=42)
    spatial_train_segments = set(spatial_train_segs["segment_id"].unique())
    holdout_mask = np.array([seg not in spatial_train_segments for seg in segment_order])
    train_mask = ~holdout_mask

    rows = []

    logger.info("Training: best config (1 head + residual) - temporal")
    model_temporal = train_gat_temporal(x_seq, edge_index, y_test_year, epochs=200, **BEST_CONFIG)
    pred_temporal = predict_gat_temporal(model_temporal, x_seq, edge_index)
    rows.append({"split": "temporal", "model": "gat_best_config", **evaluate_predictions(y_test_year, pred_temporal)})

    logger.info("Training: best config (1 head + residual) - spatiotemporal (train mask excludes holdout segments)")
    model_spatiotemporal = train_gat_temporal(
        x_seq, edge_index, y_test_year, train_mask=train_mask, epochs=200, **BEST_CONFIG
    )
    pred_spatiotemporal = predict_gat_temporal(model_spatiotemporal, x_seq, edge_index)
    rows.append({
        "split": "spatiotemporal", "model": "gat_best_config",
        **evaluate_predictions(y_test_year[holdout_mask], pred_spatiotemporal[holdout_mask]),
    })

    logger.info("Re-running ablations WITH the best config (1 head + residual)")
    model_no_graph = train_gat_temporal(
        x_seq, edge_index, y_test_year, epochs=200, use_graph=False, use_temporal=True, heads=1
    )  # use_residual is a no-op without use_graph, see gat_temporal.py
    pred_no_graph = predict_gat_temporal(model_no_graph, x_seq, edge_index)
    rows.append({"split": "temporal", "model": "gat_no_graph (1 head)", **evaluate_predictions(y_test_year, pred_no_graph)})

    model_no_temporal = train_gat_temporal(
        x_seq, edge_index, y_test_year, epochs=200, use_graph=True, use_temporal=False, **BEST_CONFIG
    )
    pred_no_temporal = predict_gat_temporal(model_no_temporal, x_seq, edge_index)
    rows.append({"split": "temporal", "model": "gat_no_temporal (1 head + residual)", **evaluate_predictions(y_test_year, pred_no_temporal)})

    results_df = pd.DataFrame(rows).set_index(["split", "model"])
    out_path = ROOT / "reports" / "westminster" / "gat_best_config_validation.csv"
    results_df.to_csv(out_path)

    print("\n=== Best-config validation (reference: gat_no_graph temporal=0.360, spatiotemporal untested; xgboost temporal=0.332, spatiotemporal=0.347) ===")
    print(results_df.to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
