"""Fast GAT architecture sweep — investigates *why* gat_no_graph beats the
full GAT+GRU model (confirmed across six runs, see docs/decision_log.md),
by testing candidate fixes one at a time: a residual/skip connection
(over-smoothing hypothesis), fewer/more attention heads, and a larger
hidden dimension.

Reuses the already-built Westminster feature table and cached road graph
(`data/processed/westminster/segment_year_table.parquet`,
`data/interim/westminster_graph.graphml`) instead of re-running the full
~90-second ingestion pipeline, so each architecture variant trains in
seconds. Run `scripts/run_pipeline.py` first if those files don't exist yet.

Run from the project root: `python scripts/experiment_gat_architecture.py`
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from greyspot.eval.metrics import evaluate_predictions  # noqa: E402
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
logger = logging.getLogger("experiment_gat_architecture")

YEARS = [2021, 2022, 2023, 2024, 2025]
TEST_YEAR = 2024
TABLE_PATH = ROOT / "data" / "processed" / "westminster" / "segment_year_table.parquet"
GRAPH_CACHE = ROOT / "data" / "interim" / "westminster_graph.graphml"

# Each variant changes exactly one thing relative to "baseline" (the
# pipeline's current default GAT config) so any performance difference is
# attributable to that one change.
VARIANTS = {
    "baseline (no residual, 4 heads, hidden=16)": dict(use_residual=False, heads=4, gat_hidden=16),
    "+ residual connection": dict(use_residual=True, heads=4, gat_hidden=16),
    "1 head (instead of 4)": dict(use_residual=False, heads=1, gat_hidden=16),
    "1 head + residual": dict(use_residual=True, heads=1, gat_hidden=16),
    "larger hidden (64, 4 heads)": dict(use_residual=False, heads=4, gat_hidden=64),
    "larger hidden + residual": dict(use_residual=True, heads=4, gat_hidden=64),
}


def main() -> None:
    if not TABLE_PATH.exists() or not GRAPH_CACHE.exists():
        raise FileNotFoundError(
            f"Run scripts/run_pipeline.py first to build {TABLE_PATH} and {GRAPH_CACHE}."
        )

    table = pd.read_parquet(TABLE_PATH)
    graph = build_westminster_graph(cache_path=GRAPH_CACHE)
    edges = graph_to_edges_gdf(graph)

    segment_order = table["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)
    sequence_years = [y for y in YEARS if y < TEST_YEAR]
    x_seq = build_temporal_feature_tensor(table, segment_order, FEATURE_COLUMNS, years=sequence_years)
    y_test_year = build_target_vector(table, segment_order, year=TEST_YEAR)

    logger.info(
        "Sweeping %d GAT variants on %d segments, %d timesteps, %d features",
        len(VARIANTS), len(segment_order), len(sequence_years), len(FEATURE_COLUMNS),
    )

    rows = []
    for variant_name, kwargs in VARIANTS.items():
        logger.info("Training: %s", variant_name)
        model = train_gat_temporal(x_seq, edge_index, y_test_year, epochs=200, **kwargs)
        pred = predict_gat_temporal(model, x_seq, edge_index)
        rows.append({"variant": variant_name, **evaluate_predictions(y_test_year, pred)})

    results_df = pd.DataFrame(rows).set_index("variant")
    out_path = ROOT / "reports" / "westminster" / "gat_architecture_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path)

    print("\n=== GAT architecture sweep (temporal split, gat_no_graph reference: PR-AUC 0.360) ===")
    print(results_df.to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
