"""One-off diagnostic (2026-09-01): four architecturally different GAT
configurations (ZIP heads=1, ZINB heads=1, ZIP heads=3, ZIP 2-layer) all
produced IDENTICAL MAE/RMSE/ZR/AccHR@20 to 4 decimal places on the
original-window UCL comparison - suspicious enough (per this project's own
"too consistent to be right" instinct) to check directly whether the
models are actually learning different predictions at all, rather than
trusting that "different config -> presumably different result."

Builds the segment-day table and instances ONCE, then trains two
configurations in the same process and compares their raw held-out
point-estimate arrays directly (correlation, max absolute difference,
whether either model collapsed to a near-constant prediction).

Run from the project root: python scripts/diagnose_architecture_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.features.build_features import aggregate_exposure_features  # noqa: E402
from greyspot.features.daily_features import (  # noqa: E402
    attach_rolling_collision_features,
    attach_static_exposure_features,
    build_segment_day_table,
    collision_severity_counts_by_segment_day,
)
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    build_daily_multistep_instances,
    fit_feature_standardizer,
)
from greyspot.features.graph_temporal import build_segment_adjacency, edge_index_from_line_graph  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.network import build_borough_graph, graph_to_edges_gdf, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
from greyspot.ingest.stats19 import load_casualty_years, load_local_authority_collisions  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

YEARS = [2022, 2023, 2024]
START_DATE, END_DATE = "2022-01-01", "2024-12-31"
INPUT_WINDOW, HORIZON, STRIDE_DAYS = 20, 14, 90
ROLLING_WINDOWS = (7, 14, 30)
FEATURE_COLUMNS = [
    "length", "day_of_week", "u_degree", "v_degree",
    "collision_count", "collision_count_7d", "collision_count_14d", "collision_count_30d",
    "n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
    "n_pedestrian_casualties", "n_cyclist_casualties",
    "aadf_all_motor_vehicles", "aadf_pedal_cycles", "has_aadf",
]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"


def main(borough_name: str = "Westminster") -> None:
    borough = get_borough(borough_name)
    borough_slug = slug(borough.name)
    collisions = load_local_authority_collisions(borough.ons_code, YEARS, RAW_DIR)
    graph = build_borough_graph(borough.osm_place, cache_path=INTERIM_DIR / f"{borough_slug}_graph.graphml")
    edges = graph_to_edges_gdf(graph)
    snapped = snap_collisions_to_graph(collisions, graph)
    casualties = load_casualty_years(YEARS, RAW_DIR)
    daily_counts = collision_severity_counts_by_segment_day(snapped, casualties)

    aadf_path = RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        snapped_aadf = snap_points_to_graph(aadf, graph, max_distance_m=100)
        if snapped_aadf["segment_id"].notna().sum() > 0:
            exposure_agg = aggregate_exposure_features(snapped_aadf)

    table = build_segment_day_table(edges, daily_counts, graph, start_date=START_DATE, end_date=END_DATE)
    table = attach_rolling_collision_features(table, windows=ROLLING_WINDOWS)
    if exposure_agg is not None:
        table = attach_static_exposure_features(table, exposure_agg)
    else:
        table["aadf_all_motor_vehicles"] = 0.0
        table["aadf_pedal_cycles"] = 0.0
        table["has_aadf"] = 0

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    segment_order = edges["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)

    instances = build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
    )
    train_x_seqs = [x for x, _, _ in instances[:-1]]
    mean, std = fit_feature_standardizer(train_x_seqs)
    instances = [(apply_feature_standardizer(x, mean, std), y, d) for x, y, d in instances]
    train_instances = [(x, y.T) for x, y, _ in instances[:-1]]
    held_out_x, held_out_y, _ = instances[-1]

    print(f"Held-out y stats: sum={held_out_y.sum()}, nonzero cells={int((held_out_y > 0).sum())}")

    configs = {
        "heads=1,layers=1,ZIP": dict(heads=1, gat_layers=1, negative_binomial=False),
        "heads=3,layers=1,ZIP": dict(heads=3, gat_layers=1, negative_binomial=False),
        "heads=1,layers=2,ZIP": dict(heads=1, gat_layers=2, negative_binomial=False),
        "heads=1,layers=1,ZINB": dict(heads=1, gat_layers=1, negative_binomial=True),
    }

    preds = {}
    for name, kwargs in configs.items():
        model = train_gat_temporal_walkforward(
            train_instances, edge_index, epochs=200, horizon=HORIZON, zero_inflated=True,
            use_residual=True, gat_hidden=16, gru_hidden=32, **kwargs,
        )
        pred = predict_gat_temporal(model, held_out_x, edge_index)  # [N, horizon]
        preds[name] = pred
        print(f"\n{name}: pred stats: mean={pred.mean():.6e}, max={pred.max():.6e}, "
              f"std={pred.std():.6e}, n_nonzero={(pred > 1e-6).sum()}/{pred.size}")

    names = list(preds.keys())
    print("\n--- Pairwise comparisons ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = preds[names[i]].flatten(), preds[names[j]].flatten()
            corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
            max_diff = np.abs(a - b).max()
            identical = np.allclose(a, b, atol=1e-10)
            print(f"{names[i]} vs {names[j]}: corr={corr:.6f}, max_abs_diff={max_diff:.3e}, "
                  f"bit-identical={identical}")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
