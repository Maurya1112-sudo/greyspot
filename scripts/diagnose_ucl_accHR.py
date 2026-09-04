"""One-off diagnostic (2026-09-01): the enriched-feature GAT run on the
original window/stride jumped AccHR@20 from 41.3% to 89.13% - large enough
to interrogate before trusting, per this project's own established
"too good to be true" discipline (docs/decision_log.md). This script
does NOT retrain anything; it rebuilds the exact same held-out data the
comparison run used and checks:

1. How many actual (day, segment) crash events the metric's denominator
   is built from (`total_crash_days` in `ucl_metrics.accuracy_hit_rate`) -
   a small denominator means the metric is high-variance and a handful of
   hits/misses can swing it by tens of points.
2. How a *trivial* baseline (rank segments purely by their own recent
   raw collision history, no model at all) scores on the identical
   held-out window - if a trivial rule scores nearly as well as the
   trained GAT, the win is coming from "the feature exists" more than
   from the model's architecture, an important, honest caveat either way.

Run from the project root: python scripts/diagnose_ucl_accHR.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.build_features import aggregate_exposure_features  # noqa: E402
from greyspot.features.daily_features import (  # noqa: E402
    attach_rolling_collision_features,
    attach_static_exposure_features,
    build_segment_day_table,
    collision_severity_counts_by_segment_day,
)
from greyspot.features.daily_temporal import build_daily_feature_tensor, build_daily_multistep_instances  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.network import build_borough_graph, graph_to_edges_gdf, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
from greyspot.ingest.stats19 import load_casualty_years, load_local_authority_collisions  # noqa: E402

YEARS = [2022, 2023, 2024]
START_DATE, END_DATE = "2022-01-01", "2024-12-31"
INPUT_WINDOW = 20
HORIZON = 14
STRIDE_DAYS = 90
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

    instances = build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
    )
    held_out_x, held_out_y, held_out_start = instances[-1]
    print(f"Held-out window starts {held_out_start.date()}, shape y={held_out_y.shape}")

    # --- Check 1: how many actual crash events is AccHR@20 built from? ---
    n = held_out_y.shape[1]
    crash_days_total = int((held_out_y > 0).sum())
    print(f"Total (day, segment) crash events in held-out window: {crash_days_total} "
          f"out of {held_out_y.shape[0]} days x {n} segments = {held_out_y.size} cells "
          f"({100 * crash_days_total / held_out_y.size:.3f}% nonzero)")
    per_day_crashes = (held_out_y > 0).sum(axis=1)
    print(f"Crashes per day in held-out window: {per_day_crashes.tolist()}")

    # --- Check 2: trivial baseline - rank purely by the segment's own
    # raw historical collision total over the ENTIRE input window (no
    # model, no graph, no temporal encoder at all) - same score every day.
    collision_count_idx = FEATURE_COLUMNS.index("collision_count")
    naive_score = held_out_x[:, :, collision_count_idx].sum(axis=0)  # [N] - total collisions in the 20-day input window
    naive_pred = np.tile(naive_score, (HORIZON, 1))  # [horizon, N] - same ranking every day
    naive_acchr = accuracy_hit_rate(held_out_y, naive_pred, top_fraction=0.20)
    print(f"Trivial baseline (rank by raw historical collision_count sum, no model): AccHR@20 = {naive_acchr:.4f}")

    # --- Check 3: an even more trivial baseline using ANY history at all
    # (has this segment EVER had a collision in the input window, y/n) -
    # tests whether "having history vs not" alone (not even how MUCH
    # history) explains most of the effect.
    has_history = (held_out_x[:, :, collision_count_idx].sum(axis=0) > 0).astype(float)
    # Break ties randomly among "has history" segments so the top-20% cut
    # isn't an arbitrary index-order artifact when many segments tie at 1.0.
    rng = np.random.default_rng(0)
    tie_break = rng.random(n) * 1e-6
    has_history_pred = np.tile(has_history + tie_break, (HORIZON, 1))
    has_history_acchr = accuracy_hit_rate(held_out_y, has_history_pred, top_fraction=0.20)
    n_with_history = int(has_history.sum())
    print(f"Segments with ANY collision in the 20-day input window: {n_with_history}/{n} "
          f"({100 * n_with_history / n:.1f}%) - top-20% cutoff is {int(round(n * 0.20))} segments")
    print(f"Even more trivial baseline (binary 'has any history'): AccHR@20 = {has_history_acchr:.4f}")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
