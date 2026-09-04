"""Second-borough validation - built 2026-09-01 immediately after the
Westminster finding that neither the TCR (severity-weighted) target nor
the paper's own ZITD/Tweedie decoder beat this project's simpler
ZIP+plain-count baseline (docs/decision_log.md's "The full comparison"
table: ZIP+count=49.57%, ZIP+TCR=50.17%, ZITD+TCR=45.76%). Every one of
those numbers was measured on Westminster ONLY - this script re-runs
the same two comparisons (ZIP+TCR and ZITD/Tweedie+TCR) on Lambeth, the
paper's own other directly-relevant case (76.59% AccHR@20 reported),
to check whether "Tweedie doesn't help" is a genuine finding or a
Westminster-specific fluke - the same generalisation check this
project's own methodology already applied to the heads=3 finding
(49.57% Westminster / 47.31% Lambeth, confirming that one was real).

Reuses the same expanding-window walk-forward methodology as
`run_ucl_comparison_multiwindow.py` (see that file's docstring for why a
single held-out window is too coarse to trust) - N_WINDOWS held-out
periods, train on everything strictly before each one, report mean+std
across all of them, not one point estimate.

Known duplication (disclosed, not hidden): re-implements the same
feature-building steps as the other `run_ucl_comparison*.py` scripts
rather than importing a shared function - see those files' own
docstrings for why (a real, disclosed piece of tech debt, not hidden).

Run from the project root: python scripts/run_ucl_comparison_multiwindow_lambeth_tcr.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import ucl_metric_suite  # noqa: E402
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
from greyspot.models.conformal import manual_split_conformal_interval  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ucl_comparison_multiwindow_lambeth_tcr")
TARGET_COL = "tcr_score"


# Deliberately the ORIGINAL 2022-2024/quarterly-stride protocol (12 total
# instances), not the 128-instance dense 2021-2025 config
# `run_ucl_comparison.py` otherwise defaults to - each held-out window
# here needs its OWN full training run (expanding window), so N_WINDOWS
# separate trainings per candidate. At the dense config's instance counts
# (~30-127 training instances per window), a single candidate's full
# multi-window sweep would take hours; at this config's 5-11 training
# instances per window, it takes minutes - the right tradeoff when the
# question is "does architecture X or Y rank better," not "what's the
# absolute best score from every possible data-richness lever at once."
YEARS = [2022, 2023, 2024]
START_DATE, END_DATE = "2022-01-01", "2024-12-31"
INPUT_WINDOW, HORIZON, STRIDE_DAYS = 20, 14, 90
ROLLING_WINDOWS = (7, 14, 30)
N_WINDOWS = 6  # number of held-out windows to evaluate (expanding training window each time)
FEATURE_COLUMNS = [
    "length", "day_of_week", "u_degree", "v_degree",
    "collision_count", "collision_count_7d", "collision_count_14d", "collision_count_30d",
    "n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
    "n_pedestrian_casualties", "n_cyclist_casualties",
    "aadf_all_motor_vehicles", "aadf_pedal_cycles", "has_aadf",
]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

# Both Westminster candidates, side by side, so the same window budget
# directly answers "does the ranking between them flip on a second
# borough" rather than just re-confirming one of them alone.
CANDIDATES = {
    "heads=3, ZIP decoder, TCR target (Lambeth validation)": dict(
        heads=3, gat_layers=1, negative_binomial=False,
    ),
    "heads=3, ZITD/Tweedie decoder, TCR target (Lambeth validation)": dict(
        heads=3, gat_layers=1, negative_binomial=False, tweedie=True, zero_inflated=False,
    ),
}


def build_instances(borough_name: str):
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

    if TARGET_COL not in table.columns:
        raise ValueError(
            f"'{TARGET_COL}' column missing from the segment-day table - "
            "collision_severity_counts_by_segment_day only computes it when the "
            "snapped collisions carry a 'collision_severity' column (real STATS19 "
            "data always does; a synthetic/test fixture might not)."
        )
    instances = build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
        target_col=TARGET_COL,
    )
    return instances, edge_index


def evaluate_config(instances, edge_index, config: dict, n_windows: int) -> list[dict]:
    """Expanding-window walk-forward: for each of the last `n_windows`
    instances, standardise features (fit on everything strictly before
    it), train on everything strictly before it, evaluate on it."""
    results = []
    n = len(instances)
    for held_out_idx in range(n - n_windows, n):
        train_slice = instances[:held_out_idx]
        held_out_x_raw, held_out_y, held_out_start = instances[held_out_idx]

        train_x_seqs = [x for x, _, _ in train_slice]
        mean, std = fit_feature_standardizer(train_x_seqs)
        train_instances = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
        held_out_x = apply_feature_standardizer(held_out_x_raw, mean, std)

        defaults = dict(epochs=200, horizon=HORIZON, zero_inflated=True, use_residual=True, gat_hidden=16, gru_hidden=32)
        model = train_gat_temporal_walkforward(
            train_instances, edge_index, **{**defaults, **config},
        )
        held_out_pred = predict_gat_temporal(model, held_out_x, edge_index)
        y_true = held_out_y
        y_pred = held_out_pred.T

        calib_x_raw, calib_y, _ = train_slice[-1]
        calib_x = apply_feature_standardizer(calib_x_raw, mean, std)
        calib_pred = predict_gat_temporal(model, calib_x, edge_index)
        lower, upper = manual_split_conformal_interval(
            calib_y.T.flatten(), calib_pred.flatten(), held_out_pred.flatten(), confidence_level=0.9
        )
        lower, upper = lower.reshape(held_out_pred.shape).T, upper.reshape(held_out_pred.shape).T

        metrics = ucl_metric_suite(y_true, y_pred, lower, upper, top_fraction=0.20)
        metrics["held_out_start"] = held_out_start
        metrics["n_train_instances"] = len(train_slice)
        results.append(metrics)
        logger.info(
            "  window %s (trained on %d instances): AccHR@20=%.4f",
            held_out_start.date(), len(train_slice), metrics["AccHR"],
        )
    return results


def main(borough_name: str = "Westminster") -> None:
    logger.info("=== Multi-window AccHR@20 evaluation: %s ===", borough_name)
    instances, edge_index = build_instances(borough_name)
    logger.info("Built %d total instances; evaluating the last %d as held-out windows", len(instances), N_WINDOWS)
    if len(instances) < N_WINDOWS + 2:
        raise ValueError(f"Only {len(instances)} instances - need at least {N_WINDOWS + 2} for {N_WINDOWS} held-out windows plus training history.")

    reports_dir = ROOT / "reports" / slug(get_borough(borough_name).name)
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    # Now that the target genuinely matches the paper's (tcr_score, not a
    # plain count), every one of their Table 4 metrics is worth reporting
    # here, not just AccHR@20 - the whole point of this script.
    metric_names = ["MAE", "RMSE", "ZR", "AccHR", "MAPE_excl_zeros", "MPIW", "PICP"]
    for name, config in CANDIDATES.items():
        logger.info("--- Candidate: %s ---", name)
        results = evaluate_config(instances, edge_index, config, N_WINDOWS)
        for r in results:
            all_rows.append({"candidate": name, **r})
        summary = {"candidate": name, "n_windows": len(results)}
        for m in metric_names:
            values = [r[m] for r in results if m in r]
            if values:
                summary[f"{m}_mean"] = float(np.mean(values))
                summary[f"{m}_std"] = float(np.std(values))
        summary_rows.append(summary)
        logger.info(
            "=== %s across %d windows: MAE=%.4f RMSE=%.4f ZR=%.4f AccHR=%.4f (std %.4f) PICP=%.4f ===",
            name, len(results), summary.get("MAE_mean", float("nan")), summary.get("RMSE_mean", float("nan")),
            summary.get("ZR_mean", float("nan")), summary.get("AccHR_mean", float("nan")),
            summary.get("AccHR_std", float("nan")), summary.get("PICP_mean", float("nan")),
        )

    pd.DataFrame(all_rows).to_csv(reports_dir / "ucl_multiwindow_per_window_tcr_tweedie.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "ucl_multiwindow_summary_tcr_tweedie.csv", index=False)
    logger.info("Written to %s", reports_dir / "ucl_multiwindow_summary_tcr_tweedie.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
