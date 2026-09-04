"""k-fold-style early-stopping sweep - built 2026-09-01 directly after the
single-held-out-instance version of early stopping measurably HURT
results (26.37% mean AccHR@20 vs the fixed-200-epoch 49.57%, see
docs/decision_log.md's "think, think, think" entry). Diagnosed cause:
at only 5-10 training instances per expanding window, ONE held-out
validation instance is too noisy a generalisation signal - it can fail
to improve by chance well before the model has actually converged,
triggering premature stopping (every window finished in ~8-10 seconds).

This script tests the fix that diagnosis implies:
`early_stopping_val_instances` (added to `train_gat_temporal_walkforward`
alongside this script) holds out the LAST several instances instead of
one, and validates against their MEAN loss each epoch - a k-fold-style
noise reduction without the expense of actually training k separate
models. If the diagnosis is right, val_instances=2 or 3 should recover
some or all of the gap back toward the 49.57% fixed-epoch baseline
while still gaining early stopping's real benefit (no hand-picked epoch
count, per-window adaptive training length).

Same expanding-window walk-forward methodology and duplication caveat
as `run_ucl_comparison_multiwindow_earlystop.py` (this file is a copy of
it, CANDIDATES/output paths changed - the same disclosed, not hidden,
tech debt as every other sweep script in this family; see
docs/decision_log.md).

Run from the project root: python scripts/run_ucl_comparison_multiwindow_earlystop_kfold.py
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
logger = logging.getLogger("run_ucl_comparison_multiwindow_earlystop_kfold")


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

# Candidates to compare - the current best-known config (from the single-
# window sweep, docs/decision_log.md) plus one real architectural
# departure worth testing at multi-window resolution now that a single
# window has been shown too coarse to distinguish them.
CANDIDATES = {
    # 2026-09-01: results from the first sweep (see docs/decision_log.md) -
    # heads=3 alone (mean 49.57% vs baseline's 42.25% across 6 windows) was
    # the single best lever found so far, clearly ahead of 2-layer alone
    # (46.96%). This second sweep tests whether they compound, and whether
    # the corrected training hyperparameters found in the actual STTD
    # reference codebase (github.com/STTDAnonymous/STTD - lr=1e-3,
    # weight_decay=1e-4 - more trustworthy than the possibly-OCR-garbled
    # "0.01"/"0.01" figures read from the PDF text) add anything further.
    # 2026-09-01, third sweep: heads=3+2-layer (41.67%) and heads=3+hidden=42
    # +lr=1e-3 (20.55%, likely undertrained at that much lower LR) both did
    # WORSE than heads=3 alone (49.57%) - see docs/decision_log.md. heads=3
    # alone is the best config found; this run validates it generalises to
    # a SECOND borough (Lambeth - the paper's own other directly-relevant
    # case, with its own reported target: 76.59% AccHR@20), the real test
    # of whether this is a genuine finding or a Westminster-specific fluke.
    # 2026-09-01: the single-held-out-instance version of early stopping
    # (val_instances=1, implicitly, in the earlier sweep) scored 26.37% -
    # clearly worse than the fixed-200-epoch 49.57%. This sweep tests
    # whether holding out MORE instances and averaging their validation
    # loss (k-fold-style, `early_stopping_val_instances`) recovers the
    # gap, at two held-out sizes. At only 5-10 training instances per
    # window in this protocol, val_instances=3 already gives up a
    # sizeable fraction of the training set to validation - this is a
    # real, disclosed tradeoff being tested, not assumed to be free.
    "heads=3 + early_stopping(patience=10, val_instances=2, max_epochs=300)": dict(
        heads=3, gat_layers=1, negative_binomial=False, epochs=300,
        early_stopping_patience=10, early_stopping_val_instances=2,
    ),
    "heads=3 + early_stopping(patience=10, val_instances=3, max_epochs=300)": dict(
        heads=3, gat_layers=1, negative_binomial=False, epochs=300,
        early_stopping_patience=10, early_stopping_val_instances=3,
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

    instances = build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
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
    for name, config in CANDIDATES.items():
        logger.info("--- Candidate: %s ---", name)
        results = evaluate_config(instances, edge_index, config, N_WINDOWS)
        for r in results:
            all_rows.append({"candidate": name, **r})
        acchr_values = [r["AccHR"] for r in results]
        summary_rows.append({
            "candidate": name,
            "AccHR_mean": float(np.mean(acchr_values)),
            "AccHR_std": float(np.std(acchr_values)),
            "AccHR_min": float(np.min(acchr_values)),
            "AccHR_max": float(np.max(acchr_values)),
            "n_windows": len(acchr_values),
        })
        logger.info(
            "=== %s: AccHR@20 across %d windows: mean=%.4f std=%.4f (min=%.4f, max=%.4f) ===",
            name, len(acchr_values), np.mean(acchr_values), np.std(acchr_values),
            np.min(acchr_values), np.max(acchr_values),
        )

    pd.DataFrame(all_rows).to_csv(reports_dir / "ucl_multiwindow_per_window_earlystop_kfold.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "ucl_multiwindow_summary_earlystop_kfold.csv", index=False)
    logger.info("Written to %s", reports_dir / "ucl_multiwindow_summary_earlystop_kfold.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
