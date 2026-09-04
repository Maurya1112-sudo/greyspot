"""Daily-granularity, 14-day-multi-step evaluation, built specifically to
be genuinely comparable to Gao et al. (2024)'s STZITD-GNN (arXiv:2309.05072v4)
- see docs/publication_readiness.md for the full comparison writeup this
script's output feeds into.

Deliberate, disclosed differences from the paper's own setup (every one
recorded here, not discovered later):
  - **Years**: this project's cached STATS19 data is 2021-2025; the paper
    uses 2019. We now use all five (2021-2025 inclusive, extended
    2026-09-01 from the original 2022-2024 - see
    docs/decision_log.md's "data-richness pass" entry for why: more
    calendar coverage means more walk-forward training instances at a
    fixed stride, which is the whole point of this extension) - a
    different period, not a worse one; see docs/publication_readiness.md
    for why year-matching is not actually necessary for a valid
    comparison of the KIND this script produces.
  - **Input window**: the paper states "N = 20" in its experiment setup
    (Section 4.2) without disambiguating whether N is input sequence
    length or (as the same symbol, confusingly, means elsewhere in the
    paper's own metric formulas) node/road count - we read it as input
    sequence length (20 days of history), the more common meaning of N
    in a sequence-model hyperparameter list, and use that reading here.
  - **Walk-forward stride**: 14 days (2026-09-01: reduced from a 90-day
    quarterly stride) - the shortest stride that still keeps every
    instance's target window genuinely disjoint from every other's
    (stride == horizon == 14 means zero overlap and zero gap between
    consecutive target windows), maximising the number of training
    instances the fixed calendar range yields without violating the
    walk-forward discipline (`docs/decision_log.md`'s 2026-08-31
    correction entry). This was one of this project's own previously-named
    candidate reasons for the AccHR@20 gap ("only 11 walk-forward training
    instances... far less supervision than a continuous rolling window
    would give" - `docs/publication_readiness.md`) - now addressed rather
    than left as a documented limitation.

Run from the project root:
    python scripts/run_ucl_comparison.py Westminster
"""
from __future__ import annotations

import logging
import os
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
logger = logging.getLogger("run_ucl_comparison")

YEARS = [2021, 2022, 2023, 2024, 2025]
START_DATE, END_DATE = "2021-01-01", "2025-12-31"
INPUT_WINDOW = 20  # days - see module docstring on the paper's "N = 20"
HORIZON = 14  # days - the paper's own stated forecast window, p=14
STRIDE_DAYS = 14  # == HORIZON - see module docstring: maximum non-overlapping instance density
ROLLING_WINDOWS = (7, 14, 30)  # days - trailing collision-count aggregates, see daily_features.py

# One-off ablation switch (2026-09-01, see the FEATURE_COLUMNS switch below
# for the full rationale): GREYSPOT_ABLATION_ORIGINAL_WINDOW=1 reverts
# *only* the date range/stride to this project's original comparison
# protocol (2022-2024, quarterly stride) - the exact configuration that
# produced the AccHR@20=41.3% figure in docs/publication_readiness.md -
# while leaving FEATURE_COLUMNS at whatever the other switch resolves to.
# Combined with the feature switch, this makes all four cells of the
# 2x2 (old/new window x old/new features) reproducible from one script
# instead of hand-edited constants that are easy to leave in a stale state.
if os.environ.get("GREYSPOT_ABLATION_ORIGINAL_WINDOW") == "1":
    YEARS = [2022, 2023, 2024]
    START_DATE, END_DATE = "2022-01-01", "2024-12-31"
    STRIDE_DAYS = 90

# Architecture/training hyperparameters - defaults are the winner of a
# real, multi-window-verified sweep (2026-09-01, "research paper approach"
# entry in docs/decision_log.md), not an assumption. The annual pipeline's
# earlier finding (1 head + residual, tuned on ~2-3 walk-forward instances)
# does NOT transfer to this daily pipeline's much larger instance count:
# `GAT_HEADS=3` (Gao et al. (2024)'s own reported value, Section 4.2) beat
# the annual-pipeline's `heads=1` by a real, replicated margin - mean
# AccHR@20 49.57% vs 42.25% across 6 expanding-window held-out periods on
# Westminster, and 47.31% on Lambeth (a second borough, confirming this
# isn't a Westminster-specific fluke) - both far too large a gap, and
# checked across too many independent windows, to be sampling noise.
# Every OTHER paper-matched value tried (2-layer GNN, hidden dim 42,
# weight decay 0.01/1e-4, the paper's own lr=1e-3, more epochs, a shorter
# symmetric 14-day input window) either did nothing or made things worse
# once actually measured the same way - see decision_log.md's full sweep
# table before assuming any other paper value would help without testing
# it exactly this way first.
GAT_HEADS = int(os.environ.get("GREYSPOT_GAT_HEADS", "3"))
GAT_LAYERS = int(os.environ.get("GREYSPOT_GAT_LAYERS", "1"))
GAT_HIDDEN = int(os.environ.get("GREYSPOT_GAT_HIDDEN", "16"))
GRU_HIDDEN = int(os.environ.get("GREYSPOT_GRU_HIDDEN", "32"))
GAT_WEIGHT_DECAY = float(os.environ.get("GREYSPOT_GAT_WEIGHT_DECAY", "0.0"))
GAT_EPOCHS = int(os.environ.get("GREYSPOT_GAT_EPOCHS", "200"))
GAT_DECODER = os.environ.get("GREYSPOT_GAT_DECODER", "zip")  # "zip" or "nb" (zero-inflated negative binomial)

# 2026-09-01 data-richness pass: was ["length", "day_of_week", "u_degree",
# "v_degree"] only - a model with *no* historical collision signal at all
# as an input feature (see attach_rolling_collision_features's docstring
# for why this was almost certainly the single biggest driver of the
# AccHR@20 gap recorded in docs/publication_readiness.md). Now includes the
# raw daily count, three trailing rolling sums, the same severity/
# vulnerable-user breakdown the annual pipeline already uses, and static
# AADF exposure - every one of these was already-downloaded STATS19/DfT
# data, not a new external source (see docs/decision_log.md for the one
# genuinely new source considered and why it was deferred - going further
# back than 2021 would require the un-split 1979-latest bulk file, not the
# per-year files this project's downloader targets).
FEATURE_COLUMNS = [
    "length", "day_of_week", "u_degree", "v_degree",
    "collision_count", "collision_count_7d", "collision_count_14d", "collision_count_30d",
    "n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
    "n_pedestrian_casualties", "n_cyclist_casualties",
    "aadf_all_motor_vehicles", "aadf_pedal_cycles", "has_aadf",
]

# One-off ablation switch (2026-09-01): the first enriched run's AccHR@20
# came back *lower* than the pre-enrichment baseline (31.25% vs 41.3%), but
# extending YEARS/START_DATE also moved the held-out window itself
# (2024-10-07 -> 2025-12-18) at the same time the feature set changed - two
# variables moved together, so that comparison is confounded and cannot
# say which change (if either) caused the drop. Setting
# GREYSPOT_ABLATION_BASELINE_FEATURES=1 reverts *only* FEATURE_COLUMNS to
# the original 4-column set while keeping the new YEARS/STRIDE_DAYS/window
# unchanged, isolating the feature-enrichment effect on the identical
# held-out window. See docs/decision_log.md's data-richness entry for the
# result of this ablation - this switch is a diagnostic tool, not a
# permanent configuration knob, and should be removed once the ablation
# question is answered.
if os.environ.get("GREYSPOT_ABLATION_BASELINE_FEATURES") == "1":
    FEATURE_COLUMNS = ["length", "day_of_week", "u_degree", "v_degree"]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"


def main(borough_name: str = "Westminster") -> None:
    borough = get_borough(borough_name)
    borough_slug = slug(borough.name)
    reports_dir = ROOT / "reports" / borough_slug
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== UCL-comparable daily/14-day evaluation: %s ===", borough.name)

    logger.info("Step 1/6: loading STATS19 collisions, years=%s", YEARS)
    collisions = load_local_authority_collisions(borough.ons_code, YEARS, RAW_DIR)
    logger.info("Loaded %d collisions", len(collisions))

    logger.info("Step 2/6: road network graph (OSMnx, cached)")
    graph_cache = INTERIM_DIR / f"{borough_slug}_graph.graphml"
    graph = build_borough_graph(borough.osm_place, cache_path=graph_cache)
    edges = graph_to_edges_gdf(graph)
    logger.info("Graph has %d nodes, %d edges", graph.number_of_nodes(), len(edges))

    logger.info("Step 3/6: snapping collisions to segments")
    snapped = snap_collisions_to_graph(collisions, graph)
    logger.info("Snapped %d/%d (%.1f%%)", len(snapped), len(collisions), 100 * len(snapped) / max(len(collisions), 1))

    logger.info("Step 3a/6: loading casualty severity + AADF exposure for enrichment (data-richness pass)")
    casualties = load_casualty_years(YEARS, RAW_DIR)
    daily_counts = collision_severity_counts_by_segment_day(snapped, casualties)

    aadf_path = RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        snapped_aadf = snap_points_to_graph(aadf, graph, max_distance_m=100)
        n_matched = snapped_aadf["segment_id"].notna().sum()
        logger.info("Snapped %d/%d AADF count-point-years to a segment within 100m", n_matched, len(snapped_aadf))
        if n_matched > 0:
            exposure_agg = aggregate_exposure_features(snapped_aadf)
    else:
        logger.warning("AADF file not found at %s; exposure features will be all-zero/has_aadf=0", aadf_path)

    logger.info("Step 4/6: building enriched segment-day table (%s to %s)", START_DATE, END_DATE)
    table = build_segment_day_table(edges, daily_counts, graph, start_date=START_DATE, end_date=END_DATE)
    table = attach_rolling_collision_features(table, windows=ROLLING_WINDOWS)
    if exposure_agg is not None:
        table = attach_static_exposure_features(table, exposure_agg)
    else:
        table["aadf_all_motor_vehicles"] = 0.0
        table["aadf_pedal_cycles"] = 0.0
        table["has_aadf"] = 0
    logger.info("Segment-day table: %d rows, %d feature columns", len(table), len(FEATURE_COLUMNS))

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    segment_order = edges["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)

    logger.info(
        "Step 5/6: building multi-step walk-forward instances (input=%d days, horizon=%d, stride=%d)",
        INPUT_WINDOW, HORIZON, STRIDE_DAYS,
    )
    instances = build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
    )
    if len(instances) < 2:
        raise ValueError(f"Only {len(instances)} instance(s) - need at least 2 (>=1 train, 1 held-out).")
    logger.info("Built %d instances; training on %d, holding out the last", len(instances), len(instances) - 1)

    # Standardise features (fit on TRAINING instances only, applied to
    # every instance with those same stats) - added 2026-09-01 after
    # discovering the enriched feature set's AADF columns (tens of
    # thousands) versus collision-count columns (single digits) is a 4-5
    # order-of-magnitude scale mismatch fed into a neural network with no
    # normalisation anywhere in this pipeline before now - see
    # `daily_temporal.fit_feature_standardizer`'s docstring and
    # docs/decision_log.md for the full finding. Must happen before the
    # train/held-out split below only in the sense that the *fit* uses
    # training instances - the transform itself is applied to all of them.
    train_x_seqs = [x for x, _, _ in instances[:-1]]
    feature_mean, feature_std = fit_feature_standardizer(train_x_seqs)
    instances = [
        (apply_feature_standardizer(x, feature_mean, feature_std), y, d) for x, y, d in instances
    ]

    # y_multistep from build_daily_multistep_instances is [horizon, N]
    # (time-major, matching every other tensor in this codebase - see
    # daily_temporal.py's module docstring); the model's own output is
    # [N, horizon] (see gat_temporal.py's GATTemporal docstring) - a
    # single, clearly-commented transpose at this integration boundary.
    train_instances = [(x, y.T) for x, y, _ in instances[:-1]]
    held_out_x, held_out_y, held_out_start = instances[-1]

    logger.info(
        "Step 6/6: training the multi-step GAT+GRU+%s model (walk-forward, GPU if available) - "
        "heads=%d layers=%d gat_hidden=%d gru_hidden=%d weight_decay=%s epochs=%d",
        GAT_DECODER.upper(), GAT_HEADS, GAT_LAYERS, GAT_HIDDEN, GRU_HIDDEN, GAT_WEIGHT_DECAY, GAT_EPOCHS,
    )
    model = train_gat_temporal_walkforward(
        train_instances, edge_index, epochs=GAT_EPOCHS, horizon=HORIZON,
        zero_inflated=True, negative_binomial=(GAT_DECODER == "nb"),
        heads=GAT_HEADS, gat_layers=GAT_LAYERS, gat_hidden=GAT_HIDDEN, gru_hidden=GRU_HIDDEN,
        weight_decay=GAT_WEIGHT_DECAY,
        use_residual=True,  # the over-smoothing fix from the annual pipeline - carried over as a starting point
    )
    held_out_pred = predict_gat_temporal(model, held_out_x, edge_index)  # [N, horizon]

    # held_out_y is already [horizon, N] (daily_temporal.py's native,
    # time-major convention - matches ucl_metrics.py's [p, N] directly, no
    # transpose needed). held_out_pred is [N, horizon] (GATTemporal's
    # native output convention - see gat_temporal.py) - this one DOES
    # need transposing to line up with y_true. (Caught by a real
    # ValueError on the first run - both arrays were the wrong way round
    # here initially; see docs/decision_log.md.)
    y_true = held_out_y
    y_pred = held_out_pred.T

    # Manual split-conformal interval, reusing this project's existing
    # conformal machinery (models/conformal.py) - calibrated on the
    # second-to-last training instance, exactly mirroring the annual
    # pipeline's own conformal step (see scripts/run_pipeline.py).
    calib_x, calib_y, _ = instances[-2] if len(instances) >= 3 else instances[0]
    calib_pred = predict_gat_temporal(model, calib_x, edge_index)
    lower, upper = manual_split_conformal_interval(calib_y.T.flatten(), calib_pred.flatten(), held_out_pred.flatten(), confidence_level=0.9)
    lower = lower.reshape(held_out_pred.shape).T
    upper = upper.reshape(held_out_pred.shape).T

    metrics = ucl_metric_suite(y_true, y_pred, lower, upper, top_fraction=0.20)

    logger.info("=== %s: UCL-comparable metrics, held-out window starting %s ===", borough.name, held_out_start.date())
    for key, value in metrics.items():
        logger.info("  %s = %.4f", key, value)

    pd.DataFrame([{"borough": borough.name, "held_out_start": held_out_start, **metrics}]).to_csv(
        reports_dir / "ucl_comparison_metrics.csv", index=False
    )
    logger.info("Written to %s", reports_dir / "ucl_comparison_metrics.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
