"""Gao et al.'s OWN evaluation protocol AND their OWN road network,
finally combined - built 2026-09-02.

**Why this specific combination is worth running.** Both halves have
been tested before, but never together, and the earlier conclusion is
now known to have rested on a confound:

- `run_2019_replication.py` tested their 2019 same-year 6:2:2 protocol
  and concluded "the temporal-protocol difference is NOT the
  explanation for the gap" (46.64% Westminster, 50.04% Lambeth). But
  that script ran on the **OSMnx** network - its own docstring says so
  explicitly, disclosing the OS Open Roads gap as unresolved because
  the GeoPackage was believed missing.
- Later the same day, the real OS Open Roads network turned out to be
  on disk all along, and proved to be **the single largest driver of
  AccHR@20 in the whole investigation** (+11.59 points, p=0.0010 across
  18 paired windows and all three boroughs).

So the 2019 protocol was ruled out while the dominant variable was
held at the wrong setting. Re-running it on the real network is the
one remaining combination where BOTH the evaluation protocol and the
data source match the paper simultaneously - the closest this project
can get to a like-for-like replication, and the fairest possible test
of whether the residual gap is method or data.

Identical to `run_2019_replication.py` in every other respect (2019
data only, 6:2:2 within-year split, the same fixed-epoch-budget
default with `GREYSPOT_2019_EARLY_STOPPING=1` available to switch to
the thesis's literal patience-10 early stopping) - only the network
source changes.

Run from the project root: python scripts/run_2019_replication_os_open_roads.py <Borough>
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
    collision_counts_by_segment_day,
)
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    build_daily_multistep_instances,
    fit_feature_standardizer,
)
from greyspot.features.graph_temporal import build_segment_adjacency, edge_index_from_line_graph  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.network import graph_to_edges_gdf, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
from greyspot.ingest.os_open_roads import borough_bbox_wgs84, borough_polygon_wgs84, build_borough_graph_os_open_roads  # noqa: E402
from greyspot.ingest.stats19 import filter_to_local_authority  # noqa: E402
from greyspot.models.conformal import manual_split_conformal_interval  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_2019_replication_os_open_roads")

OS_OPEN_ROADS_GPKG = ROOT / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"

YEAR = 2019
TRAIN_START, TRAIN_END = "2019-01-01", "2019-07-20"   # ~219 days, 60%
VAL_START, VAL_END = "2019-07-21", "2019-10-01"       # ~73 days, 20%
TEST_START, TEST_END = "2019-10-02", "2019-12-31"     # ~73 days, 20%
INPUT_WINDOW, HORIZON, STRIDE_DAYS = 20, 14, 14
ROLLING_WINDOWS = (7, 14, 30)
EARLY_STOPPING_PATIENCE = 10  # matches the thesis exactly (page 232)
# 2026-09-02: the first run of this script (WITH early stopping, matching
# the thesis literally) finished training in ~18 seconds and scored a
# surprisingly LOW 29.07% - the exact same premature-stopping failure
# mode this project already diagnosed at small instance counts
# (docs/decision_log.md's "think, think, think" entry: single-instance
# validation stops training almost immediately). With only 12 fit
# instances here (comparable to the light protocol's 5-11), that
# diagnosis plausibly applies again even with 3 validation instances
# (better than 1, still small in absolute terms) - conflating the
# temporal-split question this script exists to test with an already-
# known-unreliable stopping mechanism. GREYSPOT_2019_EARLY_STOPPING=0
# (the default here) disables it in favour of the project's other
# established best practice (a fixed epoch budget) specifically so the
# temporal-split variable can be tested in isolation; set it to 1 to
# reproduce the thesis's literal patience=10 setup instead.
USE_EARLY_STOPPING = os.environ.get("GREYSPOT_2019_EARLY_STOPPING", "0") == "1"
FIXED_EPOCHS = 200  # this project's own established best-known epoch count when not early-stopping
FEATURE_COLUMNS = [
    "length", "day_of_week", "u_degree", "v_degree",
    "collision_count", "collision_count_7d", "collision_count_14d", "collision_count_30d",
    "aadf_all_motor_vehicles", "aadf_pedal_cycles", "has_aadf",
]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"


def load_2019_collisions(borough) -> pd.DataFrame:
    """2019 has rolled out of DfT's "last 5 years" per-year download
    window (only 2021-2025 are available as individual files as of
    2026-09-02) - filtered here from the full historical archive
    (`dft-road-casualty-statistics-collision-1979-latest-published-year.csv`,
    downloaded once to `data/raw/collision-historical-full.csv`) instead.
    Reuses `filter_to_local_authority` (the same function every other
    script in this project uses) rather than reimplementing the ONS-code
    filter and `datetime` construction separately, so this stays
    consistent with the rest of the pipeline by construction, not by
    coincidence."""
    historical_path = RAW_DIR / "collision-historical-full.csv"
    if not historical_path.exists():
        raise FileNotFoundError(
            f"{historical_path} not found - download it from "
            "https://data.dft.gov.uk/road-accidents-safety-data/"
            "dft-road-casualty-statistics-collision-1979-latest-published-year.csv "
            "(large, ~1.5GB) before running this script."
        )
    chunks = []
    for chunk in pd.read_csv(historical_path, chunksize=200_000, low_memory=False):
        chunk_2019 = chunk[chunk["collision_year"] == YEAR]
        if len(chunk_2019):
            chunks.append(chunk_2019)
    all_2019 = pd.concat(chunks, ignore_index=True)
    all_2019["datetime"] = pd.to_datetime(
        all_2019["date"] + " " + all_2019["time"].fillna("00:00"),
        format="%d/%m/%Y %H:%M", errors="coerce",
    )
    return filter_to_local_authority(all_2019, borough.ons_code)


def build_split_instances(table, segment_order, start, end):
    dates = pd.date_range(start, end, freq="D")
    return build_daily_multistep_instances(
        table, segment_order, FEATURE_COLUMNS, dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
    )


def main(borough_name: str = "Westminster") -> None:
    borough = get_borough(borough_name)
    borough_slug = slug(borough.name)
    logger.info("=== 2019 replication (thesis's own 6:2:2 protocol): %s ===", borough.name)

    collisions = load_2019_collisions(borough)
    graph = build_borough_graph_os_open_roads(
        borough_bbox_wgs84(borough.osm_place), OS_OPEN_ROADS_GPKG,
        cache_path=INTERIM_DIR / f"{borough_slug}_os_open_roads_graph.graphml",
        polygon_wgs84=borough_polygon_wgs84(borough.osm_place),
    )
    edges = graph_to_edges_gdf(graph)
    snapped = snap_collisions_to_graph(collisions, graph)
    daily_counts = collision_counts_by_segment_day(snapped)

    aadf_path = RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        snapped_aadf = snap_points_to_graph(aadf, graph, max_distance_m=100)
        if snapped_aadf["segment_id"].notna().sum() > 0:
            exposure_agg = aggregate_exposure_features(snapped_aadf)

    table = build_segment_day_table(edges, daily_counts, graph, start_date=TRAIN_START, end_date=TEST_END)
    table = attach_rolling_collision_features(table, windows=ROLLING_WINDOWS)
    if exposure_agg is not None:
        table = attach_static_exposure_features(table, exposure_agg)
    else:
        table["aadf_all_motor_vehicles"] = 0.0
        table["aadf_pedal_cycles"] = 0.0
        table["has_aadf"] = 0

    segment_order = edges["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)

    train_split = build_split_instances(table, segment_order, TRAIN_START, TRAIN_END)
    val_split = build_split_instances(table, segment_order, VAL_START, VAL_END)
    test_split = build_split_instances(table, segment_order, TEST_START, TEST_END)
    logger.info(
        "Split instance counts: train=%d val=%d test=%d (input_window=%d, horizon=%d, stride=%d)",
        len(train_split), len(val_split), len(test_split), INPUT_WINDOW, HORIZON, STRIDE_DAYS,
    )
    if not train_split or not val_split or not test_split:
        raise ValueError("One of train/val/test splits produced zero instances - widen the date ranges or shorten input_window+horizon.")

    # Standardise on TRAIN only (never val/test) - this project's own
    # established leakage discipline, unchanged from every other script.
    train_x_seqs = [x for x, _, _ in train_split]
    mean, std = fit_feature_standardizer(train_x_seqs)

    def standardize(split):
        return [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in split]

    train_instances = standardize(train_split)
    val_instances = standardize(val_split)
    test_instances_raw = test_split  # keep raw x for prediction; y needed both raw and .T

    defaults = dict(
        heads=3, gat_layers=1, negative_binomial=False,
        horizon=HORIZON, zero_inflated=True, use_residual=True,
        gat_hidden=16, gru_hidden=32,
    )
    if USE_EARLY_STOPPING:
        # Real early stopping on the FULL validation split (every
        # instance, not just one) - matches the thesis's own patience=10
        # exactly. See the module-level comment above USE_EARLY_STOPPING
        # for why this is NOT the default here.
        combined_train_and_val = train_instances + val_instances
        logger.info(
            "Training heads=3 WITH early stopping (thesis-literal): %d fit instances, %d validation instances, patience=%d",
            len(train_instances), len(val_instances), EARLY_STOPPING_PATIENCE,
        )
        model = train_gat_temporal_walkforward(
            combined_train_and_val, edge_index, **defaults,
            epochs=300, early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_val_instances=len(val_instances),
        )
    else:
        # Fixed-epoch training (val_instances folded into the fit set too,
        # not wasted - there is no early-stopping validation split to
        # reserve them for) - isolates the temporal-split-protocol
        # variable this script exists to test from the separate,
        # already-diagnosed early-stopping question.
        fit_instances = train_instances + val_instances
        logger.info(
            "Training heads=3 with a FIXED %d-epoch budget (isolating the temporal-split variable): %d fit instances",
            FIXED_EPOCHS, len(fit_instances),
        )
        model = train_gat_temporal_walkforward(fit_instances, edge_index, **defaults, epochs=FIXED_EPOCHS)

    # Evaluate on every test-split instance, report mean+std (this
    # project's own multi-window reporting convention) rather than one
    # point estimate, even though this is a single contiguous held-out
    # block, not a walk-forward.
    calib_x_raw, calib_y, _ = val_split[-1]
    calib_x = apply_feature_standardizer(calib_x_raw, mean, std)
    calib_pred = predict_gat_temporal(model, calib_x, edge_index)

    results = []
    for test_x_raw, test_y, test_start in test_instances_raw:
        test_x = apply_feature_standardizer(test_x_raw, mean, std)
        test_pred = predict_gat_temporal(model, test_x, edge_index)
        y_true = test_y  # already [horizon, N], matching ucl_metric_suite's expected shape
        y_pred = test_pred.T  # predict_gat_temporal returns [N, horizon] for horizon>1

        lower, upper = manual_split_conformal_interval(
            calib_y.T.flatten(), calib_pred.flatten(), test_pred.flatten(), confidence_level=0.9
        )
        lower, upper = lower.reshape(test_pred.shape).T, upper.reshape(test_pred.shape).T

        metrics = ucl_metric_suite(y_true, y_pred, lower, upper, top_fraction=0.20)
        metrics["test_window_start"] = test_start
        results.append(metrics)
        logger.info("  test window %s: AccHR@20=%.4f", test_start.date(), metrics["AccHR"])

    acchr_values = [r["AccHR"] for r in results]
    reports_dir = ROOT / "reports" / borough_slug
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(reports_dir / "2019_replication_os_open_roads_per_window.csv", index=False)
    logger.info(
        "=== 2019 replication, %s: AccHR@20 across %d test windows: mean=%.4f std=%.4f (min=%.4f, max=%.4f) ===",
        borough.name, len(acchr_values), np.mean(acchr_values), np.std(acchr_values),
        np.min(acchr_values), np.max(acchr_values),
    )


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
