"""CLIPPED z-scoring (+/- 5 sigma) - the third and best-motivated
attempt at fixing a measured numerical defect, 2026-09-03.

**The defect, measured on real Lambeth data**: this project's z-scored
features reach an absolute maximum of **1179.63**. A 99.98%-zero count
column has a near-zero standard deviation, so every rare non-zero
becomes an enormous z-score. Feeding values of that magnitude into GAT
attention logits is the condition that produced attention-parameter
NaNs at lr=0.01 and forced the paper's own encoder order down to a 10x
lower learning rate to train at all.

**Two rejected fixes, both tested rather than reasoned away**:
  - *Max-value scaling*, which the reference implementation itself uses
    (github.com/ZhuangDingyi/STZINB scales inputs by the training-set
    maximum and never z-scores). It bounds everything to [-1, 1] but
    CRUSHES the variance of exactly the sparse columns that matter -
    almost every value becomes 0. Measured: 34.70% on window 1 against
    a 75.76% baseline.
  - *log1p-then-z-score*, the textbook count-data transform. Verified
    NOT to help, for a reason worth recording: with one non-zero in
    ~500 cells, both the mean and the standard deviation scale with
    that value, so any monotone transform leaves the z-score
    essentially unchanged. **The extremeness comes from the sparsity,
    not the magnitude** - so no rescaling of values can fix it.

**Clipping addresses it directly**: typical values keep their z-scored
unit variance (good gradient flow, signal intact - verified: values
within +/-5 pass through unchanged, only ~0.03% of values are capped),
while the rare extremes are bounded to a magnitude attention softmax
can handle. clip=5.0 is a standard 5-sigma choice, fixed once on that
basis and NOT tuned against the evaluation windows.

Identical to `run_ucl_comparison_multiwindow_os_open_roads.py` in every
other respect, so the comparison isolates the scaler alone.
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
    attach_poi_features,
    attach_rolling_collision_features,
    attach_socio_demographic_features,
    attach_static_exposure_features,
    build_segment_day_table,
    collision_severity_counts_by_segment_day,
)
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    build_daily_multistep_instances,
    apply_clipped_standardizer,
    fit_feature_standardizer,
)
from greyspot.features.graph_temporal import build_segment_adjacency, edge_index_from_line_graph  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.network import graph_to_edges_gdf, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
from greyspot.ingest.os_open_roads import borough_bbox_wgs84, borough_polygon_wgs84, build_borough_graph_os_open_roads  # noqa: E402
from greyspot.ingest.poi import POI_COUNT_COLUMNS, assert_poi_counts_plausible, count_pois_near_segments, download_borough_pois  # noqa: E402
from greyspot.ingest.socio_demographic import (  # noqa: E402
    SOCIO_DEMOGRAPHIC_COLUMNS,
    load_lsoa_boundaries,
    load_lsoa_socio_demographics,
    snap_segments_to_lsoa,
)
from greyspot.ingest.stats19 import load_casualty_years, load_local_authority_collisions  # noqa: E402
from greyspot.models.conformal import manual_split_conformal_interval  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ucl_comparison_clipscale")


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
    *POI_COUNT_COLUMNS, "has_poi",
    *SOCIO_DEMOGRAPHIC_COLUMNS, "has_socio_demographic",
]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
BOUNDARIES_DIR = RAW_DIR / "boundaries" / "lsoa_bgc"
IMD_PATH = RAW_DIR / "imd2019_london_lsoa.xlsx"

# The current best-known config (heads=3 + POI + socio-demographic),
# unchanged - this run answers ONE question ("does the REAL OS Open
# Roads network change AccHR@20 vs OSMnx"), isolated to the network
# source alone.
CANDIDATES = {
    "heads=3 + REAL OS network + CLIPPED z-scoring (+/-5 sigma)": dict(
        heads=3, gat_layers=1, negative_binomial=False,
    ),
}

OS_OPEN_ROADS_GPKG = ROOT / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"


def build_instances(borough_name: str):
    borough = get_borough(borough_name)
    borough_slug = slug(borough.name)
    collisions = load_local_authority_collisions(borough.ons_code, YEARS, RAW_DIR)
    bbox = borough_bbox_wgs84(borough.osm_place)
    polygon = borough_polygon_wgs84(borough.osm_place)
    graph = build_borough_graph_os_open_roads(
        bbox, OS_OPEN_ROADS_GPKG,
        cache_path=INTERIM_DIR / f"{borough_slug}_os_open_roads_graph.graphml",
        polygon_wgs84=polygon,
    )
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

    # POI - cached to disk after the first fetch (an Overpass API call,
    # ~75s for a borough) rather than re-downloaded on every run.
    poi_cache_path = INTERIM_DIR / f"{borough_slug}_poi_counts_os_open_roads.csv"
    if poi_cache_path.exists():
        poi_counts = pd.read_csv(poi_cache_path)
    else:
        pois = download_borough_pois(borough.osm_place)
        poi_counts = count_pois_near_segments(pois, graph)
        assert_poi_counts_plausible(poi_counts, borough.osm_place)
        poi_counts.to_csv(poi_cache_path, index=False)
        logger.info("Cached POI counts to %s", poi_cache_path)
    table = attach_poi_features(table, poi_counts)

    # Socio-demographic (LSOA boundaries + IMD 2019) - both files are
    # local/already-downloaded, so no caching layer needed here.
    # London Datastore's own filenames use underscores for multi-word
    # borough names (e.g. "LSOA_2011_BGC_Tower_Hamlets.shp") - found
    # 2026-09-02 extending this to Tower Hamlets (Westminster/Lambeth
    # are single words, so this mismatch never surfaced before).
    lsoa_shapefile = BOUNDARIES_DIR / f"LSOA_2011_BGC_{borough.name.replace(' ', '_')}.shp"
    if lsoa_shapefile.exists() and IMD_PATH.exists():
        lsoa_boundaries = load_lsoa_boundaries(lsoa_shapefile)
        lsoa_socio = load_lsoa_socio_demographics(IMD_PATH, lsoa_boundaries)
        segment_socio = snap_segments_to_lsoa(edges, lsoa_socio)
        table = attach_socio_demographic_features(table, segment_socio)
    else:
        logger.warning("LSOA boundary/IMD files missing for %s - socio-demographic features zero-filled", borough.name)
        for col in SOCIO_DEMOGRAPHIC_COLUMNS:
            table[col] = 0.0
        table["has_socio_demographic"] = 0

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
        train_instances = [(apply_clipped_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
        held_out_x = apply_clipped_standardizer(held_out_x_raw, mean, std)

        defaults = dict(epochs=200, horizon=HORIZON, zero_inflated=True, use_residual=True, gat_hidden=16, gru_hidden=32)
        model = train_gat_temporal_walkforward(
            train_instances, edge_index, **{**defaults, **config},
        )
        held_out_pred = predict_gat_temporal(model, held_out_x, edge_index)
        y_true = held_out_y
        y_pred = held_out_pred.T

        calib_x_raw, calib_y, _ = train_slice[-1]
        calib_x = apply_clipped_standardizer(calib_x_raw, mean, std)
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

    pd.DataFrame(all_rows).to_csv(reports_dir / "ucl_multiwindow_per_window_clipscale.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "ucl_multiwindow_summary_clipscale.csv", index=False)
    logger.info("Written to %s", reports_dir / "ucl_multiwindow_summary_clipscale.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
