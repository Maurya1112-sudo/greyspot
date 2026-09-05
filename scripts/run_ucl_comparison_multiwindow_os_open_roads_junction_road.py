"""JUNCTION RISK REDISTRIBUTION + the real OS Open Roads network -
implementing a methodological detail from Gao et al.'s own text that
this project had never replicated, found 2026-09-02 by re-reading their
Section 7.2.1 rather than only their results/hyperparameter tables.

**Their stated method**: "To further address the complexity of crash
occurrences at intersections, where multiple road segments converge,
the methodology implements a balanced weighting scheme, whereas the
associated risk value is distributed equally among all connected road
segments."

**What this project did instead, until now**: attributed each crash
wholly to its single nearest road segment.

**Why this is worth a run, quantified before building it** (not
assumed): STATS19's own `junction_detail` field (0 = "not at or within
20 metres of a junction") says **63-72% of all collisions in the three
study boroughs are junction-related** - Westminster 71.9%, Lambeth
67.7%, Tower Hamlets 63.2%, measured over 2022-2024. Snapping all of
that to one arm of the junction concentrates two-thirds of the total
signal onto an essentially arbitrary choice between several segments
that share the risk in reality. This is the largest unimplemented
methodological difference remaining against the reference paper.

Implementation: `ingest.network.redistribute_junction_crashes` assigns
each at-junction crash to the graph node nearest its coordinate, then
emits it once per incident directed edge with weight 1/n (non-junction
and missing-`junction_detail` crashes keep weight 1.0 on their original
segment - unknown is NOT treated as at-junction). Total weight per
crash stays exactly 1.0, so this redistributes risk without inflating
or deflating it; verified end-to-end on real Lambeth data (3,085
crashes in, total weight 3,085.0 out, zero unmatched segment_ids).
`collision_severity_counts_by_segment_day` then aggregates on that
weight rather than row counts.

**CRITICAL evaluation-fairness design decision** (caught 2026-09-02
after a first version of this script produced a misleadingly bad
result): redistribution changes the TARGET, and AccHR@20 defines
"actual crashes" as `y_true > 0`. If the redistributed target is also
used for scoring, a single junction crash becomes 8 fractional entries
across 8 segments, and the model must now land ALL EIGHT in the top
20% instead of one - the metric becomes mechanically much harder, and
any score drop says nothing about whether redistribution helped. The
first run of this script scored 50.83% on its opening window (vs the
75.76% baseline) purely from that artefact.

This script therefore builds **two parallel targets**: the
redistributed (fractional) one is the TRAINING signal, while the
ORIGINAL un-redistributed one is the EVALUATION ground truth - the
same ground truth every other experiment in this project is scored
against, so AccHR@20 stays directly comparable. That isolates the
actual question ("does spreading the training signal across a
junction's arms help the model rank where crashes really occur?")
rather than conflating it with a change of measuring stick. The two
instance lists are built through the same windowing function and
asserted to align one-to-one on their held-out start dates, so they
cannot silently drift apart.

Otherwise identical to `run_ucl_comparison_multiwindow_os_open_roads.py`
(real network, POI + socio-demographic features, heads=3, ZIP decoder,
light protocol) - isolating this one variable against the current
best-known configuration.

Run from the project root: python scripts/run_ucl_comparison_multiwindow_os_open_roads_junction.py
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
    fit_feature_standardizer,
)
from greyspot.features.graph_temporal import build_segment_adjacency, edge_index_from_line_graph  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.network import graph_to_edges_gdf, redistribute_junction_crashes, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
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
logger = logging.getLogger("run_ucl_comparison_multiwindow_os_open_roads_junction_road")


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
    "heads=3 + POI + socio-demographic + REAL OS network + JUNCTION redistribution (per PHYSICAL ROAD)": dict(
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

    # TWO targets, deliberately (see this script's docstring on the
    # evaluation-fairness problem):
    #   - `daily_counts` (redistributed, fractional) = the TRAINING signal
    #   - `eval_counts` (original, un-redistributed) = the EVALUATION
    #     ground truth, identical to every other run's, so AccHR@20 stays
    #     directly comparable across experiments.
    redistributed = redistribute_junction_crashes(snapped, graph, per_physical_road=True)
    daily_counts = collision_severity_counts_by_segment_day(redistributed, casualties, weight_col="crash_weight")
    eval_counts = collision_severity_counts_by_segment_day(snapped, casualties)

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

    # A parallel instance list built from the ORIGINAL (un-redistributed)
    # counts, using the identical segment order, dates and windowing - so
    # `eval_instances[i]`'s target is the same held-out period as
    # `instances[i]`'s, just measured against real single-segment crash
    # locations rather than the redistributed training signal. Features
    # are irrelevant here (only the y arrays are used), but building it
    # through the same function guarantees the windowing/alignment cannot
    # silently drift from the training instances'.
    eval_table = build_segment_day_table(edges, eval_counts, graph, start_date=START_DATE, end_date=END_DATE)
    eval_instances = build_daily_multistep_instances(
        eval_table, segment_order, ["length", "day_of_week"], all_dates,
        input_window=INPUT_WINDOW, horizon=HORIZON, stride=STRIDE_DAYS,
    )
    if len(eval_instances) != len(instances):
        raise ValueError(
            f"Evaluation instances ({len(eval_instances)}) and training instances ({len(instances)}) "
            "must align one-to-one; they were built with identical windowing so a mismatch means a real bug."
        )
    for (_, _, train_start), (_, _, eval_start) in zip(instances, eval_instances):
        if train_start != eval_start:
            raise ValueError(f"Instance window misalignment: training starts {train_start}, evaluation starts {eval_start}")

    return instances, eval_instances, edge_index


def evaluate_config(instances, eval_instances, edge_index, config: dict, n_windows: int) -> list[dict]:
    """Expanding-window walk-forward, with the training signal and the
    evaluation ground truth deliberately drawn from DIFFERENT targets:
    trains on `instances` (junction-redistributed, fractional) but scores
    against `eval_instances` (original, un-redistributed) - see this
    script's docstring for why scoring against the redistributed target
    would be an unfair, mechanically-harder comparison rather than a
    measurement of whether redistribution helps."""
    results = []
    n = len(instances)
    for held_out_idx in range(n - n_windows, n):
        train_slice = instances[:held_out_idx]
        held_out_x_raw, _held_out_y_redistributed, held_out_start = instances[held_out_idx]
        _, held_out_y_eval, _ = eval_instances[held_out_idx]

        train_x_seqs = [x for x, _, _ in train_slice]
        mean, std = fit_feature_standardizer(train_x_seqs)
        train_instances = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
        held_out_x = apply_feature_standardizer(held_out_x_raw, mean, std)

        defaults = dict(epochs=200, horizon=HORIZON, zero_inflated=True, use_residual=True, gat_hidden=16, gru_hidden=32)
        model = train_gat_temporal_walkforward(
            train_instances, edge_index, **{**defaults, **config},
        )
        held_out_pred = predict_gat_temporal(model, held_out_x, edge_index)
        y_true = held_out_y_eval  # ORIGINAL crash locations, not the redistributed signal
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
    instances, eval_instances, edge_index = build_instances(borough_name)
    logger.info("Built %d total instances; evaluating the last %d as held-out windows", len(instances), N_WINDOWS)
    if len(instances) < N_WINDOWS + 2:
        raise ValueError(f"Only {len(instances)} instances - need at least {N_WINDOWS + 2} for {N_WINDOWS} held-out windows plus training history.")

    reports_dir = ROOT / "reports" / slug(get_borough(borough_name).name)
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    for name, config in CANDIDATES.items():
        logger.info("--- Candidate: %s ---", name)
        results = evaluate_config(instances, eval_instances, edge_index, config, N_WINDOWS)
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

    pd.DataFrame(all_rows).to_csv(reports_dir / "ucl_multiwindow_per_window_os_open_roads_junction_road.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "ucl_multiwindow_summary_os_open_roads_junction_road.csv", index=False)
    logger.info("Written to %s", reports_dir / "ucl_multiwindow_summary_os_open_roads_junction_road.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
