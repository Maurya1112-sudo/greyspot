"""THE REAL network-source test - OS Open Roads (the paper's actual road
network), not an OSMnx approximation. Built 2026-09-02 after discovering
the "2GB source GeoPackage is gone from local disk" note in project
memory was simply WRONG - the file (`oproad_gpkg_gb/Data/oproad_gb.gpkg`,
1.02GB compressed, downloaded by the user 2026-09-01) was on disk the
whole time, just never actually exercised end-to-end until now.

**A real bug was found and fixed getting here** (see `ingest.os_open_roads`'s
own docstring/`docs/decision_log.md`): the existing loader only ever
bbox-clipped, never polygon-clipped, to the real administrative
boundary. For Westminster this pulled in 11,347 road links from a bbox
that extends well past the actual borough (e.g. across the Thames) -
polygon-clipping (endpoint-inside test, matching OSMnx's own
`graph_from_place` convention) cuts this to 5,549 links (11,098
directed edges). This is HIGHER than this project's own OSMnx graph
(7,552 directed edges) for the same borough - the OPPOSITE of the
"OS Open Roads is ~1.5x coarser" reading of the paper's thesis Table 7.2
that motivated the (separately, already-tested-negative) network-
consolidation experiment. Either the paper counts "roads" by some
different convention (e.g. named-street aggregation, not individual
link/junction-to-junction segments), or by a different boundary
definition - unresolved, but irrelevant to this script's actual
purpose: testing the REAL network's effect on AccHR@20 directly,
rather than reasoning from segment counts that may not be comparable
at all.

Otherwise identical to `run_ucl_comparison_multiwindow_poi_socio.py`
(same light protocol, same POI+socio-demographic features, same
heads=3 config) - only the road network source changes, isolating this
one variable against the current best-known configuration.

POI counts are cached under a NEW, network-specific key
(`{borough}_poi_counts_os_open_roads.csv`) since OS Open Roads assigns
completely different segment_ids than OSMnx - reusing the OSMnx POI
cache would silently key-mismatch (the same bug class caught and fixed
in the network-consolidation script).

Run from the project root: python scripts/run_ucl_comparison_multiwindow_os_open_roads.py
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
    attach_long_history_features,
    parse_stats19_date,
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
logger = logging.getLogger("run_s5_deep_history")


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
# 2021 is loaded ONLY so the 365-day rolling history feature is complete
# for instances at the start of 2022 - without it the table's 2021 rows
# would contain zero crashes and the 365d counts would be silently
# undercounted (plausible-looking output, wrong feature). No 2021 date
# is ever used as an evaluation window; the instance grid still starts
# 2022-01-01.
YEARS = [2021, 2022, 2023, 2024]

# MULTI-YEAR history, added 2026-09-03. Collisions ONLY (no casualty
# files needed, and none exist before 2021) loaded purely to deepen the
# crash-history features. Measured on real Lambeth data, ranking
# segments by history alone on this project's six evaluation windows:
#   30d 21.86% | 365d 50.50% | 730d 60.26% | 1095d 70.15%
# There is NO plateau - the curve is still climbing steeply at three
# years, because a 365-day window leaves 94.2% of segments tied at
# exactly zero while the top-20% bucket needs 2,319 of them. Longer
# horizons break those ties with real signal rather than arbitrary
# tie-ordering. (An earlier LSOA-level proxy suggested a plateau at
# 365d; that was misleading - LSOAs are ~58x denser than segments.)
#
# These windows are computed by `attach_long_history_features` from the
# sparse collision list, NOT by rolling over the segment-day table -
# extending the table back to 2018 would add ~30M rows and exhaust
# memory. 2018-2020 were extracted from DfT's 1979-2025 historical
# archive (identical 44-column schema, verified).
# 2016-2017 included so the 1825-day (5-year) window is COMPLETE for
# every training instance, not just the evaluation ones. Without them
# the earliest instances get truncated partial sums while later ones get
# full windows - the same feature silently meaning different things at
# different points in the walk-forward, which would bias training.
# 2012-2015 added 2026-09-05 so the 3285-day (9-year) lookback is COMPLETE
# rather than truncated. The table starts 2021-01-01, so a 9-year window
# reaches back to 2012; without those years the deepest features are
# partial sums and the experiment cannot test what it claims to test.
# (The first launch of S5 hit exactly this truncation warning and was
# stopped - the same failure mode already fixed once for the 1825d
# features, and not checked for here. See rule R4.)
HISTORY_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# S5 - EXTEND THE HISTORY CEILING (2026-09-05).
#
# S2 produced direct evidence that the current 1825-day (5-year) ceiling
# leaves signal unused: a parameter-free crash-count sort scores 83.94%
# with ~8 years of history but only 80.99% when capped at 1825 days -
# a +2.95 gain from the extra three years alone, on the same windows.
#
# The GNN's deepest feature is 1825d, so it cannot see that signal. This
# adds 2555d (7yr) and 3285d (9yr) lookbacks. Collision data is loaded
# from 2016, and `attach_long_history_features` computes any lookback
# from the sparse collision list via a cumulative-sum matrix, so deeper
# horizons cost no extra memory.
#
# **Honest expectation**: the gain should be smaller than +2.95, because
# the GNN must learn to use the new columns from 6-11 training instances
# whereas the baseline uses them directly. If it is NEGATIVE, that is
# itself informative - it would mean added sparse columns hurt more than
# the extra history helps, consistent with the pattern seen for road
# class, traffic exposure and casualty breakdown.
LONG_LOOKBACKS = (730, 1095, 1825, 2555, 3285)
# The TABLE is built from 2021-01-01 so the new 365-day rolling feature
# has a full year of history available for every instance. The INSTANCE
# GRID is still anchored at 2022-01-01 (EVAL_START_DATE), exactly as
# every other run in this project - so the six held-out windows are
# bit-identical to every previous result and remain directly comparable.
#
# This separation is deliberate and load-bearing. Simply moving
# START_DATE back would silently shift the stride-90 grid by 5 days
# (365 % 90 = 5), producing held-out windows of 2023-07-10 instead of
# 2023-07-15 - a bug already caught once in this project (see
# docs/decision_log.md, dense-training grid misalignment). Paired
# significance testing merges on `held_out_start` and would have found
# ZERO matching windows, turning every comparison into a silent
# unpaired contrast between different evaluation periods.
TABLE_START_DATE = "2021-01-01"
START_DATE, END_DATE = "2022-01-01", "2024-12-31"
INPUT_WINDOW, HORIZON, STRIDE_DAYS = 20, 14, 90
# LONG-HORIZON crash history, added 2026-09-03. Every previous feature
# experiment added a NEW data source (POI classes, weather, road class)
# and was null. This instead extends the horizon of the signal that is
# already the most important one present - past crashes on this segment.
#
# Motivated by a direct measurement on real Lambeth data, not a hunch
# (see docs/decision_log.md): ranking by crash history over the prior N
# days, walk-forward on the same six evaluation windows, at LSOA
# granularity as a cheap proxy:
#   30d -> 35.74% | 90d -> 42.10% | 180d -> 41.07%
#   365d -> 46.08% | 730d -> 46.00% (plateau)
# A 365-day lookback is worth ~+10 points over the 30-day maximum this
# project's feature set has always been capped at. Long-horizon crash
# history is also the single strongest known predictor of road-level
# crash risk in the safety literature - 30 days at ~2-3 crashes/day
# across ~11.6k segments is simply too sparse to estimate segment risk.
ROLLING_WINDOWS = (7, 14, 30, 90, 365)
N_WINDOWS = 6  # number of held-out windows to evaluate (expanding training window each time)
FEATURE_COLUMNS = [
    "length", "day_of_week", "u_degree", "v_degree",
    "collision_count", "collision_count_7d", "collision_count_14d", "collision_count_30d",
    "collision_count_90d", "collision_count_365d",
    "collision_count_730d", "collision_count_1095d", "collision_count_1825d",
    "collision_count_2555d", "collision_count_3285d",
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
    "deep history (up to 3285d / 9 years)": dict(
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

    # Deep history: collisions ONLY, back to HISTORY_YEARS[0], snapped to
    # the same graph. Used exclusively to compute LONG_LOOKBACKS features -
    # never to build the target, which stays on YEARS so it is identical to
    # every previous run and remains directly comparable.
    history_collisions = load_local_authority_collisions(borough.ons_code, HISTORY_YEARS, RAW_DIR)
    history_snapped = snap_collisions_to_graph(history_collisions, graph)
    history_snapped = history_snapped.dropna(subset=["segment_id"]).copy()
    history_snapped["date"] = parse_stats19_date(history_snapped["date"])
    logger.info(
        "Deep history: %d collisions snapped over %s (target still uses %s)",
        len(history_snapped), HISTORY_YEARS, YEARS,
    )
    casualties = load_casualty_years(YEARS, RAW_DIR)
    daily_counts = collision_severity_counts_by_segment_day(snapped, casualties)

    aadf_path = RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        snapped_aadf = snap_points_to_graph(aadf, graph, max_distance_m=100)
        if snapped_aadf["segment_id"].notna().sum() > 0:
            exposure_agg = aggregate_exposure_features(snapped_aadf)

    table = build_segment_day_table(edges, daily_counts, graph, start_date=TABLE_START_DATE, end_date=END_DATE)
    table = attach_rolling_collision_features(table, windows=ROLLING_WINDOWS)
    table = attach_long_history_features(table, history_snapped, lookbacks=LONG_LOOKBACKS)
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
    # Enforce the invariant the comment on TABLE_START_DATE describes: the
    # instance grid must start where every other run's does, regardless of
    # how far back the feature table itself reaches.
    assert all_dates[0] == pd.Timestamp("2022-01-01"), (
        f"instance grid must start 2022-01-01 to stay paired-comparable, got {all_dates[0]}"
    )
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

    pd.DataFrame(all_rows).to_csv(reports_dir / "s5_deep_history_per_window.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "s5_deep_history_summary.csv", index=False)
    logger.info("Written to %s", reports_dir / "s5_deep_history_summary.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
