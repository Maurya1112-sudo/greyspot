"""DENSE protocol (the paper's own exact temporal density: YEARS
2021-2025, INPUT_WINDOW=20 days, HORIZON=14 days, STRIDE_DAYS=14 -
copied verbatim from `run_ucl_comparison.py`'s own constants, not
re-derived) + the confirmed-best POI + socio-demographic feature set -
built 2026-09-02 at the user's explicit request to "use the same
dataset as the ucl one... exactly the same and dense form."

**What "exactly the same" means here, honestly**: the TEMPORAL protocol
(5 years, N=20-day input, p=14-day horizon, non-overlapping stride) now
matches the paper's own stated setup exactly - this project's earlier
"light" 6-window scripts deliberately used a coarser 90-day stride
instead, for faster iteration during architecture/feature testing, not
because the dense protocol was unknown. The DATA SOURCES cannot be made
literally identical without blocked inputs (OS Open Roads network,
commercial OS Points of Interest, a literal Census 2011 pull) - see
`docs/decision_log.md`/`docs/publication_readiness.md` for the
disclosed substitutes each of those already has (OSMnx, OSM POI tags,
2019 IMD + LSOA population density respectively). This script closes
the temporal-protocol gap, not the data-source gap.

**Fixes a real confound found in the ORIGINAL dense-protocol attempt**
(`docs/decision_log.md`'s 2026-09-02 entry, before POI+socio existed):
that run's naive "last N_WINDOWS instances" convention packed all 3
held-out windows into a single 4-week span at the very end of the
5-year range (stride=14 means the last 6 instances span only 84 days).
Here, held-out indices are chosen EVENLY SPACED across the full usable
range (after reserving `MIN_TRAIN_INSTANCES` instances of initial
training history) instead of just the trailing N - still a genuine
walk-forward (each held-out window trains only on strictly-earlier
data, an expanding window each time), but now testing across the whole
2021-2025 span/all seasons, not one clustered month.

Reuses `build_instances` verbatim from `run_ucl_comparison_multiwindow_poi_socio.py`
(same feature-building steps) with only the date-range/stride constants
and the held-out-window selection changed.

Cost warning: at dense density, training sets range from
~`MIN_TRAIN_INSTANCES` up to ~125+ instances per window (vs 6-11 at the
light protocol) - each window costs substantially more GPU time; this
is a real, accepted tradeoff for matching the paper's protocol, not
free.

Run from the project root: python scripts/run_ucl_comparison_dense_poi_socio.py
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
from greyspot.ingest.network import build_borough_graph, graph_to_edges_gdf, snap_collisions_to_graph, snap_points_to_graph  # noqa: E402
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
logger = logging.getLogger("run_ucl_comparison_dense_poi_socio")


# The paper's own exact temporal density - copied verbatim from
# `run_ucl_comparison.py` (2021-2025, N=20-day input, p=14-day horizon,
# stride=14 == HORIZON for maximum non-overlapping instance density).
# Accepted cost: each held-out window trains on up to ~125+ instances
# (vs 5-11 at the "light" protocol used elsewhere), so a full sweep here
# takes substantially longer - the point of this script is matching the
# paper's protocol, not fast iteration.
YEARS = [2021, 2022, 2023, 2024, 2025]
START_DATE, END_DATE = "2021-01-01", "2025-12-31"
INPUT_WINDOW, HORIZON, STRIDE_DAYS = 20, 14, 14
ROLLING_WINDOWS = (7, 14, 30)
N_WINDOWS = 6  # number of held-out windows to evaluate (expanding training window each time)
MIN_TRAIN_INSTANCES = 20  # minimum training history before the earliest held-out window - avoids a near-empty first training set
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

# Just the current best-known config (heads=3, plain-count target, ZIP
# decoder) - this run answers ONE question ("do POI + socio-demographic
# features help this specific config's AccHR@20"), not a fresh
# architecture sweep on top of new features at once.
CANDIDATES = {
    "heads=3 + POI + socio-demographic, DENSE protocol (paper's own N=20/p=14/stride=14, calendar-spread windows)": dict(
        heads=3, gat_layers=1, negative_binomial=False,
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

    # POI - cached to disk after the first fetch (an Overpass API call,
    # ~75s for a borough) rather than re-downloaded on every run.
    poi_cache_path = INTERIM_DIR / f"{borough_slug}_poi_counts.csv"
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


def choose_held_out_indices(n_instances: int, n_windows: int, min_train: int) -> list[int]:
    """Evenly-spaced held-out indices across the whole usable range,
    instead of just the trailing `n_windows` - fixes the calendar-
    clustering confound found in the original dense-protocol attempt
    (all held-out windows landing in the same ~1-month span at the end
    of the range, since a fixed 14-day stride packs `n_windows` trailing
    instances into only `n_windows * 14` days). Each index still trains
    only on strictly-earlier instances (a genuine expanding-window walk-
    forward), but the held-out points themselves now span the full
    2021-2025 range/every season, not one clustered month."""
    if n_instances < min_train + n_windows:
        raise ValueError(
            f"Only {n_instances} instances - need at least {min_train + n_windows} "
            f"({min_train} minimum training history + {n_windows} held-out windows)."
        )
    return sorted(set(np.linspace(min_train, n_instances - 1, n_windows).round().astype(int).tolist()))


def evaluate_config(instances, edge_index, config: dict, held_out_indices: list[int]) -> list[dict]:
    """Expanding-window walk-forward over `held_out_indices` (evenly
    spaced across the calendar, see `choose_held_out_indices`) -
    standardise features (fit on everything strictly before the held-out
    point), train on everything strictly before it, evaluate on it."""
    results = []
    for held_out_idx in held_out_indices:
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
    logger.info("=== DENSE multi-window AccHR@20 evaluation: %s ===", borough_name)
    instances, edge_index = build_instances(borough_name)
    held_out_indices = choose_held_out_indices(len(instances), N_WINDOWS, MIN_TRAIN_INSTANCES)
    logger.info(
        "Built %d total instances; evaluating %d calendar-spread held-out windows "
        "(training set sizes at each: %s)",
        len(instances), len(held_out_indices), held_out_indices,
    )

    reports_dir = ROOT / "reports" / slug(get_borough(borough_name).name)
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    for name, config in CANDIDATES.items():
        logger.info("--- Candidate: %s ---", name)
        results = evaluate_config(instances, edge_index, config, held_out_indices)
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

    pd.DataFrame(all_rows).to_csv(reports_dir / "ucl_dense_per_window_poi_socio.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(reports_dir / "ucl_dense_summary_poi_socio.csv", index=False)
    logger.info("Written to %s", reports_dir / "ucl_dense_summary_poi_socio.csv")


if __name__ == "__main__":
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    main(borough_arg)
