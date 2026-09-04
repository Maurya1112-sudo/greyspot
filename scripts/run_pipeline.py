"""End-to-end Greyspot research-core pipeline, parameterised by borough.

ingest STATS19 (+ casualty/vehicle/IMD/AADF enrichment) -> build a road
network graph (OSMnx by default, or OS Open Roads - see below) -> snap
collisions -> build features -> train historical-rate baseline, XGBoost
and a GAT+GRU graph-temporal model (+ ablations) -> evaluate all of them
on temporal, spatial and spatiotemporal held-out splits -> conformal
uncertainty layers (MAPIE for XGBoost, manual split-conformal for the GAT)
-> export a risk map.

Westminster is the primary target (see docs/proposal.md); Lambeth is the
first expansion test (see `greyspot.ingest.boroughs` and
docs/project_management.md's "expanding to London" section) - it is also
the case-study borough of the project's closest academic precedent, so its
results are directly comparable to published literature.

Run from the project root:
    python scripts/run_pipeline.py                        # Westminster, OSMnx (default)
    python scripts/run_pipeline.py Lambeth                 # any registered borough
    python scripts/run_pipeline.py Westminster os_open_roads  # OS Open Roads instead of OSMnx

OS Open Roads (the dossier's originally-preferred official network - see
`ingest/os_open_roads.py`'s module docstring) requires the GB-wide
GeoPackage downloaded manually from OS Data Hub and extracted to
`oproad_gpkg_gb/Data/oproad_gb.gpkg` at the project root (2026-09-01);
falls back to a clear error, not a silent OSMnx substitution, if it's
missing. Output for an os_open_roads run goes to a separate
`{borough}_os_open_roads/` directory under `data/processed/` and
`reports/`, so it never overwrites the OSMnx-based run for the same
borough - both stay available side by side for comparison.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.metrics import evaluate_predictions  # noqa: E402
from greyspot.eval.splits import spatial_split, temporal_split  # noqa: E402
from greyspot.features.build_features import (  # noqa: E402
    aggregate_enriched_features,
    aggregate_exposure_features,
    build_segment_year_table,
    collision_counts_by_segment_year,
)
from greyspot.features.graph_temporal import (  # noqa: E402
    build_segment_adjacency,
    build_target_vector,
    build_temporal_feature_tensor,
    build_walkforward_instances,
    edge_index_from_line_graph,
)
from greyspot.ingest.boroughs import Borough, get_borough, slug  # noqa: E402
from greyspot.ingest.exposure import load_local_authority_aadf  # noqa: E402
from greyspot.ingest.imd import attach_imd_to_collisions, load_imd_lookup  # noqa: E402
from greyspot.ingest.network import (  # noqa: E402
    build_borough_graph,
    graph_to_edges_gdf,
    snap_collisions_to_graph,
    snap_points_to_graph,
)
from greyspot.ingest.os_open_roads import (  # noqa: E402
    borough_bbox_wgs84,
    build_borough_graph_os_open_roads,
)
from greyspot.ingest.stats19 import (  # noqa: E402
    collision_severity_and_vulnerable_user_features,
    load_casualty_years,
    load_local_authority_collisions,
    load_vehicle_years,
    vehicle_mix_features,
)
from greyspot.models.baseline import historical_rate_score  # noqa: E402
from greyspot.models.conformal import (  # noqa: E402
    empirical_coverage,
    fit_split_conformal,
    manual_split_conformal_interval,
    mean_interval_width,
    predict_with_interval,
)
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal,
    train_gat_temporal,
    train_gat_temporal_walkforward,
)
from greyspot.models.xgboost_model import FEATURE_COLUMNS, predict_xgboost, train_xgboost  # noqa: E402
from greyspot.viz.map_demo import build_risk_map, save_map  # noqa: E402
from greyspot.viz.maplibre_map import build_maplibre_map  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

YEARS = [2021, 2022, 2023, 2024, 2025]
# 2025 is a provisional/partial-year release (smaller file, incomplete
# months) at the time this pipeline was built - see docs/decision_log.md.
# Use 2024 (a complete year) as the held-out test year; 2025 rows still
# flow through ingestion/features but are excluded from this evaluation.
TEST_YEAR = 2024

# Regularisation was tested 2026-08-31 after reading the academic
# precedent's training setup (Gao et al. 2024 use dropout 0.2, weight decay
# 0.01), on the hypothesis that the GAT branch was overfitting harder than
# its own "no graph" ablation. Tested at the precedent's exact strength AND
# at a lighter strength (dropout 0.1, weight decay 0.001): both made every
# GAT variant *worse* than no regularisation at all, in a clean monotonic
# dose-response (more regularisation -> worse, consistently). The
# hypothesis is refuted, not just for one setting - reverted to 0/0, the
# actual best-performing configuration found. Full comparison in
# docs/decision_log.md.
GAT_DROPOUT = 0.0
GAT_WEIGHT_DECAY = 0.0

# Architecture fix for the over-smoothing problem, found 2026-08-31 via
# scripts/experiment_gat_architecture.py and confirmed on the
# spatiotemporal split via scripts/experiment_gat_best_config_validation.py:
# 1 attention head (down from 4) + a residual/skip connection around the
# GAT layer. This is the first configuration all session where the full
# GAT+GRU model beats its own "no graph" ablation - on Westminster,
# temporal PR-AUC went 0.261 -> 0.424 (vs gat_no_graph's 0.360, vs
# XGBoost's 0.332) and spatiotemporal 0.250 -> 0.350 (vs XGBoost's 0.347).
# Full before/after tables and the diagnostic reasoning in
# docs/decision_log.md.
GAT_HEADS = 1
GAT_USE_RESIDUAL = True

# STATS19/AADF/IMD source files are national/regional, not borough-specific
# - one shared raw/ cache serves every borough. Graph cache, processed
# features and reports ARE borough-specific, so two boroughs never clobber
# each other's output.
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

# OS Open Roads GeoPackage (GB-wide, ~2GB) - downloaded manually by the user
# from OS Data Hub 2026-09-01, the upgrade this pipeline named as a planned
# swap-in since its very first version (see ingest/network.py's module
# docstring). Only used when network_source="os_open_roads"; the default
# stays OSMnx so every existing cached graph/report is unaffected.
OS_OPEN_ROADS_GPKG = ROOT / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"


def main(borough_name: str = "Westminster", network_source: str = "osmnx") -> None:
    if network_source not in ("osmnx", "os_open_roads"):
        raise ValueError(f"network_source must be 'osmnx' or 'os_open_roads', got {network_source!r}")

    borough: Borough = get_borough(borough_name)
    # A borough run with a different network source is not the same
    # dataset - keep OS Open Roads output alongside, not overwriting, the
    # existing OSMnx-based reports for the same borough, so both remain
    # available for comparison.
    output_slug = slug(borough.name) + ("_os_open_roads" if network_source == "os_open_roads" else "")
    borough_slug = slug(borough.name)
    processed_dir = ROOT / "data" / "processed" / output_slug
    reports_dir = ROOT / "reports" / output_slug
    logger.info(
        "=== Running Greyspot pipeline for %s (%s), network_source=%s ===",
        borough.name, borough.ons_code, network_source,
    )

    logger.info("Step 1/6: loading STATS19 collisions for %s, years=%s", borough.name, YEARS)
    collisions = load_local_authority_collisions(borough.ons_code, YEARS, RAW_DIR)
    logger.info("Loaded %d %s collisions", len(collisions), borough.name)

    if network_source == "os_open_roads":
        logger.info("Step 2/6: building %s road network graph (OS Open Roads)", borough.name)
        if not OS_OPEN_ROADS_GPKG.exists():
            raise FileNotFoundError(
                f"OS Open Roads GeoPackage not found at {OS_OPEN_ROADS_GPKG} - download it from "
                "OS Data Hub (https://osdatahub.os.uk/downloads/open/OpenRoads) and extract it to "
                "the project root, or use network_source='osmnx'."
            )
        graph_cache = INTERIM_DIR / f"{borough_slug}_graph_os_open_roads.graphml"
        bbox = borough_bbox_wgs84(borough.osm_place)
        graph = build_borough_graph_os_open_roads(bbox, OS_OPEN_ROADS_GPKG, cache_path=graph_cache)
    else:
        logger.info("Step 2/6: building %s road network graph (OSMnx)", borough.name)
        graph_cache = INTERIM_DIR / f"{borough_slug}_graph.graphml"
        graph = build_borough_graph(borough.osm_place, cache_path=graph_cache)
    edges = graph_to_edges_gdf(graph)
    logger.info("Graph has %d nodes, %d edges", graph.number_of_nodes(), len(edges))

    logger.info("Step 3/6: snapping collisions to nearest road segment")
    snapped = snap_collisions_to_graph(collisions, graph)
    join_rate = len(snapped) / max(len(collisions), 1)
    logger.info("Snapped %d/%d collisions (%.1f%%)", len(snapped), len(collisions), join_rate * 100)

    logger.info("Step 4a/6: loading casualty/vehicle tables + IMD 2019 for enrichment")
    casualties_national = load_casualty_years(YEARS, RAW_DIR)
    vehicles_national = load_vehicle_years(YEARS, RAW_DIR)
    casualty_feats = collision_severity_and_vulnerable_user_features(casualties_national)
    vehicle_feats = vehicle_mix_features(vehicles_national)

    imd_path = RAW_DIR / "imd2019_london_lsoa.xlsx"
    snapped_enriched = snapped.merge(casualty_feats, on="collision_index", how="left")
    snapped_enriched = snapped_enriched.merge(vehicle_feats, on="collision_index", how="left")
    if imd_path.exists():
        imd_lookup = load_imd_lookup(imd_path)
        snapped_enriched = attach_imd_to_collisions(snapped_enriched, imd_lookup)
    else:
        logger.warning("IMD file not found at %s; skipping equity context feature", imd_path)
        snapped_enriched["imd_decile"] = pd.NA

    logger.info("Step 4c/6: loading DfT AADF traffic-count data (exposure) for %s", borough.name)
    aadf_path = RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        # AADF count points sit on major roads and are sparse relative to
        # the full segment set - a 100m cutoff avoids wrongly attaching a
        # count point to an unrelated nearby minor road (see
        # network.snap_points_to_graph's docstring).
        snapped_aadf = snap_points_to_graph(aadf, graph, max_distance_m=100)
        n_matched = snapped_aadf["segment_id"].notna().sum()
        logger.info("Snapped %d/%d AADF count-point-years to a segment within 100m", n_matched, len(snapped_aadf))
        if n_matched == 0:
            logger.warning(
                "No AADF count points matched for %s - check that AADF's local_authority_name "
                "field really uses %r (see docs/decision_log.md before trusting this silently).",
                borough.name, borough.name,
            )
        exposure_agg = aggregate_exposure_features(snapped_aadf)
    else:
        logger.warning("AADF file not found at %s; skipping exposure feature", aadf_path)

    logger.info("Step 4b/6: building segment-year feature table (with lagged enrichment + exposure)")
    counts = collision_counts_by_segment_year(snapped)
    enriched_counts = aggregate_enriched_features(snapped_enriched)
    table = build_segment_year_table(
        edges, counts, graph, years=YEARS, enriched=enriched_counts, exposure=exposure_agg
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed_dir / "segment_year_table.parquet", index=False)
    logger.info("Feature table: %d rows (segments x years)", len(table))

    logger.info("Step 5/6: temporal, spatial and spatiotemporal held-out evaluation")
    # Temporal: same segments, unseen year (2024) - "does last year predict next year?"
    temporal_train, temporal_test = temporal_split(table, test_year=TEST_YEAR)

    # Spatial: same years, unseen *segments* - "does this generalise to unseen roads?"
    # Split first, then restrict each side to the temporal-train period so no
    # split leaks TEST_YEAR data into training.
    spatial_train_segs, spatial_holdout_segs = spatial_split(table, holdout_frac=0.2, seed=42)
    spatial_train = spatial_train_segs[spatial_train_segs["year"] < TEST_YEAR]
    spatial_test = spatial_holdout_segs[spatial_holdout_segs["year"] < TEST_YEAR]

    # Spatiotemporal: unseen segments AND unseen year - the hardest, most
    # realistic generalisation test (a genuinely new road, next year).
    spatiotemporal_test = spatial_holdout_segs[spatial_holdout_segs["year"] == TEST_YEAR]

    logger.info(
        "temporal train/test = %d/%d | spatial train/test = %d/%d | spatiotemporal test = %d",
        len(temporal_train), len(temporal_test), len(spatial_train), len(spatial_test), len(spatiotemporal_test),
    )

    def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, split_name: str) -> list[dict]:
        baseline_score = historical_rate_score(test_df)
        model = train_xgboost(train_df)
        xgb_score = predict_xgboost(model, test_df)
        y_true = test_df["collision_count"].to_numpy()
        return [
            {"split": split_name, "model": "historical_rate", **evaluate_predictions(y_true, baseline_score)},
            {"split": split_name, "model": "xgboost", **evaluate_predictions(y_true, xgb_score)},
        ], model, xgb_score

    all_results = []
    temporal_rows, _temporal_model, xgb_score = evaluate_split(temporal_train, temporal_test, "temporal")
    all_results += temporal_rows
    spatial_rows, _, _ = evaluate_split(spatial_train, spatial_test, "spatial")
    all_results += spatial_rows
    # Spatiotemporal test reuses the spatial-split model (trained only on
    # spatial_train, which already excludes TEST_YEAR) for a clean unseen
    # segment x unseen year comparison.
    spatiotemporal_model = train_xgboost(spatial_train)
    st_score = predict_xgboost(spatiotemporal_model, spatiotemporal_test)
    st_baseline = historical_rate_score(spatiotemporal_test)
    st_true = spatiotemporal_test["collision_count"].to_numpy()
    all_results += [
        {"split": "spatiotemporal", "model": "historical_rate", **evaluate_predictions(st_true, st_baseline)},
        {"split": "spatiotemporal", "model": "xgboost", **evaluate_predictions(st_true, st_score)},
    ]

    logger.info("Step 5b/6: GAT+GRU research model (graph-temporal comparator)")
    # The GAT operates on the segment-level line graph (Section 9 "why a
    # graph?"). Its "temporal" evaluation uses WALK-FORWARD training
    # (`build_walkforward_instances` / `train_gat_temporal_walkforward`):
    # train on an earlier (window, target) transition, evaluate on a later,
    # disjoint one the model never saw a label for. This fixes a real bug
    # found 2026-08-31 (docs/decision_log.md): training directly against
    # the exact year later used for evaluation is an in-sample fit, not a
    # held-out temporal test, however many epochs are used - confirmed by a
    # 2000-epoch run of the old code collapsing to a suspicious PR-AUC of
    # 0.85. "spatiotemporal" below needs no such fix: it already properly
    # excludes held-out segments' labels from the loss (a standard
    # transductive-GNN setup), which is a genuine held-out test on its own.
    segment_order = table["segment_id"].drop_duplicates().tolist()
    line_graph = build_segment_adjacency(edges)
    edge_index = edge_index_from_line_graph(line_graph, segment_order)
    sequence_years = [y for y in YEARS if y < TEST_YEAR]
    x_seq = build_temporal_feature_tensor(table, segment_order, FEATURE_COLUMNS, years=sequence_years)
    y_test_year = build_target_vector(table, segment_order, year=TEST_YEAR)
    logger.info(
        "GAT input: %d timesteps (%s) x %d segments x %d features, line graph has %d edges",
        len(sequence_years), sequence_years, len(segment_order), len(FEATURE_COLUMNS), edge_index.shape[1],
    )

    walkforward_years = [y for y in YEARS if y <= TEST_YEAR]  # excludes the partial 2025 year
    walkforward_window = 2
    walkforward_instances = build_walkforward_instances(
        table, segment_order, FEATURE_COLUMNS, years=walkforward_years, window=walkforward_window
    )
    wf_train_instances = [(x, y) for x, y, target_year in walkforward_instances[:-1]]
    wf_held_out_x, wf_held_out_y, wf_held_out_year = walkforward_instances[-1]
    assert wf_held_out_year == TEST_YEAR, f"walk-forward held-out year {wf_held_out_year} != TEST_YEAR {TEST_YEAR}"
    logger.info(
        "Walk-forward: %d training transition(s) (window=%d), held-out target year %d",
        len(wf_train_instances), walkforward_window, wf_held_out_year,
    )

    def train_walkforward(**kwargs) -> "GATTemporal":  # noqa: F821 - type only, avoids importing the class here
        return train_gat_temporal_walkforward(
            wf_train_instances, edge_index, epochs=200,
            dropout=GAT_DROPOUT, weight_decay=GAT_WEIGHT_DECAY, heads=GAT_HEADS, use_residual=GAT_USE_RESIDUAL,
            **kwargs,
        )

    gat_full = train_walkforward(use_graph=True, use_temporal=True)
    gat_temporal_pred = predict_gat_temporal(gat_full, wf_held_out_x, edge_index)
    all_results.append(
        {"split": "temporal", "model": "gat_temporal", **evaluate_predictions(wf_held_out_y, gat_temporal_pred)}
    )

    spatial_train_segments = set(spatial_train_segs["segment_id"].unique())
    holdout_mask = np.array([seg not in spatial_train_segments for seg in segment_order])
    train_mask = ~holdout_mask
    gat_spatial = train_gat_temporal(
        x_seq, edge_index, y_test_year, train_mask=train_mask, epochs=200,
        dropout=GAT_DROPOUT, weight_decay=GAT_WEIGHT_DECAY, heads=GAT_HEADS, use_residual=GAT_USE_RESIDUAL,
    )
    gat_st_pred = predict_gat_temporal(gat_spatial, x_seq, edge_index)
    all_results.append(
        {
            "split": "spatiotemporal", "model": "gat_temporal",
            **evaluate_predictions(y_test_year[holdout_mask], gat_st_pred[holdout_mask]),
        }
    )

    logger.info("Step 5b-ii/6: ablations - remove the graph, remove the temporal encoder (dossier Section 9)")
    # Same walk-forward training/evaluation as the full model above - only
    # one architectural switch differs per row, so any performance gap is
    # attributable to that one component, not to a leakier evaluation.
    gat_no_graph = train_walkforward(use_graph=False, use_temporal=True)
    gat_no_graph_pred = predict_gat_temporal(gat_no_graph, wf_held_out_x, edge_index)
    all_results.append(
        {"split": "temporal", "model": "gat_no_graph", **evaluate_predictions(wf_held_out_y, gat_no_graph_pred)}
    )

    gat_no_temporal = train_walkforward(use_graph=True, use_temporal=False)
    gat_no_temporal_pred = predict_gat_temporal(gat_no_temporal, wf_held_out_x, edge_index)
    all_results.append(
        {"split": "temporal", "model": "gat_no_temporal", **evaluate_predictions(wf_held_out_y, gat_no_temporal_pred)}
    )

    results_df = pd.DataFrame(all_results).set_index(["split", "model"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "baseline_vs_xgboost_results.csv")

    print(f"\n=== {borough.name}: held-out results (temporal / spatial / spatiotemporal + ablations) ===")
    print(results_df.to_string())

    logger.info("Step 5c/6: conformal (MAPIE) uncertainty layer on the XGBoost point model")
    # A genuinely held-out calibration year (2023), disjoint from the years
    # used to fit the point model (2021-2022), so the empirical coverage
    # check below is a fair test - not training-data reuse dressed up as
    # calibration (see models/conformal.py's module docstring).
    conformal_fit = temporal_train[temporal_train["year"] < 2023]
    conformal_calib = temporal_train[temporal_train["year"] == 2023]
    conformal, _ = fit_split_conformal(conformal_fit, conformal_calib, confidence_level=0.9)
    _, lower, upper = predict_with_interval(conformal, temporal_test)
    coverage = empirical_coverage(temporal_test["collision_count"].to_numpy(), lower, upper)
    width = mean_interval_width(lower, upper)
    logger.info(
        "Conformal 90%% target -> empirical coverage=%.3f, mean interval width=%.2f, n=%d",
        coverage, width, len(temporal_test),
    )

    logger.info("Step 5d/6: manual split-conformal on the GAT+GRU model (MAPIE doesn't fit a transductive GNN)")
    # Same fit/calibrate/test year split as the XGBoost conformal step above
    # (fit on an earlier sequence, calibrate on 2023, evaluate on 2024), so
    # the two conformal results are methodologically comparable.
    calib_sequence_years = [y for y in sequence_years if y < 2023]
    x_seq_calib = build_temporal_feature_tensor(table, segment_order, FEATURE_COLUMNS, years=calib_sequence_years)
    gat_calib_model = train_gat_temporal(
        x_seq_calib, edge_index, build_target_vector(table, segment_order, year=2023),
        epochs=200, dropout=GAT_DROPOUT, weight_decay=GAT_WEIGHT_DECAY, heads=GAT_HEADS, use_residual=GAT_USE_RESIDUAL,
    )
    gat_calib_pred = predict_gat_temporal(gat_calib_model, x_seq_calib, edge_index)
    gat_calib_true = build_target_vector(table, segment_order, year=2023)
    gat_lower, gat_upper = manual_split_conformal_interval(
        gat_calib_true, gat_calib_pred, gat_temporal_pred, confidence_level=0.9
    )
    gat_coverage = empirical_coverage(y_test_year, gat_lower, gat_upper)
    gat_width = mean_interval_width(gat_lower, gat_upper)
    logger.info(
        "GAT conformal 90%% target -> empirical coverage=%.3f, mean interval width=%.2f, n=%d",
        gat_coverage, gat_width, len(y_test_year),
    )
    pd.DataFrame(
        [
            {"confidence_level": 0.9, "empirical_coverage": coverage, "mean_interval_width": width, "n_test": len(temporal_test), "model": "xgboost"},
            {"confidence_level": 0.9, "empirical_coverage": gat_coverage, "mean_interval_width": gat_width, "n_test": len(y_test_year), "model": "gat_temporal"},
        ]
    ).to_csv(reports_dir / "conformal_calibration.csv", index=False)
    print(f"=== {borough.name} conformal: xgboost coverage={coverage:.3f} width={width:.2f} | gat coverage={gat_coverage:.3f} width={gat_width:.2f} ===")

    logger.info("Step 6/6: exporting risk maps (temporal-test model)")
    test_with_score = temporal_test.copy()
    test_with_score["score"] = xgb_score

    # Folium/Leaflet map: kept as the fast, dependency-light sanity check.
    fmap = build_risk_map(edges, test_with_score[["segment_id", "score"]], collisions)
    map_path = save_map(fmap, reports_dir / "figures" / f"{borough_slug}_risk_map.html")
    logger.info("Folium map written to %s", map_path)

    # MapLibre GL JS map: the primary, dossier-specified map (Section 7) -
    # GPU-accelerated data-driven styling, collision clustering, a
    # provenance panel, and a confidence-vs-risk-separate legend.
    maplibre_path = build_maplibre_map(
        edges, test_with_score[["segment_id", "score"]], collisions,
        borough_name=borough.name, model_version=f"xgboost (temporal, {borough.name}, test_year={TEST_YEAR})",
        test_year=TEST_YEAR, out_path=reports_dir / "figures" / f"{borough_slug}_risk_map_maplibre.html",
    )
    logger.info("MapLibre map written to %s", maplibre_path)


if __name__ == "__main__":
    # Usage: python scripts/run_pipeline.py [Borough] [osmnx|os_open_roads]
    borough_arg = sys.argv[1] if len(sys.argv) > 1 else "Westminster"
    network_source_arg = sys.argv[2] if len(sys.argv) > 2 else "osmnx"
    main(borough_arg, network_source=network_source_arg)
