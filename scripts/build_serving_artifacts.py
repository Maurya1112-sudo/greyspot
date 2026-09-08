"""Precompute the artifacts the API/frontend actually serve.

**Why precompute rather than train-per-request.** The old `services/api`
trained an XGBoost model on first request (`data_service.py`, now
superseded). The real final model (`GATTemporal`, GAT+GRU+ZIP) takes
minutes to train and needs the full OS Open Roads + STATS19 + POI +
socio-demographic pipeline to build its 35 features - not something to run
inside an HTTP request. This script runs that pipeline ONCE per borough
and writes a static artifact the API loads at process start.

**What "current risk" means here, and why it is not invented.** There is
no ground truth for a window that has not happened yet, so "live" risk in
the ordinary sense cannot be validated. Instead this uses the SAME
held-out window already reported in the paper - the most recent of the six
walk-forward windows (2024-10-07) - trained and evaluated by the exact
protocol in `evaluate_config` (train on every instance strictly before it,
calibrate conformal intervals on the last training instance). The
per-segment scores shown in the product are therefore the same computation
already validated in `reports/<borough>/*.csv`, not a new untested code
path.

**Why both a model score and a baseline score are served.** The paper's
central finding is that a parameter-free crash-count sort matches or beats
this model. Serving only the model's ranking would misrepresent that
finding to a product user. Both rankings are computed and exposed; the
frontend shows both rather than picking a favourite.

**Duplication note.** The pipeline steps below mirror
`scripts/run_ucl_comparison_multiyear.py::build_instances` line for line,
using the SAME bound functions and constants from that module (imported,
not retyped) so there is no risk of the two silently drifting apart - the
kind of bug class R3 exists to catch. The function is duplicated rather
than the source module modified because `build_instances` is called by
~70 other scripts and changing its return signature is not a change worth
making for this.

Run: python scripts/build_serving_artifacts.py [Borough ...]
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.models.conformal import manual_split_conformal_interval  # noqa: E402
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal,
    train_gat_temporal_walkforward,
)
from greyspot.product.priority_score import DEFAULT_POLICY_PROFILE, compute_priority_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_serving_artifacts")

MODEL_VERSION = "gat-gru-zip-final-v1"
SEED = 42  # matches the headline/reported figures for cross-checking
DEFAULT_BOROUGHS = ["Westminster", "Lambeth", "Tower Hamlets"]
OUT_DIR = ROOT / "data" / "served"

# Loaded once: the run_ucl_comparison_multiyear module, for its bound
# functions and constants (YEARS, HISTORY_YEARS, FEATURE_COLUMNS, ...).
#
# BUG FOUND AND FIXED: this import mutates sys.argv (`m`'s own module-level
# code expects it) at IMPORT TIME - i.e. before this script's own
# `if __name__ == "__main__"` block below has a chance to read the borough
# name off the command line. The first run silently ignored "Lambeth" on
# the command line and built Westminster (DEFAULT_BOROUGHS[0]) instead,
# because sys.argv had already been overwritten to ["x"] by the time the
# argument was read. Save and restore it around the import.
_argv_before_import = sys.argv[:]
_spec = importlib.util.spec_from_file_location(
    "m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
m = importlib.util.module_from_spec(_spec)
sys.argv = ["x"]
_spec.loader.exec_module(m)
sys.argv = _argv_before_import


def build_full(borough_name: str):
    """Identical body to `m.build_instances`, additionally returning the
    intermediate `edges`, `table` and `history_snapped` the API needs for
    evidence fields and the baseline ranker. See module docstring."""
    borough = get_borough(borough_name)
    borough_slug = slug(borough.name)
    collisions = m.load_local_authority_collisions(borough.ons_code, m.YEARS, m.RAW_DIR)
    bbox = m.borough_bbox_wgs84(borough.osm_place)
    polygon = m.borough_polygon_wgs84(borough.osm_place)
    graph = m.build_borough_graph_os_open_roads(
        bbox, m.OS_OPEN_ROADS_GPKG,
        cache_path=m.INTERIM_DIR / f"{borough_slug}_os_open_roads_graph.graphml",
        polygon_wgs84=polygon,
    )
    edges = m.graph_to_edges_gdf(graph)
    snapped = m.snap_collisions_to_graph(collisions, graph)

    history_collisions = m.load_local_authority_collisions(borough.ons_code, m.HISTORY_YEARS, m.RAW_DIR)
    history_snapped = m.snap_collisions_to_graph(history_collisions, graph)
    history_snapped = history_snapped.dropna(subset=["segment_id"]).copy()
    history_snapped["date"] = m.parse_stats19_date(history_snapped["date"])

    casualties = m.load_casualty_years(m.YEARS, m.RAW_DIR)
    daily_counts = m.collision_severity_counts_by_segment_day(snapped, casualties)

    aadf_path = m.RAW_DIR / "aadf_raw" / "dft_traffic_counts_aadf.csv"
    exposure_agg = None
    if aadf_path.exists():
        aadf = m.load_local_authority_aadf(aadf_path, local_authority_name=borough.name)
        snapped_aadf = m.snap_points_to_graph(aadf, graph, max_distance_m=100)
        if snapped_aadf["segment_id"].notna().sum() > 0:
            exposure_agg = m.aggregate_exposure_features(snapped_aadf)

    table = m.build_segment_day_table(edges, daily_counts, graph, start_date=m.TABLE_START_DATE, end_date=m.END_DATE)
    table = m.attach_rolling_collision_features(table, windows=m.ROLLING_WINDOWS)
    table = m.attach_long_history_features(table, history_snapped, lookbacks=m.LONG_LOOKBACKS)
    if exposure_agg is not None:
        table = m.attach_static_exposure_features(table, exposure_agg)
    else:
        table["aadf_all_motor_vehicles"] = 0.0
        table["aadf_pedal_cycles"] = 0.0
        table["has_aadf"] = 0

    poi_cache_path = m.INTERIM_DIR / f"{borough_slug}_poi_counts_os_open_roads.csv"
    if poi_cache_path.exists():
        poi_counts = pd.read_csv(poi_cache_path)
    else:
        pois = m.download_borough_pois(borough.osm_place)
        poi_counts = m.count_pois_near_segments(pois, graph)
        m.assert_poi_counts_plausible(poi_counts, borough.osm_place)
        poi_counts.to_csv(poi_cache_path, index=False)
    table = m.attach_poi_features(table, poi_counts)

    lsoa_shapefile = m.BOUNDARIES_DIR / f"LSOA_2011_BGC_{borough.name.replace(' ', '_')}.shp"
    if lsoa_shapefile.exists() and m.IMD_PATH.exists():
        lsoa_boundaries = m.load_lsoa_boundaries(lsoa_shapefile)
        lsoa_socio = m.load_lsoa_socio_demographics(m.IMD_PATH, lsoa_boundaries)
        segment_socio = m.snap_segments_to_lsoa(edges, lsoa_socio)
        table = m.attach_socio_demographic_features(table, segment_socio)
    else:
        for col in m.SOCIO_DEMOGRAPHIC_COLUMNS:
            table[col] = 0.0
        table["has_socio_demographic"] = 0

    all_dates = pd.date_range(m.START_DATE, m.END_DATE, freq="D")
    assert all_dates[0] == pd.Timestamp("2022-01-01")
    segment_order = edges["segment_id"].drop_duplicates().tolist()
    line_graph = m.build_segment_adjacency(edges)
    edge_index = m.edge_index_from_line_graph(line_graph, segment_order)

    instances = m.build_daily_multistep_instances(
        table, segment_order, m.FEATURE_COLUMNS, all_dates,
        input_window=m.INPUT_WINDOW, horizon=m.HORIZON, stride=m.STRIDE_DAYS,
    )
    return instances, edge_index, edges, table, history_snapped, segment_order


def serve_borough(borough_name: str, seed: int = SEED) -> None:
    logger.info("=== Building serving artifact: %s ===", borough_name)
    instances, edge_index, edges, table, history_snapped, segment_order = build_full(borough_name)
    n = len(instances)
    held_out_idx = n - 1  # the most recent of the 6 reported evaluation windows
    train_slice = instances[:held_out_idx]
    held_out_x_raw, held_out_y, held_out_start = instances[held_out_idx]
    logger.info("held-out window (most recent reported): %s, trained on %d instances",
                held_out_start.date(), len(train_slice))

    mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
    train_instances = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
    held_out_x = apply_feature_standardizer(held_out_x_raw, mean, std)

    defaults = dict(epochs=200, horizon=m.HORIZON, zero_inflated=True,
                    use_residual=True, gat_hidden=16, gru_hidden=32)
    config = dict(heads=3, gat_layers=1, negative_binomial=False)
    model = train_gat_temporal_walkforward(train_instances, edge_index, **{**defaults, **config}, seed=seed)

    held_out_pred = predict_gat_temporal(model, held_out_x, edge_index)  # shape (N, H) - see note below
    y_true = held_out_y  # (H, N) natively, per build_daily_multistep_instances's own docstring
    y_pred = held_out_pred.T  # (H, N), matching y_true - the exact pattern evaluate_config uses
    acchr = float(accuracy_hit_rate(y_true, y_pred, top_fraction=0.20))
    logger.info("this window's AccHR@20 = %.4f (for cross-check against reports/)", acchr)

    calib_x_raw, calib_y, _ = train_slice[-1]
    calib_x = apply_feature_standardizer(calib_x_raw, mean, std)
    calib_pred = predict_gat_temporal(model, calib_x, edge_index)
    lower, upper = manual_split_conformal_interval(
        calib_y.T.flatten(), calib_pred.flatten(), held_out_pred.flatten(), confidence_level=0.9
    )
    lower = lower.reshape(held_out_pred.shape)  # (N, H) - held_out_pred's OWN (native) shape
    upper = upper.reshape(held_out_pred.shape)

    # A single "risk over the next 14 days" number per segment: sum the
    # per-day expectation and the per-day interval bounds over the
    # horizon (axis=1 here - held_out_pred is (N, H), confirmed by the
    # ValueError this line originally raised: summing over the wrong axis
    # produced a length-H array where a length-N one was needed, which
    # pandas caught immediately rather than silently broadcasting).
    # Summing independent per-day intervals is a conservative
    # (wider-than-necessary) approximation to a genuinely joint interval
    # on the 14-day total, not a jointly-calibrated one - stated in
    # model_info.json rather than left implicit.
    model_score = held_out_pred.sum(axis=1)          # (N,)
    interval_lower = np.clip(lower, 0, None).sum(axis=1)
    interval_upper = upper.sum(axis=1)

    # Baseline: cumulative crash count over ALL available history, no cap -
    # exactly `raw count (no shrinkage)` in scripts/run_s2_empirical_bayes.py.
    hist = history_snapped[history_snapped.date < held_out_start]
    baseline_counts = hist.groupby("segment_id").size()
    baseline_score = pd.Series(0.0, index=segment_order)
    baseline_score.update(baseline_counts.astype(float))
    baseline_score = baseline_score.to_numpy()

    # Evidence fields, "as observed the day before the held-out window
    # starts" - static columns (u_degree, v_degree, highway, length) plus
    # the model's own 365d/730d rolling counts, read directly off `table`
    # rather than re-derived, so they cannot silently disagree with what
    # the model itself saw.
    asof = held_out_start - pd.Timedelta(days=1)
    asof_rows = table[table["date"] == asof].set_index("segment_id")
    asof_rows = asof_rows.reindex(segment_order)

    # Prior-year (365d) casualty breakdown: `table`'s casualty columns are
    # DAILY counts (collision_severity_counts_by_segment_day), not rolling
    # sums, so this is a direct 365-day filter+groupby - correct and cheap
    # at this scale, not the cumsum-matrix trick used for the wide
    # multi-window sweeps elsewhere in the project.
    window = table[(table["date"] >= held_out_start - pd.Timedelta(days=365)) & (table["date"] < held_out_start)]
    casualty_cols = ["n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
                     "n_pedestrian_casualties", "n_cyclist_casualties"]
    prior_year_casualties = window.groupby("segment_id")[casualty_cols].sum().reindex(segment_order).fillna(0.0)

    # Road name, straight from OS Open Roads' own "name_1" attribute (see
    # ingest/os_open_roads.py's graph_to_edges_gdf) - present for every
    # Westminster segment when checked directly, but was never carried
    # past `edges` into the served output until this fix (2026-09-08 user
    # report: the priority queue and evidence panel showed nothing but a
    # raw segment UUID, on every row, for every borough). `edges` can have
    # duplicate segment_id rows (hence edges_geo's own drop_duplicates
    # below), so build the lookup the same way rather than a plain
    # set_index, which would raise on a non-unique index.
    name_lookup = edges.drop_duplicates(subset="segment_id").set_index("segment_id")["name"]
    # A genuinely unnamed segment (service roads, tracks, some minor stubs
    # - about 7% of Westminster) comes back from OS Open Roads as float
    # NaN, not None. Left as NaN, both pyarrow (parquet) and Fiona/GDAL
    # (GeoJSON) were observed to stringify it to the literal text "nan"
    # rather than emit a real null (2026-09-08: caught in the served
    # output itself - a priority-queue row read "nan" as its road name).
    # `.where(notna, None)` swaps in real Python `None`, which both
    # writers correctly serialise as null.
    name_lookup = name_lookup.where(name_lookup.notna(), None)

    assert len(model_score) == len(segment_order), (
        f"model_score length {len(model_score)} != segment_order length {len(segment_order)} - "
        f"held_out_pred shape was {held_out_pred.shape}, check which axis is N vs H"
    )
    out = pd.DataFrame({
        "segment_id": segment_order,
        "model_score": model_score,
        "baseline_score": baseline_score,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "interval_width": interval_upper - interval_lower,
        "u_degree": asof_rows["u_degree"].to_numpy(),
        "v_degree": asof_rows["v_degree"].to_numpy(),
        "highway": asof_rows["highway"].to_numpy() if "highway" in asof_rows.columns else None,
        "name": name_lookup.reindex(segment_order).to_numpy(),
        "length": asof_rows["length"].to_numpy() if "length" in asof_rows.columns else None,
        "prior_year_count": asof_rows["collision_count_365d"].to_numpy(),
        "prior_2yr_avg": asof_rows["collision_count_730d"].to_numpy() / 2.0,
        "aadf_all_motor_vehicles": asof_rows["aadf_all_motor_vehicles"].to_numpy(),
        "aadf_pedal_cycles": asof_rows["aadf_pedal_cycles"].to_numpy(),
        "has_aadf": asof_rows["has_aadf"].to_numpy(),
        "prior_year_n_fatal_casualties": prior_year_casualties["n_fatal_casualties"].to_numpy(),
        "prior_year_n_serious_casualties": prior_year_casualties["n_serious_casualties"].to_numpy(),
        "prior_year_n_slight_casualties": prior_year_casualties["n_slight_casualties"].to_numpy(),
        "prior_year_n_pedestrian_casualties": prior_year_casualties["n_pedestrian_casualties"].to_numpy(),
        "prior_year_n_cyclist_casualties": prior_year_casualties["n_cyclist_casualties"].to_numpy(),
    })

    scored = compute_priority_score(
        out, risk_score_col="model_score", profile=DEFAULT_POLICY_PROFILE, interval_width_col="interval_width",
    )
    # A second priority ranking using the baseline score in place of the
    # model - so the frontend can show "if you trusted the trivial sort
    # instead" without a second full model run.
    scored["baseline_priority_score"] = compute_priority_score(
        out, risk_score_col="baseline_score", profile=DEFAULT_POLICY_PROFILE,
    )["priority_score"]

    bslug = slug(get_borough(borough_name).name)
    out_dir = OUT_DIR / bslug
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_geo = edges[["segment_id", "geometry"]].drop_duplicates(subset="segment_id")
    merged = edges_geo.merge(scored, on="segment_id", how="left")
    # MapView.tsx keys its colour scale on `has_score` (0/1), not on
    # `priority_score` alone - a segment absent from the merge (should not
    # happen given segment_order covers every edge, but must not be
    # silently mis-rendered as "zero risk" if it ever does) needs an
    # explicit flag rather than a NaN priority_score the frontend would
    # otherwise have to special-case.
    merged["has_score"] = merged["priority_score"].notna().astype(int)
    merged.to_file(out_dir / "segments.geojson", driver="GeoJSON")
    # `highway` used to be dropped here on the (undocumented, and wrong)
    # assumption that the geojson was the only place the API needed it -
    # data_service.py loads `scored_table` from THIS parquet, not from the
    # geojson, so every priority-queue row and evidence response was
    # silently getting `highway: None` and falling back to "Unclassified
    # Road" regardless of the segment's real class (2026-09-08 user
    # report: every single row showed "Unclassified Road"). Kept now.
    scored.to_parquet(out_dir / "segments.parquet", index=False)

    # Historical performance for this exact window, from the already
    # validated per-window CSVs - not re-derived, so the product's stated
    # accuracy figure always matches the paper's.
    reported_acchr, reported_source = _lookup_reported_acchr(bslug, held_out_start)

    info = {
        "borough": get_borough(borough_name).name,
        "model_version": MODEL_VERSION,
        "seed": seed,
        "held_out_start": held_out_start.date().isoformat(),
        "held_out_horizon_days": int(m.HORIZON),
        "n_segments": len(scored),
        "this_run_acchr_at_20": round(acchr, 4),
        "reported_acchr_at_20": reported_acchr,
        "reported_acchr_source": reported_source,
        "conformal_confidence_level": 0.9,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "research_finding": (
            "A parameter-free ranking of segments by cumulative past crash "
            "count is statistically indistinguishable from this model at "
            "matched history depth, and beats it once given more history "
            "than the model was trained with. Both rankings are shown "
            "below; see the accompanying paper (paper/arxiv/main.tex) for "
            "the full comparison."
        ),
        "interval_note": (
            "The 14-day interval is the sum of per-day conformal bounds, "
            "which is a conservative approximation to a jointly-calibrated "
            "14-day interval, not one itself."
        ),
        "non_negotiable_boundary": (
            "This is a research prototype. It does not predict a specific "
            "collision, does not label any location 'safe' or 'unsafe', "
            "and its ranking has not been validated as a policy tool."
        ),
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Written to %s (%d segments)", out_dir, len(scored))
    logger.info("model_info: %s", json.dumps(info, indent=2))


def _lookup_reported_acchr(bslug: str, held_out_start: pd.Timestamp) -> tuple[float | None, str | None]:
    for fname in ("headline_multiseed_per_window.csv", "ucl_multiwindow_per_window_multiyear.csv"):
        p = ROOT / "reports" / bslug / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["held_out_start"] = pd.to_datetime(d["held_out_start"])
        row = d[d["held_out_start"] == held_out_start]
        if "seed" in row.columns:
            row = row[row["seed"] == SEED]
        if not row.empty:
            return round(float(row.iloc[0]["AccHR"]), 4), fname
    return None, None


if __name__ == "__main__":
    for name in (sys.argv[1:] or DEFAULT_BOROUGHS):
        serve_borough(name)
