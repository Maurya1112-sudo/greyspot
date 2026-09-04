"""In-memory data/model service backing the FastAPI endpoints.

Deliberately simple for this prototype phase (dossier Section 17: "a clean
Dockerised prototype... [is] more valuable than prematurely implementing a
complex cloud microservices estate"): on first request for a borough, load
its cached feature table + road graph (built by `scripts/run_pipeline.py`,
which must be run at least once first), train XGBoost, compute the
transparent Safety Priority Score, and cache the result in memory for the
life of the process. No database yet - `data/processed/{borough}/` and
`data/interim/{borough}_graph.graphml` already function as the versioned
artefact store the dossier's audit-trail requirement calls for; a real
PostGIS-backed version is future work (see docs/scaling_to_london.md-style
staged plan, not yet written for the product layer).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import geopandas as gpd
import pandas as pd

from greyspot.eval.metrics import evaluate_predictions
from greyspot.eval.splits import temporal_split
from greyspot.ingest.boroughs import Borough, get_borough, slug
from greyspot.ingest.network import build_borough_graph, graph_to_edges_gdf
from greyspot.models.conformal import empirical_coverage, fit_split_conformal, mean_interval_width, predict_with_interval
from greyspot.models.xgboost_model import predict_xgboost, train_xgboost
from greyspot.product.priority_score import DEFAULT_POLICY_PROFILE, compute_priority_score

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
TEST_YEAR = 2024
MODEL_VERSION = "xgboost-temporal-v1"  # bump when the training procedure changes


@dataclass
class BoroughData:
    borough: Borough
    edges: gpd.GeoDataFrame
    scored_table: pd.DataFrame  # test-year rows with model score + priority score + components
    metrics: dict
    conformal_coverage: float
    conformal_width: float
    generated_at: str


class BoroughDataCache:
    """Process-lifetime cache - avoids re-training on every request. Not
    thread-safe against concurrent first-loads of the *same* borough
    (acceptable for a single-worker dev/demo server; a production version
    would need a lock or a precomputed offline store)."""

    _cache: ClassVar[dict[str, BoroughData]] = {}

    @classmethod
    def get(cls, borough_name: str) -> BoroughData:
        borough = get_borough(borough_name)  # raises ValueError with a helpful message if unknown
        if borough.name not in cls._cache:
            logger.info("Loading and scoring %s for the first time this process...", borough.name)
            cls._cache[borough.name] = cls._load(borough)
        return cls._cache[borough.name]

    @classmethod
    def _load(cls, borough: Borough) -> BoroughData:
        bslug = slug(borough.name)
        table_path = ROOT / "data" / "processed" / bslug / "segment_year_table.parquet"
        graph_path = ROOT / "data" / "interim" / f"{bslug}_graph.graphml"
        if not table_path.exists() or not graph_path.exists():
            raise FileNotFoundError(
                f"No processed data for {borough.name} yet. Run "
                f"`python scripts/run_pipeline.py {borough.name}` first."
            )

        table = pd.read_parquet(table_path)
        graph = build_borough_graph(borough.osm_place, cache_path=graph_path)
        edges = graph_to_edges_gdf(graph)

        train, test = temporal_split(table, test_year=TEST_YEAR)
        model = train_xgboost(train)
        test = test.copy()
        test["model_score"] = predict_xgboost(model, test)

        # Conformal calibration (same fit/calibrate/test protocol as
        # scripts/run_pipeline.py - see models/conformal.py's module
        # docstring for why the calibration set must be genuinely disjoint).
        conformal_fit = train[train["year"] < 2023]
        conformal_calib = train[train["year"] == 2023]
        conformal, _ = fit_split_conformal(conformal_fit, conformal_calib, confidence_level=0.9)
        _, lower, upper = predict_with_interval(conformal, test)
        test["interval_width"] = upper - lower
        coverage = empirical_coverage(test["collision_count"].to_numpy(), lower, upper)
        width = mean_interval_width(lower, upper)

        scored = compute_priority_score(
            test, risk_score_col="model_score", profile=DEFAULT_POLICY_PROFILE, interval_width_col="interval_width",
        )
        metrics = evaluate_predictions(test["collision_count"].to_numpy(), test["model_score"].to_numpy())

        return BoroughData(
            borough=borough,
            edges=edges,
            scored_table=scored,
            metrics=metrics,
            conformal_coverage=coverage,
            conformal_width=width,
            generated_at=pd.Timestamp.now("UTC").isoformat(),
        )
