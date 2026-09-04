"""Greyspot API — the FastAPI backend for the Minimum-Viable product layer
(dossier Section 16's REST resource list, scoped down to what this phase's
real, tested model outputs can actually support: road risk + priority
score + evidence + audit trail. Scenario/portfolio/copilot endpoints are
explicitly NOT here yet - they need features this phase doesn't build).

Run: `uvicorn app.main:app --reload --app-dir services/api` from the
project root, or `python services/api/run.py`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from greyspot.ingest.boroughs import BOROUGHS  # noqa: E402

from .data_service import MODEL_VERSION, BoroughData, BoroughDataCache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("greyspot.api")

app = FastAPI(
    title="Greyspot API",
    description=(
        "Uncertainty-aware road-risk prioritisation - research/decision-support "
        "prototype. Every score is a relative-risk research signal, never a "
        "safety guarantee. See /docs for the interactive schema."
    ),
    version=MODEL_VERSION,
)

# Permissive CORS for local dev (a plain static frontend or a Vite dev
# server on a different port needs this). Tighten to specific origins
# before any real deployment - dossier Section 17's security baseline.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)


def _get_data_or_404(borough_name: str) -> BoroughData:
    """The single place borough resolution happens for every endpoint -
    deliberately NOT a global `@app.exception_handler(ValueError)`, which
    was tried first and found (via `services/api/tests/test_api.py`) to
    silently swallow an *unrelated* ValueError raised deep inside FastAPI's
    own JSON encoder (a numpy scalar it couldn't serialize) and mis-report
    it as a plain 404 "not found", hiding a real bug. Catching ValueError
    only around the one call that can legitimately raise it for "unknown
    borough" keeps every other ValueError a genuine, visible 500.
    """
    try:
        return BoroughDataCache.get(borough_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/boroughs")
def list_boroughs() -> list[dict]:
    """Registered boroughs (see greyspot.ingest.boroughs) - not all have
    been run through the pipeline yet; `/boroughs/{name}/model-info` 404s
    with a helpful message for one that hasn't."""
    return [{"name": b.name, "ons_code": b.ons_code} for b in BOROUGHS.values()]


@app.get("/boroughs/{borough_name}/model-info")
def model_info(borough_name: str) -> dict:
    """Audit/provenance endpoint (dossier Section 6: every exported number
    must be traceable to a model version, data snapshot and generation
    timestamp)."""
    data = _get_data_or_404(borough_name)
    return {
        "borough": data.borough.name,
        "ons_code": data.borough.ons_code,
        "model_version": MODEL_VERSION,
        "test_year": 2024,
        "n_segments": len(data.scored_table),
        "metrics": {k: (None if v != v else v) for k, v in data.metrics.items()},  # NaN -> null in JSON
        "conformal": {
            "target_confidence_level": 0.9,
            "empirical_coverage": data.conformal_coverage,
            "mean_interval_width": data.conformal_width,
        },
        "generated_at": data.generated_at,
        "non_negotiable_boundary": (
            "This is an investigation and prioritisation aid. It does not claim a "
            "specific collision will happen, that a location is safe or unsafe, or "
            "that any score is a validated policy rule."
        ),
    }


@app.get("/boroughs/{borough_name}/roads")
def roads_geojson(borough_name: str) -> dict:
    """Road segments as a GeoJSON FeatureCollection, coloured by priority
    score client-side (see the standalone MapLibre map in
    greyspot.viz.maplibre_map for a self-contained equivalent)."""
    data = _get_data_or_404(borough_name)
    cols = [
        "segment_id", "highway", "model_score", "priority_score",
        "component_risk_signal", "component_severity", "component_vulnerable_users",
        "component_trend", "component_network_importance", "component_data_confidence",
        "data_confidence_is_default",
    ]
    scored_cols = [c for c in cols if c in data.scored_table.columns] + ["segment_id"]
    merged = data.edges.merge(
        data.scored_table[list(dict.fromkeys(scored_cols))], on="segment_id", how="left"
    )
    merged["has_score"] = merged["priority_score"].notna().astype(int)
    import json as _json

    return _json.loads(merged.to_json())


@app.get("/boroughs/{borough_name}/roads/{segment_id}")
def road_evidence(borough_name: str, segment_id: str) -> dict:
    """The 'Why this road?' evidence panel for one segment (dossier
    Section 5.5) - structured evidence, not a bare number."""
    data = _get_data_or_404(borough_name)
    row = data.scored_table[data.scored_table["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id!r} not found for {borough_name}.")
    r = row.iloc[0]
    return {
        "segment_id": segment_id,
        "borough": data.borough.name,
        "observed_evidence": {
            "prior_year_collision_count": _safe(r.get("prior_year_count")),
            "prior_2yr_avg": _safe(r.get("prior_2yr_avg")),
            "prior_year_fatal_casualties": _safe(r.get("prior_year_n_fatal_casualties")),
            "prior_year_serious_casualties": _safe(r.get("prior_year_n_serious_casualties")),
            "prior_year_pedestrian_casualties": _safe(r.get("prior_year_n_pedestrian_casualties")),
            "prior_year_cyclist_casualties": _safe(r.get("prior_year_n_cyclist_casualties")),
        },
        "exposure": {
            "aadf_all_motor_vehicles": _safe(r.get("aadf_all_motor_vehicles")),
            "aadf_pedal_cycles": _safe(r.get("aadf_pedal_cycles")),
            "has_aadf": bool(r.get("has_aadf", 0)),
        },
        "model_evidence": {
            "model_version": MODEL_VERSION,
            "predicted_relative_risk": _safe(r.get("model_score")),
            "conformal_interval_width_90pct": _safe(r.get("interval_width")),
        },
        "priority_score": {
            "score_0_100": _safe(r.get("priority_score")),
            "policy_profile": r.get("policy_profile"),
            "components": {
                "risk_signal": _safe(r.get("component_risk_signal")),
                "severity": _safe(r.get("component_severity")),
                "vulnerable_users": _safe(r.get("component_vulnerable_users")),
                "trend": _safe(r.get("component_trend")),
                "network_importance": _safe(r.get("component_network_importance")),
                "data_confidence": _safe(r.get("component_data_confidence")),
            },
            "data_confidence_is_default": bool(r.get("data_confidence_is_default", True)),
        },
        "limitations": [
            "Relative risk research signal, not a prediction of a specific collision.",
            "Priority score weights are a labelled policy prototype, not a validated finding.",
            "Historical evidence only covers 2021-2024 for this borough.",
        ],
    }


@app.get("/boroughs/{borough_name}/priority-queue")
def priority_queue(
    borough_name: str,
    limit: int = Query(default=25, ge=1, le=500),
    min_confidence: bool | None = Query(default=None, description="If true, exclude rows with default (non-computed) confidence"),
) -> list[dict]:
    """Ranked list of segments by priority score - the core 'where should
    we investigate first?' workflow (dossier Section 6)."""
    data = _get_data_or_404(borough_name)
    table = data.scored_table
    if min_confidence:
        table = table[~table["data_confidence_is_default"]]
    top = table.sort_values("priority_score", ascending=False).head(limit)
    return [
        {
            "segment_id": row["segment_id"],
            "priority_score": _safe(row["priority_score"]),
            "model_score": _safe(row["model_score"]),
            "prior_year_count": _safe(row.get("prior_year_count")),
            "highway": row.get("highway"),
        }
        for _, row in top.iterrows()
    ]


def _safe(value):
    """NaN/NaT -> None, and any numpy scalar (np.float32/np.int64/np.bool_,
    all of which come out of every pandas `.iloc[]`/`.get()` row access
    here) -> a native Python type, so FastAPI's JSON encoder never chokes.

    Found via `services/api/tests/test_api.py` actually exercising this
    against real pipeline output: FastAPI's encoder cannot serialize
    `numpy.float32` at all, and (before a second bug was also fixed - see
    `_get_data_or_404`) that failure was being silently mis-reported as a
    plain 404 rather than surfacing as the real error it was.
    """
    try:
        if value is None:
            return None
        if hasattr(value, "item"):  # numpy scalar (float32/float64/int64/bool_/...)
            value = value.item()
        if isinstance(value, float) and value != value:  # NaN check without importing math/numpy here
            return None
        return value
    except Exception:
        return None
