"""Greyspot API — FastAPI backend serving the GAT+GRU+ZIP model's precomputed
risk rankings alongside the trivial crash-count baseline this project's own
research found to be competitive with it (see paper/arxiv/main.tex).

Run: `uvicorn app.main:app --reload --app-dir services/api` from the
project root, or `python services/api/run.py`. Requires artifacts built by
`python scripts/build_serving_artifacts.py` first.
"""
from __future__ import annotations

import logging
import math
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
        "Uncertainty-aware road-risk prioritisation - research prototype. "
        "Serves the final GAT+GRU+ZIP model's ranking AND a parameter-free "
        "crash-count baseline side by side, because this project's own "
        "replication study found the model does not reliably beat that "
        "baseline. Every score is a relative-risk research signal, never a "
        "safety guarantee. See /docs for the interactive schema."
    ),
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)


def _get_data_or_404(borough_name: str) -> BoroughData:
    """The single place borough resolution happens - deliberately NOT a
    global exception handler; see the git history for why a blanket
    `ValueError` handler once mis-reported an unrelated JSON-encoding bug
    as a plain 404 and hid it."""
    try:
        return BoroughDataCache.get(borough_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def _score_col(rank_by: str) -> tuple[str, str]:
    """(priority column, raw-score column) for the requested ranking."""
    if rank_by == "baseline":
        return "baseline_priority_score", "baseline_score"
    if rank_by == "model":
        return "priority_score", "model_score"
    raise HTTPException(status_code=422, detail="rank_by must be 'model' or 'baseline'")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/boroughs")
def list_boroughs() -> list[dict]:
    """Registered boroughs - not all have a served artifact yet;
    `/boroughs/{name}/model-info` 404s with a helpful message for one that
    doesn't."""
    return [{"name": b.name, "ons_code": b.ons_code} for b in BOROUGHS.values()]


@app.get("/boroughs/{borough_name}/model-info")
def model_info(borough_name: str) -> dict:
    """Audit/provenance endpoint: every score must be traceable to a model
    version, the exact evaluation window it was scored on, and the
    already-published accuracy figure for that window."""
    data = _get_data_or_404(borough_name)
    info = dict(data.model_info)
    info["n_segments"] = len(data.scored_table)
    return info


@app.get("/boroughs/{borough_name}/roads")
def roads_geojson(borough_name: str) -> dict:
    """Road segments as a GeoJSON FeatureCollection, coloured by priority
    score client-side."""
    data = _get_data_or_404(borough_name)
    import json as _json

    return _json.loads(data.edges.to_json())


@app.get("/boroughs/{borough_name}/roads/{segment_id}")
def road_evidence(borough_name: str, segment_id: str) -> dict:
    """The 'Why this road?' evidence panel for one segment - structured
    evidence and BOTH rankings, not a bare number from one model."""
    data = _get_data_or_404(borough_name)
    row = data.scored_table[data.scored_table["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id!r} not found for {borough_name}.")
    r = row.iloc[0]
    info = data.model_info
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
            "model_version": info.get("model_version"),
            "predicted_crashes_next_14_days": _safe(r.get("model_score")),
            "conformal_interval_90pct": [_safe(r.get("interval_lower")), _safe(r.get("interval_upper"))],
            "as_of_window": info.get("held_out_start"),
        },
        "baseline_evidence": {
            "method": "cumulative crash count, full history, no model",
            "score": _safe(r.get("baseline_score")),
            "research_note": info.get("research_finding"),
        },
        "priority_score": {
            "model_ranked": {
                "score_0_100": _safe(r.get("priority_score")),
                "components": {
                    "risk_signal": _safe(r.get("component_risk_signal")),
                    "severity": _safe(r.get("component_severity")),
                    "vulnerable_users": _safe(r.get("component_vulnerable_users")),
                    "trend": _safe(r.get("component_trend")),
                    "network_importance": _safe(r.get("component_network_importance")),
                    "data_confidence": _safe(r.get("component_data_confidence")),
                },
            },
            "baseline_ranked": {"score_0_100": _safe(r.get("baseline_priority_score"))},
            "policy_profile": r.get("policy_profile"),
            "data_confidence_is_default": bool(r.get("data_confidence_is_default", True)),
        },
        "limitations": [
            "Relative risk research signal, not a prediction of a specific collision.",
            "Priority score weights are a labelled policy prototype, not a validated finding.",
            "The model score is not reliably better than the baseline shown above - see baseline_evidence.",
            "Scored as of a single historical evaluation window, not live data.",
        ],
    }


@app.get("/boroughs/{borough_name}/priority-queue")
def priority_queue(
    borough_name: str,
    limit: int = Query(default=25, ge=1, le=500),
    rank_by: str = Query(default="model", description="'model' (GAT+GRU+ZIP) or 'baseline' (crash-count sort)"),
) -> list[dict]:
    """Ranked list of segments - the core 'where should we investigate
    first?' workflow. `rank_by=baseline` serves the parameter-free
    crash-count ranking through the identical response shape, so a
    frontend toggle needs no special-casing."""
    data = _get_data_or_404(borough_name)
    priority_col, score_col = _score_col(rank_by)
    table = data.scored_table
    top = table.sort_values(priority_col, ascending=False).head(limit)
    return [
        {
            "segment_id": row["segment_id"],
            "priority_score": _safe(row[priority_col]),
            "model_score": _safe(row[score_col]),
            "prior_year_count": _safe(row.get("prior_year_count")),
            "highway": row.get("highway"),
            "rank_by": rank_by,
        }
        for _, row in top.iterrows()
    ]


def _safe(value):
    """NaN/NaT -> None, numpy scalar -> native Python, so FastAPI's JSON
    encoder never chokes (it cannot serialize numpy.float32 at all)."""
    try:
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    except Exception:
        return None
