"""Integration tests for the Greyspot API against the precomputed serving
artifacts. Run `python scripts/build_serving_artifacts.py Westminster` (and
any other borough exercised below) first if `data/served/westminster/`
doesn't exist. Deliberately integration-style rather than mocked: the
point is "do these real objects survive parquet -> pandas -> JSON without
corrupting NaN/None handling", which a mock can't test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services" / "api"))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_boroughs_includes_westminster_and_lambeth():
    resp = client.get("/boroughs")
    assert resp.status_code == 200
    names = {b["name"] for b in resp.json()}
    assert "Westminster" in names
    assert "Lambeth" in names


def test_unknown_borough_returns_404_not_500():
    resp = client.get("/boroughs/Notaplace/model-info")
    assert resp.status_code == 404


def test_model_info_westminster_has_expected_shape():
    resp = client.get("/boroughs/Westminster/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["borough"] == "Westminster"
    assert "held_out_start" in body
    assert "research_finding" in body  # the trivial-baseline finding must always be present
    assert "non_negotiable_boundary" in body


def test_roads_geojson_is_a_valid_feature_collection():
    resp = client.get("/boroughs/Westminster/roads")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) > 0
    first = body["features"][0]
    assert first["type"] == "Feature"
    assert "segment_id" in first["properties"]


def test_priority_queue_is_sorted_descending_and_respects_limit():
    resp = client.get("/boroughs/Westminster/priority-queue?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 10
    scores = [row["priority_score"] for row in body]
    assert scores == sorted(scores, reverse=True)


def test_priority_queue_rows_carry_a_real_street_name():
    """2026-09-08 regression test: every row previously showed nothing but
    a raw segment UUID (the `name` column was computed in `edges` but
    dropped before it reached the served artifact - see
    scripts/build_serving_artifacts.py). OS Open Roads names the large
    majority of segments, so most rows in a real batch must carry one -
    not just "the field exists and is null everywhere", which would pass
    a shallower `"name" in row` check just as easily."""
    body = client.get("/boroughs/Westminster/priority-queue?limit=20").json()
    assert all("name" in row for row in body)
    named = [row for row in body if row["name"]]
    assert len(named) >= len(body) * 0.5, (
        f"only {len(named)}/{len(body)} rows had a name - expected most OS Open Roads "
        "segments to be named; check build_serving_artifacts.py's name_lookup merge."
    )


def test_priority_queue_baseline_ranking_is_a_different_but_valid_order():
    """The research finding this API exists to be honest about: a
    parameter-free baseline ranking must be servable through the identical
    endpoint shape, not bolted on as a special case."""
    model_ranked = client.get("/boroughs/Westminster/priority-queue?limit=20&rank_by=model").json()
    baseline_ranked = client.get("/boroughs/Westminster/priority-queue?limit=20&rank_by=baseline").json()
    assert len(model_ranked) == len(baseline_ranked) == 20
    for row in baseline_ranked:
        assert row["rank_by"] == "baseline"
    scores = [row["priority_score"] for row in baseline_ranked]
    assert scores == sorted(scores, reverse=True)


def test_priority_queue_rejects_unknown_rank_by():
    resp = client.get("/boroughs/Westminster/priority-queue?rank_by=nonsense")
    assert resp.status_code == 422


def test_road_evidence_for_a_real_segment_has_full_structure():
    queue = client.get("/boroughs/Westminster/priority-queue?limit=1").json()
    segment_id = queue[0]["segment_id"]

    resp = client.get(f"/boroughs/Westminster/roads/{segment_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["segment_id"] == segment_id
    assert "name" in body  # may be null for a genuinely unnamed segment, but the key must exist
    for section in ["observed_evidence", "exposure", "model_evidence", "baseline_evidence",
                     "priority_score", "limitations"]:
        assert section in body


def test_road_evidence_for_unknown_segment_is_404():
    resp = client.get("/boroughs/Westminster/roads/does_not_exist_0_0")
    assert resp.status_code == 404


def test_response_json_never_contains_bare_nan_tokens():
    # A literal `NaN` in a JSON body is invalid JSON per spec - exactly
    # what `_safe()` in main.py exists to prevent silently reappearing.
    resp = client.get("/boroughs/Westminster/priority-queue?limit=500")
    assert "NaN" not in resp.text
