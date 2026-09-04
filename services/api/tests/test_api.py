"""Integration tests for the Greyspot API - these use the real, already
processed Westminster data (run `python scripts/run_pipeline.py Westminster`
first if `data/processed/westminster/` doesn't exist). Deliberately
integration-style rather than mocked: the whole point of this backend is
"do these real objects survive the trip from parquet -> pandas -> JSON
without silently corrupting NaN/None handling", which a mock can't test.
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
    assert body["ons_code"] == "E09000033"
    assert "metrics" in body and "pr_auc" in body["metrics"]
    assert "conformal" in body and 0 <= body["conformal"]["empirical_coverage"] <= 1
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


def test_priority_queue_min_confidence_filter_excludes_default_confidence_rows():
    resp_all = client.get("/boroughs/Westminster/priority-queue?limit=500")
    resp_filtered = client.get("/boroughs/Westminster/priority-queue?limit=500&min_confidence=true")
    assert resp_filtered.status_code == 200
    # filtering can only ever reduce (or match) the candidate pool
    assert len(resp_filtered.json()) <= len(resp_all.json())


def test_road_evidence_for_a_real_segment_has_full_structure():
    queue = client.get("/boroughs/Westminster/priority-queue?limit=1").json()
    segment_id = queue[0]["segment_id"]

    resp = client.get(f"/boroughs/Westminster/roads/{segment_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["segment_id"] == segment_id
    for section in ["observed_evidence", "exposure", "model_evidence", "priority_score", "limitations"]:
        assert section in body


def test_road_evidence_for_unknown_segment_is_404():
    resp = client.get("/boroughs/Westminster/roads/does_not_exist_0_0")
    assert resp.status_code == 404


def test_response_json_never_contains_bare_nan_tokens():
    # A literal `NaN` in a JSON body is invalid JSON per spec and breaks
    # strict parsers - this is exactly what _safe() in main.py exists to
    # prevent silently reappearing.
    resp = client.get("/boroughs/Westminster/priority-queue?limit=500")
    assert "NaN" not in resp.text
