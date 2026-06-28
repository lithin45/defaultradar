"""Serving API tests: validation (always), and scoring + latency (integration)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from defaultradar.config import CONFIG
from defaultradar.serving.app import app

client = TestClient(app)

VALID_APPLICATION = {
    "revenue": 65000.0,
    "dti_n": 16.06,
    "loan_amnt": 24700.0,
    "fico_n": 717.0,
    "experience_c": 1,
    "emp_length": "10+ years",
    "purpose": "small_business",
    "home_ownership_n": "MORTGAGE",
    "addr_state": "SD",
    "zip_code": "577xx",
    "title": "Business",
    "desc": "",
}


# --- Always-on (no model required) ------------------------------------------
def test_health_always_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_invalid_payload() -> None:
    """Pydantic validation: missing required fields -> 422."""
    r = client.post("/predict", json={"revenue": 50000})  # missing many required
    assert r.status_code == 422


def test_predict_rejects_outcome_derived_field() -> None:
    """An outcome-derived field (e.g. int_rate) must be rejected, not ignored."""
    payload = {**VALID_APPLICATION, "int_rate": 13.5, "loan_status": "Charged Off"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# --- Integration (require a served Production model) -------------------------
@pytest.fixture
def served():
    from defaultradar.serving.service import SERVICE

    if not SERVICE.ensure_loaded():
        pytest.skip("No served model available (run make up + make train + make promote).")


@pytest.mark.integration
def test_model_info(served) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["registered_model_name"] == CONFIG.registered_model_name
    assert body["n_features"] == 15
    assert "version" in body and "stage" in body


@pytest.mark.integration
def test_predict_returns_calibrated_probability(served) -> None:
    r = client.post("/predict", json=VALID_APPLICATION)
    assert r.status_code == 200
    body = r.json()
    p = body["default_probability"]
    assert 0.0 < p < 1.0
    assert body["model_version"]
    assert body["explanation"] is None  # not requested


@pytest.mark.integration
def test_predict_with_shap_explanation(served) -> None:
    r = client.post("/predict?explain=true", json=VALID_APPLICATION)
    assert r.status_code == 200
    exp = r.json()["explanation"]
    assert exp is not None
    assert "base_value" in exp
    assert len(exp["top_contributions"]) > 0
    assert {"feature", "shap"} <= set(exp["top_contributions"][0])


@pytest.mark.integration
def test_predict_latency_p95_under_threshold(served) -> None:
    # warm up
    for _ in range(5):
        client.post("/predict", json=VALID_APPLICATION)
    lat = []
    for _ in range(100):
        t = time.perf_counter()
        client.post("/predict", json=VALID_APPLICATION)
        lat.append((time.perf_counter() - t) * 1000)
    p95 = float(np.percentile(lat, 95))
    assert p95 < CONFIG.latency_p95_ms, f"p95={p95:.1f}ms exceeds {CONFIG.latency_p95_ms}ms"
