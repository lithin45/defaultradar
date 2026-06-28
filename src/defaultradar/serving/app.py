"""FastAPI application for the DefaultRadar scorer.

Endpoints
---------
* ``GET  /health``      — liveness; always 200 once the process is up.
* ``GET  /model-info``  — served version/stage + metrics (503 until a model exists).
* ``POST /predict``     — Pydantic-validated application -> calibrated P(default),
                          with an optional per-prediction SHAP explanation.
* ``POST /reload``      — refresh the cached model after a promotion.

The service loads the current Production model from the MLflow registry (falling
back to the latest version when nothing is promoted yet). ``/health`` stays green
even before any model is trained, so the container is healthy on a fresh clone.
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from defaultradar import __version__
from defaultradar.config import CONFIG
from defaultradar.serving.schemas import LoanApplication, ModelInfoResponse, PredictionResponse
from defaultradar.serving.service import SERVICE

app = FastAPI(
    title="DefaultRadar Scorer",
    version=__version__,
    summary="Calibrated loan-default probabilities from the Production model.",
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by the compose healthcheck."""
    return {"status": "ok", "service": "defaultradar-scorer", "version": __version__}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> JSONResponse | ModelInfoResponse:
    """Report the served model version + metrics."""
    if not SERVICE.ensure_loaded():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "No served model is available yet.",
                "registered_model_name": CONFIG.registered_model_name,
                "hint": "Run `make train` then `make promote`, or `make demo`.",
            },
        )
    return SERVICE.model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    application: LoanApplication,
    explain: bool = Query(False, description="Include a per-prediction SHAP explanation"),
) -> JSONResponse | PredictionResponse:
    """Score a loan application -> calibrated default probability (+ optional SHAP)."""
    if not SERVICE.ensure_loaded():
        return JSONResponse(
            status_code=503,
            content={"detail": "No served model is available yet. Train + promote first."},
        )
    return SERVICE.predict(application, explain=explain)


@app.post("/reload", tags=["ops"])
def reload_model() -> JSONResponse:
    """Refresh the cached model (call after a promotion)."""
    ok = SERVICE.reload()
    status = 200 if ok else 503
    return JSONResponse(
        status_code=status,
        content={"reloaded": ok, "version": SERVICE._version, "stage": SERVICE._stage},
    )
