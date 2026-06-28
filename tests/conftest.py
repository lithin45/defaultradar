"""Shared pytest fixtures for DefaultRadar.

Two cross-cutting concerns live here:

* ``data_source`` — resolves the best available data source (full Parquet if
  ``make data`` has run, else the committed CI sample) so the same tests run in
  CI and locally.
* ``require_mlflow`` — skips integration tests gracefully when the MLflow server
  is not reachable, so ``make test`` is green whether or not the stack is up.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from defaultradar.config import CONFIG


def _mlflow_reachable(timeout: float = 2.0) -> bool:
    """Return True if the MLflow server answers /health with 200."""
    url = CONFIG.mlflow_tracking_uri.rstrip("/") + "/health"
    try:
        resp = httpx.get(url, timeout=timeout)
    except (httpx.HTTPError, OSError):
        return False
    return resp.status_code == 200


@pytest.fixture(scope="session")
def data_source() -> Path:
    """Best available data source: full Parquet if present, else CI sample."""
    from defaultradar.data.duckdb_summary import resolve_source

    return resolve_source()


@pytest.fixture
def require_mlflow() -> None:
    """Skip the test unless the MLflow server is reachable."""
    if not _mlflow_reachable():
        pytest.skip(
            f"MLflow not reachable at {CONFIG.mlflow_tracking_uri} "
            "(run `make up` / `make up-core` to enable integration tests)."
        )
