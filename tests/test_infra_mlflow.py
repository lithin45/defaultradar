"""Integration tests: the MLflow tracking server is reachable + usable.

These require the docker stack (``make up`` / ``make up-core``). They auto-skip
via the ``require_mlflow`` fixture when the server is not up, so ``make test``
stays green in environments without the stack.
"""

from __future__ import annotations

import httpx
import pytest

from defaultradar.config import CONFIG

pytestmark = pytest.mark.integration


def test_mlflow_health(require_mlflow) -> None:
    url = CONFIG.mlflow_tracking_uri.rstrip("/") + "/health"
    resp = httpx.get(url, timeout=5.0)
    assert resp.status_code == 200


def test_mlflow_tracking_api(require_mlflow) -> None:
    """The tracking API answers and we can create/find an experiment."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    client = MlflowClient()

    # search_experiments hits the REST API backed by Postgres.
    experiments = client.search_experiments()
    assert isinstance(experiments, list)

    name = "defaultradar-smoke"
    exp = client.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else client.create_experiment(name)
    assert exp_id is not None
