"""Phase 6: the closed loop — retrain -> register -> gate -> promote -> served update.

Integration tests (require the MLflow stack + a built feature store). They retrain
on the full expanded window, so they are local-only and skipped in CI.
"""

from __future__ import annotations

import pytest

# These retrain on the full expanded window, so they are also `slow`: excluded
# from the default `make test`, run via `pytest -m slow` / `make demo`.
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def feature_store_ready(require_mlflow):
    from defaultradar.features.pipeline import load_split

    try:
        load_split("train")
    except FileNotFoundError:
        pytest.skip("feature store not built; run `make features`")


def test_retrain_produces_new_passing_version(feature_store_ready) -> None:
    from defaultradar.training import retrain_and_log

    result = retrain_and_log(log_to_mlflow=True, register=True)
    assert result.model_version is not None
    # The expanded-window model still clears the honest ROC-AUC gate on test.
    assert result.gate.passed
    assert result.metrics["test"]["roc_auc"] >= 0.67


def test_retraining_flow_gates_and_promotes(feature_store_ready) -> None:
    from defaultradar.monitoring import retraining_flow
    from defaultradar.registry import current_production

    out = retraining_flow()
    assert out["retrained"]["model_version"] is not None
    assert out["promotion"]["promoted"] is True
    # The promoted version is now the served Production model.
    prod = current_production()
    assert prod is not None and prod[0] == out["promotion"]["version"]


def test_full_lifecycle_updates_served_version(feature_store_ready) -> None:
    """drift -> retrain -> promote changes the served Production version."""
    from defaultradar.demo import run_demo

    result = run_demo(inject=True, sample_size=5000)
    assert result.drift_detected
    assert result.promoted
    assert result.served_version_changed
    assert result.version_after == result.retrained_version
