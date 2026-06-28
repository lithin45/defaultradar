"""Training pipeline tests.

* A fast, offline smoke (sample-backed) that exercises the full
  train -> calibrate -> evaluate -> gate machinery — this is the model-quality
  gate that runs in CI.
* An integration test that logs a run and registers a model version against a
  live MLflow server (auto-skips when the stack is down).
"""

from __future__ import annotations

import pytest

from defaultradar.training import GateResult, ModelParams, train_and_log

SAMPLE = "tests/fixtures/lc_sample.csv"
# Tiny, fast model for CI; we assert the machinery, not a production AUC.
FAST = ModelParams(n_estimators=60, max_depth=4)


def test_train_smoke_produces_metrics_and_gate() -> None:
    """End-to-end train on the committed sample without MLflow (CI gate)."""
    result = train_and_log(params=FAST, source=SAMPLE, log_to_mlflow=False)

    # Metrics computed for valid + test, with all expected keys.
    for split in ("valid", "test"):
        m = result.metrics[split]
        assert {"roc_auc", "pr_auc", "ks", "brier"} <= set(m)
        assert m["roc_auc"] > 0.5  # better than random on a separable sample

    # The gate machinery returns a structured result.
    assert isinstance(result.gate, GateResult)
    assert "roc_auc>=0.67" in next(iter(result.gate.checks)) or any(
        "roc_auc" in k for k in result.gate.checks
    )
    # No MLflow -> no run/version.
    assert result.run_id is None and result.model_version is None


def test_calibration_produces_valid_probabilities() -> None:
    """Calibrated + uncalibrated Brier are valid; no catastrophic regression.

    On the tiny sample (valid=340 rows) isotonic calibration is noisy, so we only
    assert sane probabilities here. Strict improvement is asserted on full data by
    ``test_calibration_improves_brier_full`` (slow) and shown by ``make train``.
    """
    result = train_and_log(params=FAST, source=SAMPLE, log_to_mlflow=False)
    cal = result.metrics["test"]["brier"]
    assert 0.0 <= cal <= 1.0
    assert 0.0 <= result.brier_uncalibrated <= 1.0
    assert cal <= result.brier_uncalibrated + 0.03  # no catastrophic regression


@pytest.mark.slow
def test_calibration_improves_brier_full() -> None:
    """On the full dataset, isotonic calibration strictly improves test Brier."""
    result = train_and_log(log_to_mlflow=False)  # full splits, full params
    assert result.metrics["test"]["brier"] <= result.brier_uncalibrated
    assert result.metrics["test"]["roc_auc"] >= 0.67  # the honest gate


@pytest.mark.integration
def test_train_and_log_registers_version(require_mlflow, monkeypatch) -> None:
    """A run is logged and a model version is registered in MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from defaultradar.config import CONFIG

    # Register under a throwaway name so the real model's registry stays clean.
    test_name = "defaultradar-ci-test-model"
    monkeypatch.setattr(CONFIG, "registered_model_name", test_name)

    result = train_and_log(params=FAST, source=SAMPLE, log_to_mlflow=True, register=True)
    assert result.run_id is not None
    assert result.model_version is not None

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{test_name}'")
    assert any(v.version == result.model_version for v in versions)
    # The run carries the feature-config hash tag (lineage).
    run = client.get_run(result.run_id)
    assert run.data.tags.get("feature_config_hash") == result.config_hash
