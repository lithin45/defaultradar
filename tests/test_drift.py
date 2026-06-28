"""Drift-gate tests: PSI correctness + injected drift fires, baseline does not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from defaultradar.features import build_feature_matrix, load_feature_contract
from defaultradar.monitoring import (
    evaluate_drift,
    inject_drift,
    population_stability_index,
    prediction_psi,
)


def _features(data_source) -> pd.DataFrame:
    raw = (
        pd.read_parquet(data_source)
        if data_source.suffix == ".parquet"
        else pd.read_csv(data_source)
    )
    return build_feature_matrix(raw, load_feature_contract())


# --- PSI unit tests ---------------------------------------------------------
def test_psi_zero_for_identical_numeric() -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 5000))
    assert population_stability_index(x, x.copy(), categorical=False) < 1e-6


def test_psi_numeric_detects_large_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(2.5, 1, 5000))  # mean shifted by 2.5 sigma
    assert population_stability_index(ref, cur, categorical=False) > 0.2


def test_psi_categorical_detects_shift() -> None:
    ref = pd.Series(["a"] * 800 + ["b"] * 200)
    cur = pd.Series(["a"] * 200 + ["b"] * 800)
    assert population_stability_index(ref, cur, categorical=True) > 0.2


def test_prediction_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.0, 0.3, 5000)
    cur = rng.uniform(0.4, 0.8, 5000)
    assert prediction_psi(ref, cur) > 0.2


# --- gate behaviour ---------------------------------------------------------
def test_injected_drift_fires_gate(data_source) -> None:
    feats = _features(data_source)
    drifted = inject_drift(feats)
    result = evaluate_drift(feats, drifted, threshold=0.2)
    assert result.drift_detected
    # The shifted key features must be flagged.
    assert "fico_n" in result.drifted_features
    assert result.feature_psi["fico_n"] > 0.2


def test_no_drift_baseline_does_not_fire(data_source) -> None:
    feats = _features(data_source)
    result = evaluate_drift(feats, feats.copy(), threshold=0.2)
    assert not result.drift_detected
    assert result.drifted_features == []


def test_evidently_reports_generated(data_source, tmp_path) -> None:
    """Evidently produces an HTML drift report artifact."""
    from defaultradar.monitoring import evidently_reports

    feats = _features(data_source).head(1000)
    drifted = inject_drift(feats)
    paths = evidently_reports(feats, drifted, out_dir=tmp_path, prefix="test_drift")
    assert "html" in paths
    from pathlib import Path

    assert Path(paths["html"]).exists() and Path(paths["html"]).stat().st_size > 0


@pytest.mark.integration
def test_run_monitoring_full_injected_fires() -> None:
    """End-to-end monitoring cycle on the full feature store fires on injection."""
    from defaultradar.features.pipeline import load_split

    try:
        load_split("train")
    except FileNotFoundError:
        pytest.skip("feature store not built; run `make features`")

    from defaultradar.monitoring import run_monitoring

    result = run_monitoring(inject=True, write_reports=False, sample_size=5000)
    assert result.drift_detected
