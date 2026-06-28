"""Unit tests for metrics + the shared model-quality gate (the CI gate logic)."""

from __future__ import annotations

import numpy as np

from defaultradar.training.metrics import (
    classification_metrics,
    ks_statistic,
    quality_gate,
)


def _toy_scores(sep: float = 1.0, n: int = 400, seed: int = 0):
    """Two gaussian score clusters; `sep` controls class separability."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = rng.normal(0.0, 1.0, n) + y * sep
    # squash to (0,1) so it looks like a probability
    p = 1 / (1 + np.exp(-scores))
    return y, p


def test_classification_metrics_keys_and_ranges() -> None:
    y, p = _toy_scores(sep=2.0)
    m = classification_metrics(y, p)
    assert {"roc_auc", "pr_auc", "ks", "brier", "base_rate", "n"} <= set(m)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert 0.0 <= m["ks"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] == len(y)


def test_ks_statistic_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0


def test_quality_gate_blocks_below_threshold() -> None:
    # Below the floor -> fail.
    g = quality_gate(test_metrics={"roc_auc": 0.60}, roc_auc_min=0.67)
    assert not g.passed
    # At/above the floor -> pass.
    g2 = quality_gate(test_metrics={"roc_auc": 0.69}, roc_auc_min=0.67)
    assert g2.passed


def test_quality_gate_calibration_is_reported_not_blocking() -> None:
    # Calibration improvement is reported but does NOT block (the promotion path
    # has no uncalibrated model to compare, so it can only be a quality signal).
    g = quality_gate(
        test_metrics={"roc_auc": 0.70},
        brier_uncalibrated=0.15,
        brier_calibrated=0.18,  # worsened
        roc_auc_min=0.67,
    )
    assert g.passed  # ROC-AUC passes -> gate passes regardless of calibration
    assert "calibration_improves_brier" in g.reports
    assert g.reports["calibration_improves_brier"][0] is False

    g2 = quality_gate(
        test_metrics={"roc_auc": 0.70},
        brier_uncalibrated=0.18,
        brier_calibrated=0.15,  # improved
        roc_auc_min=0.67,
    )
    assert g2.passed
    assert g2.reports["calibration_improves_brier"][0] is True


def test_quality_gate_latency_check() -> None:
    g = quality_gate(test_metrics={"roc_auc": 0.70}, latency_p95_ms=500.0, roc_auc_min=0.67)
    assert not g.passed  # 500ms > 300ms threshold
    g2 = quality_gate(test_metrics={"roc_auc": 0.70}, latency_p95_ms=50.0, roc_auc_min=0.67)
    assert g2.passed
