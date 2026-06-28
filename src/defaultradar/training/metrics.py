"""Evaluation metrics + the shared model-quality gate.

The same :func:`quality_gate` is used by ``make eval`` and by the Staging->
Production promotion function, so the bar a model must clear is defined exactly
once. Metrics are reported honestly — Lending Club default prediction is hard
and we do not inflate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from defaultradar.config import CONFIG


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic = max separation of TPR and FPR (credit KS)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    """ROC-AUC, PR-AUC, KS and Brier for a probability vector."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


@dataclass
class GateResult:
    """Outcome of the model-quality gate (used by eval + promotion)."""

    passed: bool
    checks: dict[str, tuple[bool, str]]  # hard, blocking: name -> (passed, msg)
    reports: dict[str, tuple[bool, str]] = field(default_factory=dict)  # informational

    def table(self) -> str:
        lines = ["check                       status  detail", "-" * 60]
        for name, (ok, msg) in self.checks.items():
            lines.append(f"{name:27s} {'PASS' if ok else 'FAIL':6s}  {msg}")
        for name, (ok, msg) in self.reports.items():
            lines.append(f"{name:27s} {'(ok)' if ok else '(info)':6s}  {msg}")
        lines.append("-" * 60)
        lines.append(f"{'OVERALL':27s} {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def quality_gate(
    *,
    test_metrics: dict[str, float],
    brier_uncalibrated: float | None = None,
    brier_calibrated: float | None = None,
    latency_p95_ms: float | None = None,
    roc_auc_min: float | None = None,
) -> GateResult:
    """Evaluate the model-quality gate.

    HARD criteria (block promotion):
      * test ROC-AUC >= roc_auc_min (default CONFIG.roc_auc_min = 0.67 — the
        honest achievable floor on this dataset; see config.py for the rationale)
      * serving p95 latency < threshold (when supplied; measured in Phase 4)

    INFORMATIONAL (reported, not blocking — the deployed model is calibrated and
    the promotion path has no uncalibrated model to compare against, so this can
    only be a quality signal, not a hard gate):
      * calibration improved Brier (calibrated <= uncalibrated), when both supplied
    """
    roc_auc_min = roc_auc_min if roc_auc_min is not None else CONFIG.roc_auc_min
    checks: dict[str, tuple[bool, str]] = {}
    reports: dict[str, tuple[bool, str]] = {}

    auc = test_metrics["roc_auc"]
    checks[f"roc_auc>={roc_auc_min:.2f}"] = (auc >= roc_auc_min, f"test ROC-AUC = {auc:.4f}")

    if latency_p95_ms is not None:
        ok = latency_p95_ms < CONFIG.latency_p95_ms
        checks[f"latency_p95<{int(CONFIG.latency_p95_ms)}ms"] = (
            ok,
            f"p95 = {latency_p95_ms:.1f} ms",
        )

    if brier_uncalibrated is not None and brier_calibrated is not None:
        improved = brier_calibrated <= brier_uncalibrated + 1e-6
        reports["calibration_improves_brier"] = (
            improved,
            f"brier {brier_uncalibrated:.5f} -> {brier_calibrated:.5f}",
        )

    passed = all(ok for ok, _ in checks.values())
    return GateResult(passed=passed, checks=checks, reports=reports)


def metrics_table(named_metrics: dict[str, dict[str, float]]) -> str:
    """Render a compact metrics table for several splits (valid/test)."""
    cols = ("roc_auc", "pr_auc", "ks", "brier", "base_rate", "n")
    header = f"{'split':8s}" + "".join(f"{c:>12s}" for c in cols)
    lines = [header, "-" * len(header)]
    for split, m in named_metrics.items():
        row = f"{split:8s}"
        for c in cols:
            v = m.get(c, float("nan"))
            row += f"{int(v):>12d}" if c == "n" else f"{v:>12.4f}"
        lines.append(row)
    return "\n".join(lines)
