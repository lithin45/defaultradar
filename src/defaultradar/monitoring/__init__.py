"""Evidently drift + PSI monitoring and Prefect flows (Phase 5/6)."""

from defaultradar.monitoring.drift import (
    KEY_FEATURES,
    DriftResult,
    evaluate_drift,
    evidently_reports,
    feature_psi,
    population_stability_index,
    prediction_psi,
)
from defaultradar.monitoring.run import run_monitoring
from defaultradar.monitoring.simulate import inject_drift

# The Prefect flows are imported lazily so that the drift/PSI utilities can be
# used (e.g. in a lightweight dashboard) without requiring Prefect installed.
_LAZY_FLOWS = {"lifecycle_flow", "monitoring_flow", "retraining_flow"}


def __getattr__(name: str):
    if name in _LAZY_FLOWS:
        from defaultradar.monitoring import flows

        return getattr(flows, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KEY_FEATURES",
    "DriftResult",
    "evaluate_drift",
    "evidently_reports",
    "feature_psi",
    "inject_drift",
    "lifecycle_flow",
    "monitoring_flow",
    "population_stability_index",
    "prediction_psi",
    "retraining_flow",
    "run_monitoring",
]
