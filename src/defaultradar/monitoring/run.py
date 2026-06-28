"""Monitoring run: score an 'incoming' batch, compute drift, persist reports.

This is the plain-Python core that ``make monitor`` and the Prefect monitoring
flow both call. The model's training distribution (train split) is the reference;
the test split is the "incoming" batch, optionally drift-injected to demonstrate
the gate firing.
"""

from __future__ import annotations

import json
from pathlib import Path

from defaultradar.config import CONFIG
from defaultradar.features.pipeline import FEATURE_COLUMNS, load_split
from defaultradar.monitoring.drift import DriftResult, evaluate_drift, evidently_reports
from defaultradar.monitoring.simulate import inject_drift


def _load_production_model():
    """Load the CURRENT PRODUCTION model (strict).

    Returns ``(None, None)`` when nothing is in Production, so prediction-drift is
    simply skipped rather than being scored against a non-Production model.
    """
    try:
        import mlflow

        from defaultradar.registry import current_production

        prod = current_production()
        if prod is None:
            return None, None
        version = prod[0]
        mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
        model = mlflow.sklearn.load_model(f"models:/{CONFIG.registered_model_name}/Production")
        return model, version
    except Exception:
        return None, None


def run_monitoring(
    *,
    inject: bool = True,
    write_reports: bool = True,
    sample_size: int = 20_000,
    threshold: float | None = None,
    seed: int | None = None,
) -> DriftResult:
    """Run one monitoring cycle and return the drift result (with retrain flag)."""
    seed = CONFIG.seed if seed is None else seed

    reference = load_split("train")[list(FEATURE_COLUMNS)]
    incoming = load_split("test")[list(FEATURE_COLUMNS)]

    # Subsample for a fast, stable PSI estimate (deterministic).
    if len(reference) > sample_size:
        reference = reference.sample(sample_size, random_state=seed)
    if len(incoming) > sample_size:
        incoming = incoming.sample(sample_size, random_state=seed)

    current = inject_drift(incoming) if inject else incoming

    # Prediction drift (if a Production model is available).
    ref_scores = cur_scores = None
    model, version = _load_production_model()
    if model is not None:
        # The prediction-PSI baseline must be OUT-OF-SAMPLE: score the held-out
        # valid split (the base model never trained on it), NOT the training data,
        # whose in-sample scores are systematically over-confident.
        valid = load_split("valid")[list(FEATURE_COLUMNS)]
        if len(valid) > sample_size:
            valid = valid.sample(sample_size, random_state=seed)
        ref_scores = model.predict_proba(valid)[:, 1]
        cur_scores = model.predict_proba(current)[:, 1]

    result = evaluate_drift(
        reference, current, ref_scores=ref_scores, cur_scores=cur_scores, threshold=threshold
    )

    if write_reports:
        prefix = "drift_injected" if inject else "drift_baseline"
        try:
            result.reports = evidently_reports(reference, current, prefix=prefix)
        except Exception as exc:  # pragma: no cover - Evidently optional at gate time
            result.reports = {"evidently_error": str(exc)}
        _write_summary(result, prefix=prefix, served_version=version)

    return result


def _write_summary(result: DriftResult, *, prefix: str, served_version: str | None) -> Path:
    CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
    path = CONFIG.reports_dir / f"{prefix}_summary.json"
    payload = {
        "served_version": served_version,
        "threshold": result.threshold,
        "feature_psi": result.feature_psi,
        "prediction_psi": result.prediction_psi,
        "drifted_features": result.drifted_features,
        "drift_detected": result.drift_detected,
        "n_reference": result.n_reference,
        "n_current": result.n_current,
        "reports": result.reports,
    }
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
