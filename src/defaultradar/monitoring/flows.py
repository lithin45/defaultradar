"""Prefect flows for scheduled monitoring (and, in Phase 6, retraining).

The monitoring flow scores an incoming batch, computes drift, persists reports
and returns the retrain decision. It is schedulable via Prefect (``flow.serve``)
and is the entrypoint the automated retraining loop hooks into.
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from defaultradar.monitoring.drift import DriftResult
from defaultradar.monitoring.run import run_monitoring


@task(name="score-and-detect-drift")
def _detect_drift(inject: bool, write_reports: bool, sample_size: int) -> DriftResult:
    return run_monitoring(inject=inject, write_reports=write_reports, sample_size=sample_size)


@flow(name="defaultradar-monitoring")
def monitoring_flow(
    inject: bool = True,
    write_reports: bool = True,
    sample_size: int = 20_000,
) -> dict:
    """Scheduled monitoring cycle: drift detection over an incoming batch.

    Returns a small dict with the drift decision so an orchestration layer (or the
    retraining flow) can branch on ``drift_detected``.
    """
    logger = get_run_logger()
    result = _detect_drift(inject, write_reports, sample_size)

    logger.info(
        "drift_detected=%s drifted_features=%s prediction_psi=%s",
        result.drift_detected,
        result.drifted_features,
        result.prediction_psi,
    )
    return {
        "drift_detected": result.drift_detected,
        "drifted_features": result.drifted_features,
        "prediction_psi": result.prediction_psi,
        "feature_psi": result.feature_psi,
        "reports": result.reports,
    }


@task(name="retrain")
def _retrain() -> dict:
    from defaultradar.training import retrain_and_log

    result = retrain_and_log(log_to_mlflow=True, register=True)
    return {
        "model_version": result.model_version,
        "test_roc_auc": result.metrics["test"]["roc_auc"],
        "gate_passed": result.gate.passed,
    }


@task(name="gate-and-promote")
def _gate_and_promote(version: str | None) -> dict:
    from defaultradar.registry import promote_model

    result = promote_model(version=version)
    return {
        "version": result.version,
        "promoted": result.promoted,
        "stage": result.stage,
        "gate_passed": result.gate.passed,
    }


@flow(name="defaultradar-retraining")
def retraining_flow() -> dict:
    """Retrain on the expanded window -> register -> gate -> promote.

    Returns the new version + whether it was promoted. Called by the lifecycle
    flow when drift is detected.
    """
    logger = get_run_logger()
    trained = _retrain()
    logger.info(
        "retrained version=%s gate_passed=%s", trained["model_version"], trained["gate_passed"]
    )
    promo = _gate_and_promote(trained["model_version"])
    logger.info("promotion: version=%s promoted=%s", promo["version"], promo["promoted"])
    return {"retrained": trained, "promotion": promo}


@flow(name="defaultradar-lifecycle")
def lifecycle_flow(inject: bool = True, sample_size: int = 20_000) -> dict:
    """The full closed loop: monitor -> (if drift) retrain -> gate -> promote.

    This is the heart of DefaultRadar: drift detection automatically triggers
    retraining and gated promotion, so the served Production model self-heals.
    """
    logger = get_run_logger()
    monitoring = monitoring_flow(inject=inject, write_reports=True, sample_size=sample_size)

    if not monitoring["drift_detected"]:
        logger.info("no drift -> no retraining")
        return {"monitoring": monitoring, "retraining": None, "action": "none"}

    logger.info("drift detected -> triggering retraining")
    retraining = retraining_flow()
    return {"monitoring": monitoring, "retraining": retraining, "action": "retrained"}


if __name__ == "__main__":  # pragma: no cover
    # `python -m defaultradar.monitoring.flows` runs one monitoring cycle locally.
    monitoring_flow()
