"""``make eval`` — evaluate the registered model on test and enforce the gate.

Loads the current model from the MLflow Model Registry (Production if present,
else the latest version), scores the held-out test split, prints an honest
metrics table, and returns a non-zero exit code if the hard gate is missed. The
gate is the same :func:`quality_gate` used by the promotion function.
"""

from __future__ import annotations

from dataclasses import dataclass

from defaultradar.config import CONFIG
from defaultradar.features.pipeline import get_xy, load_split
from defaultradar.features.schema import load_feature_contract
from defaultradar.training.metrics import (
    GateResult,
    classification_metrics,
    metrics_table,
    quality_gate,
)


@dataclass
class EvalResult:
    model_version: str | None
    stage: str | None
    metrics: dict[str, float]
    gate: GateResult


def load_registered_model(prefer_production: bool = True):
    """Load the served model: Production alias/stage if present, else latest version.

    Returns ``(model, version, stage)``.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    client = MlflowClient()
    name = CONFIG.registered_model_name

    if prefer_production:
        prod = client.get_latest_versions(name, stages=["Production"])
        if prod:
            mv = prod[0]
            model = mlflow.sklearn.load_model(f"models:/{name}/Production")
            return model, mv.version, "Production"

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions for model {name!r}. Run `make train` first.")
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
    return model, latest.version, latest.current_stage


def evaluate_on_test(model, *, latency_p95_ms: float | None = None) -> EvalResult:
    """Score the test split with a loaded model and apply the quality gate."""
    contract = load_feature_contract()
    X_test, y_test = get_xy(load_split("test"), contract)
    p = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, p)
    gate = quality_gate(test_metrics=metrics, latency_p95_ms=latency_p95_ms)
    return EvalResult(model_version=None, stage=None, metrics=metrics, gate=gate)


def run_eval(prefer_production: bool = True) -> int:
    """Entry point for ``make eval``. Returns a process exit code (0 = pass)."""
    model, version, stage = load_registered_model(prefer_production=prefer_production)
    result = evaluate_on_test(model)
    result.model_version, result.stage = version, stage

    print("=" * 64)
    print("DefaultRadar — make eval (model-quality gate)")
    print("=" * 64)
    print(f"model_version : {version}  (stage: {stage})")
    print()
    print(metrics_table({"test": result.metrics}))
    print()
    print(result.gate.table())
    print("=" * 64)
    return 0 if result.gate.passed else 1
