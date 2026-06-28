"""Metric-gated Staging -> Production promotion (MLflow Model Registry).

Promotion is **code, not a manual click**: a model version is moved to Staging,
evaluated on the held-out test split, and only transitioned to Production if it
clears the SAME :func:`quality_gate` used by ``make eval``. The previously-served
Production version is archived. This is the gate the automated retraining loop
(Phase 6) calls after registering a fresh version.

Uses MLflow 2.x registry *stages* (Staging/Production), as specified.
"""

from __future__ import annotations

from dataclasses import dataclass

from defaultradar.config import CONFIG
from defaultradar.training.evaluate import evaluate_on_test
from defaultradar.training.metrics import GateResult


@dataclass
class PromotionResult:
    version: str
    promoted: bool
    stage: str
    gate: GateResult
    metrics: dict[str, float]
    archived: list[str]

    def report(self) -> str:
        lines = [
            "=" * 60,
            "DefaultRadar — promotion gate (Staging -> Production)",
            "=" * 60,
            f"candidate version : {self.version}",
            f"final stage       : {self.stage}",
            f"promoted          : {self.promoted}",
        ]
        if self.archived:
            lines.append(f"archived previous : {', '.join(self.archived)}")
        lines += ["", self.gate.table(), "=" * 60]
        return "\n".join(lines)


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    return MlflowClient()


def _latest_version(client, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions for {name!r}. Run `make train`.")
    return max(versions, key=lambda v: int(v.version)).version


def promote_model(
    version: str | None = None,
    *,
    archive_existing: bool = True,
    dry_run: bool = False,
) -> PromotionResult:
    """Evaluate a version and promote Staging -> Production iff the gate passes.

    Parameters
    ----------
    version:
        Registry version to consider; defaults to the latest registered version.
    archive_existing:
        Archive the current Production version(s) when promoting.
    dry_run:
        Evaluate + report but do not transition to Production.
    """
    import mlflow

    client = _client()
    name = CONFIG.registered_model_name
    version = version or _latest_version(client, name)
    current_stage = client.get_model_version(name, version).current_stage or "None"

    # Evaluate FIRST — the gate decides. No stage is mutated before we know the
    # result, so a rejected candidate is a true no-op and a re-promote of the live
    # Production model never demotes it through Staging (a self-inflicted outage).
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    eval_result = evaluate_on_test(model)
    gate = eval_result.gate

    archived: list[str] = []
    final_stage = current_stage
    promoted = False

    if gate.passed and not dry_run:
        # Provenance tags first (stage-agnostic), so the stage transition is the
        # last, committing step.
        client.set_model_version_tag(name, version, "promoted", "true")
        client.set_model_version_tag(
            name, version, "test_roc_auc", f"{eval_result.metrics['roc_auc']:.4f}"
        )
        # Staging -> Production audit hop — but skip Staging if the candidate is
        # already Production (idempotent re-promote must not blink the live model).
        if current_stage != "Production":
            client.transition_model_version_stage(name, version, "Staging")
        if archive_existing:
            archived = [
                mv.version
                for mv in client.get_latest_versions(name, stages=["Production"])
                if mv.version != version
            ]
        client.transition_model_version_stage(
            name, version, "Production", archive_existing_versions=archive_existing
        )
        promoted = True
        final_stage = "Production"
    # Gate failed (or dry_run): no stage mutation — the candidate stays put.

    return PromotionResult(
        version=version,
        promoted=promoted,
        stage=final_stage,
        gate=gate,
        metrics=eval_result.metrics,
        archived=archived,
    )


def current_production(client=None) -> tuple[str, dict] | None:
    """Return (version, tags) of the current Production model, or None."""
    client = client or _client()
    prod = client.get_latest_versions(CONFIG.registered_model_name, stages=["Production"])
    if not prod:
        return None
    mv = prod[0]
    return mv.version, dict(mv.tags)
