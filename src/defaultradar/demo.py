"""End-to-end lifecycle demo: train -> promote -> drift -> retrain -> promote.

``make demo`` runs this. It proves the closed loop by showing the served
Production version change after a drift event triggers an automatic, gate-checked
retrain. Idempotent: it bootstraps an initial Production model if none exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemoResult:
    version_before: str | None
    version_after: str | None
    drift_detected: bool
    retrained_version: str | None
    promoted: bool

    @property
    def served_version_changed(self) -> bool:
        return (
            self.version_before is not None
            and self.version_after is not None
            and self.version_before != self.version_after
        )


def _hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def run_demo(*, inject: bool = True, sample_size: int = 20_000) -> DemoResult:
    """Drive the full lifecycle loop and return what changed."""
    from defaultradar.features.pipeline import load_split
    from defaultradar.registry import current_production, promote_model
    from defaultradar.training import retrain_and_log, train_and_log

    # 0) Ensure the feature store + an initial Production model exist.
    _hr("STEP 0 — ensure a Production model (bootstrap if needed)")
    try:
        load_split("train")
    except FileNotFoundError:
        from defaultradar.features.pipeline import build_feature_store

        build_feature_store(write=True)
        print("[demo] built feature store")

    if current_production() is None:
        print("[demo] no Production model — training + promoting v1 ...")
        t = train_and_log()
        promote_model(t.model_version)
    before = current_production()
    version_before = before[0] if before else None
    print(f"[demo] served Production version BEFORE: {version_before}")

    # 1) Monitor an incoming batch with injected drift.
    _hr("STEP 1 — monitor incoming batch (drift injected)")
    from defaultradar.monitoring import run_monitoring

    drift = run_monitoring(inject=inject, write_reports=True, sample_size=sample_size)
    print(drift.summary())

    retrained_version = None
    promoted = False
    if drift.drift_detected:
        # 2) Drift -> retrain on the expanded window -> register a new version.
        _hr("STEP 2 — drift detected: retrain on expanded window (train+valid)")
        rt = retrain_and_log()
        retrained_version = rt.model_version
        print(
            f"[demo] retrained -> version {retrained_version}  "
            f"(test ROC-AUC {rt.metrics['test']['roc_auc']:.4f}, gate={rt.gate.passed})"
        )

        # 3) Gate + promote the new version.
        _hr("STEP 3 — promotion gate on the retrained version")
        promo = promote_model(retrained_version)
        print(promo.report())
        promoted = promo.promoted
    else:
        _hr("STEP 2 — no drift detected: no retraining")

    after = current_production()
    version_after = after[0] if after else None

    _hr("RESULT — served version transition")
    print(f"served Production version: {version_before}  ->  {version_after}")
    print(f"served version changed   : {version_before != version_after}")

    return DemoResult(
        version_before=version_before,
        version_after=version_after,
        drift_detected=drift.drift_detected,
        retrained_version=retrained_version,
        promoted=promoted,
    )
