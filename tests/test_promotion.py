"""Promotion-gate tests: promote only when the quality gate passes (integration)."""

from __future__ import annotations

import pytest

from defaultradar.config import CONFIG

pytestmark = pytest.mark.integration


def test_promotion_passes_when_gate_met(require_mlflow) -> None:
    """Dry-run promotion of the latest version: gate passes, no side effects."""
    from defaultradar.registry import current_production, promote_model

    result = promote_model(dry_run=True)
    assert result.gate.passed, result.gate.table()
    assert result.promoted is False  # dry run does not transition
    # A Production model exists (promoted earlier in the lifecycle).
    assert current_production() is not None


def test_promotion_blocked_when_gate_fails(require_mlflow, monkeypatch) -> None:
    """An unreachable ROC-AUC floor must block promotion (gate is enforced)."""
    from defaultradar.registry import promote_model

    monkeypatch.setattr(CONFIG, "roc_auc_min", 0.99)
    result = promote_model(dry_run=True)
    assert not result.gate.passed
    assert result.promoted is False


def test_real_promotion_sets_production(require_mlflow) -> None:
    """A passing model is actually transitioned to Production."""
    from defaultradar.registry import current_production, promote_model

    result = promote_model()  # latest; passes the 0.67 gate
    assert result.promoted is True
    assert result.stage == "Production"
    prod = current_production()
    assert prod is not None and prod[0] == result.version


def test_failing_promotion_does_not_demote_production(require_mlflow, monkeypatch) -> None:
    """A gate failure on a re-promote must NOT demote the live Production model."""
    from defaultradar.registry import current_production, promote_model

    # Ensure there is a Production model first.
    promoted = promote_model()
    assert promoted.promoted

    # Now a real (non-dry-run) promote that fails the gate must be a no-op.
    monkeypatch.setattr(CONFIG, "roc_auc_min", 0.99)
    result = promote_model(version=promoted.version)
    assert result.promoted is False
    assert result.stage == "Production"  # not demoted to Staging
    prod = current_production()
    assert prod is not None and prod[0] == promoted.version
