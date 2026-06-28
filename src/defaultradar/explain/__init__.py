"""SHAP global importance + per-prediction explanations (Phase 3/4)."""

from defaultradar.explain.shap_utils import (
    ShapArtifacts,
    global_shap_importance,
    per_prediction_shap,
)

__all__ = ["ShapArtifacts", "global_shap_importance", "per_prediction_shap"]
