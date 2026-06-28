"""SHAP explainability for the XGBoost model.

Phase 3 uses :func:`global_shap_importance` to log a global feature-importance
artifact at training time. Phase 4 reuses the same TreeExplainer machinery for
per-prediction explanations exposed by the serving API.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


@dataclass
class ShapArtifacts:
    importance: pd.DataFrame  # feature, mean_abs_shap (sorted desc)
    figure: plt.Figure


def _explainer_and_values(base_pipeline: Pipeline, X: pd.DataFrame):
    """Return (feature_names, shap_values 2D, transformed X) for the base model.

    SHAP is computed on the *uncalibrated* XGBoost over the preprocessed feature
    space (the calibration wrapper only rescales the output probability and does
    not change attributions).
    """
    pre = base_pipeline.named_steps["preprocess"]
    clf = base_pipeline.named_steps["clf"]
    x_trans = pre.transform(X)
    feature_names = list(pre.get_feature_names_out())

    explainer = shap.TreeExplainer(clf)
    values = np.asarray(explainer.shap_values(x_trans))
    # Binary XGBClassifier returns 2D (n_samples, n_features). Defensive handling
    # if a SHAP version returns 3D: modern SHAP uses (n_samples, n_features,
    # n_classes) -> take the positive class on the LAST axis.
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape[1] != len(feature_names):  # pragma: no cover - guard
        raise ValueError(
            f"SHAP values width {values.shape[1]} != {len(feature_names)} feature names"
        )
    return feature_names, values, x_trans, explainer


def global_shap_importance(
    base_pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    max_samples: int = 2000,
    top_n: int = 20,
    seed: int = 42,
) -> ShapArtifacts:
    """Compute global SHAP importance (mean |SHAP|) + a horizontal bar figure."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)

    feature_names, values, _, _ = _explainer_and_values(base_pipeline, X)
    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b6fb0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Global feature importance (top {len(top)})")
    fig.tight_layout()
    return ShapArtifacts(importance=importance, figure=fig)


def per_prediction_shap(
    base_pipeline: Pipeline,
    X_row: pd.DataFrame,
    *,
    top_n: int = 8,
) -> dict:
    """Per-prediction SHAP explanation for a single engineered feature row.

    Returns the model's base value, the per-feature SHAP contributions, and the
    top contributors by absolute impact — suitable for a serving API response.
    """
    feature_names, values, _, explainer = _explainer_and_values(base_pipeline, X_row)
    contribs = values[0]  # single row

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[-1])
    else:
        base_value = float(base_value)

    pairs = sorted(
        ({"feature": f, "shap": float(s)} for f, s in zip(feature_names, contribs, strict=True)),
        key=lambda d: abs(d["shap"]),
        reverse=True,
    )
    return {
        "base_value": base_value,
        "top_contributions": pairs[:top_n],
        "n_features": len(feature_names),
    }
