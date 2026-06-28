"""Model pipeline: scikit-learn preprocessing + XGBoost + probability calibration.

The estimator is a two-stage object:

1. A ``Pipeline`` of a ``ColumnTransformer`` (passthrough numerics with native
   NaN handling + bounded one-hot for categoricals) and an ``XGBClassifier``.
2. A calibration wrapper (isotonic by default) fit on the **validation** split
   via a frozen base estimator, so calibration never sees the training labels
   it was fit on and never touches test.

Everything is seeded for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from defaultradar.config import CONFIG
from defaultradar.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Categoricals (incl. high-cardinality zip_code ~900, addr_state 51) are encoded
# with scikit-learn's cross-fitted TargetEncoder: it captures each level's
# default-rate signal in a single column without a one-hot dimensionality blow-up,
# and the internal cross-fitting prevents target leakage during fit.
CATEGORICAL_ENCODING = "target_encoder"


@dataclass
class ModelParams:
    """XGBoost hyperparameters (pinned for reproducibility, not over-tuned)."""

    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid"
    seed: int = field(default_factory=lambda: CONFIG.seed)

    def xgb_kwargs(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "tree_method": "hist",
            "eval_metric": "auc",
            "missing": np.nan,  # XGBoost learns split direction for NaN
            "random_state": self.seed,
            "n_jobs": -1,
        }


def build_preprocessor(seed: int | None = None) -> ColumnTransformer:
    """ColumnTransformer: passthrough numerics + cross-fitted target-encoded cats."""
    seed = CONFIG.seed if seed is None else seed
    target_enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=seed)
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(NUMERIC_FEATURES)),
            ("cat", target_enc, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_base_pipeline(params: ModelParams | None = None) -> Pipeline:
    """The uncalibrated preprocessing + XGBoost pipeline."""
    params = params or ModelParams()
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(params.seed)),
            ("clf", XGBClassifier(**params.xgb_kwargs())),
        ]
    )


def _freeze(estimator):
    """Freeze a fitted estimator for calibration (sklearn>=1.6 FrozenEstimator)."""
    try:
        from sklearn.frozen import FrozenEstimator

        return FrozenEstimator(estimator)
    except ImportError:  # pragma: no cover - older sklearn fallback
        return estimator


def calibrate(base_pipeline: Pipeline, X_valid, y_valid, method: str = "isotonic"):
    """Fit a probability calibrator on validation data over a frozen base model.

    The base pipeline must already be fit on train. Returns a fitted
    ``CalibratedClassifierCV`` whose ``predict_proba`` yields calibrated
    probabilities.
    """
    calibrator = CalibratedClassifierCV(_freeze(base_pipeline), method=method)
    calibrator.fit(X_valid, y_valid)
    return calibrator
