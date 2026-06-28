"""Model service: loads the served model from the registry and scores requests.

Serves the current **Production** model (falling back to the latest version when
nothing is promoted yet, so a fresh ``make train`` is demoable). The model is
cached; ``reload()`` refreshes it after a promotion so the scorer picks up a
newly-promoted version without a restart.
"""

from __future__ import annotations

import threading

import pandas as pd

from defaultradar.config import CONFIG
from defaultradar.explain.shap_utils import per_prediction_shap
from defaultradar.features.pipeline import FEATURE_COLUMNS, build_feature_matrix
from defaultradar.serving.schemas import (
    Explanation,
    LoanApplication,
    ModelInfoResponse,
    PredictionResponse,
)

_METRIC_KEYS = ("test_roc_auc", "test_pr_auc", "test_ks", "test_brier")


def _extract_base_pipeline(calibrated):
    """Reach the uncalibrated base Pipeline inside a CalibratedClassifierCV (for SHAP)."""
    try:
        est = calibrated.calibrated_classifiers_[0].estimator
        return getattr(est, "estimator", est)  # unwrap FrozenEstimator
    except (AttributeError, IndexError):
        return None


class ModelService:
    """Thread-safe lazy loader + scorer for the served model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._model = None
        self._base = None
        self._version: str | None = None
        self._stage: str | None = None
        self._run_id: str | None = None
        self._config_hash: str | None = None
        self._metrics: dict[str, float] = {}

    # --- loading ------------------------------------------------------------
    def _load(self) -> bool:
        from defaultradar.training.evaluate import load_registered_model

        try:
            model, version, stage = load_registered_model(prefer_production=True)
        except Exception:
            return False

        self._model = model
        self._base = _extract_base_pipeline(model)
        self._version = version
        self._stage = stage or "None"
        self._hydrate_metadata(version)
        return True

    def _hydrate_metadata(self, version: str) -> None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
            client = MlflowClient()
            mv = client.get_model_version(CONFIG.registered_model_name, version)
            self._run_id = mv.run_id
            self._config_hash = dict(mv.tags).get("feature_config_hash")
            if mv.run_id:
                run = client.get_run(mv.run_id)
                self._metrics = {
                    k: float(run.data.metrics[k]) for k in _METRIC_KEYS if k in run.data.metrics
                }
        except Exception:
            pass

    def ensure_loaded(self) -> bool:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()
        return self._model is not None

    def reload(self) -> bool:
        """Drop the cache and reload (call after a promotion)."""
        with self._lock:
            self._reset()
        return self.ensure_loaded()

    # --- scoring ------------------------------------------------------------
    def predict(self, application: LoanApplication, *, explain: bool = False) -> PredictionResponse:
        if not self.ensure_loaded():
            raise RuntimeError("No served model available")

        raw = pd.DataFrame([application.model_dump()])
        X = build_feature_matrix(raw)
        prob = float(self._model.predict_proba(X)[:, 1][0])

        explanation = None
        if explain and self._base is not None:
            raw_shap = per_prediction_shap(self._base, X)
            explanation = Explanation(**raw_shap)

        return PredictionResponse(
            default_probability=prob,
            model_version=self._version or "unknown",
            model_stage=self._stage or "unknown",
            explanation=explanation,
        )

    def model_info(self) -> ModelInfoResponse:
        if not self.ensure_loaded():
            raise RuntimeError("No served model available")
        return ModelInfoResponse(
            registered_model_name=CONFIG.registered_model_name,
            version=self._version or "unknown",
            stage=self._stage or "unknown",
            n_features=len(FEATURE_COLUMNS),
            feature_config_hash=self._config_hash,
            metrics=self._metrics,
        )


# Process-wide singleton used by the FastAPI app.
SERVICE = ModelService()
