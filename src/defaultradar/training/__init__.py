"""XGBoost training, probability calibration, MLflow tracking (Phase 3)."""

from defaultradar.training.evaluate import (
    EvalResult,
    evaluate_on_test,
    load_registered_model,
    run_eval,
)
from defaultradar.training.metrics import (
    GateResult,
    classification_metrics,
    ks_statistic,
    metrics_table,
    quality_gate,
)
from defaultradar.training.model import (
    ModelParams,
    build_base_pipeline,
    build_preprocessor,
    calibrate,
)
from defaultradar.training.train import (
    TrainingResult,
    format_report,
    retrain_and_log,
    train_and_log,
)

__all__ = [
    "EvalResult",
    "GateResult",
    "ModelParams",
    "TrainingResult",
    "build_base_pipeline",
    "build_preprocessor",
    "calibrate",
    "classification_metrics",
    "evaluate_on_test",
    "format_report",
    "ks_statistic",
    "load_registered_model",
    "metrics_table",
    "quality_gate",
    "retrain_and_log",
    "run_eval",
    "train_and_log",
]
