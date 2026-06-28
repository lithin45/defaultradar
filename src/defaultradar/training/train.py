"""Train -> calibrate -> evaluate -> log to MLflow -> register a model version.

This is the heart of Phase 3. ``make train`` calls :func:`train_and_log`, which:

* builds the leakage-safe feature store (if needed) and loads the time splits,
* fits the XGBoost pipeline on train and calibrates probabilities on validation,
* evaluates honestly on the test split (ROC-AUC, PR-AUC, KS, Brier),
* logs params/metrics/artifacts (calibration curve, SHAP global importance,
  metrics JSON, the feature-config hash) to MLflow, and
* registers a new Model Registry version.

Determinism: everything is seeded; the feature-config hash ties the model to the
exact feature contract it was trained under.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from defaultradar.config import CONFIG
from defaultradar.explain.shap_utils import global_shap_importance
from defaultradar.features.pipeline import FEATURE_COLUMNS, build_feature_store, get_xy, load_split
from defaultradar.features.schema import load_feature_contract
from defaultradar.training.metrics import (
    GateResult,
    classification_metrics,
    metrics_table,
    quality_gate,
)
from defaultradar.training.model import (
    CATEGORICAL_ENCODING,
    ModelParams,
    build_base_pipeline,
    calibrate,
)
from defaultradar.training.plots import reliability_curve_figure


@dataclass
class TrainingResult:
    run_id: str | None
    model_version: str | None
    metrics: dict[str, dict[str, float]]  # split -> metrics
    brier_uncalibrated: float
    gate: GateResult
    config_hash: str


def _load_splits(source: str | Path | None):
    """Load train/valid/test engineered frames.

    * Explicit ``source`` (e.g. the CI sample): build in-memory from it WITHOUT
      writing, so the full-data feature store on disk is never clobbered.
    * No source: use the cached full-data split Parquets, building them on first
      run.
    """
    if source is not None:
        return build_feature_store(source=source, write=False).splits
    try:
        return {s: load_split(s) for s in ("train", "valid", "test")}
    except FileNotFoundError:
        return build_feature_store(source=None, write=True).splits


def _fit_eval_log(
    *,
    X_base,
    y_base,
    X_cal,
    y_cal,
    X_test,
    y_test,
    params: ModelParams,
    contract,
    n_rows: dict[str, int],
    log_to_mlflow: bool,
    register: bool,
    window: str,
) -> TrainingResult:
    """Core: fit base on X_base, calibrate on X_cal, evaluate on test, log+register.

    Shared by :func:`train_and_log` (base=train, calib=valid) and
    :func:`retrain_and_log` (base=80% of train+valid, calib=holdout).
    """
    base = build_base_pipeline(params)
    base.fit(X_base, y_base)
    calibrated = calibrate(base, X_cal, y_cal, method=params.calibration_method)

    p_test_uncal = base.predict_proba(X_test)[:, 1]
    p_test_cal = calibrated.predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_cal)[:, 1]

    metrics = {
        "valid": classification_metrics(y_cal, p_cal),  # calibration/validation set
        "test": classification_metrics(y_test, p_test_cal),
    }
    brier_uncal = classification_metrics(y_test, p_test_uncal)["brier"]
    gate = quality_gate(
        test_metrics=metrics["test"],
        brier_uncalibrated=brier_uncal,
        brier_calibrated=metrics["test"]["brier"],
    )

    result = TrainingResult(
        run_id=None,
        model_version=None,
        metrics=metrics,
        brier_uncalibrated=brier_uncal,
        gate=gate,
        config_hash=contract.config_hash,
    )
    if log_to_mlflow:
        _log_run(
            params=params,
            contract_hash=contract.config_hash,
            calibrated=calibrated,
            base=base,
            X_train=X_base,
            X_test=X_test,
            y_test=y_test,
            p_test_cal=p_test_cal,
            p_test_uncal=p_test_uncal,
            metrics=metrics,
            brier_uncal=brier_uncal,
            gate=gate,
            register=register,
            result=result,
            n_rows=n_rows,
            window=window,
        )
    return result


def train_and_log(
    *,
    params: ModelParams | None = None,
    source: str | Path | None = None,
    log_to_mlflow: bool = True,
    register: bool = True,
) -> TrainingResult:
    """Train base on the train split, calibrate on valid, evaluate + log + register."""
    params = params or ModelParams()
    contract = load_feature_contract()
    splits = _load_splits(source)

    X_train, y_train = get_xy(splits["train"], contract)
    X_valid, y_valid = get_xy(splits["valid"], contract)
    X_test, y_test = get_xy(splits["test"], contract)

    return _fit_eval_log(
        X_base=X_train,
        y_base=y_train,
        X_cal=X_valid,
        y_cal=y_valid,
        X_test=X_test,
        y_test=y_test,
        params=params,
        contract=contract,
        n_rows={k: len(v) for k, v in splits.items()},
        log_to_mlflow=log_to_mlflow,
        register=register,
        window="train",
    )


def retrain_and_log(
    *,
    params: ModelParams | None = None,
    source: str | Path | None = None,
    log_to_mlflow: bool = True,
    register: bool = True,
    calib_frac: float = 0.15,
) -> TrainingResult:
    """Retrain on the EXPANDED recent window (train+valid).

    Simulates "the validation-period labels are now available": the base model is
    refit on train+valid combined (so it sees the more-recent 2016 data), with a
    seeded stratified hold-out carved off for calibration. Produces a genuinely
    different model version, still gated on the untouched test split.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    params = params or ModelParams()
    contract = load_feature_contract()
    splits = _load_splits(source)

    pool = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    X_pool, y_pool = get_xy(pool, contract)
    X_test, y_test = get_xy(splits["test"], contract)

    X_base, X_cal, y_base, y_cal = train_test_split(
        X_pool, y_pool, test_size=calib_frac, random_state=params.seed, stratify=y_pool
    )

    return _fit_eval_log(
        X_base=X_base,
        y_base=y_base,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        params=params,
        contract=contract,
        n_rows={"train": len(X_base), "valid": len(X_cal), "test": len(X_test)},
        log_to_mlflow=log_to_mlflow,
        register=register,
        window="train+valid",
    )


def _log_run(
    *,
    params,
    contract_hash,
    calibrated,
    base,
    X_train,
    X_test,
    y_test,
    p_test_cal,
    p_test_uncal,
    metrics,
    brier_uncal,
    gate,
    register,
    result,
    n_rows,
    window="train",
) -> None:
    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)

    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # --- params ---
        mlflow.log_params(
            {
                **asdict(params),
                "n_features": len(FEATURE_COLUMNS),
                "categorical_encoding": CATEGORICAL_ENCODING,
                "training_window": window,
                "feature_config_hash": contract_hash,
                "n_train": n_rows.get("train"),
                "n_valid": n_rows.get("valid"),
                "n_test": n_rows.get("test"),
            }
        )

        # --- metrics ---
        flat = {f"{split}_{k}": v for split, m in metrics.items() for k, v in m.items()}
        flat["test_brier_uncalibrated"] = brier_uncal
        flat["gate_passed"] = float(gate.passed)
        mlflow.log_metrics(flat)
        mlflow.set_tags({"feature_config_hash": contract_hash, "gate_passed": str(gate.passed)})

        # --- artifacts: calibration curve, SHAP global, metrics json ---
        rel_fig = reliability_curve_figure(
            {
                "uncalibrated": (y_test.to_numpy(), p_test_uncal),
                "calibrated": (y_test.to_numpy(), p_test_cal),
            }
        )
        rel_path = tmp / "calibration_curve.png"
        rel_fig.savefig(rel_path, dpi=120)
        plt.close(rel_fig)

        shap_art = global_shap_importance(base, X_train, seed=params.seed)
        shap_png = tmp / "shap_global_importance.png"
        shap_art.figure.savefig(shap_png, dpi=120)
        plt.close(shap_art.figure)
        shap_csv = tmp / "shap_importance.csv"
        shap_art.importance.to_csv(shap_csv, index=False)

        metrics_json = tmp / "metrics.json"
        metrics_json.write_text(
            json.dumps(
                {"metrics": metrics, "brier_uncalibrated": brier_uncal, "gate_passed": gate.passed},
                indent=2,
                default=float,
            )
        )
        for f in (rel_path, shap_png, shap_csv, metrics_json):
            mlflow.log_artifact(str(f))

        # --- model ---
        signature = infer_signature(X_test.head(5), p_test_cal[:5])
        mlflow.sklearn.log_model(
            sk_model=calibrated,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(2),
        )

        result.run_id = run.info.run_id
        if register:
            mv = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=CONFIG.registered_model_name,
                tags={"feature_config_hash": contract_hash, "gate_passed": str(gate.passed)},
            )
            result.model_version = mv.version


def format_report(result: TrainingResult) -> str:
    """Human-readable training/eval report (used by make train / make eval)."""
    lines = [
        "=" * 64,
        "DefaultRadar — training & evaluation report",
        "=" * 64,
        f"run_id        : {result.run_id}",
        f"model_version : {result.model_version}",
        f"config_hash   : {result.config_hash[:12]}",
        "",
        metrics_table(result.metrics),
        f"\ntest Brier  uncalibrated={result.brier_uncalibrated:.5f}  "
        f"calibrated={result.metrics['test']['brier']:.5f}",
        "",
        result.gate.table(),
        "=" * 64,
    ]
    return "\n".join(lines)
