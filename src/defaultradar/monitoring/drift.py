"""Drift detection: PSI (the gate metric) + Evidently reports.

The retraining gate is driven by the **Population Stability Index (PSI)**: if PSI
exceeds the configurable threshold (default 0.2) on any *key* feature, or the
prediction distribution drifts, retraining is flagged. Evidently produces the
rich HTML/JSON data-drift + data-summary reports for human inspection; PSI is
computed here directly so the gate is deterministic and transparent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from defaultradar.config import CONFIG
from defaultradar.features.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

# Features whose drift most threatens the model; PSI>threshold on any one fires
# the gate. (The strongest predictors per SHAP/importance.)
KEY_FEATURES: tuple[str, ...] = ("fico_n", "dti_n", "loan_to_income", "revenue", "loan_amnt")

_EPS = 1e-6


def _psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) < bins or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # near-constant reference -> undefined / no meaningful drift
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_p = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_p = np.clip(ref_p, _EPS, None)
    cur_p = np.clip(cur_p, _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    # Treat NaN as an explicit "__NA__" category so a shift in the *missing rate*
    # (a classic upstream-field-dropout signal) contributes to PSI rather than
    # being silently dropped by value_counts(dropna=True).
    ref = ref.astype("string").fillna("__NA__")
    cur = cur.astype("string").fillna("__NA__")
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    ref_p = ref.value_counts(normalize=True)
    cur_p = cur.value_counts(normalize=True)
    categories = set(ref_p.index) | set(cur_p.index)
    psi = 0.0
    for c in categories:
        r = max(float(ref_p.get(c, 0.0)), _EPS)
        k = max(float(cur_p.get(c, 0.0)), _EPS)
        psi += (k - r) * np.log(k / r)
    return float(psi)


def population_stability_index(ref: pd.Series, cur: pd.Series, *, categorical: bool) -> float:
    """PSI between a reference and current series (numeric or categorical)."""
    return _psi_categorical(ref, cur) if categorical else _psi_numeric(ref, cur)


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """PSI for every engineered feature column."""
    out: dict[str, float] = {}
    cat = set(CATEGORICAL_FEATURES)
    for col in FEATURE_COLUMNS:
        out[col] = population_stability_index(reference[col], current[col], categorical=col in cat)
    return out


def prediction_psi(ref_scores: np.ndarray, cur_scores: np.ndarray) -> float:
    """PSI between two predicted-probability distributions."""
    return _psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores))


@dataclass
class DriftResult:
    feature_psi: dict[str, float]
    prediction_psi: float | None
    threshold: float
    drifted_features: list[str]
    key_drift: bool
    prediction_drift: bool
    drift_detected: bool
    n_reference: int
    n_current: int
    reports: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DefaultRadar — drift report",
            "=" * 60,
            f"reference rows : {self.n_reference:,}   current rows : {self.n_current:,}",
            f"PSI threshold  : {self.threshold}",
            "",
            "feature PSI (sorted):",
        ]
        for f, v in sorted(self.feature_psi.items(), key=lambda kv: kv[1], reverse=True):
            flag = "  <== DRIFT" if v > self.threshold else ""
            lines.append(f"  {f:18s} {v:7.4f}{flag}")
        if self.prediction_psi is not None:
            lines.append(f"\nprediction PSI : {self.prediction_psi:.4f}")
        lines += [
            "",
            f"drifted key features : {self.drifted_features or 'none'}",
            f"DRIFT DETECTED       : {self.drift_detected}  -> "
            f"{'RETRAIN' if self.drift_detected else 'no action'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def evaluate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    ref_scores: np.ndarray | None = None,
    cur_scores: np.ndarray | None = None,
    threshold: float | None = None,
    key_features: tuple[str, ...] = KEY_FEATURES,
) -> DriftResult:
    """Compute PSI-based drift and the retrain decision.

    Drift fires when PSI exceeds ``threshold`` on any *key* feature OR the
    prediction distribution drifts beyond ``threshold``.
    """
    threshold = threshold if threshold is not None else CONFIG.psi_threshold
    fpsi = feature_psi(reference, current)
    # Fail loudly if a key feature is not actually computed (e.g. a typo), rather
    # than silently disabling its gate via a 0.0 default.
    missing = [f for f in key_features if f not in fpsi]
    if missing:
        raise KeyError(f"KEY_FEATURES not present in computed feature PSI: {missing}")
    drifted = [f for f in key_features if fpsi[f] > threshold]

    pred_psi = None
    prediction_drift = False
    if ref_scores is not None and cur_scores is not None:
        pred_psi = prediction_psi(ref_scores, cur_scores)
        prediction_drift = pred_psi > threshold

    key_drift = len(drifted) > 0
    return DriftResult(
        feature_psi=fpsi,
        prediction_psi=pred_psi,
        threshold=threshold,
        drifted_features=drifted,
        key_drift=key_drift,
        prediction_drift=prediction_drift,
        drift_detected=key_drift or prediction_drift,
        n_reference=len(reference),
        n_current=len(current),
    )


def evidently_reports(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    prefix: str = "drift",
) -> dict[str, str]:
    """Build Evidently data-drift + data-summary reports (HTML + JSON)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    out_dir = out_dir or CONFIG.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = list(FEATURE_COLUMNS)
    data_def = DataDefinition(
        numerical_columns=list(NUMERIC_FEATURES),
        categorical_columns=list(CATEGORICAL_FEATURES),
    )
    ref_ds = Dataset.from_pandas(reference[cols].copy(), data_definition=data_def)
    cur_ds = Dataset.from_pandas(current[cols].copy(), data_definition=data_def)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(cur_ds, ref_ds)  # (current, reference)

    html_path = out_dir / f"{prefix}_report.html"
    json_path = out_dir / f"{prefix}_report.json"
    snapshot.save_html(str(html_path))
    try:
        snapshot.save_json(str(json_path))
    except Exception:  # pragma: no cover - older/newer API variance
        json_path = None  # type: ignore[assignment]

    paths = {"html": str(html_path)}
    if json_path is not None:
        paths["json"] = str(json_path)
    return paths
