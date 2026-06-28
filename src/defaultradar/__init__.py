"""DefaultRadar — a local, production-grade MLOps platform for loan-default scoring.

The package is organised by lifecycle stage:

* :mod:`defaultradar.data`       — download + DuckDB analytics over the raw data
* :mod:`defaultradar.features`   — leakage-safe feature pipeline + time-based split
* :mod:`defaultradar.training`   — XGBoost training, calibration, MLflow logging
* :mod:`defaultradar.registry`   — Model Registry registration + promotion gate
* :mod:`defaultradar.serving`    — FastAPI scorer loading the Production model
* :mod:`defaultradar.monitoring` — Evidently drift + PSI, Prefect flows
* :mod:`defaultradar.explain`    — SHAP global + per-prediction explanations
"""

from defaultradar.config import CONFIG, Settings

__all__ = ["CONFIG", "Settings", "__version__"]

__version__ = "0.1.0"
