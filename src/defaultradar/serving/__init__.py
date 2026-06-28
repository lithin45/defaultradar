"""FastAPI scorer that serves the current Production model (Phase 4).

Phase 1 ships a minimal-but-honest app: ``/health`` is always green so the
container becomes healthy in the compose stack, while ``/predict`` and
``/model-info`` return 503 until a model is promoted to Production. Phase 4
replaces the placeholders with real registry-backed scoring + SHAP.
"""

from defaultradar.serving.app import app

__all__ = ["app"]
