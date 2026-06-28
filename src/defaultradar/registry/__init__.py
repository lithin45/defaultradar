"""Model Registry registration + metric-gated Staging->Production promotion (Phase 4)."""

from defaultradar.registry.promote import (
    PromotionResult,
    current_production,
    promote_model,
)

__all__ = ["PromotionResult", "current_production", "promote_model"]
