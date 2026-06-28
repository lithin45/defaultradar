"""Pydantic request/response schemas for the scorer API.

The request mirrors the **raw application-time fields** a lender knows at decision
time (it deliberately excludes any outcome-derived field — those are not even
accepted). The serving layer engineers features from these via the same
leakage-safe pipeline used in training.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """A single loan application (raw, application-time inputs only)."""

    revenue: float = Field(..., description="Stated annual income", ge=0, examples=[65000.0])
    dti_n: float = Field(..., description="Debt-to-income ratio", examples=[16.06])
    loan_amnt: float = Field(..., description="Requested loan amount", gt=0, examples=[24700.0])
    fico_n: float = Field(..., description="FICO score at application", examples=[717.0])
    experience_c: int = Field(1, description="Credit-experience indicator (0/1)", examples=[1])
    emp_length: str = Field(
        "10+ years", description="Employment length, e.g. '10+ years', '< 1 year', 'NI'"
    )
    purpose: str = Field(..., description="Loan purpose", examples=["small_business"])
    home_ownership_n: str = Field(..., description="Home ownership", examples=["MORTGAGE"])
    addr_state: str = Field(..., description="US state", examples=["SD"])
    zip_code: str = Field(..., description="3-digit ZIP prefix, e.g. '577xx'", examples=["577xx"])
    title: str | None = Field(None, description="Optional loan title")
    desc: str | None = Field(None, description="Optional free-text description")

    model_config = {
        # Reject unknown fields (HTTP 422) so an outcome-derived field such as
        # int_rate / grade / loan_status can never sneak in as a serving input.
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "revenue": 65000.0,
                "dti_n": 16.06,
                "loan_amnt": 24700.0,
                "fico_n": 717.0,
                "experience_c": 1,
                "emp_length": "10+ years",
                "purpose": "small_business",
                "home_ownership_n": "MORTGAGE",
                "addr_state": "SD",
                "zip_code": "577xx",
                "title": "Business",
                "desc": "",
            }
        },
    }


class ShapContribution(BaseModel):
    feature: str
    shap: float


class Explanation(BaseModel):
    base_value: float
    top_contributions: list[ShapContribution]
    n_features: int


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Calibrated P(default)")
    model_version: str
    model_stage: str
    explanation: Explanation | None = None


class ModelInfoResponse(BaseModel):
    registered_model_name: str
    version: str
    stage: str
    n_features: int
    feature_config_hash: str | None = None
    metrics: dict[str, float] = {}
