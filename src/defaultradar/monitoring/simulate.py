"""Controlled drift injection for the offline monitoring demo.

Because this project runs offline, ``make monitor`` injects a *controlled*
distribution shift into the "incoming" batch so the drift -> retrain -> promote
loop can be demonstrated deterministically. The shift simulates an economic
downturn (lower FICO, higher debt-to-income, lower incomes), which moves the key
features enough to push PSI past the gate threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_drift(
    df: pd.DataFrame,
    *,
    fico_shift: float = -60.0,
    dti_scale: float = 1.6,
    income_scale: float = 0.6,
) -> pd.DataFrame:
    """Return a copy of the engineered feature frame with a controlled shift.

    Operates on the engineered features and keeps ``loan_to_income`` consistent
    with the shifted income. Deterministic (no randomness) so the demo is
    reproducible.
    """
    out = df.copy()
    if "fico_n" in out:
        out["fico_n"] = out["fico_n"] + fico_shift
    if "dti_n" in out:
        out["dti_n"] = out["dti_n"] * dti_scale
    if "revenue" in out:
        out["revenue"] = out["revenue"] * income_scale
    # Recompute the ratio so it stays internally consistent with shifted income.
    if {"loan_amnt", "revenue"} <= set(out.columns):
        rev = out["revenue"].replace(0, np.nan)
        out["loan_to_income"] = (out["loan_amnt"] / rev).replace([np.inf, -np.inf], np.nan)
    return out
