"""Reliability (calibration) curve artifact for training runs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def reliability_curve_figure(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int = 10,
    title: str = "Reliability curve (test split)",
) -> plt.Figure:
    """Overlay reliability curves for several models (e.g. calibrated vs not).

    ``curves`` maps a label -> (y_true, y_prob). The legend shows each model's
    Brier score so the calibration improvement is visible at a glance.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")

    for label, (y_true, y_prob) in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(
            mean_pred, frac_pos, marker="o", linewidth=1.5, label=f"{label} (Brier={brier:.4f})"
        )

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
