"""DefaultRadar interactive demo dashboard.

A standalone, free, read-only showcase of the DefaultRadar project. It runs a
bundled, pre-trained model directly in the browser session. There is no external
API, no keys and no cost: visitors clicking around does nothing but run the model
locally. The full self-updating system (MLflow, FastAPI, Prefect, Docker) lives
in the GitHub repository.

Author: Linga Reddy Gudisha
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
# Make the project package importable without installing it.
sys.path.insert(0, str(HERE.parent / "src"))

from defaultradar.features.pipeline import build_feature_matrix  # noqa: E402
from defaultradar.monitoring.drift import KEY_FEATURES, evaluate_drift  # noqa: E402
from defaultradar.monitoring.simulate import inject_drift  # noqa: E402

REPO_URL = "https://github.com/lithin45/defaultradar"

st.set_page_config(page_title="DefaultRadar", page_icon="📡", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ASSETS / "model.joblib")


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "sample.csv")


model = load_model()

# --- Header -----------------------------------------------------------------
st.title("📡 DefaultRadar")
st.markdown(
    "**A self updating machine learning system for loan default prediction** "
    "&nbsp;·&nbsp; by Linga Reddy Gudisha"
)
st.info(
    "This is a live demo running a bundled, pre trained model in your browser session. "
    "There is no external API, no keys and no cost. The full self updating system "
    f"(MLflow, FastAPI, Prefect, Docker) is on [GitHub]({REPO_URL})."
)

tab_overview, tab_score, tab_perf, tab_drift = st.tabs(
    ["Overview", "Score a loan", "Model performance", "Drift monitor demo"]
)

# --- Overview ---------------------------------------------------------------
with tab_overview:
    st.subheader("What this project is")
    st.markdown(
        """
DefaultRadar predicts whether a loan applicant will repay or default, and then
keeps itself healthy over time. The model is only a small part of it. The system
builds its own features, trains the model, ships it, serves live predictions,
watches the incoming data for changes, and retrains itself automatically when the
world shifts. The whole thing runs on a laptop with one command.

**The full lifecycle loop**
"""
    )
    c = st.columns(7)
    steps = [
        (
            "1 Features",
            "Clean the data with a guard that blocks any field the bank would only know after the loan.",
        ),
        ("2 Train", "XGBoost learns to predict default, then calibrates its probabilities."),
        ("3 Register", "Every model version is saved with its score and history."),
        ("4 Gate", "Code, not a person, promotes a model only if it clears the quality bar."),
        ("5 Serve", "A live service returns predictions and explains each one."),
        ("6 Monitor", "Watches the data for drift and measures how far it shifted."),
        (
            "7 Retrain",
            "On drift it retrains, re checks the gate, and promotes the new model by itself.",
        ),
    ]
    for col, (title, body) in zip(c, steps, strict=True):
        col.markdown(f"**{title}**")
        col.caption(body)
    st.markdown(
        "Step 7 loops back to step 2. That closing loop is the heart of the project: "
        "the system notices when the world changes and fixes itself, with no one touching it."
    )
    st.markdown(
        "**Tech used:** Python, XGBoost, scikit learn, MLflow, FastAPI, Prefect, "
        "Evidently, DuckDB, SHAP, Docker, GitHub Actions."
    )

# --- Score a loan -----------------------------------------------------------
with tab_score:
    st.subheader("Score a loan application")
    st.caption(
        "Enter an application using only what a lender knows at decision time. "
        "The model returns a calibrated probability of default and explains the drivers."
    )

    col1, col2, col3 = st.columns(3)
    revenue = col1.number_input("Annual income ($)", 5_000, 1_000_000, 65_000, step=1_000)
    loan_amnt = col1.number_input("Loan amount ($)", 500, 50_000, 15_000, step=500)
    fico_n = col2.number_input("FICO credit score", 300, 850, 690, step=5)
    dti_n = col2.number_input("Debt to income ratio", 0.0, 60.0, 17.0, step=0.5)
    emp_length = col3.selectbox(
        "Employment length",
        [
            "10+ years",
            "< 1 year",
            "1 year",
            "2 years",
            "3 years",
            "4 years",
            "5 years",
            "6 years",
            "7 years",
            "8 years",
            "9 years",
            "NI",
        ],
    )
    purpose = col3.selectbox(
        "Loan purpose",
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "small_business",
            "major_purchase",
            "medical",
            "car",
            "moving",
            "house",
            "vacation",
            "other",
        ],
    )
    col4, col5, col6 = st.columns(3)
    home = col4.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    state = col5.selectbox(
        "State", ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "SD", "WA", "NJ"]
    )
    zip_code = col6.text_input("ZIP prefix", "900xx")

    if st.button("Score this application", type="primary"):
        raw = pd.DataFrame(
            [
                {
                    "revenue": revenue,
                    "dti_n": dti_n,
                    "loan_amnt": loan_amnt,
                    "fico_n": fico_n,
                    "experience_c": 1,
                    "emp_length": emp_length,
                    "purpose": purpose,
                    "home_ownership_n": home,
                    "addr_state": state,
                    "zip_code": zip_code,
                    "title": "",
                    "desc": "",
                }
            ]
        )
        X = build_feature_matrix(raw)
        prob = float(model.predict_proba(X)[:, 1][0])

        left, right = st.columns([1, 2])
        left.metric("Probability of default", f"{prob:.1%}")
        band = "Lower risk" if prob < 0.15 else "Moderate risk" if prob < 0.30 else "Higher risk"
        left.markdown(f"**Risk band:** {band}")

        try:
            base = model.calibrated_classifiers_[0].estimator.estimator
            from defaultradar.explain.shap_utils import per_prediction_shap

            exp = per_prediction_shap(base, X, top_n=6)
            contribs = pd.DataFrame(exp["top_contributions"])
            contribs["pushes"] = contribs["shap"].apply(
                lambda s: "toward default" if s > 0 else "toward repaying"
            )
            right.markdown("**Why the model scored it this way (top drivers)**")
            right.dataframe(
                contribs[["feature", "shap", "pushes"]].round(3),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:  # explanation is a bonus; never break scoring
            right.caption(f"(Explanation unavailable in this environment: {exc})")

# --- Model performance ------------------------------------------------------
with tab_perf:
    st.subheader("Honest performance")
    st.markdown(
        "On strictly application time data, split by date so the model never sees the "
        "future, it scores about **0.68 ROC AUC**. That is a realistic number for this "
        "problem. I chose not to inflate it with leaky data, and documented why."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC (test)", "0.688")
    m2.metric("PR AUC", "0.358")
    m3.metric("KS statistic", "0.271")
    m4.metric("Brier (calibrated)", "0.156")
    st.divider()
    g1, g2 = st.columns(2)
    g1.image(
        str(ASSETS / "calibration_curve.png"),
        caption="Calibration: the orange line hugs the perfect line, so the "
        "probabilities can be trusted.",
    )
    g2.image(
        str(ASSETS / "shap_global_importance.png"),
        caption="What the model pays attention to: credit score and the engineered "
        "loan to income ratio lead.",
    )

# --- Drift monitor demo -----------------------------------------------------
with tab_drift:
    st.subheader("Drift monitor demo")
    st.markdown(
        "In production, the system watches incoming data and retrains itself when the "
        "data drifts too far. Here you can simulate an economic downturn and watch the "
        "drift gate fire. The measure is **PSI** (population stability index); anything "
        "above **0.2** on a key feature trips the retraining gate."
    )
    severity = st.slider(
        "Downturn severity",
        0.0,
        1.0,
        0.8,
        0.1,
        help="0 means no change. Higher means a bigger shift in credit "
        "scores, debt ratios and incomes.",
    )

    raw = load_sample()
    reference = build_feature_matrix(raw)
    current = inject_drift(
        reference,
        fico_shift=-75 * severity,
        dti_scale=1 + 0.75 * severity,
        income_scale=1 - 0.5 * severity,
    )
    result = evaluate_drift(reference, current, threshold=0.2)

    psi_df = pd.Series(result.feature_psi).sort_values(ascending=False).rename("PSI").to_frame()
    st.bar_chart(psi_df, horizontal=True, height=420)

    if result.drift_detected:
        st.error(
            f"DRIFT DETECTED on key features {result.drifted_features}. "
            "In the live system this automatically triggers a retrain, a quality gate "
            "check, and promotion of the new model."
        )
    else:
        st.success("No significant drift on key features. The model keeps serving as is.")
    st.caption(f"Key features watched by the gate: {', '.join(KEY_FEATURES)}")

st.divider()
st.caption(f"DefaultRadar · by Linga Reddy Gudisha · full project at {REPO_URL}")
