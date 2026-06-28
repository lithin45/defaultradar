# DefaultRadar demo dashboard

A standalone, free Streamlit app that showcases the project. It runs a bundled,
pre-trained model directly in the browser session. **There is no external API, no
keys and no cost** to host or share it. The full self-updating system (MLflow,
FastAPI, Prefect, Docker) lives in the repository root.

## What it shows
- **Overview** of the lifecycle loop.
- **Score a loan**: enter an application and get a calibrated probability of default plus a SHAP explanation.
- **Model performance**: honest metrics, the calibration curve, and global SHAP importance.
- **Drift monitor demo**: simulate an economic downturn and watch the PSI drift gate fire.

## Run it locally
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/streamlit_app.py
```

## Deploy it free on Streamlit Community Cloud
1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick this repository and branch `main`.
4. Set the **main file path** to `dashboard/streamlit_app.py`.
5. Under advanced settings, set the requirements file to `dashboard/requirements.txt`.
6. Deploy. The first build takes a few minutes; after that it loads quickly.

The app bundles `assets/model.joblib` (the trained model) and a small data sample,
so it needs nothing else to run.
