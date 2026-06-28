"""Leakage-safe feature pipeline: raw -> clean/engineer -> time-split Parquet.

Design
------
The pipeline produces an *interpretable* engineered feature store (semantic
columns, categoricals kept as strings). Categorical one-hot encoding is deferred
to the scikit-learn model pipeline (Phase 3), fit on the **train** split only, so
encodings never leak validation/test information.

Two leakage defenses, enforced by :func:`assert_no_leakage` and the Phase-2
tests:

1. The pipeline only *reads* allowlisted raw columns (:data:`RAW_INPUT_COLUMNS`
   ⊆ contract.allowed).
2. No banned/outcome-derived column ever appears in the engineered feature set
   (:data:`FEATURE_COLUMNS` ∩ contract.banned == ∅).

The split is strictly time-based on the parsed ``issue_date`` (never the raw
``issue_d`` string, never random), so no future information bleeds backwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from defaultradar.config import CONFIG
from defaultradar.features.schema import (
    FeatureContract,
    LeakageError,
    load_feature_contract,
)

# --- Engineered feature columns (the model's X), grouped for the encoder -----
NUMERIC_FEATURES: tuple[str, ...] = (
    "revenue",
    "dti_n",
    "loan_amnt",
    "loan_to_income",  # engineered: loan_amnt / revenue (credit signal)
    "fico_n",
    "experience_c",
    "emp_length_years",  # parsed from emp_length
    "has_desc",  # engineered from desc
    "desc_len",
    "has_title",  # engineered from title
    "title_len",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "purpose",
    "home_ownership_n",
    "addr_state",
    "zip_code",
)
FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# Raw columns the pipeline is allowed to *read* (must be ⊆ contract.allowed).
RAW_INPUT_COLUMNS: tuple[str, ...] = (
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "experience_c",
    "emp_length",
    "purpose",
    "home_ownership_n",
    "addr_state",
    "zip_code",
    "title",
    "desc",
)

_EMP_LENGTH_RE = re.compile(r"(\d+)")
_SPLIT_NAMES = ("train", "valid", "test")


# --- Cleaning helpers --------------------------------------------------------
def parse_emp_length(series: pd.Series) -> pd.Series:
    """Parse Lending Club ``emp_length`` strings to numeric years.

    ``"< 1 year" -> 0``, ``"10+ years" -> 10``, ``"N years" -> N``,
    ``"NI"`` / missing -> ``NaN`` (XGBoost handles NaN natively).
    """

    def _one(v: object) -> float:
        # pd.isna covers None, np.nan, pd.NA and NaT uniformly — important so a
        # pandas <NA> does not stringify to "<NA>" and match the "<" rule below.
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in {"NI", "n/a", "NA", ""}:
            return np.nan
        if s.startswith("<"):
            return 0.0
        if s.startswith("10+"):
            return 10.0
        m = _EMP_LENGTH_RE.search(s)
        return float(m.group(1)) if m else np.nan

    return series.map(_one).astype("float64")


def _ensure_issue_date(df: pd.DataFrame, contract: FeatureContract) -> pd.Series:
    """Return the parsed split DATE, deriving it from raw ``issue_d`` if absent.

    The full Parquet already has ``issue_date``; the committed CSV sample only
    has the raw ``issue_d`` ("Mon-YYYY") string, so we parse it here.
    """
    col = contract.split_date_column
    # errors="coerce" -> bad/unparseable dates become NaT (consistent with the
    # DuckDB layer's null-safe try_strptime); time_split then fails loudly on NaT.
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.to_datetime(df[contract.time_column], format="%b-%Y", errors="coerce")


# --- Feature engineering -----------------------------------------------------
def build_feature_matrix(df: pd.DataFrame, contract: FeatureContract | None = None) -> pd.DataFrame:
    """Build ONLY the engineered feature columns (the model's X) from raw rows.

    Reads exclusively from allowlisted raw inputs and runs the leakage guard. Used
    both by :func:`engineer_features` (which adds metadata) and by the serving
    layer (which has no target/date at inference time).
    """
    contract = contract or load_feature_contract()

    # Read ONLY allowlisted raw columns, via an explicit subframe. Any attempt to
    # use a column not in RAW_INPUT_COLUMNS now raises KeyError by construction,
    # so the "consumed columns" guard can never drift from the real data access.
    raw = df[list(RAW_INPUT_COLUMNS)]

    desc = raw["desc"].fillna("").astype("string")
    title = raw["title"].fillna("").astype("string")

    revenue = pd.to_numeric(raw["revenue"], errors="coerce")
    loan_amnt = pd.to_numeric(raw["loan_amnt"], errors="coerce")
    # loan-to-income ratio; guard against /0 and inf -> NaN (XGBoost handles NaN).
    loan_to_income = (loan_amnt / revenue.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    def _cat(name: str) -> pd.Series:
        # Fill missing categoricals with an explicit sentinel so the target
        # encoder treats "missing" as its own level rather than erroring.
        return raw[name].astype("string").fillna("__NA__")

    out = pd.DataFrame(
        {
            # passthrough numerics
            "revenue": revenue,
            "dti_n": pd.to_numeric(raw["dti_n"], errors="coerce"),
            "loan_amnt": loan_amnt,
            "loan_to_income": loan_to_income,
            "fico_n": pd.to_numeric(raw["fico_n"], errors="coerce"),
            "experience_c": pd.to_numeric(raw["experience_c"], errors="coerce"),
            # parsed ordinal
            "emp_length_years": parse_emp_length(raw["emp_length"]),
            # engineered text flags (free text itself is never used)
            "has_desc": (desc.str.strip().str.len() > 0).astype("int8"),
            "desc_len": desc.str.len().fillna(0).astype("int32"),
            "has_title": (title.str.strip().str.len() > 0).astype("int8"),
            "title_len": title.str.len().fillna(0).astype("int32"),
            # categoricals (target-encoded later, in the model pipeline)
            "purpose": _cat("purpose"),
            "home_ownership_n": _cat("home_ownership_n"),
            "addr_state": _cat("addr_state"),
            "zip_code": _cat("zip_code"),
        },
        index=df.index,
    )

    # Validate what was ACTUALLY produced, not a hardcoded constant: the engineered
    # columns must match the declared FEATURE_COLUMNS exactly (fail loudly on
    # drift), and the real columns must pass the leakage guard.
    produced = tuple(out.columns)
    if produced != FEATURE_COLUMNS:
        raise LeakageError(
            f"engineered feature columns {produced} != declared FEATURE_COLUMNS {FEATURE_COLUMNS}"
        )
    assert_no_leakage(list(out.columns), contract, consumed=list(RAW_INPUT_COLUMNS))
    return out


def engineer_features(df: pd.DataFrame, contract: FeatureContract | None = None) -> pd.DataFrame:
    """Build the engineered feature frame plus split/label metadata.

    Returns a frame with :data:`FEATURE_COLUMNS` plus two metadata columns
    (``Default`` target and ``issue_date`` for the split) — metadata are NOT
    features and are excluded from :data:`FEATURE_COLUMNS`. Requires the target
    and time columns in ``df`` (training/eval use); serving uses
    :func:`build_feature_matrix` instead.
    """
    contract = contract or load_feature_contract()
    out = build_feature_matrix(df, contract)
    # Metadata (kept for splitting / labels; never part of FEATURE_COLUMNS).
    out[contract.target] = pd.to_numeric(df[contract.target]).astype("int8")
    out[contract.split_date_column] = _ensure_issue_date(df, contract)
    return out


def assert_no_leakage(
    feature_columns: list[str] | tuple[str, ...],
    contract: FeatureContract | None = None,
    *,
    consumed: list[str] | tuple[str, ...] | None = None,
) -> None:
    """The leakage guard. Raises :class:`LeakageError` on any violation.

    * No ``feature_columns`` may be in the banned list.
    * If ``consumed`` (the raw input columns) is given, every one must be
      allowlisted.
    """
    contract = contract or load_feature_contract()

    banned_hits = sorted(set(feature_columns) & contract.banned)
    if banned_hits:
        raise LeakageError(f"Banned/outcome-derived columns reached the feature set: {banned_hits}")

    if consumed is not None:
        not_allowed = sorted(set(consumed) - contract.allowed)
        if not_allowed:
            raise LeakageError(f"Pipeline consumes non-allowlisted raw columns: {not_allowed}")


def feature_columns_of(frame: pd.DataFrame, contract: FeatureContract | None = None) -> list[str]:
    """The non-metadata (candidate feature) columns actually present in a frame."""
    contract = contract or load_feature_contract()
    return [c for c in frame.columns if c not in contract.metadata_columns]


def get_xy(frame: pd.DataFrame, contract: FeatureContract | None = None):
    """Split an engineered frame into ``(X[FEATURE_COLUMNS], y[target])``.

    The leakage guard runs on the frame's ACTUAL non-metadata columns (not a
    constant), so a banned column present in the frame fails loudly here.
    """
    contract = contract or load_feature_contract()
    assert_no_leakage(
        feature_columns_of(frame, contract), contract, consumed=list(RAW_INPUT_COLUMNS)
    )
    return frame[list(FEATURE_COLUMNS)].copy(), frame[contract.target].copy()


# --- Time-based split --------------------------------------------------------
def time_split(
    frame: pd.DataFrame, contract: FeatureContract | None = None
) -> dict[str, pd.DataFrame]:
    """Partition by ``issue_date`` into train/valid/test (no random split)."""
    contract = contract or load_feature_contract()
    d = pd.to_datetime(frame[contract.split_date_column], errors="coerce")

    # Fail loudly on undated rows rather than silently dropping them (which would
    # break the disjoint+complete guarantee).
    missing = int(d.isna().sum())
    if missing:
        raise ValueError(
            f"{missing} row(s) have NULL/unparseable {contract.split_date_column}; "
            "cannot assign them to a time split."
        )

    train_end = pd.Timestamp(CONFIG.train_end)
    valid_end = pd.Timestamp(CONFIG.valid_end)

    return {
        "train": frame[d <= train_end],
        "valid": frame[(d > train_end) & (d <= valid_end)],
        "test": frame[d > valid_end],
    }


@dataclass
class FeatureStore:
    """Result of building the feature store: split frames + lineage metadata."""

    splits: dict[str, pd.DataFrame]
    paths: dict[str, Path]
    config_hash: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in _SPLIT_NAMES:
            f = self.splits[name]
            d = pd.to_datetime(f["issue_date"])
            rows.append(
                {
                    "split": name,
                    "rows": len(f),
                    "date_min": d.min(),
                    "date_max": d.max(),
                    "default_rate": float(f["Default"].mean()) if len(f) else float("nan"),
                }
            )
        return pd.DataFrame(rows)


def build_feature_store(
    *,
    source: str | Path | None = None,
    write: bool = True,
    contract: FeatureContract | None = None,
) -> FeatureStore:
    """Build the engineered, leakage-checked, time-split feature store.

    Reads the raw Parquet (or committed sample), engineers features, runs the
    leakage guard, splits by time, and (optionally) writes train/valid/test
    Parquet partitions under ``data/features/``.
    """
    contract = contract or load_feature_contract()
    from defaultradar.data.duckdb_summary import resolve_source

    src = Path(source) if source is not None else resolve_source()
    raw = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    frame = engineer_features(raw, contract)
    # Hard leakage gate on the ACTUAL engineered frame before it is persisted, so
    # a banned column can never reach the on-disk feature store / load_split.
    assert_no_leakage(
        feature_columns_of(frame, contract), contract, consumed=list(RAW_INPUT_COLUMNS)
    )

    splits = time_split(frame, contract)

    paths: dict[str, Path] = {}
    if write:
        CONFIG.features_dir.mkdir(parents=True, exist_ok=True)
        for name, f in splits.items():
            out = CONFIG.features_dir / f"{name}.parquet"
            f.to_parquet(out, index=False)
            paths[name] = out

    return FeatureStore(splits=splits, paths=paths, config_hash=contract.config_hash)


def load_split(name: str) -> pd.DataFrame:
    """Load a previously built split Parquet (``train``/``valid``/``test``)."""
    if name not in _SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {_SPLIT_NAMES}")
    path = CONFIG.features_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not built yet: {path}. Run the feature pipeline.")
    return pd.read_parquet(path)
