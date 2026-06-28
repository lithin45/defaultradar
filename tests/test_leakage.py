"""The leakage guard — the project's hard contract test.

These tests MUST fail loudly if any banned/outcome-derived column can reach the
feature set. They cover both the real engineered features and adversarial
injections (e.g. swapping in the full Kaggle dataset that contains int_rate /
grade / payment history).
"""

from __future__ import annotations

import pandas as pd
import pytest

from defaultradar.features import (
    FEATURE_COLUMNS,
    RAW_INPUT_COLUMNS,
    LeakageError,
    assert_no_leakage,
    engineer_features,
    feature_columns_of,
    get_xy,
    load_feature_contract,
)

# Outcome-derived columns that exist in the FULL Lending Club (Kaggle) dataset
# but must never become features.
KAGGLE_LEAKAGE_COLUMNS = [
    "int_rate",
    "grade",
    "sub_grade",
    "loan_status",
    "total_pymnt",
    "recoveries",
    "last_pymnt_d",
]


def _raw(data_source) -> pd.DataFrame:
    return (
        pd.read_parquet(data_source)
        if data_source.suffix == ".parquet"
        else pd.read_csv(data_source)
    )


def test_contract_allowlist_banlist_disjoint() -> None:
    contract = load_feature_contract()
    assert contract.allowed.isdisjoint(contract.banned)
    # The pipeline only reads allowlisted raw columns.
    assert set(RAW_INPUT_COLUMNS) <= contract.allowed


def test_feature_columns_have_no_banned() -> None:
    """The declared feature set never intersects the banned list."""
    contract = load_feature_contract()
    assert set(FEATURE_COLUMNS).isdisjoint(contract.banned)
    # The guard accepts the real feature set.
    assert_no_leakage(list(FEATURE_COLUMNS), contract, consumed=list(RAW_INPUT_COLUMNS))


def test_engineered_frame_excludes_target_and_banned(data_source) -> None:
    """Engineered X columns contain neither the target nor any banned column."""
    contract = load_feature_contract()
    frame = engineer_features(_raw(data_source).head(2000), contract)

    # Every declared feature column is present...
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    # ...and the model's X (FEATURE_COLUMNS only) has no banned column.
    assert set(FEATURE_COLUMNS).isdisjoint(contract.banned)
    # Target/time are present ONLY as metadata, never as features.
    assert contract.target not in FEATURE_COLUMNS
    assert contract.time_column not in FEATURE_COLUMNS
    assert "id" not in FEATURE_COLUMNS


@pytest.mark.parametrize("banned_col", KAGGLE_LEAKAGE_COLUMNS + ["Default", "issue_d", "id"])
def test_guard_fails_loudly_on_injected_banned_column(banned_col: str) -> None:
    """Injecting ANY banned column into the feature set raises LeakageError."""
    poisoned = [*FEATURE_COLUMNS, banned_col]
    with pytest.raises(LeakageError, match="[Bb]anned"):
        assert_no_leakage(poisoned)


def test_guard_fails_on_non_allowlisted_consumed_column() -> None:
    """Reading a raw column that is not on the allowlist raises LeakageError."""
    with pytest.raises(LeakageError, match="non-allowlisted"):
        assert_no_leakage(
            list(FEATURE_COLUMNS),
            consumed=[*RAW_INPUT_COLUMNS, "int_rate"],
        )


def test_pipeline_ignores_extra_leakage_columns_in_raw(data_source) -> None:
    """If the raw data is swapped for one WITH leakage columns, they are dropped.

    Simulates feeding the full Kaggle schema: extra outcome columns are present
    in the input but the pipeline only reads RAW_INPUT_COLUMNS, so none of them
    survive into the engineered feature frame.
    """
    contract = load_feature_contract()
    raw = _raw(data_source).head(1000).copy()
    clean_features = engineer_features(raw, contract)[list(FEATURE_COLUMNS)]

    poisoned = raw.copy()
    # Inject every banned column we can into the raw input.
    poisoned["int_rate"] = 13.5
    poisoned["grade"] = "C"
    poisoned["loan_status"] = "Charged Off"
    poisoned["recoveries"] = 100.0
    poisoned_features = engineer_features(poisoned, contract)[list(FEATURE_COLUMNS)]

    # 1) No banned column survives by name.
    leaked = set(poisoned_features.columns) & contract.banned
    assert not leaked, f"leakage columns survived into features: {leaked}"
    # 2) Behavioural proof of independence: the engineered features are byte-for-
    #    byte identical with and without the injected outcome columns, so no
    #    feature can possibly depend on a banned source.
    pd.testing.assert_frame_equal(clean_features, poisoned_features)


def test_get_xy_guard_inspects_real_frame_columns(data_source) -> None:
    """The guard checks the frame's ACTUAL columns, not a constant.

    If a banned column is somehow present in an engineered frame, ``get_xy``
    must raise — this defends against a future ``engineer_features`` edit that
    leaks a banned column into the frame / on-disk feature store.
    """
    contract = load_feature_contract()
    frame = engineer_features(_raw(data_source).head(500), contract)

    # A clean frame passes and excludes metadata from the feature set.
    feats = feature_columns_of(frame, contract)
    assert set(feats) == set(FEATURE_COLUMNS)
    assert contract.target not in feats and contract.split_date_column not in feats
    get_xy(frame, contract)  # no raise

    # Inject a banned column into the frame -> guard fires loudly.
    poisoned = frame.copy()
    poisoned["int_rate"] = 13.5
    with pytest.raises(LeakageError, match="[Bb]anned"):
        get_xy(poisoned, contract)


def test_engineer_features_rejects_missing_allowlisted_column(data_source) -> None:
    """Dropping a required allowlisted raw column fails fast (KeyError)."""
    raw = _raw(data_source).head(100).drop(columns=["revenue"])
    with pytest.raises(KeyError):
        engineer_features(raw)
