"""Time-based split tests — no temporal leakage across train/valid/test."""

from __future__ import annotations

import pandas as pd
import pytest

from defaultradar.config import CONFIG
from defaultradar.features import (
    engineer_features,
    load_feature_contract,
    parse_emp_length,
    time_split,
)


def _frame(data_source) -> pd.DataFrame:
    raw = (
        pd.read_parquet(data_source)
        if data_source.suffix == ".parquet"
        else pd.read_csv(data_source)
    )
    return engineer_features(raw, load_feature_contract())


def test_parse_emp_length_mapping() -> None:
    s = pd.Series(["< 1 year", "1 year", "9 years", "10+ years", "NI", None])
    out = parse_emp_length(s).tolist()
    assert out[0] == 0.0
    assert out[1] == 1.0
    assert out[2] == 9.0
    assert out[3] == 10.0
    assert pd.isna(out[4])  # "NI" -> NaN
    assert pd.isna(out[5])  # None -> NaN


def test_split_is_time_ordered(data_source) -> None:
    frame = _frame(data_source)
    splits = time_split(frame)
    train_end = pd.Timestamp(CONFIG.train_end)
    valid_end = pd.Timestamp(CONFIG.valid_end)

    train_d = pd.to_datetime(splits["train"]["issue_date"])
    valid_d = pd.to_datetime(splits["valid"]["issue_date"])
    test_d = pd.to_datetime(splits["test"]["issue_date"])

    # Every split sits strictly inside its time window.
    assert train_d.max() <= train_end
    assert valid_d.min() > train_end
    assert valid_d.max() <= valid_end
    assert test_d.min() > valid_end

    # And the windows are globally ordered: train < valid < test (no overlap).
    assert train_d.max() < valid_d.min()
    assert valid_d.max() < test_d.min()


def test_splits_are_disjoint_and_complete(data_source) -> None:
    # No pre-filtering: assert completeness against the ORIGINAL frame so a silent
    # drop of any row (dated or undated) would fail this test.
    frame = _frame(data_source)
    splits = time_split(frame)

    idx_train = set(splits["train"].index)
    idx_valid = set(splits["valid"].index)
    idx_test = set(splits["test"].index)

    # Pairwise disjoint (no row in two splits) ...
    assert idx_train.isdisjoint(idx_valid)
    assert idx_train.isdisjoint(idx_test)
    assert idx_valid.isdisjoint(idx_test)
    # ... and together they cover EVERY row (nothing silently dropped).
    assert len(idx_train) + len(idx_valid) + len(idx_test) == len(frame)


def test_time_split_fails_loudly_on_undated_row(data_source) -> None:
    """A NULL/unparseable split date must raise, not be silently dropped."""
    frame = _frame(data_source).head(50).copy()
    frame.loc[frame.index[0], "issue_date"] = pd.NaT
    with pytest.raises(ValueError, match="NULL/unparseable"):
        time_split(frame)


def test_all_splits_non_empty(data_source) -> None:
    """The committed sample spans all years, so every split is populated."""
    splits = time_split(_frame(data_source))
    for name in ("train", "valid", "test"):
        assert len(splits[name]) > 0, f"{name} split is empty"
