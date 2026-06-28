"""Tests for the DuckDB analytics layer (base-rate + cohort summaries)."""

from __future__ import annotations

import pandas as pd
import pytest

from defaultradar.data.duckdb_summary import (
    base_rate_summary,
    cohort_summary,
    print_smoke_summary,
    resolve_source,
    yearly_cohort_summary,
)


def test_resolve_source_returns_existing(data_source) -> None:
    assert resolve_source().exists()
    assert resolve_source(data_source) == data_source


def test_base_rate_summary(data_source) -> None:
    summary = base_rate_summary(data_source)
    assert summary["n_rows"] > 0
    assert summary["n_default"] + summary["n_repaid"] == summary["n_rows"]
    # A real probability strictly inside (0, 1).
    assert 0.0 < summary["base_rate"] < 1.0


def test_cohort_summary_runs(data_source) -> None:
    df = cohort_summary(data_source, by="purpose")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"cohort", "n", "default_rate"} <= set(df.columns)
    # default_rate is a valid probability per cohort.
    assert (df["default_rate"].between(0.0, 1.0)).all()
    # Sorted descending by default_rate.
    assert df["default_rate"].is_monotonic_decreasing


def test_yearly_cohort_summary(data_source) -> None:
    df = yearly_cohort_summary(data_source)
    assert not df.empty
    assert {"issue_year", "n", "default_rate"} <= set(df.columns)
    assert df["issue_year"].is_monotonic_increasing


def test_print_smoke_summary(data_source, capsys: pytest.CaptureFixture[str]) -> None:
    base = print_smoke_summary(data_source)
    out = capsys.readouterr().out
    assert "DuckDB data summary" in out
    assert "Base rate" in out
    assert base["n_rows"] > 0
