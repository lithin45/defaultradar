"""Tests for the data acquisition layer.

The fast tests run against the committed CI sample (no network). A ``slow`` test
exercises the real Zenodo download + Parquet build and is opt-in via
``pytest -m slow`` / a full ``make data``.
"""

from __future__ import annotations

import duckdb
import pytest

from defaultradar.config import CONFIG, TARGET_COLUMN, TIME_COLUMN
from defaultradar.data.download import DATASET

EXPECTED_COLUMNS = {
    "id",
    "issue_d",
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
    "Default",
    "title",
    "desc",
}


def test_dataset_info_constants() -> None:
    assert DATASET.record_id == "11295916"
    assert DATASET.size_bytes == 167_468_415
    assert len(DATASET.md5) == 32
    assert DATASET.url.startswith("https://zenodo.org/")
    assert DATASET.license == "CC-BY-4.0"


def test_data_source_available_and_typed(data_source) -> None:
    """A usable data source resolves (committed sample in CI) with the right schema."""
    assert data_source.exists()
    con = duckdb.connect()
    try:
        if data_source.suffix == ".parquet":
            scan = f"read_parquet('{data_source.as_posix()}')"
        else:
            scan = f"read_csv_auto('{data_source.as_posix()}', header=true)"
        cols = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
        n_rows = con.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()[0]
    finally:
        con.close()

    assert cols >= EXPECTED_COLUMNS, f"missing columns: {EXPECTED_COLUMNS - cols}"
    assert TARGET_COLUMN in cols
    assert TIME_COLUMN in cols
    assert n_rows > 0


def test_target_is_binary(data_source) -> None:
    con = duckdb.connect()
    try:
        scan = (
            f"read_parquet('{data_source.as_posix()}')"
            if data_source.suffix == ".parquet"
            else f"read_csv_auto('{data_source.as_posix()}', header=true)"
        )
        distinct = {
            r[0]
            for r in con.execute(
                f'SELECT DISTINCT CAST("{TARGET_COLUMN}" AS INTEGER) FROM {scan}'
            ).fetchall()
        }
    finally:
        con.close()
    assert distinct <= {0, 1}, distinct


@pytest.mark.slow
def test_full_download_and_parquet() -> None:
    """End-to-end: download the real dataset and build the Parquet feature store."""
    from defaultradar.data.download import ensure_raw_parquet

    parquet = ensure_raw_parquet()
    assert parquet.exists()
    assert parquet == CONFIG.raw_parquet_path

    con = duckdb.connect()
    try:
        n_rows, n_dates = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT issue_date)
            FROM read_parquet('{parquet.as_posix()}')
            """
        ).fetchone()
    finally:
        con.close()
    # ~1.35M records; issue_date parsed from the Mon-YYYY strings.
    assert n_rows > 1_000_000
    assert n_dates > 12
