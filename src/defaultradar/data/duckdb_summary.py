"""Analytical SQL over the local feature store using DuckDB.

DuckDB gives us fast, zero-server SQL directly over the Parquet/CSV feature
store. We use it for cohort analysis, base-rate / label analysis and feature
aggregations — making the SQL usage explicit and reviewable rather than hiding
it behind pandas. Phase 1 ships the base-rate + cohort smoke summary (the
``make data`` smoke test); later phases reuse this layer for label backfill and
SQL feature aggregations.

All functions accept either the full raw Parquet or the small committed CSV
sample, so the exact same queries run in CI and in a full local environment.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from defaultradar.config import CONFIG, TARGET_COLUMN, TIME_COLUMN


def resolve_source(source: str | Path | None = None) -> Path:
    """Pick a data source: explicit > full raw Parquet > committed CI sample.

    Raises ``FileNotFoundError`` if nothing is available so callers fail loudly
    rather than silently summarising an empty set.
    """
    if source is not None:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Requested source does not exist: {p}")
        return p
    if CONFIG.raw_parquet_path.exists():
        return CONFIG.raw_parquet_path
    if CONFIG.sample_csv_path.exists():
        return CONFIG.sample_csv_path
    raise FileNotFoundError(
        "No data source found. Run `make data` to download the dataset, "
        f"or provide the committed sample at {CONFIG.sample_csv_path}."
    )


def _scan_expr(source: Path) -> str:
    """Return a DuckDB table-function call for either Parquet or CSV input."""
    posix = source.as_posix()
    if source.suffix.lower() == ".parquet":
        return f"read_parquet('{posix}')"
    return f"read_csv_auto('{posix}', header=true, sample_size=-1)"


# An issue_date expression that works whether or not a typed issue_date column
# already exists (the CSV sample only has the raw `issue_d` string).
_ISSUE_YEAR_FROM_RAW = f"year(try_strptime({TIME_COLUMN}, '%b-%Y'))"


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def base_rate_summary(source: str | Path | None = None) -> dict[str, float | int]:
    """Compute overall row count, default count and the base default rate."""
    src = resolve_source(source)
    con = _connect()
    try:
        row = con.execute(
            f"""
            SELECT
                COUNT(*)                            AS n_rows,
                SUM(CAST("{TARGET_COLUMN}" AS BIGINT)) AS n_default,
                AVG(CAST("{TARGET_COLUMN}" AS DOUBLE)) AS base_rate
            FROM {_scan_expr(src)}
            """
        ).fetchone()
    finally:
        con.close()
    n_rows, n_default, base_rate = row
    return {
        "source": str(src),
        "n_rows": int(n_rows),
        "n_default": int(n_default),
        "n_repaid": int(n_rows) - int(n_default),
        "base_rate": float(base_rate),
    }


def cohort_summary(
    source: str | Path | None = None,
    *,
    by: str = "purpose",
    min_count: int = 1,
) -> pd.DataFrame:
    """Default rate and volume by cohort (e.g. by ``purpose`` or ``home_ownership_n``).

    Returns a pandas DataFrame sorted by descending default rate.
    """
    src = resolve_source(source)
    con = _connect()
    try:
        df = con.execute(
            f"""
            SELECT
                "{by}"                                AS cohort,
                COUNT(*)                              AS n,
                AVG(CAST("{TARGET_COLUMN}" AS DOUBLE)) AS default_rate
            FROM {_scan_expr(src)}
            GROUP BY "{by}"
            HAVING COUNT(*) >= {int(min_count)}
            ORDER BY default_rate DESC
            """
        ).df()
    finally:
        con.close()
    return df


def yearly_cohort_summary(source: str | Path | None = None) -> pd.DataFrame:
    """Default rate by issue *year* — the key view for the time-based split."""
    src = resolve_source(source)
    con = _connect()
    try:
        df = con.execute(
            f"""
            SELECT
                {_ISSUE_YEAR_FROM_RAW}                AS issue_year,
                COUNT(*)                              AS n,
                AVG(CAST("{TARGET_COLUMN}" AS DOUBLE)) AS default_rate
            FROM {_scan_expr(src)}
            WHERE {_ISSUE_YEAR_FROM_RAW} IS NOT NULL
            GROUP BY issue_year
            ORDER BY issue_year
            """
        ).df()
    finally:
        con.close()
    return df


def print_smoke_summary(source: str | Path | None = None) -> dict[str, float | int]:
    """Print the base-rate + cohort tables — the ``make data`` smoke test.

    Returns the base-rate dict so callers/tests can assert on it.
    """
    src = resolve_source(source)
    base = base_rate_summary(src)

    print("=" * 64)
    print("DefaultRadar — DuckDB data summary (smoke test)")
    print("=" * 64)
    print(f"Source        : {base['source']}")
    print(f"Rows          : {base['n_rows']:,}")
    print(f"Defaults      : {base['n_default']:,}  (repaid: {base['n_repaid']:,})")
    print(f"Base rate     : {base['base_rate']:.4f}  ({base['base_rate'] * 100:.2f}%)")

    print("\nDefault rate by issue year (time-split view):")
    with pd.option_context("display.max_rows", None):
        print(yearly_cohort_summary(src).to_string(index=False))

    print("\nDefault rate by loan purpose (top cohorts):")
    print(cohort_summary(src, by="purpose", min_count=1).head(12).to_string(index=False))

    print("\nDefault rate by home ownership:")
    print(cohort_summary(src, by="home_ownership_n", min_count=1).to_string(index=False))
    print("=" * 64)
    return base
