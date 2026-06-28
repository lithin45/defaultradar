"""Data acquisition + DuckDB analytics layer.

Phase 1 implements the download/cache + DuckDB cohort summary. Later phases read
the cached Parquet produced here as the source of truth for feature building.
"""

from defaultradar.data.download import (
    DatasetInfo,
    download_dataset,
    ensure_raw_parquet,
    make_ci_sample,
)
from defaultradar.data.duckdb_summary import base_rate_summary, cohort_summary

__all__ = [
    "DatasetInfo",
    "download_dataset",
    "ensure_raw_parquet",
    "make_ci_sample",
    "base_rate_summary",
    "cohort_summary",
]
