#!/usr/bin/env python
"""Scripted dataset step for ``make data``.

Downloads + caches the Zenodo dataset, materialises the raw Parquet feature
store, (re)generates the small committed CI sample, and prints the DuckDB
base-rate/cohort summary smoke test. Thin wrapper over
``defaultradar.data`` so the logic is unit-testable.

Usage:
    uv run python scripts/download_data.py [--force] [--no-sample]
"""

from __future__ import annotations

import argparse

from defaultradar.data.download import ensure_raw_parquet, make_ci_sample
from defaultradar.data.duckdb_summary import print_smoke_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Download + summarise the dataset.")
    parser.add_argument("--force", action="store_true", help="re-download / rebuild Parquet")
    parser.add_argument(
        "--no-sample", action="store_true", help="skip regenerating the CI sample CSV"
    )
    parser.add_argument(
        "--per-class-per-year",
        type=int,
        default=170,
        help="rows per (class, issue-year) cell in the CI sample",
    )
    args = parser.parse_args()

    parquet = ensure_raw_parquet(force=args.force)
    print(f"[data] raw Parquet ready: {parquet}")

    if not args.no_sample:
        sample = make_ci_sample(per_class_per_year=args.per_class_per_year)
        print(f"[data] CI sample written: {sample}")

    print_smoke_summary(parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
