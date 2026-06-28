"""Download + cache the leakage-safe Lending Club dataset and materialise Parquet.

Source: Zenodo record 11295916 — *"Lending Club loan dataset for granting models"*
(CC-BY-4.0). This is the **granting-model** version of Lending Club data: it
contains only *application-time* features (income, DTI, FICO, loan amount,
employment length, purpose, home ownership, state, ZIP prefix) plus a binary
``Default`` target. Outcome-derived fields (interest rate, grade, sub-grade,
recoveries, ...) are **absent by construction**, which is precisely why we use it
to avoid label leakage. See ``config/features.yaml`` for the explicit allow/ban
lists and the leakage test that enforces them.

This module:

* streams the CSV from Zenodo with integrity checks (size + MD5),
* converts it to columnar Parquet via DuckDB (streaming, low-memory), adding a
  typed ``issue_date`` parsed from the ``Mon-YYYY`` ``issue_d`` strings, and
* can carve out a small, deterministic, stratified CSV sample committed to the
  repo so CI runs fast without the 167 MB download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import requests

from defaultradar.config import CONFIG, TARGET_COLUMN, TIME_COLUMN


@dataclass(frozen=True)
class DatasetInfo:
    """Immutable metadata describing the canonical Zenodo dataset file."""

    record_id: str = "11295916"
    filename: str = "LC_loans_granting_model_dataset.csv"
    size_bytes: int = 167_468_415
    md5: str = "b019384d6bc65bf2a3e839362e4ff502"
    license: str = "CC-BY-4.0"
    citation: str = (
        "Lending Club loan dataset for granting models (Zenodo record 11295916), CC-BY-4.0."
    )

    @property
    def url(self) -> str:
        return f"https://zenodo.org/api/records/{self.record_id}/files/{self.filename}/content"


DATASET = DatasetInfo()


# --- Integrity helpers -------------------------------------------------------
def _md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_cached(path: Path, expected_size: int) -> bool:
    """Cheap cache check: file exists and matches the expected byte count."""
    return path.exists() and path.stat().st_size == expected_size


# --- Download ----------------------------------------------------------------
def download_dataset(
    *,
    force: bool = False,
    verify_md5: bool = True,
    info: DatasetInfo = DATASET,
) -> Path:
    """Download the raw CSV to ``CONFIG.raw_csv_path`` (cached, idempotent).

    Parameters
    ----------
    force:
        Re-download even if a byte-size-matching cache exists.
    verify_md5:
        After download, verify the MD5 against the Zenodo-published checksum and
        raise if it does not match.

    Returns
    -------
    Path to the cached CSV.
    """
    CONFIG.ensure_dirs()
    dest = CONFIG.raw_csv_path

    if not force and _is_cached(dest, info.size_bytes):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        downloaded = 0
        with requests.get(info.url, stream=True, timeout=(30, 300)) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

        if downloaded != info.size_bytes:
            raise OSError(
                f"Downloaded size {downloaded} != expected {info.size_bytes}. "
                "The Zenodo file may have changed; verify the record."
            )

        if verify_md5:
            actual = _md5_of_file(tmp)
            if actual != info.md5:
                raise OSError(
                    f"MD5 mismatch: got {actual}, expected {info.md5}. Refusing to cache."
                )

        tmp.replace(dest)
    except BaseException:
        # Never leave a partial/corrupt .part behind on any failure (network
        # drop, size/MD5 mismatch, interrupt) so the next run retries cleanly.
        tmp.unlink(missing_ok=True)
        raise
    return dest


# --- CSV -> Parquet ----------------------------------------------------------
# issue_d arrives as e.g. "Dec-2015"; parse to a real DATE (first of month) so
# the time-based split and DuckDB cohort queries are correct and fast.
_ISSUE_DATE_EXPR = f"try_strptime({TIME_COLUMN}, '%b-%Y')::DATE AS issue_date"


def ensure_raw_parquet(*, force: bool = False) -> Path:
    """Ensure the raw Parquet feature-store source exists; build it if needed.

    Downloads the CSV if absent, then uses DuckDB to stream-convert it to Parquet
    (typed, columnar), adding the parsed ``issue_date`` column. Returns the
    Parquet path. This is the deterministic source of truth all later phases read.
    """
    CONFIG.ensure_dirs()
    out = CONFIG.raw_parquet_path
    if out.exists() and not force:
        return out

    # Propagate force so `--force` re-downloads the CSV, not just the Parquet.
    csv_path = download_dataset(force=force)
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    {_ISSUE_DATE_EXPR}
                FROM read_csv_auto('{csv_path.as_posix()}', header=true, sample_size=-1)
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )
    finally:
        con.close()
    return out


# --- Small committed CI sample ----------------------------------------------
def make_ci_sample(
    *,
    per_class_per_year: int = 170,
    source_parquet: Path | None = None,
    dest_csv: Path | None = None,
) -> Path:
    """Write a small, deterministic, class- AND time-stratified CSV sample for CI.

    CI must run fast and offline, so we commit a tiny slice (a few thousand rows)
    rather than the 167 MB original. The sample is stratified on **both**
    ``Default`` and issue *year* so the committed sample actually spans the whole
    2007-2018 range (and therefore populates the train/validation/test windows of
    the time-based split). Selection within each (class, year) cell is ordered by
    ``hash(id)`` — deterministic and reproducible, but spread across the id range
    rather than skewed to the earliest loans (low ids correlate with early dates).

    The output CSV keeps the raw ``issue_d`` (``Mon-YYYY``) string and drops the
    derived ``issue_date`` column, so the sample mirrors the raw schema.
    """
    src = source_parquet or CONFIG.raw_parquet_path
    dest = dest_csv or CONFIG.sample_csv_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                WITH ranked AS (
                    SELECT
                        * EXCLUDE (issue_date),
                        row_number() OVER (
                            PARTITION BY "{TARGET_COLUMN}", year(issue_date)
                            ORDER BY hash(id)
                        ) AS _rn
                    FROM read_parquet('{src.as_posix()}')
                    WHERE issue_date IS NOT NULL
                )
                SELECT * EXCLUDE (_rn)
                FROM ranked
                WHERE _rn <= {int(per_class_per_year)}
                ORDER BY id
            ) TO '{dest.as_posix()}' (FORMAT CSV, HEADER true);
            """
        )
    finally:
        con.close()
    return dest
