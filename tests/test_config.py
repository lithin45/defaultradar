"""Sanity checks on central config + the feature contract file."""

from __future__ import annotations

import yaml

from defaultradar.config import CONFIG, SPLIT_DATE_COLUMN, TARGET_COLUMN, TIME_COLUMN


def test_thresholds_are_sane() -> None:
    # These are the hard gates referenced by make eval and the promotion fn.
    # roc_auc_min = 0.67 is the honest achievable floor on this leakage-safe,
    # time-split, application-time dataset (model ~0.684); see README.
    assert CONFIG.roc_auc_min == 0.67
    assert CONFIG.psi_threshold == 0.20
    assert CONFIG.latency_p95_ms == 300.0
    assert CONFIG.seed == 42


def test_time_split_cutoffs_ordered() -> None:
    # train_end < valid_end so train/valid/test are non-overlapping in time.
    assert CONFIG.train_end < CONFIG.valid_end


def test_paths_resolve_under_repo() -> None:
    assert CONFIG.config_dir.is_dir()
    assert CONFIG.features_config_path.exists()
    # data/ and reports/ are created on demand.
    CONFIG.ensure_dirs()
    assert CONFIG.data_dir.is_dir()
    assert CONFIG.reports_dir.is_dir()


def test_features_yaml_contract() -> None:
    spec = yaml.safe_load(CONFIG.features_config_path.read_text())

    assert spec["target"] == TARGET_COLUMN
    assert spec["time_column"] == TIME_COLUMN
    # The split must compare against the parsed DATE column, not the raw string.
    assert spec["split_date_column"] == SPLIT_DATE_COLUMN
    assert spec["split_date_column"] != spec["time_column"]

    allowed = spec["allowed_features"]
    assert {"numeric", "categorical", "ordinal", "text"} <= set(allowed)

    banned = set(spec["banned_columns"])
    # The target / time / split-date / id must be banned from the feature set.
    assert {"Default", "issue_d", "issue_date", "id"} <= banned
    assert spec["split_date_column"] in banned  # split key is never a feature
    # Classic Lending Club leakage columns must be enumerated.
    assert {"int_rate", "grade", "sub_grade", "loan_status", "recoveries"} <= banned

    # Allowlist and banlist must be disjoint (no column both allowed and banned).
    allowed_flat = {c for group in allowed.values() for c in group}
    assert allowed_flat.isdisjoint(banned), allowed_flat & banned
