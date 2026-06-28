"""Leakage-safe feature pipeline + time-based split (Phase 2)."""

from defaultradar.features.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_INPUT_COLUMNS,
    FeatureStore,
    assert_no_leakage,
    build_feature_matrix,
    build_feature_store,
    engineer_features,
    feature_columns_of,
    get_xy,
    load_split,
    parse_emp_length,
    time_split,
)
from defaultradar.features.schema import (
    FeatureContract,
    LeakageError,
    feature_config_hash,
    load_feature_contract,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "RAW_INPUT_COLUMNS",
    "FeatureStore",
    "FeatureContract",
    "LeakageError",
    "assert_no_leakage",
    "build_feature_matrix",
    "build_feature_store",
    "engineer_features",
    "feature_columns_of",
    "feature_config_hash",
    "get_xy",
    "load_feature_contract",
    "load_split",
    "parse_emp_length",
    "time_split",
]
