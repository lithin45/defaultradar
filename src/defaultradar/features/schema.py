"""The feature contract: a typed view over ``config/features.yaml``.

This is the single place that reads the allow/ban lists, so the feature pipeline
and the leakage guard share one source of truth. It also exposes a deterministic
hash of the contract file, which training logs to MLflow for full reproducibility
(a model version is tied to the exact feature contract it was built under).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from defaultradar.config import CONFIG


class LeakageError(AssertionError):
    """Raised when a banned/outcome-derived column reaches the feature set."""


@dataclass(frozen=True)
class FeatureContract:
    """Typed, validated view of ``config/features.yaml``."""

    target: str
    time_column: str
    split_date_column: str
    id_columns: tuple[str, ...]
    numeric: tuple[str, ...]
    ordinal: tuple[str, ...]
    categorical: tuple[str, ...]
    text: tuple[str, ...]
    banned: frozenset[str]
    config_hash: str = field(compare=False)

    # --- Derived views ------------------------------------------------------
    @property
    def allowed(self) -> frozenset[str]:
        """All allowed *raw* input columns (across every group)."""
        return frozenset((*self.numeric, *self.ordinal, *self.categorical, *self.text))

    @property
    def model_categorical(self) -> tuple[str, ...]:
        """Categorical raw columns that the model pipeline one-hot encodes."""
        return self.categorical

    @property
    def metadata_columns(self) -> frozenset[str]:
        """Columns kept alongside features but never used AS features.

        The label (``target``) and the parsed split key (``split_date_column``)
        live in the engineered frame for training/splitting; they are excluded
        when the leakage guard inspects the feature columns.
        """
        return frozenset({self.target, self.split_date_column})

    def assert_contract_consistent(self) -> None:
        """Allow/ban lists must be disjoint and role/time columns must be banned."""
        overlap = self.allowed & self.banned
        if overlap:
            raise LeakageError(f"Columns both allowed and banned: {sorted(overlap)}")
        # Target, raw time, parsed split date, and ids must all be banned from
        # the feature set (the split date too — adding it as a feature would be
        # temporal leakage).
        for role in (
            self.target,
            self.time_column,
            self.split_date_column,
            *self.id_columns,
        ):
            if role not in self.banned:
                raise LeakageError(f"Role column {role!r} must be in banned_columns")


def feature_config_hash(path: Path | None = None) -> str:
    """SHA-256 of the feature-contract file (logged to MLflow for lineage)."""
    p = path or CONFIG.features_config_path
    return hashlib.sha256(p.read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_feature_contract(path: Path | None = None) -> FeatureContract:
    """Load + validate the feature contract from YAML (cached per path)."""
    p = path or CONFIG.features_config_path
    spec = yaml.safe_load(p.read_text())

    allowed = spec["allowed_features"]
    contract = FeatureContract(
        target=spec["target"],
        time_column=spec["time_column"],
        split_date_column=spec["split_date_column"],
        id_columns=tuple(spec.get("id_columns", [])),
        numeric=tuple(allowed.get("numeric", [])),
        ordinal=tuple(allowed.get("ordinal", [])),
        categorical=tuple(allowed.get("categorical", [])),
        text=tuple(allowed.get("text", [])),
        banned=frozenset(spec["banned_columns"]),
        config_hash=feature_config_hash(p),
    )
    contract.assert_contract_consistent()
    return contract
