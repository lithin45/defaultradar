"""Command-line entrypoint for DefaultRadar (``uv run defaultradar ...``).

The Makefile drives the lifecycle through this single CLI so there is one
discoverable surface. Phase 1 implements the data subcommands; later phases add
``train``, ``eval``, ``promote``, ``monitor`` and ``retrain``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from defaultradar.config import CONFIG


def _cmd_data(args: argparse.Namespace) -> int:
    """Download + cache the dataset, materialise Parquet, print the summary."""
    from defaultradar.data.download import ensure_raw_parquet
    from defaultradar.data.duckdb_summary import print_smoke_summary

    parquet = ensure_raw_parquet(force=args.force)
    print(f"[data] raw Parquet ready: {parquet}")
    print_smoke_summary(parquet)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    """Print the DuckDB base-rate + cohort summary over the best available source."""
    from defaultradar.data.duckdb_summary import print_smoke_summary

    print_smoke_summary(args.source)
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    """Carve out the small committed CI sample from the full Parquet."""
    from defaultradar.data.download import ensure_raw_parquet, make_ci_sample

    ensure_raw_parquet()
    dest = make_ci_sample(per_class_per_year=args.per_class_per_year)
    print(f"[sample] wrote stratified CI sample -> {dest}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    """Build the engineered, leakage-checked, time-split feature store."""
    from defaultradar.features import FEATURE_COLUMNS, build_feature_store

    store = build_feature_store(source=args.source, write=not args.no_write)
    print(
        f"[features] {len(FEATURE_COLUMNS)} feature columns | config_hash={store.config_hash[:12]}"
    )
    if store.paths:
        for name, path in store.paths.items():
            print(f"[features] wrote {name}: {path}")
    print("\nTime-based split summary:")
    print(store.summary().to_string(index=False))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train + calibrate + log to MLflow + register a model version."""
    from defaultradar.training import format_report, train_and_log

    result = train_and_log(log_to_mlflow=not args.no_mlflow, register=not args.no_register)
    print(format_report(result))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate the registered model on test; non-zero exit if the gate fails."""
    from defaultradar.training import run_eval

    return run_eval(prefer_production=not args.latest)


def _cmd_promote(args: argparse.Namespace) -> int:
    """Metric-gated Staging->Production promotion; non-zero exit if not promoted."""
    from defaultradar.registry import promote_model

    result = promote_model(version=args.version, dry_run=args.dry_run)
    print(result.report())
    return 0 if (result.promoted or args.dry_run) else 1


def _cmd_monitor(args: argparse.Namespace) -> int:
    """Run a monitoring cycle (drift detection); inject drift by default for the demo."""
    from defaultradar.monitoring import run_monitoring

    result = run_monitoring(inject=not args.no_inject, write_reports=not args.no_reports)
    print(result.summary())
    if result.reports:
        for kind, path in result.reports.items():
            print(f"[monitor] {kind} report -> {path}")
    return 0


def _cmd_retrain(args: argparse.Namespace) -> int:
    """Retrain on the expanded window, register, gate, and promote if it passes."""
    from defaultradar.monitoring import retraining_flow

    out = retraining_flow()
    print(
        f"[retrain] version={out['retrained']['model_version']} "
        f"test_roc_auc={out['retrained']['test_roc_auc']:.4f} "
        f"promoted={out['promotion']['promoted']} stage={out['promotion']['stage']}"
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Full lifecycle demo: train -> promote -> drift -> retrain -> promote."""
    from defaultradar.demo import run_demo

    result = run_demo(inject=not args.no_inject)
    return 0 if result.served_version_changed or not result.drift_detected else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="defaultradar",
        description="DefaultRadar MLOps lifecycle CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="download dataset + build Parquet + summary")
    p_data.add_argument("--force", action="store_true", help="re-download / rebuild")
    p_data.set_defaults(func=_cmd_data)

    p_sum = sub.add_parser("summary", help="print DuckDB base-rate/cohort summary")
    p_sum.add_argument("--source", default=None, help="explicit Parquet/CSV source")
    p_sum.set_defaults(func=_cmd_summary)

    p_smp = sub.add_parser("sample", help="write the committed CI sample CSV")
    p_smp.add_argument("--per-class-per-year", type=int, default=170, dest="per_class_per_year")
    p_smp.set_defaults(func=_cmd_sample)

    p_feat = sub.add_parser("features", help="build engineered + time-split feature store")
    p_feat.add_argument("--source", default=None, help="explicit Parquet/CSV source")
    p_feat.add_argument("--no-write", action="store_true", help="don't write Parquet partitions")
    p_feat.set_defaults(func=_cmd_features)

    p_train = sub.add_parser("train", help="train + calibrate + log to MLflow + register")
    p_train.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p_train.add_argument("--no-register", action="store_true", help="log run but don't register")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate registered model on test; gate exit code")
    p_eval.add_argument("--latest", action="store_true", help="use latest version, not Production")
    p_eval.set_defaults(func=_cmd_eval)

    p_prom = sub.add_parser("promote", help="metric-gated Staging->Production promotion")
    p_prom.add_argument("--version", default=None, help="version to promote (default: latest)")
    p_prom.add_argument("--dry-run", action="store_true", help="evaluate but don't transition")
    p_prom.set_defaults(func=_cmd_promote)

    p_mon = sub.add_parser("monitor", help="run a drift-monitoring cycle (PSI + Evidently)")
    p_mon.add_argument("--no-inject", action="store_true", help="do NOT inject drift (baseline)")
    p_mon.add_argument("--no-reports", action="store_true", help="skip Evidently report generation")
    p_mon.set_defaults(func=_cmd_monitor)

    p_ret = sub.add_parser("retrain", help="retrain -> register -> gate -> promote")
    p_ret.set_defaults(func=_cmd_retrain)

    p_demo = sub.add_parser("demo", help="full lifecycle: train->promote->drift->retrain->promote")
    p_demo.add_argument("--no-inject", action="store_true", help="do NOT inject drift")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Touch CONFIG so directories exist for any subcommand that writes output.
    CONFIG.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
