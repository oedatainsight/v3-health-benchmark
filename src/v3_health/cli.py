from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from v3_health.core import config as cfg


@contextmanager
def _temporary_healthcare_config(**overrides: int | float | bool | None) -> Iterator[None]:
    saved = cfg.snapshot_config()
    try:
        for key, value in overrides.items():
            if value is not None:
                cfg.HEALTHCARE_CONFIG[key] = value
        yield
    finally:
        cfg.restore_config(saved)


def _run_command(args: argparse.Namespace) -> int:
    cfg.apply_config_bundle(args.config)
    from v3_health.evaluation.run_benchmark import run_full_benchmark

    with _temporary_healthcare_config(
        n_patients_per_phase=args.n_patients_per_phase,
        n_seeds=args.n_seeds,
    ):
        run_full_benchmark(str(args.output_dir))
    return 0


def _dashboard_command(args: argparse.Namespace) -> int:
    cfg.apply_config_bundle(args.config)
    from v3_health.visualization.dashboard import render_dashboard

    render_dashboard(args.results, args.output)
    return 0


def _sensitivity_command(args: argparse.Namespace) -> int:
    cfg.apply_config_bundle(args.config)
    from v3_health.experiments.hyperparam_sensitivity import run_sensitivity_sweep

    run_sensitivity_sweep(
        output_dir=str(args.output_dir),
        n_patients_per_phase=args.n_patients_per_phase,
        n_seeds=args.n_seeds,
        perturbation=args.perturbation,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v3-health",
        description="Standalone CLI for the v3 healthcare causal benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the benchmark and write summary outputs.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=cfg.DEFAULT_CONFIG_PATH,
        help="Path to the top-level YAML config bundle.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    run_parser.add_argument(
        "--n-patients-per-phase",
        type=int,
        default=None,
        help="Optional override for smoke tests or reduced local runs.",
    )
    run_parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Optional override for smoke tests or reduced local runs.",
    )
    run_parser.set_defaults(func=_run_command)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Render the HTML dashboard from a benchmark summary JSON file.",
    )
    dashboard_parser.add_argument(
        "--config",
        type=Path,
        default=cfg.DEFAULT_CONFIG_PATH,
        help="Path to the top-level YAML config bundle.",
    )
    dashboard_parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/results_summary.json"),
    )
    dashboard_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dashboard.html"),
    )
    dashboard_parser.set_defaults(func=_dashboard_command)

    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        help="Run the reduced hyperparameter sensitivity sweep.",
    )
    sensitivity_parser.add_argument(
        "--config",
        type=Path,
        default=cfg.DEFAULT_CONFIG_PATH,
        help="Path to the top-level YAML config bundle.",
    )
    sensitivity_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sensitivity"),
    )
    sensitivity_parser.add_argument(
        "--n-patients-per-phase",
        type=int,
        default=120,
    )
    sensitivity_parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
    )
    sensitivity_parser.add_argument(
        "--perturbation",
        type=float,
        default=0.25,
    )
    sensitivity_parser.set_defaults(func=_sensitivity_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
