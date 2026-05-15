import os
import subprocess
import sys
import json
from pathlib import Path

import yaml

from v3_health.core import config as cfg


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def test_module_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "v3_health.cli", "--help"],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "v3-health" in result.stdout
    assert "run" in result.stdout
    assert "dashboard" in result.stdout
    assert "sensitivity" in result.stdout


def test_cli_run_writes_to_repo_root_results(tmp_path):
    output_dir = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "run",
            "--output-dir",
            str(output_dir),
            "--n-patients-per-phase",
            "12",
            "--n-seeds",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "results_summary.json").exists()
    assert not (REPO_ROOT / "src" / "v3_health" / "results").exists()


def test_cli_run_honors_config_path_and_runtime_overrides(tmp_path):
    config_path = tmp_path / "custom.yaml"
    payload = yaml.safe_load(Path(cfg.DEFAULT_CONFIG_PATH).read_text())
    payload["benchmark"]["n_seeds"] = 2
    payload["benchmark"]["n_patients_per_phase"] = 99
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--n-patients-per-phase",
            "6",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "results_summary.json").read_text())
    assert summary["meta"]["n_seeds"] == 2
    assert summary["meta"]["n_patients_per_phase"] == 6


def test_cli_run_uses_configured_scenarios_and_agents(tmp_path):
    config_path = tmp_path / "subset.yaml"
    payload = yaml.safe_load(Path(cfg.DEFAULT_CONFIG_PATH).read_text())
    payload["benchmark"]["n_seeds"] = 1
    payload["benchmark"]["n_patients_per_phase"] = 5
    payload["scenarios"] = [payload["scenarios"][0]]
    payload["agents"] = [payload["agents"][0]]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "results_summary.json").read_text())
    assert summary["meta"]["scenarios"] == ["treatment_allocation_confounding"]
    assert summary["meta"]["agents"] == ["baseline"]
    assert summary["meta"]["n_seeds"] == 1
    assert summary["meta"]["n_patients_per_phase"] == 5


def test_cli_dashboard_generates_html(tmp_path):
    results_dir = tmp_path / "results"
    dashboard_path = tmp_path / "dashboard.html"

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "run",
            "--output-dir",
            str(results_dir),
            "--n-patients-per-phase",
            "12",
            "--n-seeds",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "dashboard",
            "--results",
            str(results_dir / "results_summary.json"),
            "--output",
            str(dashboard_path),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert dashboard_path.exists()


def test_cli_sensitivity_generates_csv(tmp_path):
    output_dir = tmp_path / "sensitivity"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "v3_health.cli",
            "sensitivity",
            "--output-dir",
            str(output_dir),
            "--n-patients-per-phase",
            "8",
            "--n-seeds",
            "1",
            "--perturbation",
            "0.25",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "hyperparam_sensitivity.csv").exists()
