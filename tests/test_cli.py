import os
import subprocess
import sys
from pathlib import Path


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
