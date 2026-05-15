import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v3_health_default.yaml"


def test_generate_scenario_spec_table(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tables/generate_scenario_spec_table.py",
            "--config",
            str(DEFAULT_CONFIG),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    csv_path = tmp_path / "scenario_parameter_schedule.csv"
    md_path = tmp_path / "scenario_parameter_schedule.md"
    tex_path = tmp_path / "scenario_parameter_schedule.tex"
    assert csv_path.exists()
    assert md_path.exists()
    assert tex_path.exists()

    rows = list(csv.DictReader(csv_path.open()))
    assert any(
        row["scenario"] == "treatment_allocation_confounding"
        and row["phase"] == "adversarial"
        and "s1_ses_bias_adversarial" in row["parameter_values"]
        for row in rows
    )
    assert "missing_data_bias" in md_path.read_text()


def test_generate_agent_spec_table(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tables/generate_agent_spec_table.py",
            "--config",
            str(DEFAULT_CONFIG),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    csv_path = tmp_path / "agent_parameter_summary.csv"
    md_path = tmp_path / "agent_parameter_summary.md"
    tex_path = tmp_path / "agent_parameter_summary.tex"
    assert csv_path.exists()
    assert md_path.exists()
    assert tex_path.exists()

    rows = list(csv.DictReader(csv_path.open()))
    assert any(
        row["agent"] == "structural_causal"
        and row["parameter"] == "structural_state_prior_mass"
        and row["value"] == "4.0"
        for row in rows
    )
    assert "stability_filtered" in md_path.read_text()
