import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generate_priority_a_figures(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/figures/generate_priority_a_figures.py",
            "--release-dir",
            str(REPO_ROOT / "artifacts" / "release"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for name in [
        "figure1_regime_dependence_accessible",
        "figure2_paired_seed_slopechart",
    ]:
        png = tmp_path / f"{name}.png"
        pdf = tmp_path / f"{name}.pdf"
        assert png.exists() and png.stat().st_size > 1000
        assert pdf.exists() and pdf.stat().st_size > 1000
