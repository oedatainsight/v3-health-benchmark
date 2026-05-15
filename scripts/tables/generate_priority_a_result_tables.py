from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = REPO_ROOT / "artifacts" / "release"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables"


PAIRWISE_HEADERS = [
    "scenario",
    "slice",
    "metric",
    "agent_a",
    "agent_b",
    "mean_a",
    "mean_b",
    "paired_mean_difference",
    "ci_lower",
    "ci_upper",
    "p_raw",
    "cohen_dz",
    "n_seeds",
]

ADJUSTED_HEADERS = [
    "scenario",
    "slice",
    "metric",
    "agent_a",
    "agent_b",
    "p_raw",
    "p_holm",
    "p_bh",
    "p_paired_t",
    "p_paired_t_holm",
    "p_paired_t_bh",
    "correction_family",
    "family_size",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number != number:
        return "nan"
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.6g}"


def _markdown_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _tex_escape(value: object) -> str:
    text = _format_cell(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _tex_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    colspec = "l" * len(headers)
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(row.get(h, "")) for h in headers) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _write_table_bundle(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, object]],
    headers: list[str],
) -> None:
    _write_csv(output_dir / f"{stem}.csv", rows, headers)
    (output_dir / f"{stem}.md").write_text(_markdown_table(rows, headers))
    (output_dir / f"{stem}.tex").write_text(_tex_table(rows, headers))


def _build_pairwise_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "scenario": row["scenario"],
            "slice": row["slice"],
            "metric": row["metric"],
            "agent_a": row["agent_a"],
            "agent_b": row["agent_b"],
            "mean_a": row["mean_a"],
            "mean_b": row["mean_b"],
            "paired_mean_difference": row["mean_delta"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "p_raw": row["p_value"],
            "cohen_dz": row["effect_size_dz"],
            "n_seeds": row["n_paired_seeds"],
        }
        for row in rows
    ]


def _build_adjusted_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "scenario": row["scenario"],
            "slice": row["slice"],
            "metric": row["metric"],
            "agent_a": row["agent_a"],
            "agent_b": row["agent_b"],
            "p_raw": row["p_value"],
            "p_holm": row["p_value_holm"],
            "p_bh": row["p_value_bh"],
            "p_paired_t": row["p_value_paired_t"],
            "p_paired_t_holm": row["p_value_paired_t_holm"],
            "p_paired_t_bh": row["p_value_paired_t_bh"],
            "correction_family": row["family"],
            "family_size": row["family_size"],
        }
        for row in rows
    ]


def _as_float(value: str) -> float:
    return float(value)


def _orient_delta(row: dict[str, str], focal: str, comparator: str) -> dict[str, object] | None:
    agents = {row["agent_a"], row["agent_b"]}
    if agents != {focal, comparator}:
        return None

    if row["agent_a"] == focal:
        delta = _as_float(row["mean_delta"])
        ci_lower = _as_float(row["ci_lower"])
        ci_upper = _as_float(row["ci_upper"])
        dz = _as_float(row["effect_size_dz"])
        focal_mean = row["mean_a"]
        comparator_mean = row["mean_b"]
    else:
        delta = -_as_float(row["mean_delta"])
        ci_lower = -_as_float(row["ci_upper"])
        ci_upper = -_as_float(row["ci_lower"])
        dz = -_as_float(row["effect_size_dz"])
        focal_mean = row["mean_b"]
        comparator_mean = row["mean_a"]

    return {
        "scenario": row["scenario"],
        "slice": row["slice"],
        "metric": row["metric"],
        "comparison": f"{focal} - {comparator}",
        "focal_mean": focal_mean,
        "comparator_mean": comparator_mean,
        "paired_mean_difference": delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_raw": row["p_value"],
        "p_holm": row["p_value_holm"],
        "p_bh": row["p_value_bh"],
        "cohen_dz": dz,
        "favored_agent": row["favored_agent"],
        "n_seeds": row["n_paired_seeds"],
    }


def _write_key_claim_checks(output_dir: Path, rows: list[dict[str, str]]) -> None:
    focal = "structural_causal"
    comparators = ["causal_light", "stability_filtered"]
    slices = {"overall", "adversarial", "degradation"}
    metrics = {"success_rate", "near_optimal_rate", "success_rate_drop"}
    selected: list[dict[str, object]] = []
    for row in rows:
        if row["slice"] not in slices or row["metric"] not in metrics:
            continue
        for comparator in comparators:
            oriented = _orient_delta(row, focal, comparator)
            if oriented is not None:
                selected.append(oriented)

    headers = [
        "scenario",
        "slice",
        "metric",
        "comparison",
        "focal_mean",
        "comparator_mean",
        "paired_mean_difference",
        "ci_lower",
        "ci_upper",
        "p_raw",
        "p_holm",
        "p_bh",
        "cohen_dz",
        "favored_agent",
        "n_seeds",
    ]
    note = (
        "# Key Claim Checks\n\n"
        "These paired-seed checks are generated from `artifacts/release/paired_significance.csv`. "
        "They support narrative phrasing such as \"not statistically distinguishable in this "
        "experiment\" rather than equivalence claims.\n\n"
    )
    (output_dir / "key_claim_checks.md").write_text(note + _markdown_table(selected, headers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pairwise = _read_csv(args.release_dir / "paired_significance.csv")
    _write_table_bundle(
        args.output_dir,
        "pairwise_results_with_effect_sizes",
        _build_pairwise_rows(pairwise),
        PAIRWISE_HEADERS,
    )
    _write_table_bundle(
        args.output_dir,
        "adjusted_p_values",
        _build_adjusted_rows(pairwise),
        ADJUSTED_HEADERS,
    )
    _write_key_claim_checks(args.output_dir, pairwise)
    print(f"Wrote Priority A result tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
