from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3_health.core.config import DEFAULT_CONFIG_PATH, load_config_bundle


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _markdown_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h].replace("\n", " ") for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _tex_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    colspec = "l" * len(headers)
    lines = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [row[h].replace("_", "\\_").replace("&", "\\&") for h in headers]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_rows(config_path: Path) -> list[dict[str, str]]:
    bundle = load_config_bundle(config_path)
    rows: list[dict[str, str]] = []
    for agent in bundle["agents"]:
        for parameter, value in agent.get("hyperparams", {}).items():
            rows.append(
                {
                    "agent": agent["agent"],
                    "label": agent.get("label", agent["agent"]),
                    "family": agent.get("family", ""),
                    "parameter": parameter,
                    "value": _format_value(value),
                    "source": str(agent["source_path"].relative_to(REPO_ROOT)),
                }
            )
    return rows


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = ["agent", "label", "family", "parameter", "value", "source"]
    csv_path = output_dir / "agent_parameter_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "agent_parameter_summary.md").write_text(_markdown_table(rows, headers))
    (output_dir / "agent_parameter_summary.tex").write_text(_tex_table(rows, headers))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    rows = build_rows(args.config)
    write_outputs(rows, args.output_dir)
    print(f"Wrote agent specification tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
