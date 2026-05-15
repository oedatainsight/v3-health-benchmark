#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 scripts/tables/generate_scenario_spec_table.py \
  --config configs/v3_health_default.yaml \
  --output-dir outputs/tables

python3 scripts/tables/generate_agent_spec_table.py \
  --config configs/v3_health_default.yaml \
  --output-dir outputs/tables

python3 scripts/tables/generate_priority_a_result_tables.py \
  --release-dir artifacts/release \
  --output-dir outputs/tables
