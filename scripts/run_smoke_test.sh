#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUT="${1:-results/smoke}"

python3 -m v3_health.cli run \
  --config configs/v3_health_default.yaml \
  --output-dir "$OUT" \
  --n-patients-per-phase 8 \
  --n-seeds 1

python3 -m v3_health.cli dashboard \
  --config configs/v3_health_default.yaml \
  --results "$OUT/results_summary.json" \
  --output "$OUT/dashboard.html"
