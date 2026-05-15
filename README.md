# v3-health-benchmark

`v3-health-benchmark` is a standalone reproducibility release of the
`v3_health` healthcare benchmark from the broader `causal-agent-benchmark`
project.

## Benchmark question

The benchmark asks whether stronger structural constraints help healthcare
decision agents stay more robust under confounding, missing data, and
historically biased feedback.

## Agent families

- `baseline`: fixed policy baseline
- `workflow`: adaptive workflow heuristic
- `causal_light`: structural-prior ablation
- `stability_filtered`: structurally constrained adaptive heuristic
- `structural_causal`: explicit structural causal model

## Scenario suite

- `treatment_allocation_confounding`: SES-linked distortion in observed risk
- `missing_data_bias`: unequal lab availability and missing-not-at-random risk
- `historical_bias_feedback`: biased utilization history and amplified SES risk

Each scenario is run through `normal`, `surface_shift`, and `adversarial`
phases.

## Repository layout

```text
v3-health-benchmark/
├── artifacts/release/     # Curated release snapshot and manifest
├── src/v3_health/         # Standalone healthcare benchmark package
├── tests/                 # Package and CLI verification
├── LICENSE
├── PROVENANCE.md
├── pyproject.toml
└── requirements-lock.txt
```

## Install

Target runtime: Python `3.11+`.

Canonical reproducibility-oriented install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps --no-build-isolation
```

Developer install with test extras:

```bash
pip install -e .[dev]
```

## Reproduce the benchmark

Run the full benchmark:

```bash
v3-health run --output-dir results
```

Render the dashboard:

```bash
v3-health dashboard \
  --results results/results_summary.json \
  --output results/dashboard.html
```

Run the reduced hyperparameter sensitivity sweep:

```bash
v3-health sensitivity \
  --output-dir results/sensitivity \
  --n-patients-per-phase 120 \
  --n-seeds 3 \
  --perturbation 0.25
```

Run the test suite:

```bash
pytest
```

## Expected outputs

The main benchmark run writes summary-level artifacts under `results/`,
including:

- `results_summary.json`
- `negative_result.json`
- `paired_significance.csv`
- `effect_sizes.csv`
- `seed_metrics.csv`
- `seed_summary.csv`
- `tradeoff_summary.csv`
- `audit_log.jsonl`

The dashboard command writes `results/dashboard.html`.

The canonical checked-in publication snapshot lives under
`artifacts/release/`.

## Runtime notes

The headline configuration evaluates:

- `3` scenarios
- `3` phases
- `5` agent families
- `12` seeds
- `1000` patients per phase

That is a full benchmark sweep rather than a toy smoke test, so expect minutes
rather than seconds for the canonical run.

## Rights notice

This repository is public for publication support and reproducibility review,
but it is not released under an open-source license. See [LICENSE](LICENSE)
before reuse, redistribution, or commercial use.
