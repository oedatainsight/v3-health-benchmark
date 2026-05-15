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
├── configs/               # Canonical benchmark configuration source
├── docs/                  # Reviewer-facing reproducibility documentation
├── scripts/tables/        # Config-derived table generators
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

Run a quick smoke test:

```bash
scripts/run_smoke_test.sh
```

Run the full benchmark:

```bash
v3-health run \
  --config configs/v3_health_default.yaml \
  --output-dir results
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
  --config configs/v3_health_default.yaml \
  --output-dir results/sensitivity \
  --n-patients-per-phase 120 \
  --n-seeds 3 \
  --perturbation 0.25
```

Run the test suite:

```bash
pytest
```

Regenerate camera-ready Priority A tables and figures from the checked-in
release snapshot:

```bash
scripts/rebuild_tables.sh
scripts/rebuild_figures.sh
```

The table script regenerates scenario and agent specification tables from the
same YAML config used by the runtime, plus reviewer-requested statistical
tables from `artifacts/release/paired_significance.csv`.

The figure script regenerates static PDF/PNG figures from
`artifacts/release/results_summary.json` and `artifacts/release/tradeoff_summary.csv`.

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

Camera-ready Priority A artifacts are checked in under:

- `outputs/tables/scenario_parameter_schedule.{csv,md,tex}`
- `outputs/tables/agent_parameter_summary.{csv,md,tex}`
- `outputs/tables/pairwise_results_with_effect_sizes.{csv,md,tex}`
- `outputs/tables/adjusted_p_values.{csv,md,tex}`
- `outputs/tables/key_claim_checks.md`
- `outputs/figures/figure1_regime_dependence_accessible.{pdf,png}`
- `outputs/figures/figure2_paired_seed_slopechart.{pdf,png}`
- `docs/bibliography_expansion.md`

The canonical checked-in publication snapshot lives under
`artifacts/release/`.

The canonical parameter source lives under `configs/`. See
`docs/repository_map.md`, `docs/scm_specification.md`, and
`docs/agent_specifications.md` for reviewer-facing maps from config to code.

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
