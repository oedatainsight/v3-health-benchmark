# Repository Map

This map points reviewers to the benchmark-critical source files. The canonical parameter source is the YAML bundle at `configs/v3_health_default.yaml`; Python modules execute the configured benchmark.

## Configuration

- `configs/v3_health_default.yaml`: top-level benchmark bundle, runtime scale, phase ordering, outcome model, shared population parameters, scenario references, and agent references.
- `configs/scenarios/`: scenario-specific phase schedules and mechanism parameters.
- `configs/agents/`: agent hyperparameters and family labels.
- `src/v3_health/core/config.py`: YAML loader and compatibility shim exposing `HEALTHCARE_CONFIG` and `AGENT_HYPERPARAMS`.

## Simulator and SCM

- `src/v3_health/core/scm.py`: latent patient generation, access, measurement, observed proxies, and scenario observation processes.
- `src/v3_health/core/outcome_resolver.py`: outcome model mapping `(action, latent health)` to success, treatment gap, and cost.
- `src/v3_health/core/types.py`: latent state, observed patient, action observation, and outcome record types.

## Scenarios

- `src/v3_health/scenarios/treatment_allocation.py`: wrapper for `treatment_allocation_confounding`.
- `src/v3_health/scenarios/missing_data_bias.py`: wrapper for `missing_data_bias`.
- `src/v3_health/scenarios/historical_bias_feedback.py`: wrapper for `historical_bias_feedback`.
- `configs/scenarios/*.yaml`: exact phase-level parameter schedules.

## Agents

- `src/v3_health/agents/baseline_agent.py`: policy-constrained baseline.
- `src/v3_health/agents/workflow_agent.py`: adaptive workflow heuristic.
- `src/v3_health/agents/causal_light_agent.py`: structural-prior-only ablation.
- `src/v3_health/agents/causal_agent.py`: stability-filtered adaptive heuristic.
- `src/v3_health/agents/structural_causal_agent.py`: explicit latent-state structural-causal agent.
- `configs/agents/*.yaml`: agent hyperparameters used by those implementations.

## Experiment Runner

- `src/v3_health/evaluation/run_benchmark.py`: orchestrates scenario x phase x agent x seed runs, computes summaries, and writes artifacts.
- `src/v3_health/cli.py`: command-line entrypoint with `run`, `dashboard`, and `sensitivity` commands plus optional `--config`.
- `src/v3_health/experiments/hyperparam_sensitivity.py`: reduced hyperparameter sensitivity sweep.

## Metrics and Statistics

- `src/v3_health/evaluation/fairness_metrics.py`: success, near-optimality, cost, SES fairness gap, group fairness gap, and degradation metrics.
- `src/v3_health/evaluation/causal_targeting.py`: action-alignment diagnostic from partial correlations on audit logs.
- `src/v3_health/evaluation/statistical_analysis.py`: seed-level summaries, Student-t CIs, paired sign-flip tests, paired-t p-values, Cohen's dz, and Holm/BH adjustments.

## Tables and Figures

- `scripts/tables/generate_scenario_spec_table.py`: generates scenario schedule tables from YAML.
- `scripts/tables/generate_agent_spec_table.py`: generates agent parameter tables from YAML.
- `scripts/tables/generate_priority_a_result_tables.py`: generates effect-size and adjusted-p-value tables from `artifacts/release/paired_significance.csv`.
- `scripts/figures/generate_priority_a_figures.py`: generates static camera-ready PDF/PNG figures from `artifacts/release/`.
- `scripts/rebuild_tables.sh`: one-command rebuild for checked-in Priority A table artifacts.
- `scripts/rebuild_figures.sh`: one-command rebuild for checked-in Priority A figure artifacts.
- `src/v3_health/visualization/dashboard.py`: renders the HTML dashboard from `results_summary.json`.

## Artifacts

- `artifacts/release/`: curated release snapshot.
- `results/`: default output directory for local benchmark runs.
- `outputs/tables/`: checked-in camera-ready Priority A tables.
- `outputs/figures/`: checked-in camera-ready Priority A static figures.
