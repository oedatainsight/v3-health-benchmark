# v3-health Config Source of Truth Design

## Goal

Make `configs/` the canonical source of truth for benchmark-critical parameters in `v3-health-benchmark`, while preserving current runtime behavior, CLI defaults, artifact shapes, and most existing call sites.

## Scope

This pass covers:

- externalizing benchmark-critical runtime, scenario, agent, and statistical parameters into YAML
- loading those YAML files through `v3_health.core.config`
- preserving `HEALTHCARE_CONFIG` and `AGENT_HYPERPARAMS` as compatibility globals
- adding CLI `--config` support
- generating reviewer-facing documentation and tables from the YAML source

This pass does not cover:

- figure redesign
- optional S4 hybrid scenario
- community agent submission protocol
- a repo-wide migration to typed immutable config objects
- externalizing pure presentation constants such as dashboard colors

## Requirements

- Behavior-preserving relative to the current benchmark.
- `configs/` is the single source of truth for benchmark-critical parameters.
- Human-authored docs and generation scripts are committed.
- Generated tables are rebuildable outputs, not hand-maintained source files.
- Existing benchmark code should continue to read `HEALTHCARE_CONFIG` and `AGENT_HYPERPARAMS` with minimal call-site churn.

## Current State

Benchmark-critical parameters currently live in Python literals in:

- `src/v3_health/core/config.py`

Those values are already centralized better than scattered magic numbers, but they are still embedded in code rather than versioned config artifacts. Reviewer-facing documentation and generated spec tables are also missing.

## Proposed Architecture

### Canonical Config Layout

Add modular YAML files under `configs/`:

- `configs/v3_health_default.yaml`
- `configs/scenarios/s1_treatment_allocation_confounding.yaml`
- `configs/scenarios/s2_mnar_measurement.yaml`
- `configs/scenarios/s3_historical_bias_feedback.yaml`
- `configs/agents/baseline.yaml`
- `configs/agents/workflow.yaml`
- `configs/agents/causal_light.yaml`
- `configs/agents/stability_filtered.yaml`
- `configs/agents/structural_causal.yaml`

The top-level bundle file will reference the scenario and agent files that define the benchmark. This keeps provenance clear and lets docs and table generators cite exact YAML sources.

### Runtime Config Model

`src/v3_health/core/config.py` will become a loader/shim module:

- load the default YAML bundle on import
- merge referenced scenario and agent YAML files
- normalize the merged YAML into the current runtime dict shapes
- expose:
  - `HEALTHCARE_CONFIG`
  - `AGENT_HYPERPARAMS`
- provide helper functions to:
  - load a chosen bundle path
  - reload defaults
  - snapshot and restore config state for tests and CLI overrides

This preserves the current runtime contract while moving authorship and provenance into YAML.

### Compatibility Strategy

For this pass, existing scenarios, agents, runners, and tests should continue to consume the same global dicts they use today. The refactor should avoid a wide call-site rewrite.

The YAML will therefore be nested for readability, but the loader will flatten or normalize it back into the current key/value layout where needed.

### CLI Behavior

Add optional `--config` support to:

- `v3-health run`
- `v3-health dashboard`
- `v3-health sensitivity`

Behavior:

- default: use `configs/v3_health_default.yaml`
- explicit `--config`: load that bundle before command execution
- existing CLI overrides such as `--n-patients-per-phase` and `--n-seeds` still apply on top of the loaded bundle

For `dashboard`, config loading is primarily for consistency and future-proofing. The current implementation mainly consumes `results_summary.json`, so the config argument may be documented as non-semantic for the first pass.

## Config File Responsibilities

### `configs/v3_health_default.yaml`

Owns:

- global benchmark settings
- phase ordering
- selected scenario files
- selected agent files
- outcome model selection
- shared statistical settings
- default runtime scale such as seed count and patients per phase

### `configs/scenarios/*.yaml`

Each scenario file owns:

- phase-specific parameter values
- bias mechanism magnitudes
- missingness or feedback coefficients
- scenario-local notes/labels used by documentation and table generation

### `configs/agents/*.yaml`

Each agent file owns:

- decision thresholds
- adaptation rates
- priors
- latent-state settings
- stability thresholds
- other benchmark-critical hyperparameters

## Documentation Outputs

Add committed reviewer-facing docs:

- `docs/repository_map.md`
- `docs/scm_specification.md`
- `docs/agent_specifications.md`
- `docs/examples/causal_light_vs_stability_filtered.md`
- `docs/code_data_availability.md`

These docs should point directly to the YAML files as the canonical parameter sources and to the relevant Python modules as the execution layer.

## Generated Tables and Scripts

Add committed scripts:

- `scripts/tables/generate_scenario_spec_table.py`
- `scripts/tables/generate_agent_spec_table.py`

Generated outputs:

- `outputs/tables/scenario_parameter_schedule.csv`
- `outputs/tables/scenario_parameter_schedule.md`
- `outputs/tables/scenario_parameter_schedule.tex`
- `outputs/tables/agent_parameter_summary.csv`
- `outputs/tables/agent_parameter_summary.md`
- `outputs/tables/agent_parameter_summary.tex`

These tables should be generated from YAML rather than maintained by hand.

## Data Flow

1. CLI chooses a config bundle path.
2. `v3_health.core.config` loads the top-level YAML bundle.
3. The loader resolves referenced scenario and agent YAML files.
4. The loader merges and normalizes the data into the runtime dict shapes.
5. Existing benchmark code consumes `HEALTHCARE_CONFIG` and `AGENT_HYPERPARAMS`.
6. Documentation/table scripts independently read the same YAML files to emit reviewer-facing artifacts.

## Validation and Error Handling

The loader should fail fast on:

- missing referenced config files
- unknown scenario or agent identifiers
- missing required sections
- malformed scalar types for benchmark-critical fields

Validation in this pass should be lightweight and explicit. The main goal is to prevent silent drift or partial loads, not to introduce a large schema framework.

## Testing Strategy

Add tests for:

- loading the default bundle reproduces the current runtime values
- nested YAML is normalized into the current flat runtime dict shapes
- CLI `--config` is honored
- CLI runtime overrides still beat YAML defaults
- config reset/restore helpers isolate tests cleanly
- table-generation scripts execute successfully and emit expected key rows

The existing full test suite remains the regression backstop.

## Risks

### Drift During Migration

Risk:

- YAML values may accidentally differ from the current Python literals.

Mitigation:

- add exact-value tests for representative benchmark-critical fields
- migrate by transcription first, not cleanup

### Shape Mismatch

Risk:

- nested YAML may not map cleanly onto existing runtime dict keys

Mitigation:

- define explicit normalization code in one place
- keep call sites unchanged

### Global State Leakage

Risk:

- tests and CLI commands currently mutate config globals

Mitigation:

- add snapshot/reset helpers in `config.py`
- use them in tests and CLI temporary overrides

## Recommended Implementation Order

1. Add YAML config files mirroring current values.
2. Refactor `core/config.py` into loader + compatibility globals.
3. Add tests that prove default YAML reproduces current behavior.
4. Add CLI `--config` support.
5. Add docs pointing to config and runtime files.
6. Add table-generation scripts and verify outputs.
7. Run the full test suite and smoke commands.

## Success Criteria

- `configs/` is the canonical source of truth for benchmark-critical parameters.
- Existing CLI commands still work without changed defaults.
- Existing tests remain green.
- Reviewers can inspect scenario and agent settings directly from YAML.
- Docs and tables are generated from the same config source used by runtime code.
