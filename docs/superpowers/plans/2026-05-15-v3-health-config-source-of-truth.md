# v3-health Config Source of Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move benchmark-critical `v3_health` parameters into modular YAML config files, load them as the runtime source of truth, and generate reviewer-facing docs/tables from those configs.

**Architecture:** `configs/v3_health_default.yaml` is the top-level bundle. `src/v3_health/core/config.py` loads that bundle, resolves scenario and agent YAML files, normalizes them into the existing `HEALTHCARE_CONFIG` and `AGENT_HYPERPARAMS` globals, and exposes reset/snapshot helpers. CLI commands accept `--config`, while docs and table scripts read the same YAML source.

**Tech Stack:** Python 3.11+, PyYAML, pytest, stdlib `csv/json/pathlib/subprocess`.

---

### Task 1: Config Loader Tests

**Files:**
- Create: `tests/test_config_loader.py`
- Modify later: `src/v3_health/core/config.py`

- [ ] **Step 1: Write failing tests**

Add tests that import loader helpers expected from `v3_health.core.config`:

```python
from pathlib import Path

from v3_health.core import config as cfg


def test_default_yaml_materializes_current_runtime_values():
    bundle = cfg.load_config_bundle()
    assert bundle["healthcare_config"]["n_seeds"] == 12
    assert bundle["healthcare_config"]["n_patients_per_phase"] == 1000
    assert bundle["healthcare_config"]["s1_ses_bias_adversarial"][0] == 0.25
    assert bundle["agent_hyperparams"]["action_thresholds"] == (0.25, 0.50, 0.75)
    assert bundle["agent_hyperparams"]["structural_state_prior_mass"] == 4.0


def test_apply_config_bundle_updates_compatibility_globals(tmp_path):
    cfg.reset_to_default_config()
    before = cfg.snapshot_config()
    custom = tmp_path / "custom.yaml"
    custom.write_text(Path(cfg.DEFAULT_CONFIG_PATH).read_text().replace("n_seeds: 12", "n_seeds: 2"))
    try:
        cfg.apply_config_bundle(custom)
        assert cfg.HEALTHCARE_CONFIG["n_seeds"] == 2
        assert cfg.AGENT_HYPERPARAMS["action_thresholds"] == (0.25, 0.50, 0.75)
    finally:
        cfg.restore_config(before)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_config_loader.py -q`

Expected: FAIL because loader helpers or YAML config files do not exist.

- [ ] **Step 3: Implement loader and YAML files**

Create modular YAML configs under `configs/` and update `src/v3_health/core/config.py` to load them.

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_config_loader.py -q`

Expected: PASS.

### Task 2: CLI Config Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify later: `src/v3_health/cli.py`

- [ ] **Step 1: Write failing CLI test**

Add a subprocess test that copies the default config, changes `n_seeds`, runs `v3_health.cli run --config custom.yaml --output-dir ... --n-patients-per-phase 6`, and asserts `results_summary.json` reports `n_seeds == 2` and `n_patients_per_phase == 6`.

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_cli.py::test_cli_run_honors_config_path_and_runtime_overrides -q`

Expected: FAIL because `--config` is not accepted.

- [ ] **Step 3: Add CLI `--config` support**

Load the chosen config before command execution and preserve existing runtime overrides.

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/test_cli.py::test_cli_run_honors_config_path_and_runtime_overrides -q`

Expected: PASS.

### Task 3: Table Script Tests

**Files:**
- Create: `tests/test_table_scripts.py`
- Create later: `scripts/tables/generate_scenario_spec_table.py`
- Create later: `scripts/tables/generate_agent_spec_table.py`

- [ ] **Step 1: Write failing tests**

Add subprocess tests that run both table scripts against `configs/v3_health_default.yaml` and a temporary output directory, then assert generated CSV and Markdown files exist and contain representative scenario/agent rows.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_table_scripts.py -q`

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Implement scripts**

Read YAML through the config loader and write CSV, Markdown, and TeX outputs.

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_table_scripts.py -q`

Expected: PASS.

### Task 4: Reviewer Docs

**Files:**
- Create: `docs/repository_map.md`
- Create: `docs/scm_specification.md`
- Create: `docs/agent_specifications.md`
- Create: `docs/examples/causal_light_vs_stability_filtered.md`
- Create: `docs/code_data_availability.md`

- [ ] **Step 1: Add docs referencing YAML and runtime modules**

Write concise docs that identify config source files, execution files, generated table scripts, and interpretation boundaries.

- [ ] **Step 2: Link config workflow from README**

Update `README.md` with `--config`, table generation commands, and docs pointers.

### Task 5: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_config_loader.py tests/test_cli.py tests/test_table_scripts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run smoke benchmark and table generation**

Run:

```bash
PYTHONPATH=src python3 -m v3_health run --config configs/v3_health_default.yaml --output-dir /private/tmp/v3hb-config-smoke --n-patients-per-phase 8 --n-seeds 1
PYTHONPATH=src python3 scripts/tables/generate_scenario_spec_table.py --config configs/v3_health_default.yaml --output-dir /private/tmp/v3hb-config-tables
PYTHONPATH=src python3 scripts/tables/generate_agent_spec_table.py --config configs/v3_health_default.yaml --output-dir /private/tmp/v3hb-config-tables
```

Expected: benchmark writes `results_summary.json`; scripts write table files.
