from pathlib import Path

import pytest
import yaml

from v3_health.core import config as cfg


def test_default_yaml_materializes_current_runtime_values():
    bundle = cfg.load_config_bundle()

    healthcare = bundle["healthcare_config"]
    hyperparams = bundle["agent_hyperparams"]
    metadata = bundle["metadata"]

    assert metadata["config_path"].name == "v3_health_default.yaml"
    assert healthcare["n_seeds"] == 12
    assert healthcare["n_patients_per_phase"] == 1000
    assert healthcare["phases"] == ["normal", "surface_shift", "adversarial"]
    assert healthcare["scenarios"] == [
        "treatment_allocation_confounding",
        "missing_data_bias",
        "historical_bias_feedback",
    ]
    assert healthcare["s1_ses_bias_adversarial"][0] == 0.25
    assert healthcare["s2_lab_names"][-1] == "imaging"
    assert healthcare["s3_ses_util_penalty_adversarial"][0] == 0.30
    assert healthcare["outcome_model"] == "logistic"
    assert hyperparams["action_thresholds"] == (0.25, 0.50, 0.75)
    assert hyperparams["stratum_edges"] == (0.33, 0.66)
    assert hyperparams["structural_state_prior_mass"] == 4.0


def test_apply_config_bundle_updates_compatibility_globals(tmp_path):
    cfg.reset_to_default_config()
    before = cfg.snapshot_config()
    custom = tmp_path / "custom.yaml"
    payload = yaml.safe_load(Path(cfg.DEFAULT_CONFIG_PATH).read_text())
    payload["benchmark"]["n_seeds"] = 2
    payload["benchmark"]["n_patients_per_phase"] = 17
    custom.write_text(yaml.safe_dump(payload, sort_keys=False))

    try:
        cfg.apply_config_bundle(custom)

        assert cfg.HEALTHCARE_CONFIG["n_seeds"] == 2
        assert cfg.HEALTHCARE_CONFIG["n_patients_per_phase"] == 17
        assert cfg.HEALTHCARE_CONFIG["s1_ses_bias_normal"][0] == 0.15
        assert cfg.AGENT_HYPERPARAMS["action_thresholds"] == (0.25, 0.50, 0.75)
    finally:
        cfg.restore_config(before)

    assert cfg.HEALTHCARE_CONFIG["n_seeds"] == before["healthcare_config"]["n_seeds"]
    assert cfg.AGENT_HYPERPARAMS == before["agent_hyperparams"]


def test_config_loader_fails_on_missing_reference(tmp_path):
    payload = yaml.safe_load(Path(cfg.DEFAULT_CONFIG_PATH).read_text())
    payload["scenarios"][0]["path"] = "scenarios/missing.yaml"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(FileNotFoundError):
        cfg.load_config_bundle(bad)
