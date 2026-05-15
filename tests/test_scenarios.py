"""Sanity tests for scenario wrappers."""

import numpy as np

from v3_health.scenarios import (
    treatment_allocation,
    missing_data_bias,
    historical_bias_feedback,
)


def test_each_scenario_generates_and_resolves():
    rng = np.random.default_rng(0)
    for mod in [treatment_allocation, missing_data_bias, historical_bias_feedback]:
        for phase in ["normal", "surface_shift", "adversarial"]:
            latent, observed = mod.generate(1, phase, rng)
            assert 0 <= latent.optimal_treatment <= 3
            assert 0.0 <= observed.observed_risk_score <= 1.0
            outcome = mod.resolve(latent.optimal_treatment, latent, rng)
            assert outcome.treatment_gap == 0


def test_scenarios_expose_protected_group_indicator():
    """All v3_health scenarios should now populate the protected-group
    attribute on both the latent state and the observed patient."""
    rng = np.random.default_rng(0)
    for mod in [treatment_allocation, missing_data_bias, historical_bias_feedback]:
        latent, observed = mod.generate(1, "normal", rng)
        assert latent.group in (0, 1)
        assert observed.group in (0, 1)
        assert latent.group == observed.group


if __name__ == "__main__":
    test_each_scenario_generates_and_resolves()
    print("ok: scenarios generate and resolve")
    test_scenarios_expose_protected_group_indicator()
    print("ok: scenarios expose a protected-group indicator")
