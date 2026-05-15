"""
Verify SCM invariants hold across all scenarios and phases.
"""

import numpy as np

from v3_health.core.scm import generate_scenario1, generate_scenario2, generate_scenario3
from v3_health.core.outcome_resolver import (
    resolve_outcome,
    symmetric_severity_penalty,
    treatment_mismatch_gap,
)
from v3_health.core.types import LatentPatientState


def _latent(true_health: float, optimal_treatment: int) -> LatentPatientState:
    return LatentPatientState(
        patient_id=1,
        ses=1,
        true_health=true_health,
        access_to_care=0.5,
        measurement_prob=0.5,
        optimal_treatment=optimal_treatment,
        group=0,
    )


def test_treatment_mismatch_gap_boundaries():
    latent = _latent(true_health=0.75, optimal_treatment=3)

    assert treatment_mismatch_gap(3, latent) == 0
    assert treatment_mismatch_gap(2, latent) == 1
    assert treatment_mismatch_gap(0, latent) == 3


def test_symmetric_severity_penalty_boundaries():
    latent = _latent(true_health=0.75, optimal_treatment=2)

    assert symmetric_severity_penalty(2, latent, symmetric=True) == 0.0
    assert symmetric_severity_penalty(0, latent, symmetric=True) == 1.5
    assert symmetric_severity_penalty(3, latent, symmetric=True) == 0.75
    assert symmetric_severity_penalty(3, latent, symmetric=False) == 0.0


def test_outcome_independent_of_ses():
    """CRITICAL: outcome must depend on treatment + health, NOT SES."""
    rng = np.random.default_rng(42)
    results_by_ses: dict[int, list[bool]] = {0: [], 1: [], 2: []}

    for _ in range(5000):
        for scenario_gen in [generate_scenario1, generate_scenario2, generate_scenario3]:
            latent, _ = scenario_gen(0, "normal", rng)
            outcome = resolve_outcome(latent.optimal_treatment, latent, "test", rng)
            results_by_ses[latent.ses].append(outcome.success)

    rates = {ses: float(np.mean(v)) for ses, v in results_by_ses.items()}
    max_gap = max(rates.values()) - min(rates.values())
    assert max_gap < 0.05, f"SES affects outcome even with optimal treatment: {rates}"


def test_treatment_gap_affects_outcome():
    """Better treatment match -> better outcomes."""
    rng = np.random.default_rng(42)
    results_by_gap: dict[int, list[bool]] = {0: [], 1: [], 2: [], 3: []}

    for _ in range(3000):
        latent, _ = generate_scenario1(0, "normal", rng)
        for gap in range(4):
            action = min(3, max(0, latent.optimal_treatment + gap))
            outcome = resolve_outcome(action, latent, "test", rng)
            results_by_gap[gap].append(outcome.success)

    rates = {gap: float(np.mean(v)) for gap, v in results_by_gap.items()}
    assert rates[0] > rates[1] > rates[2], f"Treatment match not monotonic: {rates}"


def test_missingness_correlates_with_ses():
    """Scenario 2: low SES should have more missing data."""
    rng = np.random.default_rng(42)
    missing_by_ses: dict[int, list[int]] = {0: [], 1: [], 2: []}

    for _ in range(2000):
        latent, observed = generate_scenario2(0, "normal", rng)
        missing_by_ses[latent.ses].append(observed.n_missing_labs)

    avg_missing = {ses: float(np.mean(v)) for ses, v in missing_by_ses.items()}
    assert avg_missing[0] > avg_missing[2], f"Low SES should have more missing data: {avg_missing}"


def test_historical_ses_bias_changes_risk_and_utilization():
    """Scenario 3 should encode SES-linked historical distortion in both
    model risk and recorded utilization, alongside a separately sampled
    protected-group indicator."""
    rng = np.random.default_rng(42)
    risk_by_ses: dict[int, list[float]] = {0: [], 1: [], 2: []}
    util_by_ses: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for _ in range(3000):
        latent, observed = generate_scenario3(0, "normal", rng)
        risk_by_ses[latent.ses].append(observed.observed_risk_score)
        util_by_ses[latent.ses].append(observed.prior_utilization)
        assert observed.group in (0, 1)

    mean_risk = {ses: float(np.mean(v)) for ses, v in risk_by_ses.items()}
    mean_util = {ses: float(np.mean(v)) for ses, v in util_by_ses.items()}
    assert mean_risk[0] > mean_risk[2], (
        f"Scenario 3 should inflate low-SES risk more than high-SES risk: {mean_risk}"
    )
    assert mean_util[0] < mean_util[2], (
        f"Scenario 3 should depress low-SES historical utilization: {mean_util}"
    )


def test_phase_shift_changes_distribution():
    """Phase shift must produce the *signed* SES-stratified bias the config asserts.

    The previous version of this test only required ``|mean_n - mean_a| > 0.01``
    or a comparable change in std. With n=1000 noisy samples that condition is
    a tautology and would still pass if the SES bias table were silently
    zeroed out. Here we instead pin the per-stratum signed shift implied by
    the config so a regression that flattens or flips the bias surfaces
    immediately.
    """
    from v3_health.core.config import HEALTHCARE_CONFIG as C

    rng_n = np.random.default_rng(42)
    rng_a = np.random.default_rng(42)

    by_phase: dict[str, dict[int, list[float]]] = {
        "normal": {0: [], 1: [], 2: []},
        "adversarial": {0: [], 1: [], 2: []},
    }

    for _ in range(4000):
        latent_n, obs_n = generate_scenario1(0, "normal", rng_n)
        latent_a, obs_a = generate_scenario1(0, "adversarial", rng_a)
        by_phase["normal"][latent_n.ses].append(obs_n.observed_risk_score)
        by_phase["adversarial"][latent_a.ses].append(obs_a.observed_risk_score)

    # Tolerance is generous on bias magnitude because clipping to [0, 1] and
    # noise both shrink the realised gap relative to the nominal config bias.
    # We test sign and a meaningful fraction of the nominal effect.
    tol = 0.40
    for ses in (0, 1, 2):
        nominal_shift = (
            C["s1_ses_bias_adversarial"][ses] - C["s1_ses_bias_normal"][ses]
        )
        observed_shift = float(np.mean(by_phase["adversarial"][ses])) - float(
            np.mean(by_phase["normal"][ses])
        )
        if abs(nominal_shift) < 1e-9:
            # SES==1 has zero bias in both phases; observed shift should be
            # near zero (within sampling noise).
            assert abs(observed_shift) < 0.03, (
                f"SES={ses}: expected ~0 shift, got {observed_shift:+.3f}"
            )
            continue
        # Signs must agree.
        assert np.sign(observed_shift) == np.sign(nominal_shift), (
            f"SES={ses}: observed shift {observed_shift:+.3f} has wrong "
            f"sign vs nominal {nominal_shift:+.3f}"
        )
        # Magnitude must be at least (1 - tol) * |nominal|, accounting for
        # clipping shrinkage at the [0, 1] edges.
        assert abs(observed_shift) >= (1.0 - tol) * abs(nominal_shift), (
            f"SES={ses}: observed shift |{observed_shift:.3f}| smaller "
            f"than {(1 - tol) * abs(nominal_shift):.3f} (nominal "
            f"{nominal_shift:+.3f}); SES bias table may be silently zeroed"
        )


def test_outcome_asymmetry_under_vs_over():
    """With the asymmetric severity penalty, under-treating critical patients
    hurts success more than over-treating them. With the (default) symmetric
    flag the gap closes substantially. The headline now uses the symmetric
    flag; this test pins both branches."""
    from v3_health.core.config import HEALTHCARE_CONFIG as C
    saved_sym = C["outcome_symmetric_severity_penalty"]
    saved_model = C["outcome_model"]
    # Pin to the clipped-linear model so the magnitude assertions below
    # don't depend on logistic-coefficient choices.
    C["outcome_model"] = "clipped_linear"
    C["outcome_symmetric_severity_penalty"] = False
    try:
        rng = np.random.default_rng(7)
        under = []
        over = []
        for _ in range(4000):
            latent, _ = generate_scenario1(0, "normal", rng)
            if latent.optimal_treatment >= 2:
                o_under = resolve_outcome(latent.optimal_treatment - 2, latent, "test", rng)
                under.append(o_under.success)
            if latent.optimal_treatment <= 1:
                o_over = resolve_outcome(latent.optimal_treatment + 2, latent, "test", rng)
                over.append(o_over.success)
        assert under and over
        asym_under = float(np.mean(under))
        asym_over = float(np.mean(over))
        assert asym_over - asym_under > 0.10, (
            f"Asymmetric mode expected: over_succ={asym_over:.3f} should beat "
            f"under_succ={asym_under:.3f} by >0.10"
        )

        C["outcome_symmetric_severity_penalty"] = True
        rng2 = np.random.default_rng(7)
        sym_under = []
        sym_over = []
        for _ in range(4000):
            latent, _ = generate_scenario1(0, "normal", rng2)
            if latent.optimal_treatment >= 2:
                o = resolve_outcome(latent.optimal_treatment - 2, latent, "test", rng2)
                sym_under.append(o.success)
            if latent.optimal_treatment <= 1:
                o = resolve_outcome(latent.optimal_treatment + 2, latent, "test", rng2)
                sym_over.append(o.success)
        gap_sym = float(np.mean(sym_over)) - float(np.mean(sym_under))
        assert gap_sym < (asym_over - asym_under) - 0.05, (
            f"Symmetric mode should shrink under/over gap; got {gap_sym:.3f}"
        )
    finally:
        C["outcome_symmetric_severity_penalty"] = saved_sym
        C["outcome_model"] = saved_model


def test_outcome_logs_clip_events():
    """Critical patients under-treated by 3 levels should trigger the clip
    under the appendix ``clipped_linear`` outcome model."""
    from v3_health.core.config import HEALTHCARE_CONFIG as C
    saved_model = C["outcome_model"]
    C["outcome_model"] = "clipped_linear"
    try:
        rng = np.random.default_rng(11)
        saw_clip = False
        for _ in range(2000):
            latent, _ = generate_scenario1(0, "normal", rng)
            if latent.optimal_treatment == 3 and latent.true_health > 0.7:
                o = resolve_outcome(0, latent, "test", rng)
                saw_clip = saw_clip or o.success_prob_clipped
                if saw_clip:
                    break
        assert saw_clip, "Clip should fire for critical patients under-treated by 3 levels"
    finally:
        C["outcome_model"] = saved_model


def test_structural_bias_regression_recovers_access_slope():
    """The structural agent's bias regression should recover the
    *direct* SES->risk slope when the data-generating process has both
    a true_health->risk path (proxied by anchor) and an SES->risk path.
    Univariate regression of (risk - anchor) on access_proxy would be
    biased by the anchor's noise; bivariate OLS partials it out.
    """
    from v3_health.agents.structural_causal_agent import StructuralCausalModel

    rng = np.random.default_rng(0)
    model = StructuralCausalModel()

    n = 600
    true_bias = -0.30  # high access_proxy (low SES) lowers raw_risk
    health = rng.uniform(0.05, 0.95, size=n)
    access_proxy = rng.uniform(0.0, 1.0, size=n)
    # Anchor is a noisy proxy of (1 - true_health), independent of access.
    # The univariate residual = raw_risk - anchor would have slope ≈ true_bias
    # only when the anchor noise is uncorrelated with access; bivariate OLS
    # additionally protects against any anchor-noise correlation that the
    # old specification could not.
    anchor = (1.0 - health) + rng.normal(0, 0.10, size=n)
    raw_risk = (1.0 - health) + true_bias * access_proxy + rng.normal(0, 0.03, size=n)

    for i in range(n):
        model._bias_buffer.append(
            (float(access_proxy[i]), float(anchor[i]), float(raw_risk[i]))
        )
    # Iterate the EWMA so it converges to the underlying OLS estimate
    # (single call would dampen it by the 0.35 blend factor).
    for _ in range(20):
        model._update_risk_bias()

    # Slope should recover ~true_bias (within EWMA + clip + noise tolerance).
    assert abs(model.risk_bias_slope - true_bias) < 0.10, (
        f"Bivariate bias regression did not recover the SES->risk slope: "
        f"got {model.risk_bias_slope:.3f}, expected ~{true_bias}"
    )


if __name__ == "__main__":
    test_outcome_independent_of_ses()
    print("ok: Outcome independent of SES")
    test_treatment_gap_affects_outcome()
    print("ok: Treatment gap affects outcome")
    test_missingness_correlates_with_ses()
    print("ok: Missingness correlates with SES")
    test_historical_ses_bias_changes_risk_and_utilization()
    print("ok: Historical SES bias changes risk and utilization")
    test_phase_shift_changes_distribution()
    print("ok: Phase shift changes distribution")
    test_outcome_asymmetry_under_vs_over()
    print("ok: Under/over asymmetry behaves as documented")
    test_outcome_logs_clip_events()
    print("ok: Clip events recorded")
    test_structural_bias_regression_recovers_access_slope()
    print("ok: Structural bias regression recovers access slope")
    print("\nAll SCM invariant tests passed.")
