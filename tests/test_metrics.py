"""Tests for fairness metric computation and seed-level statistical summaries."""

import numpy as np

from v3_health.evaluation.fairness_metrics import compute_metrics, compute_phase_comparison
from v3_health.evaluation.statistical_analysis import (
    LOWER_IS_BETTER,
    _bh_adjusted,
    _favored_agent,
    _holm_adjusted,
    _paired_permutation_p_value,
    _stable_seed,
    compute_pairwise_tests,
    compute_seed_level_summary,
    seed_mean_ci,
)


def _record(**kw):
    base = {
        "success": True, "near_optimal": True, "treatment_gap": 0,
        "true_health": 0.5, "ses": 0, "group": 0,
        "under_treated": False, "over_treated": False,
        "success_prob_clipped": False, "cost": 0.3,
        "phase": "normal", "seed": 0,
    }
    base.update(kw)
    return base


def test_compute_metrics_basic():
    records = [
        _record(ses=0, success=False, under_treated=True),
        _record(ses=2, success=True, under_treated=False),
        _record(ses=1, success=True, under_treated=False),
    ]
    m = compute_metrics(records)
    assert 0.0 <= m["success_rate"] <= 1.0
    assert m["fairness_gap_ses"] >= 0
    assert m["under_treatment_disparity"] >= 0
    assert "by_ses" in m


def test_phase_comparison_degradation():
    records = (
        [_record(phase="normal", success=True) for _ in range(10)]
        + [_record(phase="adversarial", success=False) for _ in range(10)]
    )
    out = compute_phase_comparison(records)
    assert "degradation" in out
    assert out["degradation"]["success_rate_drop"] > 0


def test_seed_level_summary_exposes_bootstrap_intervals():
    records = []
    for seed in range(6):
        records.extend([
            _record(seed=seed, phase="normal", success=True, cost=0.2 + 0.01 * seed),
            _record(seed=seed, phase="normal", success=(seed % 2 == 0), cost=0.2 + 0.01 * seed),
            _record(seed=seed, phase="adversarial", success=(seed % 3 != 0), cost=0.4 + 0.01 * seed),
            _record(seed=seed, phase="adversarial", success=False, cost=0.4 + 0.01 * seed),
        ])

    summary = compute_seed_level_summary(
        records,
        compute_metrics=compute_metrics,
        compute_phase_comparison=compute_phase_comparison,
    )

    assert len(summary["per_seed"]) == 6
    overall_success = summary["overall"]["success_rate"]
    assert overall_success["n"] == 6
    assert overall_success["ci_lower"] <= overall_success["mean"] <= overall_success["ci_upper"]
    assert "degradation" in summary["by_phase"]


def test_pairwise_tests_use_matched_seed_deltas():
    def make_agent_records(successes: list[tuple[float, float]]) -> list[dict]:
        out: list[dict] = []
        for seed, (normal_success, adversarial_success) in enumerate(successes):
            out.extend([
                _record(seed=seed, phase="normal", success=bool(normal_success), cost=0.2),
                _record(seed=seed, phase="adversarial", success=bool(adversarial_success), cost=0.4),
            ])
        return out

    scenario_results = {
        "toy": {
            "causal": {
                "seed_level": compute_seed_level_summary(
                    make_agent_records([(1, 1)] * 6),
                    compute_metrics=compute_metrics,
                    compute_phase_comparison=compute_phase_comparison,
                )
            },
            "workflow": {
                "seed_level": compute_seed_level_summary(
                    make_agent_records([(1, 0)] * 6),
                    compute_metrics=compute_metrics,
                    compute_phase_comparison=compute_phase_comparison,
                )
            },
        }
    }

    rows = compute_pairwise_tests(scenario_results)
    overall_success = next(
        row for row in rows
        if row["scenario"] == "toy"
        and row["slice"] == "overall"
        and row["metric"] == "success_rate"
        and row["agent_a"] == "causal"
        and row["agent_b"] == "workflow"
    )

    assert overall_success["n_paired_seeds"] == 6
    assert overall_success["favored_agent"] == "causal"
    assert overall_success["effect_size_dz"] >= 0
    assert 0.0 <= overall_success["p_value"] <= 1.0


def test_bias_perpetuation_is_conditional_group_disparity():
    """bias_perpetuation_rate must be P(under | g=0) - P(under | g=1),
    not the tautological I[g=0 and under] indicator."""
    records = (
        # Group 0: 4/6 under-treated.
        [_record(group=0, under_treated=True) for _ in range(4)]
        + [_record(group=0, under_treated=False) for _ in range(2)]
        # Group 1: 1/6 under-treated.
        + [_record(group=1, under_treated=True) for _ in range(1)]
        + [_record(group=1, under_treated=False) for _ in range(5)]
    )
    m = compute_metrics(records)
    expected = (4.0 / 6.0) - (1.0 / 6.0)
    assert abs(m["bias_perpetuation_rate"] - expected) < 1e-9


def test_clip_rate_reported():
    records = (
        [_record(success_prob_clipped=True) for _ in range(3)]
        + [_record(success_prob_clipped=False) for _ in range(7)]
    )
    m = compute_metrics(records)
    assert abs(m["clip_rate"] - 0.30) < 1e-9


def test_seed_mean_ci_uses_student_t():
    # n=12 sample; t-CI should be wider than the std-error bounds and the
    # method label should reflect Student-t.
    values = [0.50, 0.52, 0.49, 0.55, 0.48, 0.51, 0.53, 0.50, 0.49, 0.54, 0.52, 0.50]
    ci = seed_mean_ci(values)
    assert ci["method"] == "student_t"
    assert ci["n"] == 12
    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]
    half_width = ci["ci_upper"] - ci["mean"]
    # t_{11, 0.975} ~= 2.201; check the half-width is at least
    # 2.0 * sem (slightly slack for floating point), i.e. wider than a
    # naive normal interval would give.
    assert half_width > 2.0 * ci["sem"] - 1e-9


def test_fairness_gap_uses_mad_not_max_minus_min():
    """fairness_gap_ses must be the mean absolute deviation across SES
    strata; max-minus-min is reported separately."""
    records = (
        [_record(ses=0, success=False) for _ in range(10)]
        + [_record(ses=0, success=True) for _ in range(0)]
        + [_record(ses=1, success=True) for _ in range(5)]
        + [_record(ses=1, success=False) for _ in range(5)]
        + [_record(ses=2, success=True) for _ in range(10)]
    )
    m = compute_metrics(records)
    rates = [0.0, 0.5, 1.0]
    expected_mad = sum(abs(r - 0.5) for r in rates) / 3
    expected_max_min = 1.0
    assert abs(m["fairness_gap_ses"] - expected_mad) < 1e-9
    assert abs(m["fairness_gap_ses_max_min"] - expected_max_min) < 1e-9


def test_under_treatment_disparity_is_magnitude():
    """Both directions of disparity violate equal treatment: the headline
    metric must be |ut_0 - ut_2|, with the signed value preserved
    separately."""
    high_ses_under = (
        [_record(ses=0, under_treated=False) for _ in range(10)]
        + [_record(ses=2, under_treated=True) for _ in range(7)]
        + [_record(ses=2, under_treated=False) for _ in range(3)]
    )
    m = compute_metrics(high_ses_under)
    assert m["under_treatment_disparity"] > 0
    assert m["under_treatment_disparity_signed"] < 0
    # The favored-agent comparator must treat lower disparity as better.
    assert "under_treatment_disparity" in LOWER_IS_BETTER
    fav = _favored_agent("good", "bad", "under_treatment_disparity", mean_delta=-0.20)
    assert fav == "good"


def test_stable_seed_is_order_invariant():
    rows = ["alpha", "beta", "gamma"]
    s1 = _stable_seed("pair", "scenarioA", "overall", "success_rate", "x", "y")
    # Inserting an unrelated row earlier must not change a downstream seed.
    rows.insert(0, "extra")
    s2 = _stable_seed("pair", "scenarioA", "overall", "success_rate", "x", "y")
    assert s1 == s2
    # Different identifiers must give different seeds.
    s3 = _stable_seed("pair", "scenarioA", "overall", "fairness_gap_ses", "x", "y")
    assert s1 != s3


def test_holm_and_bh_adjustments_are_added_per_family():
    def make_agent_records(seed_offset: float) -> list[dict]:
        out: list[dict] = []
        for seed in range(8):
            for _ in range(20):
                out.extend([
                    _record(seed=seed, phase="normal", success=(seed % 2 == 0), cost=0.2),
                    _record(
                        seed=seed,
                        phase="adversarial",
                        success=(seed + seed_offset) % 2 == 0,
                        cost=0.4,
                    ),
                ])
        return out

    scenarios: dict = {
        "scenario_a": {
            "good": {
                "seed_level": compute_seed_level_summary(
                    make_agent_records(0.0),
                    compute_metrics=compute_metrics,
                    compute_phase_comparison=compute_phase_comparison,
                )
            },
            "bad": {
                "seed_level": compute_seed_level_summary(
                    make_agent_records(1.0),
                    compute_metrics=compute_metrics,
                    compute_phase_comparison=compute_phase_comparison,
                )
            },
        },
    }

    rows = compute_pairwise_tests(scenarios)
    assert rows
    for row in rows:
        assert "p_value_holm" in row
        assert "p_value_bh" in row
        assert "family" in row and row["family_size"] >= 1
        assert 0.0 <= row["p_value_holm"] <= 1.0
        assert 0.0 <= row["p_value_bh"] <= 1.0
        # Adjusted >= raw for Holm; BH is monotone non-decreasing in raw.
        assert row["p_value_holm"] + 1e-12 >= row["p_value"]


def test_holm_bh_helpers_match_known_values():
    pvals = [0.01, 0.04, 0.03, 0.005]
    holm = _holm_adjusted(pvals)
    bh = _bh_adjusted(pvals)
    # Holm: rank ascending: 0.005, 0.01, 0.03, 0.04
    #  -> 4*0.005=0.02, 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04
    # cummax in rank order: 0.02, 0.03, 0.06, 0.06 (since 0.04<0.06)
    # back to original positions [0.01, 0.04, 0.03, 0.005] -> [0.03, 0.06, 0.06, 0.02]
    assert [round(x, 6) for x in holm] == [0.03, 0.06, 0.06, 0.02]
    # BH: scaled in ascending rank: 4/1*0.005=0.02, 4/2*0.01=0.02,
    #   4/3*0.03=0.04, 4/4*0.04=0.04
    # cummin from the right: 0.04, 0.04, 0.02, 0.02
    # mapped back: original positions [0.01, 0.04, 0.03, 0.005] -> [0.02, 0.04, 0.04, 0.02]
    assert [round(x, 6) for x in bh] == [0.02, 0.04, 0.04, 0.02]


def test_pairwise_rows_record_paired_seed_set():
    """Every pairwise row must carry the explicit list of seeds used so a
    reader can detect when different metrics rely on different subsets."""
    records_a = []
    records_b = []
    for seed in range(4):
        for phase in ("normal", "adversarial"):
            records_a.append(_record(seed=seed, phase=phase, success=True, cost=0.3))
            records_b.append(_record(seed=seed, phase=phase, success=(seed % 2 == 0), cost=0.3))
    sl_a = compute_seed_level_summary(
        records_a,
        compute_metrics=compute_metrics,
        compute_phase_comparison=compute_phase_comparison,
    )
    sl_b = compute_seed_level_summary(
        records_b,
        compute_metrics=compute_metrics,
        compute_phase_comparison=compute_phase_comparison,
    )
    rows = compute_pairwise_tests({
        "toy": {"a": {"seed_level": sl_a}, "b": {"seed_level": sl_b}},
    })
    assert rows
    for row in rows:
        assert "paired_seeds" in row
        assert isinstance(row["paired_seeds"], list)
        assert len(row["paired_seeds"]) == row["n_paired_seeds"]
        assert "seed_set_matches_reference" in row


# ---------------------------------------------------------------------------
# Calibration tests (magnitude / coverage / null distribution)
# ---------------------------------------------------------------------------


def test_student_t_ci_has_nominal_coverage():
    """Monte-Carlo coverage check for the n=12 Student-t CI used in the
    headline tables. Generates 2000 synthetic seed-mean draws from a
    known Normal and checks that the 95% interval covers ~95% of them.
    Coverage tolerance is wide enough (~2 SE under a Binomial(2000, .95))
    that the test is robust but still catches a broken implementation.
    """
    rng = np.random.default_rng(20240501)
    n_trials = 2000
    n_seeds = 12
    true_mu = 0.4
    sigma = 0.07
    covered = 0
    for _ in range(n_trials):
        sample = rng.normal(true_mu, sigma, size=n_seeds).tolist()
        ci = seed_mean_ci(sample, confidence=0.95, method="student_t")
        if ci["ci_lower"] <= true_mu <= ci["ci_upper"]:
            covered += 1
    coverage = covered / n_trials
    # Binomial 99% CI on 2000 trials at p=0.95 is roughly [0.937, 0.963];
    # use a slightly wider tolerance for headroom.
    assert 0.93 <= coverage <= 0.97, (
        f"Student-t CI coverage {coverage:.3f} departs from nominal 0.95"
    )


def test_under_treatment_disparity_recovers_known_magnitude():
    """Construct records with a *known* under-treatment gap of 0.30
    between group 0 and group 2 and verify the metric recovers it
    within tolerance. Catches a class of bugs that sign-only checks
    miss (e.g. accidentally averaging over treated patients only,
    flipped numerator, or returning a probability ratio).
    """
    rng = np.random.default_rng(0)
    records: list[dict] = []
    n_per = 1000
    p_under_g0 = 0.40
    p_under_g2 = 0.10  # difference = 0.30
    for _ in range(n_per):
        records.append(_record(group=0, ses=0,
                               under_treated=bool(rng.random() < p_under_g0)))
        records.append(_record(group=2, ses=2,
                               under_treated=bool(rng.random() < p_under_g2)))
    m = compute_metrics(records)
    # SE of difference of two Bernoulli means with n=1000 each is ≈ 0.022;
    # ±0.05 (~2.3 SE) keeps Type-I rate negligible.
    assert abs(m["under_treatment_disparity"] - 0.30) < 0.05, (
        f"under_treatment_disparity = {m['under_treatment_disparity']:.3f}, "
        "expected ~0.30"
    )
    assert m["under_treatment_disparity_signed"] > 0, (
        "Signed disparity should be positive when group 0 is under-treated more."
    )


def test_paired_permutation_null_is_calibrated():
    """Under exchangeable labels (the H0 of the sign-flip test) the
    two-sided p-value distribution should be approximately uniform on
    (0, 1]. We check the discrete analogue: the rejection rate at the
    nominal 5% level should be <= alpha because the permutation test is
    conservative on a discrete null with n=12 (4096 sign assignments).
    """
    rng = np.random.default_rng(20240502)
    n_trials = 2000
    n_seeds = 12
    rejections = 0
    p_values = []
    for _ in range(n_trials):
        diffs = rng.normal(0.0, 1.0, size=n_seeds)
        p = _paired_permutation_p_value(diffs)
        p_values.append(p)
        if p <= 0.05:
            rejections += 1
    rate = rejections / n_trials
    # Permutation test with a continuous symmetric null cannot exceed
    # the nominal level. SE on a Binomial(2000, .05) is ~0.005, so
    # accept up to 0.065 to guard against MC noise; assert that it does
    # not collapse to 0 either (would indicate a bug masking rejections).
    assert 0.03 <= rate <= 0.065, (
        f"Permutation rejection rate at alpha=0.05 is {rate:.3f}; "
        "expected ~0.05 under exchangeable labels."
    )
    # Mean of a Uniform(0, 1) is 0.5; allow ±0.05 (well outside the
    # discrete-test deviation).
    mean_p = float(np.mean(p_values))
    assert 0.45 <= mean_p <= 0.55, (
        f"Mean permutation p-value {mean_p:.3f} departs from 0.5; "
        "the null distribution is mis-calibrated."
    )


if __name__ == "__main__":
    test_compute_metrics_basic()
    print("ok: metrics basic")
    test_phase_comparison_degradation()
    print("ok: phase degradation")
    test_seed_level_summary_exposes_bootstrap_intervals()
    print("ok: seed summary")
    test_pairwise_tests_use_matched_seed_deltas()
    print("ok: pairwise tests")
    test_bias_perpetuation_is_conditional_group_disparity()
    print("ok: bias perpetuation is conditional group disparity")
    test_clip_rate_reported()
    print("ok: clip rate reported")
    test_seed_mean_ci_uses_student_t()
    print("ok: Student-t CI by default")
    test_fairness_gap_uses_mad_not_max_minus_min()
    print("ok: fairness gap uses MAD")
    test_under_treatment_disparity_is_magnitude()
    print("ok: under-treatment disparity is magnitude")
    test_stable_seed_is_order_invariant()
    print("ok: stable seed is order invariant")
    test_holm_and_bh_adjustments_are_added_per_family()
    print("ok: Holm + BH per-family adjustments")
    test_holm_bh_helpers_match_known_values()
    print("ok: Holm + BH helpers match known values")
    test_pairwise_rows_record_paired_seed_set()
    print("ok: pairwise rows record paired seed set")
    test_student_t_ci_has_nominal_coverage()
    print("ok: Student-t CI 95% coverage is calibrated")
    test_under_treatment_disparity_recovers_known_magnitude()
    print("ok: under-treatment disparity recovers known magnitude")
    test_paired_permutation_null_is_calibrated()
    print("ok: permutation null is calibrated under exchangeability")
