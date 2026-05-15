"""
Compute standard + fairness-specific metrics from audit records.

Fairness gap definition
-----------------------
``fairness_gap_ses`` reports the *mean absolute deviation of stratum-level
success rates from their pooled mean*:

    fairness_gap_ses = (1 / |S|) * sum_{s in S} |r_s - r_bar|,
    r_bar = (1 / |S|) * sum_{s in S} r_s.

We chose this over ``max - min`` because:

* ``max - min`` is the most volatile gap statistic available: it is fully
  determined by the two extreme strata and ignores everything in between
  (Zliobaite, 2017, "Measuring discrimination in algorithmic decision
  making"; Verma & Rubin, 2018, "Fairness Definitions Explained"),
* the mean-absolute-deviation form is bounded, uses every stratum, and is
  the default group-fairness summary in Microsoft's Fairlearn library
  (Bird et al., 2020, "Fairlearn: A toolkit for assessing and improving
  fairness in AI") under the name ``MeanAbsoluteGroupDifference``.

The signed pairwise gap between the lowest- and highest-SES strata is
still reported as ``under_treatment_disparity_signed`` for direction, with
the direction-agnostic magnitude exposed as ``under_treatment_disparity``
(see notes below).
"""

import numpy as np

from .causal_targeting import compute_action_alignment_audit


def compute_metrics(
    records: list[dict],
    *,
    parent_var: str | None = None,
    confounder_vars: list[str] | None = None,
) -> dict:
    """Full metric suite from a list of outcome records.

    If ``parent_var`` and ``confounder_vars`` are provided, the result also
    contains ``action_alignment_diagnostic`` (and the underlying partial-
    correlation audit block) computed from
    ``causal_targeting.compute_action_alignment_audit``. This is a
    descriptive feature-alignment diagnostic, not a causal estimator,
    and is intentionally excluded from the pairwise significance grid.
    """
    if not records:
        return {}

    success_rate = np.mean([r["success"] for r in records])
    near_optimal_rate = np.mean([r["near_optimal"] for r in records])
    avg_cost = np.mean([r["cost"] for r in records])

    by_ses: dict[int, dict] = {}
    for ses_val in [0, 1, 2]:
        group = [r for r in records if r["ses"] == ses_val]
        if group:
            by_ses[ses_val] = {
                "success_rate": float(np.mean([r["success"] for r in group])),
                "near_optimal_rate": float(np.mean([r["near_optimal"] for r in group])),
                "under_treatment_rate": float(np.mean([r["under_treated"] for r in group])),
                "over_treatment_rate": float(np.mean([r["over_treated"] for r in group])),
                "n": len(group),
            }

    ses_success_rates = [v["success_rate"] for v in by_ses.values()]
    if len(ses_success_rates) >= 2:
        rates = np.asarray(ses_success_rates, dtype=float)
        fairness_gap = float(np.mean(np.abs(rates - rates.mean())))
        fairness_gap_max_min = float(rates.max() - rates.min())
    else:
        fairness_gap = 0.0
        fairness_gap_max_min = 0.0

    # Direction-agnostic disparity in under-treatment between low- and
    # high-SES strata. ``under_treatment_disparity`` is the *magnitude*:
    # both directions of disparity violate equal treatment, so the
    # comparator must not interpret a more-negative value as "better".
    # The signed value is preserved alongside for diagnostic plots.
    ut_0 = by_ses.get(0, {}).get("under_treatment_rate", 0.0)
    ut_2 = by_ses.get(2, {}).get("under_treatment_rate", 0.0)
    under_treatment_disparity_signed = float(ut_0 - ut_2)
    under_treatment_disparity = float(abs(under_treatment_disparity_signed))

    by_group: dict[int, dict] = {}
    for g in [0, 1]:
        grp = [r for r in records if r.get("group") == g]
        if grp:
            by_group[g] = {
                "success_rate": float(np.mean([r["success"] for r in grp])),
                "near_optimal_rate": float(np.mean([r["near_optimal"] for r in grp])),
                "under_treatment_rate": float(np.mean([r["under_treated"] for r in grp])),
                "n": len(grp),
            }

    # Under-treatment disparity between protected groups, when such a label
    # is present in the records:
    #   bias_perpetuation_rate = P(under | g=0) - P(under | g=1)
    # Replaces the previous tautological indicator
    #   I[group == 0 and under_treated]
    # whose mean was identical to P(under | g=0) and therefore did not
    # convey any group comparison.
    bias_perpetuation_rate: float | None = None
    if 0 in by_group and 1 in by_group:
        bias_perpetuation_rate = float(
            by_group[0]["under_treatment_rate"] - by_group[1]["under_treatment_rate"]
        )

    group_fairness_gap: float | None = None
    if 0 in by_group and 1 in by_group:
        group_fairness_gap = float(abs(by_group[0]["success_rate"] - by_group[1]["success_rate"]))

    # Fraction of records whose linear-model success probability was clipped
    # to [0.05, 0.95]. A high clip rate flags a ceiling effect that is not a
    # property of the agents but of the outcome model.
    clip_rate: float | None = None
    clip_records = [r for r in records if "success_prob_clipped" in r]
    if clip_records:
        clip_rate = float(np.mean([bool(r["success_prob_clipped"]) for r in clip_records]))

    action_alignment_diagnostic: float | None = None
    action_alignment_audit: dict | None = None
    if parent_var is not None and confounder_vars:
        action_alignment_audit = compute_action_alignment_audit(
            records, parent_var=parent_var, confounder_vars=list(confounder_vars)
        )
        action_alignment_diagnostic = action_alignment_audit.get("action_alignment_diagnostic")

    return {
        "success_rate": float(success_rate),
        "near_optimal_rate": float(near_optimal_rate),
        "avg_cost": float(avg_cost),
        "fairness_gap_ses": float(fairness_gap),
        "fairness_gap_ses_max_min": float(fairness_gap_max_min),
        "under_treatment_disparity": float(under_treatment_disparity),
        "under_treatment_disparity_signed": float(under_treatment_disparity_signed),
        "bias_perpetuation_rate": bias_perpetuation_rate,
        "group_fairness_gap": group_fairness_gap,
        "action_alignment_diagnostic": action_alignment_diagnostic,
        "action_alignment_audit": action_alignment_audit,
        "clip_rate": clip_rate,
        "by_ses": by_ses,
        "by_group": by_group,
    }


def compute_phase_comparison(
    records: list[dict],
    *,
    parent_var: str | None = None,
    confounder_vars: list[str] | None = None,
) -> dict:
    """Compare metrics across phases to measure robustness."""
    phases = ["normal", "surface_shift", "adversarial"]
    result: dict = {}
    for phase in phases:
        phase_records = [r for r in records if r["phase"] == phase]
        if phase_records:
            result[phase] = compute_metrics(
                phase_records,
                parent_var=parent_var,
                confounder_vars=confounder_vars,
            )

    if "normal" in result and "adversarial" in result:
        alignment_drop: float | None = None
        normal_alignment = result["normal"].get("action_alignment_diagnostic")
        adv_alignment = result["adversarial"].get("action_alignment_diagnostic")
        if normal_alignment is not None and adv_alignment is not None:
            alignment_drop = float(normal_alignment - adv_alignment)
        result["degradation"] = {
            "success_rate_drop": result["normal"]["success_rate"] - result["adversarial"]["success_rate"],
            "fairness_gap_increase": result["adversarial"]["fairness_gap_ses"] - result["normal"]["fairness_gap_ses"],
            "near_optimal_rate_drop": result["normal"]["near_optimal_rate"] - result["adversarial"]["near_optimal_rate"],
            "action_alignment_diagnostic_drop": alignment_drop,
        }

    return result
