"""
Outcome resolution: maps (agent_action, latent_state) -> OutcomeRecord.

CRITICAL INVARIANT:
    outcome = f(treatment_match, true_health)
    outcome != f(SES | group)

Outcome model
-------------
Two models are supported, selected via ``HEALTHCARE_CONFIG['outcome_model']``:

* ``clipped_linear`` (default, original specification):
    success_prob = clip(
        base_success_at_gap
        - health_penalty_weight * true_health
        - I[under_treated] * severity_penalty * abs_gap,
        0.05, 0.95)
  where ``severity_penalty = under_treatment_severity_penalty * true_health``.
  The hard [0.05, 0.95] clip is a known compression of the tails; we record
  ``success_prob_clipped`` per outcome so the headline numbers can be re-read
  with knowledge of how often the clip fired.

  *Asymmetry note (intentional, configurable).* Under-treating a sick patient
  reduces probability of success. Over-treating, by default, only raises
  cost; success probability is unchanged. This is a defensible clinical
  prior (failing to treat a critical patient is worse than over-treating),
  but it materially advantages agents that bias upward (red-flag overrides,
  MNAR risk bumps, severity-driven escalation). To run a sensitivity
  analysis without that advantage, set
  ``HEALTHCARE_CONFIG['outcome_symmetric_severity_penalty'] = True`` and the
  same severity penalty will apply to over-treatment.

* ``logistic``:
    eta = logistic_intercept
          - logistic_gap_penalty * abs_gap
          - logistic_health_penalty * true_health
          - I[under_treated] * logistic_undertreat_penalty * abs_gap * true_health
          - I[over_treated and symmetric] * logistic_undertreat_penalty * abs_gap * true_health
    success_prob = sigmoid(eta)
  Bounded by construction; no clip; used for sensitivity reporting.
"""

import math

import numpy as np

from .types import LatentPatientState, OutcomeRecord
from .config import HEALTHCARE_CONFIG as C


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def resolve_outcome(
    action: int,
    latent: LatentPatientState,
    scenario: str,
    rng: np.random.Generator,
) -> OutcomeRecord:
    """Resolve treatment action against ground truth."""

    optimal = latent.optimal_treatment
    gap = action - optimal
    abs_gap = abs(gap)

    under_treated = gap < 0
    over_treated = gap > 0

    symmetric_penalty = bool(C.get("outcome_symmetric_severity_penalty", False))
    model = str(C.get("outcome_model", "clipped_linear")).lower()
    severity_active = under_treated or (over_treated and symmetric_penalty)

    if model == "logistic":
        eta = (
            float(C["logistic_intercept"])
            - float(C["logistic_gap_penalty"]) * abs_gap
            - float(C["logistic_health_penalty"]) * latent.true_health
        )
        if severity_active:
            eta -= float(C["logistic_undertreat_penalty"]) * abs_gap * latent.true_health
        success_prob = _sigmoid(eta)
        clip_active = False
    else:
        base_success = C["success_probs_by_gap"].get(abs_gap, 0.10)
        health_penalty = C["health_penalty_weight"] * latent.true_health
        raw_prob = base_success - health_penalty
        if severity_active:
            severity_penalty = C["under_treatment_severity_penalty"] * latent.true_health
            raw_prob -= severity_penalty * abs_gap
        clipped = float(np.clip(raw_prob, 0.05, 0.95))
        clip_active = bool(clipped != raw_prob)
        success_prob = clipped

    success = bool(rng.random() < success_prob)

    cost = C["treatment_costs"].get(action, 1.0)
    if over_treated:
        cost *= C["over_treatment_cost_multiplier"]

    near_optimal = abs_gap <= 1

    return OutcomeRecord(
        patient_id=latent.patient_id,
        action=action,
        success=success,
        near_optimal=near_optimal,
        treatment_gap=abs_gap,
        base_success_prob=success_prob,
        true_health=latent.true_health,
        ses=latent.ses,
        group=latent.group,
        under_treated=under_treated,
        over_treated=over_treated,
        success_prob_clipped=clip_active,
        cost=cost,
    )
