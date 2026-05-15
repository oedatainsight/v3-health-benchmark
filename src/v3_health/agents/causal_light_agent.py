"""
Causal-Light Agent (registered as 'causal_light').

A deliberately stripped-down causal agent that keeps only the *structural
prior* — the policy that demographic / SES proxies must not drive
treatment — and discards every adaptive component of the full
``CausalAgent``:

* no symptom-proxy stratification or stability detection,
* no stratum-conditioned action-success override,
* no MNAR-aware risk bump,
* no online learning of feature effects.

It exists as an ablation: it isolates the contribution of the structural
prior alone from the contribution of the data-adaptive machinery in the
full causal agent. Its decision rule is:

    clinical_estimate = mean(presenting_severity, observed lab values)
    score = 0.7 * clinical_estimate + 0.3 * observed_risk_score
    action = threshold(score)

Demographics, prior utilization, and any optional group label are never
consulted.
"""

from __future__ import annotations

import numpy as np

from .base_agent import BaseAgent
from v3_health.core.types import AgentObservation
from v3_health.core.config import AGENT_HYPERPARAMS as _HP


_THRESHOLDS = tuple(_HP["action_thresholds"])
_RISK_BLEND = float(_HP["causal_light_risk_blend"])


def _score_to_action(score: float) -> int:
    if score < _THRESHOLDS[0]:
        return 0
    if score < _THRESHOLDS[1]:
        return 1
    if score < _THRESHOLDS[2]:
        return 2
    return 3


class CausalLightAgent(BaseAgent):
    """Structural-prior-only ablation of the full causal agent."""

    def __init__(self):
        super().__init__("causal_light")

    def decide(self, obs: AgentObservation) -> tuple[int, str]:
        patient = obs.patient
        steps: list[str] = ["policy: demographics + utilization excluded by structural prior"]

        symptom = float(patient.presenting_complaint_severity)
        lab_vals = [float(v) for v in patient.lab_results.values() if v is not None]
        if lab_vals:
            clinical = float(np.clip(0.5 * (symptom + float(np.mean(lab_vals))), 0.0, 1.0))
        else:
            clinical = symptom
        steps.append(
            f"clinical_estimate={clinical:.2f} (symptom + {len(lab_vals)} labs)"
        )

        risk = float(patient.observed_risk_score)
        score = float(np.clip((1.0 - _RISK_BLEND) * clinical + _RISK_BLEND * risk, 0.0, 1.0))
        action = _score_to_action(score)
        steps.append(f"score={score:.2f} -> action={action}")
        return action, " | ".join(steps)
