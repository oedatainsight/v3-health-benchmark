"""
Policy-Constrained Agent (registered as 'baseline').

Represents a realistic production deployment pattern for a clinical
decision-support agent: a model wrapped in explicit policy guardrails.

Design intent
-------------
This agent does not attempt clever inference. It mirrors how a deployed
risk-score-driven decision aid actually behaves in practice:

* It uses the institution's risk score as the primary signal.
* It enforces an explicit *permission policy*: it never uses demographic
  proxies for treatment decisions, even if they are present in the
  observation. This mirrors equity-policy guardrails.
* It has explicit *fail states*. When data is too incomplete to act on
  safely, or when the risk score and presenting symptoms disagree
  strongly, it escalates to a conservative default ("standard care")
  and weights toward the more severe of the available signals.
* It applies a small, bounded *adaptive caution* term when its recent
  decisions have been failing. This is not learning; it is a feedback
  control loop on the trust placed in the score, similar to a clinician
  becoming more cautious when the model has been miscalibrated lately.

This agent has no causal model and no online relearning of feature
importance. It is the strong, realistic baseline that a causal agent
must beat.
"""

from __future__ import annotations

from .base_agent import BaseAgent
from v3_health.core.types import AgentObservation
from v3_health.core.config import AGENT_HYPERPARAMS as _HP


# Treatment-level thresholds (score in [0, 1] -> action 0..3)
_THRESHOLDS = tuple(_HP["baseline_action_thresholds"])
# Action constants
_MONITOR, _STANDARD, _AGGRESSIVE, _CRITICAL = 0, 1, 2, 3
# Policy parameters
_MAX_MISSING_BEFORE_FAILSTATE = float(_HP["baseline_max_missing_before_failstate"])
_DISAGREEMENT_ESCALATION = float(_HP["baseline_disagreement_escalation"])
_FAILSTATE_SAFETY_MARGIN = float(_HP["baseline_failstate_safety_margin"])
_ADAPTIVE_WINDOW = int(_HP["baseline_adaptive_window"])
_ADAPTIVE_FAILURE_TRIGGER = float(_HP["baseline_adaptive_failure_trigger"])
_ADAPTIVE_CAUTION_BUMP = float(_HP["baseline_adaptive_caution_bump"])
_DECAY_TOTAL_THRESHOLD = int(_HP["baseline_decay_total_threshold"])
_DECAY_FACTOR = float(_HP["baseline_decay_factor"])


def _score_to_action(score: float) -> int:
    if score < _THRESHOLDS[0]:
        return _MONITOR
    if score < _THRESHOLDS[1]:
        return _STANDARD
    if score < _THRESHOLDS[2]:
        return _AGGRESSIVE
    return _CRITICAL


class BaselineAgent(BaseAgent):
    """Policy-constrained risk-score agent with explicit guardrails."""

    def __init__(self):
        super().__init__("baseline")
        self._recent_failures = 0
        self._recent_total = 0

    def decide(self, obs: AgentObservation) -> tuple[int, str]:
        patient = obs.patient
        steps: list[str] = ["policy: demographics excluded from decision"]

        risk = float(patient.observed_risk_score)
        symptom = float(patient.presenting_complaint_severity)

        n_total = len(patient.lab_results)
        n_missing = int(patient.n_missing_labs)
        missing_rate = (n_missing / n_total) if n_total else 1.0

        # Fail-state: too little data to trust risk score.
        if missing_rate > _MAX_MISSING_BEFORE_FAILSTATE:
            steps.append(
                f"FAILSTATE: missing_rate={missing_rate:.0%} "
                f"> {_MAX_MISSING_BEFORE_FAILSTATE:.0%}; relying on symptoms"
            )
            score = min(1.0, symptom + _FAILSTATE_SAFETY_MARGIN)
            action = max(_STANDARD, _score_to_action(score))
            steps.append(
                f"safe_default>=standard | score={score:.2f} -> action={action}"
            )
            return action, " | ".join(steps)

        # Escalation: risk score and symptoms disagree strongly.
        disagreement = abs(risk - symptom)
        if disagreement > _DISAGREEMENT_ESCALATION:
            steps.append(
                f"ESCALATE: risk/symptom disagree by {disagreement:.2f}"
            )
            # Conservative resolution: trust whichever signal is more severe.
            score = max(risk, symptom)
            action = _score_to_action(score)
            steps.append(f"max_signal={score:.2f} -> action={action}")
            return action, " | ".join(steps)

        # Adaptive caution: bias up if recent decisions are failing often.
        adjusted_risk = risk
        if self._recent_total >= _ADAPTIVE_WINDOW:
            failure_rate = self._recent_failures / self._recent_total
            if failure_rate > _ADAPTIVE_FAILURE_TRIGGER:
                adjusted_risk = min(1.0, risk + _ADAPTIVE_CAUTION_BUMP)
                steps.append(
                    f"adaptive: failure_rate={failure_rate:.0%}, "
                    f"caution+{_ADAPTIVE_CAUTION_BUMP:.2f}"
                )

        action = _score_to_action(adjusted_risk)
        steps.append(f"risk={adjusted_risk:.2f} -> action={action}")
        return action, " | ".join(steps)

    def update(self, observation, action, outcome):
        super().update(observation, action, outcome)
        self._recent_total += 1
        if not outcome.get("success", True):
            self._recent_failures += 1
        # Sliding decay so the controller adapts within a phase
        # but does not lock into early bad luck.
        if self._recent_total > _DECAY_TOTAL_THRESHOLD:
            self._recent_total = int(self._recent_total * _DECAY_FACTOR)
            self._recent_failures = int(self._recent_failures * _DECAY_FACTOR)

    def reset(self):
        super().reset()
        self._recent_failures = 0
        self._recent_total = 0
