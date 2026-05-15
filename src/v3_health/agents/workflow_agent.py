"""
Guideline-Driven Heuristic Agent (registered as 'workflow').

Represents the common 'smarter non-causal' agent architecture:
multi-step rule-based reasoning with bounded online adaptation.

Design intent
-------------
This agent encodes the kind of structured clinical workflow used by
modern decision-support tools that go beyond a single risk score:

  Step 1. Red-flag triage from presenting symptoms.
  Step 2. Lab availability and pattern aggregation
          (mean + max severity capture).
  Step 3. Signal-concordance reasoning across symptoms,
          labs, and the institutional risk score.
  Step 4. Weighted combination using adaptive weights.
  Step 5. Missing-data caution adjustment (MNAR-aware).
  Step 6. Disagreement-driven safety override
          (when signals diverge, weight toward the most severe).

It updates its weights from outcome feedback using a simple
correlational rule: signals that empirically track success more
closely get higher weight over time. It does **not** distinguish
correlation from causation, which is its key limitation versus the
causal agent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base_agent import BaseAgent
from v3_health.core.types import AgentObservation
from v3_health.core.config import AGENT_HYPERPARAMS as _HP


_THRESHOLDS = tuple(_HP["action_thresholds"])
_RED_FLAG_SYMPTOM = float(_HP["red_flag_symptom"])
_RED_FLAG_FLOOR = float(_HP["red_flag_floor"])
_LOW_AVAIL_RATE = float(_HP["low_avail_threshold"])
_DISAGREE_SPREAD = float(_HP["disagree_spread"])
_ADAPT_EVERY = int(_HP["adapt_every"])
_ADAPT_BLEND = float(_HP["adapt_blend"])
_ADAPT_WINDOW = int(_HP["adapt_window"])
_MNAR_BASE = float(_HP["mnar_bump_workflow_base"])
_MNAR_SLOPE = float(_HP["mnar_bump_workflow_slope"])
_INIT_W_SYMPTOM = float(_HP["workflow_initial_weight_symptom"])
_INIT_W_LABS = float(_HP["workflow_initial_weight_labs"])
_INIT_W_RISK = float(_HP["workflow_initial_weight_risk"])
_LAB_MEAN_WEIGHT = float(_HP["workflow_lab_mean_weight"])
_LAB_MAX_WEIGHT = float(_HP["workflow_lab_max_weight"])
_DISAGREEMENT_MARGIN = float(_HP["workflow_disagreement_margin"])
_MIN_RECORDS_FOR_REWEIGHT = int(_HP["workflow_min_records_for_reweight"])
_MIN_SIGNAL_VALUES = int(_HP["workflow_min_signal_values"])
_MIN_SPLIT_COUNT = int(_HP["workflow_min_split_count"])
_QUALITY_BASELINE = float(_HP["workflow_quality_baseline"])
_QUALITY_GAIN = float(_HP["workflow_quality_gain"])
_QUALITY_DIFF_SCALE = float(_HP["workflow_quality_diff_scale"])


def _score_to_action(score: float) -> int:
    if score < _THRESHOLDS[0]:
        return 0
    if score < _THRESHOLDS[1]:
        return 1
    if score < _THRESHOLDS[2]:
        return 2
    return 3


class WorkflowAgent(BaseAgent):
    """Multi-step heuristic clinical workflow with adaptive weighting."""

    def __init__(self):
        super().__init__("workflow")
        self._w_symptom = _INIT_W_SYMPTOM
        self._w_labs = _INIT_W_LABS
        self._w_risk = _INIT_W_RISK
        # Bounded record buffer; only the trailing ``_ADAPT_WINDOW`` records
        # are ever read by ``_reweight``.
        self._records: deque[dict] = deque(maxlen=_ADAPT_WINDOW)

    def decide(self, obs: AgentObservation) -> tuple[int, str]:
        patient = obs.patient
        steps: list[str] = []

        symptom = float(patient.presenting_complaint_severity)
        risk = float(patient.observed_risk_score)
        steps.append(f"symptom={symptom:.2f} risk={risk:.2f}")

        # Step 1: red-flag triage
        red_flag = symptom > _RED_FLAG_SYMPTOM
        if red_flag:
            steps.append("RED_FLAG: severe presenting symptoms")

        # Step 2: lab aggregation
        labs = patient.lab_results
        avail = {k: v for k, v in labs.items() if v is not None}
        n_avail, n_total = len(avail), len(labs)
        avail_rate = (n_avail / n_total) if n_total else 0.0

        if avail:
            lab_vals = list(avail.values())
            avg_lab = float(np.mean(lab_vals))
            max_lab = float(np.max(lab_vals))
            lab_signal = _LAB_MEAN_WEIGHT * avg_lab + _LAB_MAX_WEIGHT * max_lab
            steps.append(
                f"labs {n_avail}/{n_total} avg={avg_lab:.2f} max={max_lab:.2f}"
            )
        else:
            lab_signal = None
            steps.append(f"labs 0/{n_total}")

        # Step 4: weighted combination
        if lab_signal is not None:
            total_w = self._w_symptom + self._w_labs + self._w_risk
            combined = (
                self._w_symptom * symptom
                + self._w_labs * lab_signal
                + self._w_risk * risk
            ) / total_w
        else:
            total_w = self._w_symptom + self._w_risk
            combined = (
                self._w_symptom * symptom + self._w_risk * risk
            ) / total_w

        # Step 5: missing-data caution
        # Scaled conservatively: max +0.07 at zero availability so we nudge
        # rather than forcibly escalate treatment level.
        if avail_rate < _LOW_AVAIL_RATE:
            adj = _MNAR_BASE + _MNAR_SLOPE * (1.0 - avail_rate)
            combined += adj
            steps.append(
                f"MNAR_caution=+{adj:.2f} (avail={avail_rate:.0%})"
            )

        # Step 6: disagreement-driven safety override
        signals = [symptom, risk]
        if lab_signal is not None:
            signals.append(lab_signal)
        spread = max(signals) - min(signals)
        if spread > _DISAGREE_SPREAD:
            severe = max(signals)
            if severe - _DISAGREEMENT_MARGIN > combined:
                steps.append(
                    f"signal_disagree spread={spread:.2f}; "
                    f"weight toward severe={severe:.2f}"
                )
                combined = severe - _DISAGREEMENT_MARGIN

        # Red-flag override: never under-treat severe presentations.
        if red_flag:
            combined = max(combined, _RED_FLAG_FLOOR)

        combined = float(np.clip(combined, 0.0, 1.0))
        action = _score_to_action(combined)
        steps.append(
            f"weights(s,l,r)=({self._w_symptom:.2f},"
            f"{self._w_labs:.2f},{self._w_risk:.2f}) "
            f"combined={combined:.2f} -> action={action}"
        )
        return action, " | ".join(steps)

    def update(self, observation, action, outcome):
        super().update(observation, action, outcome)
        patient = observation.patient
        self._records.append(
            {
                "symptom": float(patient.presenting_complaint_severity),
                "risk": float(patient.observed_risk_score),
                "lab_avg": (
                    float(np.mean(
                        [v for v in patient.lab_results.values()
                         if v is not None]
                    ))
                    if any(v is not None for v in patient.lab_results.values())
                    else None
                ),
                "success": bool(outcome.get("success", False)),
            }
        )
        if (
            len(self._records) >= _ADAPT_EVERY
            and len(self._records) % _ADAPT_EVERY == 0
        ):
            self._reweight()

    def _reweight(self):
        # Use the entire bounded buffer as the sliding window.
        recent = list(self._records)
        if len(recent) < _MIN_RECORDS_FOR_REWEIGHT:
            return

        def quality(key: str) -> float:
            vals = [r[key] for r in recent if r.get(key) is not None]
            succ = [
                r["success"] for r in recent if r.get(key) is not None
            ]
            if len(vals) < _MIN_SIGNAL_VALUES:
                return _QUALITY_BASELINE
            arr = np.asarray(vals)
            s = np.asarray(succ, dtype=float)
            median = float(np.median(arr))
            high = arr >= median
            low = ~high
            if high.sum() < _MIN_SPLIT_COUNT or low.sum() < _MIN_SPLIT_COUNT:
                return _QUALITY_BASELINE
            # Quality proxy: how much higher is success when this signal
            # is high vs low. Scale into [0, 1].
            diff = float(s[high].mean() - s[low].mean())
            return float(
                _QUALITY_BASELINE
                + _QUALITY_GAIN * np.clip(diff / _QUALITY_DIFF_SCALE, -1.0, 1.0)
            )

        q_s = quality("symptom")
        q_l = quality("lab_avg")
        q_r = quality("risk")
        total = q_s + q_l + q_r
        # ``quality`` returns values in [0, 1]; the only way ``total``
        # can be (near) zero is if every signal is anti-predictive of
        # success in the current window. Fall back to a uniform target
        # rather than masking that case with an epsilon.
        if total < 1e-3:
            target_s = target_l = target_r = 1.0 / 3.0
        else:
            target_s = q_s / total
            target_l = q_l / total
            target_r = q_r / total
        self._w_symptom = (
            (1 - _ADAPT_BLEND) * self._w_symptom + _ADAPT_BLEND * target_s
        )
        self._w_labs = (
            (1 - _ADAPT_BLEND) * self._w_labs + _ADAPT_BLEND * target_l
        )
        self._w_risk = (
            (1 - _ADAPT_BLEND) * self._w_risk + _ADAPT_BLEND * target_r
        )

    def reset(self):
        super().reset()
        self._w_symptom = _INIT_W_SYMPTOM
        self._w_labs = _INIT_W_LABS
        self._w_risk = _INIT_W_RISK
        self._records = deque(maxlen=_ADAPT_WINDOW)
