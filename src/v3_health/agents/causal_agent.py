"""Stability-filtered heuristic agent.

The benchmark still implements this policy in ``CausalAgent`` for legacy
import compatibility, but the benchmark-facing agent name is
``stability_filtered`` because its core mechanisms are heuristic rather
than formally identified causal estimators.

The policy combines:

1. Structural exclusions: SES proxies and prior utilization are never used
    as direct treatment drivers.
2. Proxy-stratified stability auditing: observed risk can be down-weighted
    or excluded when its recent association with success drifts across
    windows.
3. Stratum-conditioned action-success tables: recent observed success rates
    within symptom strata can override the threshold action, subject to a
    safety cap. This is an observational policy table, not a treatment-
    effect estimate.
4. Informative-missingness heuristics: severe lab missingness triggers a
    bounded caution bump.

This agent does not learn a DAG, identify causal effects, or perform
counterfactual abduction. It is a structured adaptive heuristic with a
strong exclusion prior.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .base_agent import BaseAgent
from v3_health.core.types import AgentObservation
from v3_health.core.config import AGENT_HYPERPARAMS as _HP


# Three symptom strata: low / mid / high presenting severity.
_STRATUM_EDGES = tuple(_HP["stratum_edges"])


def _stratum(symptom: float) -> int:
    if symptom < _STRATUM_EDGES[0]:
        return 0
    if symptom < _STRATUM_EDGES[1]:
        return 1
    return 2


# Variables that the agent is structurally forbidden from using
# as causal drivers, regardless of empirical correlation.
_FORBIDDEN_PREFIXES = ("demo_",)
_FORBIDDEN_NAMES = {"prior_utilization"}

_THRESHOLDS = tuple(_HP["action_thresholds"])
_ACTION_SUCCESS_OVERRIDE_FLOOR = float(_HP["action_success_override_floor"])
_ACTION_SUCCESS_MAX_LEVEL_SHIFT = int(_HP["action_success_max_level_shift"])
_MIN_CELL_SAMPLES = int(_HP["causal_min_cell_samples"])
_MIN_ACTION_SUCCESS_SAMPLES = int(_HP["causal_min_action_success_samples"])
_DEFAULT_STABILITY = float(_HP["causal_default_stability"])
_STABLE_RISK_BLEND = float(_HP["causal_stable_risk_blend"])
_WARMUP_RISK_BLEND = float(_HP["causal_warmup_risk_blend"])
_UNCERTAIN_RISK_BLEND = float(_HP["causal_uncertain_risk_blend"])


def _score_to_action(score: float) -> int:
    if score < _THRESHOLDS[0]:
        return 0
    if score < _THRESHOLDS[1]:
        return 1
    if score < _THRESHOLDS[2]:
        return 2
    return 3


class CausalModel:
    """
    Tracks per-feature, per-stratum association with outcome success
    and detects unstable / confounded features by comparing two
    successive time windows.
    """

    def __init__(self, window: int | None = None):
        self.window = int(window if window is not None else _HP["causal_window"])
        self.n = 0
        # Bounded record buffer: only the last 2*window records are ever
        # read by ``_recompute``. A deque with maxlen avoids unbounded
        # memory growth across long runs.
        self._records: deque[dict] = deque(maxlen=2 * self.window)
        # Stable estimates updated when window fills.
        self.feature_effect: dict[str, float] = {}
        self.feature_stability: dict[str, float] = {}
        # Per-stratum, per-action success rate estimates.
        self.action_success: dict[tuple[int, int], list[float]] = defaultdict(
            list
        )

    def add(
        self,
        features: dict[str, float],
        stratum: int,
        action: int,
        success: bool,
    ):
        self.n += 1
        self._records.append(
            {
                "features": dict(features),
                "stratum": stratum,
                "action": action,
                "success": 1.0 if success else 0.0,
            }
        )
        self.action_success[(stratum, action)].append(
            1.0 if success else 0.0
        )
        if self.n % self.window == 0 and self.n >= 2 * self.window:
            self._recompute()

    def _stratified_effect(self, records: list[dict], feat: str) -> float | None:
        """
        Action-conditional, stratum-averaged feature effect on success.

        For each (stratum, action) cell with enough samples we compute
        the within-cell median split of ``feat`` and the contrast
        E[success | feat >= median] - E[success | feat < median]. We
        then average those cell-level contrasts across cells, weighting
        each cell by its sample count. Conditioning on action ensures
        that drift in the agent's own policy across windows does not
        masquerade as feature non-invariance.

        Returns None if no cell has enough data.
        """
        cell_contrasts: list[float] = []
        cell_weights: list[float] = []
        for s in (0, 1, 2):
            for a in (0, 1, 2, 3):
                subset = [
                    r for r in records
                    if r["stratum"] == s and r["action"] == a
                ]
                if len(subset) < _MIN_CELL_SAMPLES:
                    continue
                vals = [r["features"].get(feat) for r in subset]
                valid = [
                    (v, r["success"])
                    for v, r in zip(vals, subset)
                    if v is not None
                ]
                if len(valid) < _MIN_CELL_SAMPLES:
                    continue
                arr = np.asarray([v for v, _ in valid])
                succ = np.asarray([y for _, y in valid])
                median = float(np.median(arr))
                high_mask = arr >= median
                low_mask = ~high_mask
                min_split = max(1, _MIN_CELL_SAMPLES // 3)
                if high_mask.sum() < min_split or low_mask.sum() < min_split:
                    continue
                cell_contrasts.append(
                    float(succ[high_mask].mean() - succ[low_mask].mean())
                )
                cell_weights.append(float(len(valid)))
        if not cell_contrasts:
            return None
        weights = np.asarray(cell_weights)
        contrasts = np.asarray(cell_contrasts)
        return float(np.average(contrasts, weights=weights))

    def _recompute(self):
        recent = list(self._records)[-2 * self.window :]
        early = recent[: self.window]
        late = recent[self.window :]

        feature_keys = set()
        for r in recent:
            feature_keys.update(r["features"].keys())

        drift_norm = float(_HP["drift_normalization"])
        for feat in feature_keys:
            eff_e = self._stratified_effect(early, feat)
            eff_l = self._stratified_effect(late, feat)
            if eff_e is None or eff_l is None:
                # Default: weak prior of moderate stability.
                self.feature_stability[feat] = _DEFAULT_STABILITY
                if eff_l is not None:
                    self.feature_effect[feat] = eff_l
                continue
            self.feature_effect[feat] = eff_l
            drift = abs(eff_l - eff_e)
            self.feature_stability[feat] = float(
                max(0.0, 1.0 - min(drift / drift_norm, 1.0))
            )

    def is_stable(self, feat: str, threshold: float | None = None) -> bool:
        thr = float(threshold if threshold is not None else _HP["stable_threshold"])
        return self.feature_stability.get(feat, _DEFAULT_STABILITY) >= thr

    def is_unstable(self, feat: str, threshold: float | None = None) -> bool:
        thr = float(threshold if threshold is not None else _HP["unstable_threshold"])
        return self.feature_stability.get(feat, _DEFAULT_STABILITY) < thr

    def best_action(self, stratum: int, default: int) -> tuple[int, float]:
        """
        Return the action with highest empirical success rate in this
        stratum, plus that rate. Falls back to `default` when undersampled.
        """
        best_a = default
        best_p = -math.inf
        seen = False
        for a in (0, 1, 2, 3):
            outcomes = self.action_success.get((stratum, a), [])
            if len(outcomes) < _MIN_ACTION_SUCCESS_SAMPLES:
                continue
            seen = True
            p = float(np.mean(outcomes))
            if p > best_p:
                best_p = p
                best_a = a
        if not seen:
            return default, 0.0
        return best_a, max(0.0, best_p)


class CausalAgent(BaseAgent):
    """Stability-filtered heuristic with structural exclusions."""

    def __init__(self):
        super().__init__("stability_filtered")
        self.model = CausalModel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_features(self, patient) -> dict[str, float]:
        feats: dict[str, float] = {
            "presenting_severity": float(patient.presenting_complaint_severity),
            "observed_risk": float(patient.observed_risk_score),
            "prior_utilization": float(patient.prior_utilization),
        }
        for k, v in patient.demographics.items():
            feats[f"demo_{k}"] = float(v)
        for k, v in patient.lab_results.items():
            if v is not None:
                feats[f"lab_{k}"] = float(v)
        return feats

    def _allowed(self, name: str) -> bool:
        if name in _FORBIDDEN_NAMES:
            return False
        for pref in _FORBIDDEN_PREFIXES:
            if name.startswith(pref):
                return False
        return True

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, obs: AgentObservation) -> tuple[int, str]:
        patient = obs.patient
        steps: list[str] = []

        symptom = float(patient.presenting_complaint_severity)
        stratum = _stratum(symptom)

        # Structural prior: build a clinical estimate from policy-allowed
        # signals only (symptoms + labs). This is the causal "treatment-
        # needed" surrogate the agent is willing to act on.
        clinical_signals: list[float] = [symptom]
        labs = patient.lab_results
        lab_vals = [v for v in labs.values() if v is not None]
        if lab_vals:
            clinical_signals.append(float(np.mean(lab_vals)))
        clinical = float(np.clip(np.mean(clinical_signals), 0.0, 1.0))
        steps.append(
            f"clinical_estimate={clinical:.2f} (symptom + {len(lab_vals)} labs)"
        )

        # Audit candidate non-causal signals (e.g., observed risk score).
        # If empirically stable across windows, they may be informative;
        # if unstable, they are environment-dependent and excluded.
        risk = float(patient.observed_risk_score)
        risk_stable = self.model.is_stable("observed_risk")
        risk_unstable = self.model.is_unstable("observed_risk")
        if risk_unstable:
            steps.append("EXCLUDE observed_risk (unstable / suspected confounded)")
            score = clinical
        elif risk_stable:
            steps.append("USE observed_risk (stable across windows)")
            score = (1.0 - _STABLE_RISK_BLEND) * clinical + _STABLE_RISK_BLEND * risk
        else:
            # Insufficient evidence yet; partial trust during warmup.
            if self.model.n < self.model.window:
                steps.append("warmup: partial trust in observed_risk")
                score = (1.0 - _WARMUP_RISK_BLEND) * clinical + _WARMUP_RISK_BLEND * risk
            else:
                steps.append("uncertain observed_risk; conservative blend")
                score = (
                    (1.0 - _UNCERTAIN_RISK_BLEND) * clinical
                    + _UNCERTAIN_RISK_BLEND * risk
                )

        # MNAR-aware adjustment: high missingness is treated as informative.
        n_total = len(labs)
        n_missing = int(patient.n_missing_labs)
        mnar = (n_missing / n_total) if n_total else 0.0
        if mnar > float(_HP["mnar_threshold_causal"]):
            bump = float(_HP["mnar_bump_causal"]) * mnar
            score = float(np.clip(score + bump, 0.0, 1.0))
            steps.append(f"MNAR informative-missing +{bump:.2f}")

        # Stratum-conditioned action-success table: if we have enough data
        # within this symptom stratum to know which action has recently
        # performed best, that becomes a soft prior on the threshold action.
        thresholded = _score_to_action(score)
        emp_best, emp_rate = self.model.best_action(stratum, default=thresholded)
        if emp_best != thresholded and emp_rate > _ACTION_SUCCESS_OVERRIDE_FLOOR:
            # Prefer the empirically better action only when
            # it does not contradict the clinical estimate by more
            # than ``action_success_max_level_shift`` levels (safety cap).
            if abs(emp_best - thresholded) <= _ACTION_SUCCESS_MAX_LEVEL_SHIFT:
                steps.append(
                    f"stratum override: stratum={stratum} "
                    f"empirical_best={emp_best} (p={emp_rate:.2f})"
                )
                action = emp_best
            else:
                steps.append(
                    f"stratum override suggested {emp_best} but capped to {thresholded} "
                    f"(safety: max {_ACTION_SUCCESS_MAX_LEVEL_SHIFT} level shift)"
                )
                action = thresholded
        else:
            action = thresholded

        steps.append(
            f"score={score:.2f} stratum={stratum} -> action={action}"
        )
        return action, " | ".join(steps)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, observation, action, outcome):
        super().update(observation, action, outcome)
        patient = observation.patient
        feats = self._extract_features(patient)
        # Only audit allowed-causal candidates; demographics are not
        # tracked at all (we will not use them even if they correlate).
        audited = {k: v for k, v in feats.items() if self._allowed(k)}
        symptom = float(patient.presenting_complaint_severity)
        self.model.add(
            features=audited,
            stratum=_stratum(symptom),
            action=action,
            success=bool(outcome.get("success", False)),
        )

    def reset(self):
        super().reset()
        self.model = CausalModel()
