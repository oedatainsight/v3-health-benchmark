"""Structural-causal clinical agent exposed as `structural_causal`.

This agent uses an explicit latent-state causal parameterization:

    demographic proxies, utilization, lab availability -> access
    latent clinical need -> symptoms, labs, deconfounded risk
    (latent clinical need, action) -> outcome

Demographic proxies are used only inside the access adjustment model,
never as direct action features. Treatment selection is based on the
estimated posterior over a fixed latent clinical-need basis together with
learned interventional success tables E[success | do(action), latent_state].

The latent-state grid is deliberately fixed and data-independent so this
agent remains a fair observed-input competitor rather than inheriting the
simulator's optimal-treatment thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .base_agent import BaseAgent
from v3_health.core.types import AgentObservation
from v3_health.core.config import AGENT_HYPERPARAMS as _HP


_ACTIONS = (0, 1, 2, 3)
# Fixed quartile midpoints over [0, 1]. These are not derived from the
# simulator thresholds so the agent does not inherit privileged access to
# the benchmark's optimal-treatment discretization.
_STATE_CENTERS = np.asarray([0.125, 0.375, 0.625, 0.875], dtype=float)
_STATE_NAMES = ("monitor", "standard", "aggressive", "critical")
_FEATURE_STD_FLOOR = float(_HP["structural_feature_std_floor"])
_BIAS_WINDOW = int(_HP["structural_bias_window"])
_EXPLORATION_STEPS = int(_HP["structural_exploration_steps"])
_EXPLORATION_WEIGHT = float(_HP["structural_exploration_weight"])
# Uniform Beta(1, 1) priors on E[success | do(action), latent_state]:
# the agent must learn the interventional success table from observed outcomes.
_OUTCOME_PRIOR_ALPHA = float(_HP["structural_outcome_prior_alpha"])
_OUTCOME_PRIOR_BETA = float(_HP["structural_outcome_prior_beta"])
_BIAS_EWMA = float(_HP["structural_bias_ewma"])
_BIAS_CLIP = float(_HP["structural_bias_clip"])
_STATE_PRIOR_MASS = float(_HP["structural_state_prior_mass"])
_LIK_WEIGHT_SYMPTOM = float(_HP["structural_likelihood_weight_symptom"])
_LIK_WEIGHT_LAB = float(_HP["structural_likelihood_weight_lab"])
_LIK_WEIGHT_RISK = float(_HP["structural_likelihood_weight_risk"])
_CRIT_POSTERIOR_THRESHOLD = float(_HP["structural_critical_posterior_threshold"])
_HIGH_SEV_MASS_THRESHOLD = float(_HP["structural_high_severity_mass_threshold"])
_HIGH_SEV_NEED_THRESHOLD = float(_HP["structural_high_severity_need_threshold"])


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_log(value: float) -> float:
    return float(np.log(max(value, 1e-9)))


@dataclass
class GaussianAccumulator:
    mean: float
    weight: float = 2.0
    m2: float = 0.02

    def observe(self, value: float, responsibility: float) -> None:
        if responsibility <= 0.0:
            return
        total_weight = self.weight + responsibility
        delta = value - self.mean
        new_mean = self.mean + responsibility * delta / total_weight
        delta2 = value - new_mean
        self.m2 += responsibility * delta * delta2
        self.weight = total_weight
        self.mean = new_mean

    @property
    def variance(self) -> float:
        raw = self.m2 / max(self.weight, 1.0)
        return float(max(raw, _FEATURE_STD_FLOOR ** 2))

    def logpdf(self, value: float) -> float:
        var = self.variance
        return float(-0.5 * ((value - self.mean) ** 2 / var + np.log(2.0 * np.pi * var)))


class StructuralCausalModel:
    """Latent-state SCM approximation for the healthcare benchmark."""

    def __init__(self):
        self.state_mass = np.ones(len(_ACTIONS), dtype=float) * _STATE_PRIOR_MASS
        self.symptom_models = [GaussianAccumulator(center) for center in _STATE_CENTERS]
        self.lab_models = [GaussianAccumulator(center) for center in _STATE_CENTERS]
        self.risk_models = [GaussianAccumulator(center) for center in _STATE_CENTERS]
        # Flat Beta(1, 1) priors. The interventional success table is learned
        # from observed (action, success) outcomes weighted by the posterior
        # responsibility over latent states; nothing about the simulator's
        # reward function is hard-coded here.
        self.outcome_alpha = np.full(
            (len(_ACTIONS), len(_ACTIONS)), _OUTCOME_PRIOR_ALPHA, dtype=float
        )
        self.outcome_beta = np.full(
            (len(_ACTIONS), len(_ACTIONS)), _OUTCOME_PRIOR_BETA, dtype=float
        )
        self._bias_buffer: deque[tuple[float, float, float]] = deque(maxlen=_BIAS_WINDOW)
        self.risk_bias_slope = 0.0

    def _access_score(self, patient) -> tuple[float, float]:
        labs = patient.lab_results
        n_total = len(labs)
        avail_rate = ((n_total - int(patient.n_missing_labs)) / n_total) if n_total else 0.0
        insurance = float(patient.demographics.get("insurance_tier", 0.5))
        education = float(patient.demographics.get("education_proxy", 0.5))
        deprivation = float(patient.demographics.get("zip_deprivation", 0.5))
        access = _clip01(
            0.28 * float(patient.prior_utilization)
            + 0.22 * insurance
            + 0.18 * education
            + 0.22 * avail_rate
            + 0.10 * (1.0 - deprivation)
        )
        return access, avail_rate

    def _clinical_anchor(self, patient) -> tuple[float, float | None, list[float]]:
        symptom = float(patient.presenting_complaint_severity)
        lab_vals = [float(v) for v in patient.lab_results.values() if v is not None]
        lab_mean = float(np.mean(lab_vals)) if lab_vals else None
        anchor = symptom if lab_mean is None else float(0.65 * symptom + 0.35 * lab_mean)
        return anchor, lab_mean, lab_vals

    def _centered_access_proxy(self, access_proxy: float) -> float:
        if not self._bias_buffer:
            return access_proxy
        mean_proxy = float(np.mean([x for x, _, _ in self._bias_buffer]))
        return access_proxy - mean_proxy

    def _corrected_risk(self, patient, access_score: float, lab_mean: float | None) -> tuple[float, float]:
        access_proxy = 1.0 - access_score
        correction = self.risk_bias_slope * self._centered_access_proxy(access_proxy)
        corrected_risk = _clip01(float(patient.observed_risk_score) - correction)
        risk_reliability = _clip01(0.30 + 0.50 * access_score + (0.20 if lab_mean is not None else 0.0))
        return corrected_risk, risk_reliability

    def belief(self, patient) -> dict:
        symptom = float(patient.presenting_complaint_severity)
        clinical_anchor, lab_mean, lab_vals = self._clinical_anchor(patient)
        access_score, avail_rate = self._access_score(patient)
        corrected_risk, risk_reliability = self._corrected_risk(
            patient,
            access_score=access_score,
            lab_mean=lab_mean,
        )

        prior = self.state_mass / self.state_mass.sum()
        logp = np.asarray([_safe_log(p) for p in prior], dtype=float)
        for state in _ACTIONS:
            logp[state] += _LIK_WEIGHT_SYMPTOM * self.symptom_models[state].logpdf(symptom)
            if lab_mean is not None:
                logp[state] += _LIK_WEIGHT_LAB * self.lab_models[state].logpdf(lab_mean)
            logp[state] += _LIK_WEIGHT_RISK * risk_reliability * self.risk_models[state].logpdf(corrected_risk)

        logp -= float(np.max(logp))
        posterior = np.exp(logp)
        posterior /= posterior.sum()

        return {
            "posterior": posterior,
            "symptom": symptom,
            "lab_mean": lab_mean,
            "lab_count": len(lab_vals),
            "clinical_anchor": clinical_anchor,
            "access_score": access_score,
            "avail_rate": avail_rate,
            "access_proxy": 1.0 - access_score,
            "raw_risk": float(patient.observed_risk_score),
            "risk_adjusted": corrected_risk,
            "risk_reliability": risk_reliability,
            "expected_need": float(np.dot(posterior, np.asarray(_ACTIONS, dtype=float))),
        }

    def action_value(self, belief: dict, action: int, step: int) -> dict:
        posterior = belief["posterior"]
        means = self.outcome_alpha[:, action] / (self.outcome_alpha[:, action] + self.outcome_beta[:, action])
        counts = self.outcome_alpha[:, action] + self.outcome_beta[:, action]
        uncertainty = np.sqrt(1.0 / (counts + 1.0))
        expected_success = float(np.dot(posterior, means))
        exploration_bonus = float(np.dot(posterior, uncertainty))
        utility = expected_success
        if step < _EXPLORATION_STEPS:
            utility += _EXPLORATION_WEIGHT * exploration_bonus
        return {
            "utility": utility,
            "expected_success": expected_success,
            "exploration_bonus": exploration_bonus,
        }

    def update(self, patient, action: int, success: bool) -> dict:
        belief = self.belief(patient)
        posterior = belief["posterior"]
        success_value = 1.0 if success else 0.0

        self.state_mass += posterior
        for state, responsibility in enumerate(posterior):
            self.symptom_models[state].observe(belief["symptom"], float(responsibility))
            if belief["lab_mean"] is not None:
                self.lab_models[state].observe(float(belief["lab_mean"]), float(responsibility))
            self.risk_models[state].observe(
                belief["risk_adjusted"],
                float(responsibility) * float(belief["risk_reliability"]),
            )
            self.outcome_alpha[state, action] += float(responsibility) * success_value
            self.outcome_beta[state, action] += float(responsibility) * (1.0 - success_value)

        # Bias-regression update.
        # We want the *direct* path SES -> observed_risk (the spurious
        # SES-driven inflation), with the indirect SES -> true_health ->
        # observed_risk path partialled out. ``clinical_anchor`` is a
        # noisy proxy of true_health, so the previous formulation
        # ``residual = raw_risk - anchor; residual ~ access_proxy``
        # confounded the slope with proxy noise that itself correlates
        # with access (low access -> fewer labs -> anchor leans on
        # symptom only). We therefore regress
        #
        #     raw_risk_t = b0 + b_a * access_proxy_t + b_h * anchor_t + e_t
        #
        # using bivariate OLS over the running ``_BIAS_WINDOW`` and use
        # ``b_a`` as the access-conditional bias slope. The anchor is
        # included as a noisy regressor so that any health -> risk
        # signal is absorbed by ``b_h`` rather than leaking into ``b_a``.
        # This is partial-regression in the sense of Frisch-Waugh-Lovell.
        self._bias_buffer.append(
            (
                belief["access_proxy"],
                belief["clinical_anchor"],
                belief["raw_risk"],
            )
        )
        self._update_risk_bias()
        return belief

    def _update_risk_bias(self) -> None:
        if len(self._bias_buffer) < 24:
            return
        access_proxy = np.asarray(
            [pair[0] for pair in self._bias_buffer], dtype=float
        )
        anchor = np.asarray(
            [pair[1] for pair in self._bias_buffer], dtype=float
        )
        risk = np.asarray(
            [pair[2] for pair in self._bias_buffer], dtype=float
        )
        # Centered design matrix [access_proxy, anchor, 1].
        x1 = access_proxy - access_proxy.mean()
        x2 = anchor - anchor.mean()
        if float(np.dot(x1, x1)) < 1e-6:
            return
        # Closed-form 2-predictor OLS via Frisch-Waugh: regress access on
        # anchor, residualize, then regress risk on the residual.
        denom_x2 = float(np.dot(x2, x2))
        if denom_x2 < 1e-6:
            slope = float(np.dot(x1, risk - risk.mean()) / np.dot(x1, x1))
        else:
            beta_x1_on_x2 = float(np.dot(x1, x2) / denom_x2)
            x1_resid = x1 - beta_x1_on_x2 * x2
            denom = float(np.dot(x1_resid, x1_resid))
            if denom < 1e-6:
                return
            slope = float(np.dot(x1_resid, risk - risk.mean()) / denom)
        # EWMA smoothing across windows; clipping bounds the correction
        # to less than half the [0, 1] risk range.
        self.risk_bias_slope = float(
            np.clip(
                (1.0 - _BIAS_EWMA) * self.risk_bias_slope + _BIAS_EWMA * slope,
                -_BIAS_CLIP,
                _BIAS_CLIP,
            )
        )


class StructuralCausalAgent(BaseAgent):
    """Agent with an explicit latent-state structural causal parameterization."""

    def __init__(self):
        super().__init__("structural_causal")
        self.model = StructuralCausalModel()
        self.step = 0

    def decide(self, observation: AgentObservation) -> tuple[int, str]:
        belief = self.model.belief(observation.patient)
        utilities = {
            action: self.model.action_value(belief, action, self.step)
            for action in _ACTIONS
        }
        action = max(_ACTIONS, key=lambda a: utilities[a]["utility"])

        high_severity_mass = float(belief["posterior"][2] + belief["posterior"][3])
        if belief["posterior"][3] > _CRIT_POSTERIOR_THRESHOLD or (
            high_severity_mass > _HIGH_SEV_MASS_THRESHOLD
            and belief["expected_need"] > _HIGH_SEV_NEED_THRESHOLD
            and action < 2
        ):
            action = max(
                action,
                2 if belief["posterior"][3] <= _CRIT_POSTERIOR_THRESHOLD else 3,
            )

        posterior_text = ",".join(
            f"{name}={belief['posterior'][idx]:.2f}"
            for idx, name in enumerate(_STATE_NAMES)
        )
        utility_text = ",".join(
            f"a{candidate}:E={utilities[candidate]['expected_success']:.2f}/U={utilities[candidate]['utility']:.2f}"
            for candidate in _ACTIONS
        )
        reasoning = " | ".join([
            "graph: demographics/utilization/missingness->access; access->risk_bias; clinical_need->symptom,labs,adjusted_risk; (clinical_need,action)->outcome",
            f"adjustment: access={belief['access_score']:.2f} avail={belief['avail_rate']:.2f} bias_slope={self.model.risk_bias_slope:.2f}",
            f"observational: symptom={belief['symptom']:.2f} lab_mean={belief['lab_mean'] if belief['lab_mean'] is not None else 'NA'} raw_risk={belief['raw_risk']:.2f} adjusted_risk={belief['risk_adjusted']:.2f}",
            f"interventional posterior: {posterior_text}",
            f"do(action) values: {utility_text}",
            "action choice is mediated through the access-adjusted latent-state model; no action is chosen directly from demographic proxies",
            f"selected_action={action}",
        ])
        return action, reasoning

    def update(self, observation, action, outcome):
        super().update(observation, action, outcome)
        self.model.update(
            observation.patient,
            action=action,
            success=bool(outcome.get("success", False)),
        )
        self.step += 1

    def reset(self):
        super().reset()
        self.model = StructuralCausalModel()
        self.step = 0