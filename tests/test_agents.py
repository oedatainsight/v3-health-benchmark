"""Agent-level behavioral tests for v3_health."""

import numpy as np

from v3_health.agents.baseline_agent import BaselineAgent
from v3_health.agents.workflow_agent import WorkflowAgent
from v3_health.agents.causal_agent import CausalAgent
from v3_health.agents.causal_light_agent import CausalLightAgent
from v3_health.agents.structural_causal_agent import (
    StructuralCausalAgent,
    _STATE_CENTERS,
)
from v3_health.core.scm import optimal_treatment_state_centers
from v3_health.core.types import AgentObservation, ObservedPatient


def _obs(
    *,
    risk: float = 0.2,
    symptom: float = 0.2,
    labs: dict[str, float | None] | None = None,
    prior_utilization: float = 0.5,
    demographics: dict[str, float] | None = None,
    n_missing_labs: int | None = None,
    group: int | None = None,
) -> AgentObservation:
    if labs is None:
        labs = {
            "lab_a": 0.2,
            "lab_b": 0.2,
            "lab_c": 0.2,
            "lab_d": 0.2,
        }
    if demographics is None:
        demographics = {
            "insurance_tier": 0.5,
            "zip_deprivation": 0.5,
            "education_proxy": 0.5,
        }
    if n_missing_labs is None:
        n_missing_labs = sum(v is None for v in labs.values())

    patient = ObservedPatient(
        patient_id=1,
        observed_risk_score=risk,
        lab_results=dict(labs),
        demographics=dict(demographics),
        prior_utilization=prior_utilization,
        presenting_complaint_severity=symptom,
        n_missing_labs=n_missing_labs,
        group=group,
    )
    return AgentObservation(
        patient=patient,
        scenario="toy",
        phase="normal",
    )


def test_baseline_failstate_uses_symptom_guardrail():
    agent = BaselineAgent()
    obs = _obs(
        risk=0.05,
        symptom=0.82,
        labs={"lab_a": None, "lab_b": None, "lab_c": 0.2, "lab_d": None},
    )

    action, reasoning = agent.decide(obs)

    assert action >= 1
    assert "FAILSTATE" in reasoning
    assert "relying on symptoms" in reasoning


def test_workflow_reweight_prefers_predictive_signal():
    agent = WorkflowAgent()
    for _ in range(60):
        agent._records.append(
            {"symptom": 0.9, "risk": 0.1, "lab_avg": 0.2, "success": True}
        )
    for _ in range(60):
        agent._records.append(
            {"symptom": 0.1, "risk": 0.9, "lab_avg": 0.2, "success": False}
        )

    agent._reweight()

    assert agent._w_symptom > agent._w_risk
    assert agent._w_symptom > agent._w_labs


def test_causal_light_ignores_demographics_and_utilization():
    agent = CausalLightAgent()
    obs_a = _obs(
        risk=0.4,
        symptom=0.35,
        prior_utilization=0.1,
        demographics={
            "insurance_tier": 0.0,
            "zip_deprivation": 1.0,
            "education_proxy": 0.0,
        },
    )
    obs_b = _obs(
        risk=0.4,
        symptom=0.35,
        prior_utilization=0.95,
        demographics={
            "insurance_tier": 1.0,
            "zip_deprivation": 0.0,
            "education_proxy": 1.0,
        },
    )

    action_a, _ = agent.decide(obs_a)
    action_b, _ = agent.decide(obs_b)

    assert action_a == action_b


def test_causal_agent_excludes_unstable_risk_signal():
    agent = CausalAgent()
    agent.model.feature_stability["observed_risk"] = 0.0
    obs = _obs(risk=0.95, symptom=0.15, labs={"lab_a": 0.1, "lab_b": 0.2})

    action, reasoning = agent.decide(obs)

    assert action == 0
    assert "EXCLUDE observed_risk" in reasoning


def test_causal_agent_stratum_override_engages():
    agent = CausalAgent()
    agent.model.feature_stability["observed_risk"] = 1.0
    agent.model.best_action = lambda stratum, default: (2, 0.80)
    obs = _obs(risk=0.35, symptom=0.32, labs={"lab_a": 0.3, "lab_b": 0.3})

    action, reasoning = agent.decide(obs)

    assert action == 2
    assert "stratum override" in reasoning


def test_structural_state_centers_are_not_scm_threshold_midpoints():
    assert not np.allclose(_STATE_CENTERS, optimal_treatment_state_centers())
    assert np.allclose(_STATE_CENTERS, np.asarray([0.125, 0.375, 0.625, 0.875]))


def test_structural_agent_escalates_on_critical_posterior_mass():
    agent = StructuralCausalAgent()
    agent.model.belief = lambda patient: {
        "posterior": np.asarray([0.10, 0.15, 0.20, 0.55], dtype=float),
        "symptom": 0.45,
        "lab_mean": 0.42,
        "lab_count": 3,
        "clinical_anchor": 0.44,
        "access_score": 0.50,
        "avail_rate": 0.75,
        "access_proxy": 0.50,
        "raw_risk": 0.46,
        "risk_adjusted": 0.41,
        "risk_reliability": 0.80,
        "expected_need": 2.30,
    }
    agent.model.action_value = lambda belief, action, step: {
        "utility": 0.1 if action == 0 else 0.05,
        "expected_success": 0.1 if action == 0 else 0.05,
        "exploration_bonus": 0.0,
    }

    action, reasoning = agent.decide(_obs())

    assert action == 3
    assert "selected_action=3" in reasoning


if __name__ == "__main__":
    test_baseline_failstate_uses_symptom_guardrail()
    print("ok: baseline fail-state guardrail")
    test_workflow_reweight_prefers_predictive_signal()
    print("ok: workflow reweight prefers predictive signal")
    test_causal_light_ignores_demographics_and_utilization()
    print("ok: causal_light ignores demographics + utilization")
    test_causal_agent_excludes_unstable_risk_signal()
    print("ok: stability-filtered heuristic excludes unstable risk")
    test_causal_agent_stratum_override_engages()
    print("ok: stability-filtered heuristic applies stratum override")
    test_structural_state_centers_are_not_scm_threshold_midpoints()
    print("ok: structural state centers are data-independent")
    test_structural_agent_escalates_on_critical_posterior_mass()
    print("ok: structural agent escalates on critical posterior mass")