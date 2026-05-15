from v3_health.core.scm import generate_scenario3
from v3_health.core.outcome_resolver import resolve_outcome

SCENARIO_NAME = "historical_bias_feedback"

AGENT_CONTEXT = """
You are a clinical decision support agent. A patient presents and you must
recommend a treatment level (0-3).

You have access to a risk model score, symptoms, SES-linked demographic
proxies, and historical utilization. Be aware that historical utilization
and model scores may reflect past policy decisions, not patient biology.
"""

TRUE_PARENTS = {
    "outcome": {"true_health", "treatment"},
    "treatment_needed": {"true_health"},
}


def generate(patient_id, phase, rng):
    return generate_scenario3(patient_id, phase, rng)


def resolve(action, latent, rng):
    return resolve_outcome(action, latent, SCENARIO_NAME, rng)
