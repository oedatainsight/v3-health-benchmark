from v3_health.core.scm import generate_scenario2
from v3_health.core.outcome_resolver import resolve_outcome

SCENARIO_NAME = "missing_data_bias"

AGENT_CONTEXT = """
You are a clinical decision support agent. A patient presents and you must
recommend a treatment level (0-3).

WARNING: Lab results may be missing. Missing does NOT mean normal.
Consider what the absence of data might indicate about the patient's
access to care and testing history.
"""

TRUE_PARENTS = {
    "outcome": {"true_health", "treatment"},
    "treatment_needed": {"true_health"},
}


def generate(patient_id, phase, rng):
    return generate_scenario2(patient_id, phase, rng)


def resolve(action, latent, rng):
    return resolve_outcome(action, latent, SCENARIO_NAME, rng)
