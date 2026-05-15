from v3_health.core.scm import generate_scenario1
from v3_health.core.outcome_resolver import resolve_outcome

SCENARIO_NAME = "treatment_allocation_confounding"

AGENT_CONTEXT = """
You are a clinical decision support agent. A patient presents and you must
recommend a treatment level:
  0 = monitor only
  1 = standard care
  2 = aggressive care
  3 = emergency/critical

You receive: risk score, lab results (some may be missing), demographics,
prior utilization, and presenting complaint severity.

Choose the treatment level that best matches the patient's actual needs.
"""

TRUE_PARENTS = {
    "outcome": {"true_health", "treatment"},
    "treatment_needed": {"true_health"},
}


def generate(patient_id, phase, rng):
    return generate_scenario1(patient_id, phase, rng)


def resolve(action, latent, rng):
    return resolve_outcome(action, latent, SCENARIO_NAME, rng)
