"""
Structural Causal Model for Healthcare Benchmark.

DAG (all scenarios share core structure):

    SES -----+--------------------------+
             |                          |
             v                          v
       access_to_care --+         measurement_prob
             |          |                |
             v          v                v
      treatment_recv  true_health   observed_features
             |          |           (noisy, missing)
             +----+-----+
                  v
                outcome

    historical_policy_artifacts --> utilization / model_risk  (Scenario 3 only)

INVARIANT across all phases and scenarios:
    outcome = f(treatment_match, true_health)
    outcome != f(SES)   given correct treatment
    outcome != f(group) given correct treatment
"""

import numpy as np

from .types import LatentPatientState, ObservedPatient
from .config import HEALTHCARE_CONFIG as C


def _compute_optimal_treatment(true_health: float) -> int:
    """Deterministic mapping from true health state to optimal treatment.

    Cutpoints come from ``HEALTHCARE_CONFIG['optimal_treatment_thresholds']``
    so the latent-state centers used by the structural-causal agent (the
    bin midpoints of these thresholds) cannot drift out of sync.
    """
    thresholds = C["optimal_treatment_thresholds"]
    for level, cutoff in enumerate(thresholds):
        if true_health < cutoff:
            return level
    return len(thresholds)


def optimal_treatment_state_centers() -> tuple[float, ...]:
    """Bin midpoints of ``optimal_treatment_thresholds`` over [0, 1].

    Used by the structural-causal agent so its Gaussian latent-state
    means are derived from the same thresholds that define the optimal-
    treatment label.
    """
    edges = (0.0, *C["optimal_treatment_thresholds"], 1.0)
    return tuple(0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1))


def _generate_base_patient(rng: np.random.Generator) -> tuple[int, float, int]:
    """Generate SES, true_health, and protected-group attribute (shared across scenarios).

    The protected-group attribute is sampled independently of SES so that
    group-conditional metrics are not collinear with the SES fairness gap.
    """
    ses = int(rng.choice([0, 1, 2], p=C["ses_distribution"]))
    true_health = float(rng.beta(C["health_beta_a"], C["health_beta_b"]))
    group = int(rng.choice([0, 1], p=C["group_distribution"]))
    return ses, true_health, group


def _generate_access(ses: int, rng: np.random.Generator) -> float:
    base = C["access_base_by_ses"][ses]
    return float(np.clip(base + rng.normal(0, C["access_noise_std"]), 0, 1))


def _generate_measurement_prob(access: float, rng: np.random.Generator) -> float:
    mp = (
        C["measurement_base"]
        + C["measurement_access_weight"] * access
        + rng.normal(0, C["measurement_noise_std"])
    )
    return float(np.clip(mp, 0.1, 0.99))


def _generate_demographics(ses: int, rng: np.random.Generator) -> dict[str, float]:
    return {
        "insurance_tier": float(np.clip(ses / 2.0 + rng.normal(0, 0.05), 0, 1)),
        "zip_deprivation": float(np.clip(1.0 - ses / 2.0 + rng.normal(0, 0.08), 0, 1)),
        "education_proxy": float(np.clip(ses / 2.0 + rng.normal(0, 0.10), 0, 1)),
    }


# ---------------------------------------------
# SCENARIO 1: Treatment Allocation Confounding
# ---------------------------------------------

def generate_scenario1(
    patient_id: int,
    phase: str,
    rng: np.random.Generator,
) -> tuple[LatentPatientState, ObservedPatient]:
    """
    Confounding: SES affects both observed_risk_score AND treatment access.
    Creates spurious correlation between observed risk and outcomes.
    A small group-linked additive bias on the observed risk score is also
    injected so the protected-group fairness metrics are populated; group
    never enters the outcome law.
    """
    ses, true_health, group = _generate_base_patient(rng)
    access = _generate_access(ses, rng)
    measurement_prob = _generate_measurement_prob(access, rng)
    optimal = _compute_optimal_treatment(true_health)

    presenting = float(np.clip(true_health + rng.normal(0, 0.15), 0, 1))

    lab_specs = {
        "blood_pressure": (0.3, 0.5, 0.10),
        "glucose":        (0.2, 0.4, 0.12),
        "creatinine":     (0.1, 0.6, 0.08),
        "hemoglobin":     (0.8, -0.3, 0.10),
        "inflammatory":   (0.1, 0.7, 0.15),
    }
    labs: dict[str, float | None] = {}
    n_missing = 0
    for name, (intercept, slope, noise) in lab_specs.items():
        true_val = intercept + slope * true_health + rng.normal(0, noise)
        if rng.random() < measurement_prob:
            labs[name] = float(np.clip(true_val, 0, 1))
        else:
            labs[name] = None
            n_missing += 1

    if phase == "normal":
        ses_bias = C["s1_ses_bias_normal"][ses]
        noise = C["s1_risk_noise_normal"]
        group_bias = C["group_risk_bias_normal"][group]
    elif phase == "surface_shift":
        ses_bias = 0.0
        noise = C["s1_risk_noise_shift"]
        group_bias = C["group_risk_bias_normal"][group]
    elif phase == "adversarial":
        ses_bias = C["s1_ses_bias_adversarial"][ses]
        noise = C["s1_risk_noise_adversarial"]
        group_bias = C["group_risk_bias_adversarial"][group]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    observed_risk = float(np.clip(
        true_health + ses_bias + group_bias + rng.normal(0, noise), 0, 1
    ))
    prior_util = float(np.clip(0.2 + 0.6 * access + rng.normal(0, 0.1), 0, 1))

    latent = LatentPatientState(
        patient_id=patient_id, ses=ses, true_health=true_health,
        access_to_care=access, measurement_prob=measurement_prob,
        optimal_treatment=optimal, group=group,
    )
    observed = ObservedPatient(
        patient_id=patient_id, observed_risk_score=observed_risk,
        lab_results=labs, demographics=_generate_demographics(ses, rng),
        prior_utilization=prior_util, presenting_complaint_severity=presenting,
        n_missing_labs=n_missing, group=group,
    )
    return latent, observed


# ---------------------------------------------
# SCENARIO 2: Missing Data Bias (MNAR)
# ---------------------------------------------

def generate_scenario2(
    patient_id: int,
    phase: str,
    rng: np.random.Generator,
) -> tuple[LatentPatientState, ObservedPatient]:
    """
    Missing Not At Random: low-SES patients have fewer tests.
    Missing labs look like "normal" to naive agents.
    """
    ses, true_health, group = _generate_base_patient(rng)
    access = _generate_access(ses, rng)
    optimal = _compute_optimal_treatment(true_health)

    if phase == "normal":
        measurement_prob = 0.2 + 0.7 * access
    elif phase == "surface_shift":
        symptom_driven = 0.3 * true_health
        measurement_prob = 0.2 + 0.4 * access + symptom_driven
    elif phase == "adversarial":
        ses_penalty = C["s2_adversarial_ses_penalty"][ses]
        measurement_prob = 0.2 + 0.5 * access - ses_penalty
    else:
        raise ValueError(f"Unknown phase: {phase}")

    measurement_prob = float(np.clip(measurement_prob + rng.normal(0, 0.05), 0.05, 0.95))

    labs: dict[str, float | None] = {}
    n_missing = 0
    for lab_name in C["s2_lab_names"]:
        true_val = 0.2 + 0.6 * true_health + rng.normal(0, 0.12)
        if rng.random() < measurement_prob:
            labs[lab_name] = float(np.clip(true_val, 0, 1))
        else:
            labs[lab_name] = None
            n_missing += 1

    available = [v for v in labs.values() if v is not None]
    if available:
        observed_risk = float(np.mean(available))
    else:
        observed_risk = C["s2_default_risk_when_no_labs"]

    # Phase-dependent group-linked bias on the observed risk aggregate.
    if phase == "adversarial":
        group_bias = C["group_risk_bias_adversarial"][group]
    else:
        group_bias = C["group_risk_bias_normal"][group]
    observed_risk = float(np.clip(observed_risk + group_bias, 0, 1))

    presenting = float(np.clip(true_health + rng.normal(0, 0.15), 0, 1))
    prior_util = float(np.clip(0.2 + 0.6 * access + rng.normal(0, 0.1), 0, 1))

    latent = LatentPatientState(
        patient_id=patient_id, ses=ses, true_health=true_health,
        access_to_care=access, measurement_prob=measurement_prob,
        optimal_treatment=optimal, group=group,
    )
    observed = ObservedPatient(
        patient_id=patient_id, observed_risk_score=observed_risk,
        lab_results=labs, demographics=_generate_demographics(ses, rng),
        prior_utilization=prior_util, presenting_complaint_severity=presenting,
        n_missing_labs=n_missing, group=group,
    )
    return latent, observed


# ---------------------------------------------
# SCENARIO 3: Historical SES Bias in Utilization / Model Risk
# ---------------------------------------------

def generate_scenario3(
    patient_id: int,
    phase: str,
    rng: np.random.Generator,
) -> tuple[LatentPatientState, ObservedPatient]:
    """
    Historical SES-linked policy artifacts distort recorded utilization and
    model risk. Low-SES patients appear riskier and less historically served
    even when their true health is held fixed.
    """
    ses, true_health, group = _generate_base_patient(rng)
    access = _generate_access(ses, rng)
    measurement_prob = _generate_measurement_prob(access, rng)
    optimal = _compute_optimal_treatment(true_health)

    if phase == "normal":
        util_penalty = C["s3_ses_util_penalty_normal"][ses]
        risk_bias = C["s3_ses_risk_bias_normal"][ses]
        group_bias = C["group_risk_bias_normal"][group]
    elif phase == "surface_shift":
        util_penalty = C["s3_ses_util_penalty_surface_shift"][ses]
        risk_bias = C["s3_ses_risk_bias_normal"][ses]
        group_bias = C["group_risk_bias_normal"][group]
    elif phase == "adversarial":
        util_penalty = C["s3_ses_util_penalty_adversarial"][ses]
        risk_bias = C["s3_ses_risk_bias_adversarial"][ses]
        group_bias = C["group_risk_bias_adversarial"][group]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    model_risk = float(np.clip(
        true_health + risk_bias + group_bias + rng.normal(0, 0.10), 0, 1
    ))

    hist_util = float(np.clip(
        0.2 + 0.6 * access - util_penalty + 0.2 * true_health + rng.normal(0, 0.1),
        0, 1
    ))

    presenting = float(np.clip(true_health + rng.normal(0, 0.15), 0, 1))

    lab_specs = {
        "blood_pressure": (0.3, 0.5, 0.10),
        "glucose":        (0.2, 0.4, 0.12),
        "creatinine":     (0.1, 0.6, 0.08),
        "hemoglobin":     (0.8, -0.3, 0.10),
        "inflammatory":   (0.1, 0.7, 0.15),
    }
    labs: dict[str, float | None] = {}
    n_missing = 0
    for name, (intercept, slope, noise) in lab_specs.items():
        true_val = intercept + slope * true_health + rng.normal(0, noise)
        if rng.random() < measurement_prob:
            labs[name] = float(np.clip(true_val, 0, 1))
        else:
            labs[name] = None
            n_missing += 1

    latent = LatentPatientState(
        patient_id=patient_id, ses=ses, true_health=true_health,
        access_to_care=access, measurement_prob=measurement_prob,
        optimal_treatment=optimal, group=group,
    )
    observed = ObservedPatient(
        patient_id=patient_id, observed_risk_score=model_risk,
        lab_results=labs, demographics=_generate_demographics(ses, rng),
        prior_utilization=hist_util, presenting_complaint_severity=presenting,
        n_missing_labs=n_missing, group=group,
    )
    return latent, observed
