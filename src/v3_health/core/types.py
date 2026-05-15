from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    NORMAL = "normal"
    SURFACE_SHIFT = "surface_shift"
    ADVERSARIAL = "adversarial"


@dataclass
class LatentPatientState:
    """Ground truth. Agents NEVER see this directly."""
    patient_id: int
    ses: int                        # 0 (low), 1 (medium), 2 (high)
    true_health: float              # 0.0 (healthy) to 1.0 (critical)
    access_to_care: float           # 0.0 to 1.0
    measurement_prob: float         # probability each lab is observed
    optimal_treatment: int          # 0-3 treatment level
    group: int = 0                  # binary protected-group attribute, sampled independently of SES


@dataclass
class ObservedPatient:
    """What the agent sees. Noisy, biased, incomplete."""
    patient_id: int
    observed_risk_score: float
    lab_results: dict[str, Optional[float]]     # None = missing
    demographics: dict[str, float]              # SES proxies (insurance, zip, education)
    prior_utilization: float                    # historical healthcare use
    presenting_complaint_severity: float        # noisy signal of true_health
    n_missing_labs: int = 0
    group: Optional[int] = None                 # protected-group indicator exposed to the agent (None if withheld)


@dataclass
class OutcomeRecord:
    """Result of agent's treatment decision."""
    patient_id: int
    action: int
    success: bool
    # True iff |action - optimal_treatment| <= 1. This is *not* an alignment-audit
    # measure; it is a near-optimality indicator over the action grid. The
    # parent-vs-confounder alignment audit is computed post-hoc from the log.
    near_optimal: bool
    treatment_gap: int
    base_success_prob: float
    true_health: float
    ses: int
    group: int = 0
    under_treated: bool = False
    over_treated: bool = False
    # True iff the linear-model success probability fell outside [0.05, 0.95]
    # and was clipped. A high clip rate means the bounded-linear model is
    # masking real differentiation between agents and the logistic alternative
    # should be preferred.
    success_prob_clipped: bool = False
    cost: float = 0.0


@dataclass
class AgentObservation:
    """Packaged observation given to an agent at decision time."""
    patient: ObservedPatient
    scenario: str
    phase: str
    history: list[dict] = field(default_factory=list)
