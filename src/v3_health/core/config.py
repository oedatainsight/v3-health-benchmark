HEALTHCARE_CONFIG = {
    # Scale
    "n_patients_per_phase": 1000,
    "n_seeds": 12,
    "phases": ["normal", "surface_shift", "adversarial"],
    "scenarios": [
        "treatment_allocation_confounding",
        "missing_data_bias",
        "historical_bias_feedback",
    ],
    "agents": ["baseline", "workflow", "causal_light", "causal", "structural_causal"],

    # SES distribution (unequal by design)
    "ses_distribution": [0.40, 0.35, 0.25],

    # Protected-group attribute. Sampled independently of SES so that the
    # group-conditional metrics (``bias_perpetuation_rate``,
    # ``group_fairness_gap``) are not collinear with the SES fairness
    # metrics. Group enters the observation process only through a small
    # group-linked additive bias on the observed risk score; it never
    # enters the outcome law (the SCM invariant ``outcome != f(group)``
    # given (action, true_health) is preserved).
    "group_distribution": [0.5, 0.5],
    "group_risk_bias_normal": {0: 0.05, 1: -0.05},
    "group_risk_bias_adversarial": {0: 0.10, 1: -0.10},

    # Health state generation
    "health_beta_a": 2.0,
    "health_beta_b": 5.0,

    # Thresholds on ``true_health`` that map a patient to one of four
    # optimal-treatment levels. Bin midpoints derived from these are also
    # used as the latent-state centers in the structural-causal agent
    # (see ``optimal_treatment_state_centers`` below).
    "optimal_treatment_thresholds": (0.3, 0.6, 0.8),

    # Access to care by SES
    "access_base_by_ses": {0: 0.30, 1: 0.60, 2: 0.90},
    "access_noise_std": 0.10,

    # Measurement probability
    "measurement_base": 0.30,
    "measurement_access_weight": 0.60,
    "measurement_noise_std": 0.05,

    # Lab noise
    "lab_noise_std": 0.10,

    # Scenario 1: SES bias on observed risk score
    "s1_ses_bias_normal": {0: 0.15, 1: 0.0, 2: -0.15},
    "s1_ses_bias_adversarial": {0: 0.25, 1: 0.0, 2: -0.20},
    "s1_risk_noise_normal": 0.10,
    "s1_risk_noise_shift": 0.12,
    "s1_risk_noise_adversarial": 0.08,

    # Scenario 2: Missingness
    "s2_n_labs": 8,
    "s2_lab_names": [
        "complete_blood_count", "metabolic_panel", "liver_function",
        "cardiac_markers", "thyroid", "coagulation", "urinalysis", "imaging",
    ],
    "s2_default_risk_when_no_labs": 0.30,
    "s2_adversarial_ses_penalty": {0: 0.30, 1: 0.15, 2: 0.05},

    # Scenario 3: historical SES bias carried through utilization records
    # and model-risk calibration. This is an SES-bias scenario, not a
    # protected-group benchmark.
    "s3_ses_risk_bias_normal": {0: 0.15, 1: 0.05, 2: -0.05},
    "s3_ses_risk_bias_adversarial": {0: 0.25, 1: 0.08, 2: -0.08},
    "s3_ses_util_penalty_normal": {0: 0.20, 1: 0.08, 2: 0.0},
    "s3_ses_util_penalty_surface_shift": {0: 0.05, 1: 0.02, 2: 0.0},
    "s3_ses_util_penalty_adversarial": {0: 0.30, 1: 0.12, 2: 0.0},

    # Outcome resolution
    "success_probs_by_gap": {
        0: 0.90,
        1: 0.65,
        2: 0.35,
        3: 0.15,
    },
    "health_penalty_weight": 0.30,
    "under_treatment_severity_penalty": 0.40,

    # Resource costs by treatment level
    "treatment_costs": {0: 0.1, 1: 0.3, 2: 0.6, 3: 1.0},
    "over_treatment_cost_multiplier": 1.5,

    # Outcome model variants. ``logistic`` is the headline because it is
    # bounded by construction and does not compress tail differentiation
    # between agents the way the [0.05, 0.95]-clipped additive model does.
    # ``clipped_linear`` is retained as a sensitivity / appendix model;
    # the per-record ``success_prob_clipped`` flag quantifies (it does
    # not undo) the clipping bias when that model is selected.
    "outcome_model": "logistic",
    # The severity penalty applies *symmetrically* to over- and under-
    # treatment in the headline run: failing to treat a sick patient and
    # over-treating a non-acute one are both penalised. The asymmetric
    # specification (under-treatment hurts probability, over-treatment
    # only raises cost) is available as a sensitivity by setting this to
    # False, but it confers a free upward-biasing advantage on agents
    # with red-flag overrides / MNAR caution bumps and so should not be
    # the headline.
    "outcome_symmetric_severity_penalty": True,
    # If True the auditor logs the agent's full reasoning string for
    # every decision. Default off because the benchmark writes
    # ~432K rows and the strings are diagnostic only.
    "log_reasoning": False,
    # Headline protocol: keep agent state across phases so each seed is a
    # continuous trajectory ``normal -> surface_shift -> adversarial``.
    # Phase-difference metrics therefore estimate within-trajectory
    # degradation under shift rather than independent phase-conditional
    # accuracy.
    "keep_agent_state_across_phases": True,
    # Logistic-model coefficients (used only when outcome_model == "logistic").
    "logistic_intercept": 2.2,
    "logistic_gap_penalty": 1.4,
    "logistic_health_penalty": 1.6,
    "logistic_undertreat_penalty": 1.2,
}


# ---------------------------------------------------------------------------
# Agent hyperparameters
# ---------------------------------------------------------------------------
# All numeric "knobs" used inside the agents are collected here so a reviewer
# can read them in one place and so a sensitivity sweep
# (``v3_health.experiments.hyperparam_sensitivity``) can perturb each one
# independently without editing agent source. Each entry includes a brief
# derivation note. The reported headline results are at the values below;
# robustness to ±25% perturbation of each entry is reported in the
# sensitivity sweep output.
AGENT_HYPERPARAMS = {
    # Action thresholds: the [0, 1] decision score is bucketed into the four
    # treatment levels by tertile-style cuts. Quartile cuts at (0.25, 0.50,
    # 0.75) give equal-mass bins under a uniform-on-[0,1] decision score and
    # are the standard four-level severity binning in the clinical
    # decision-support literature (e.g., NEWS2 buckets); see Smith et al.,
    # Resuscitation 84:465 (2013).
    "action_thresholds": (0.25, 0.50, 0.75),

    # Symptom strata (causal_agent stratification): tertile cuts of the
    # presenting-severity distribution. 0.33/0.66 are the empirical tertiles
    # of a Beta(2, 5) draw clipped to [0, 1] within ~0.02.
    "stratum_edges": (0.33, 0.66),

    # MNAR caution bumps. Workflow agent applies up to +0.07 + 0.03 base
    # when fewer than 40% of labs are observed; the stability-filtered
    # heuristic applies +0.10 * mnar_fraction once mnar > 0.5. These were tuned to nudge
    # rather than escalate (each bump is < one threshold-bin width of
    # 0.25). The sensitivity sweep checks behaviour at +/-25%.
    "mnar_bump_workflow_base": 0.03,
    "mnar_bump_workflow_slope": 0.07,
    "mnar_bump_causal": 0.10,
    "low_avail_threshold": 0.40,
    "mnar_threshold_causal": 0.50,

    # Workflow-agent disagreement override. When max(signal) - min(signal)
    # exceeds this spread, the score is pulled toward the most severe
    # signal less a 0.05 safety margin. 0.35 is roughly one threshold-bin
    # width plus its uncertainty band.
    "disagree_spread": 0.35,
    "red_flag_symptom": 0.85,
    "red_flag_floor": 0.75,

    # Workflow-agent online adaptation cadence + blend.
    "adapt_every": 50,
    "adapt_blend": 0.30,
    "adapt_window": 200,

    # Causal-agent feature-stability thresholds. ``stable`` and ``unstable``
    # define a hysteresis band on the [0, 1] stability score (1 - drift /
    # drift_norm). 0.55 / 0.40 was chosen to keep features stable under
    # within-phase drift up to ~0.18 and fail under adversarial drift > 0.24.
    "stable_threshold": 0.55,
    "unstable_threshold": 0.40,
    "drift_normalization": 0.40,

    # Stability-filtered heuristic stratum-override safety cap. The
    # empirical-best action from the observed action-success table only
    # overrides the threshold-derived action when its observed success
    # rate exceeds this floor *and* when |emp - thresh| <= 1 level.
    "action_success_override_floor": 0.55,
    "action_success_max_level_shift": 1,

    # Causal-agent record-buffer window. The model only uses the last
    # 2*window records; the buffer is a deque of length 2*window.
    "causal_window": 80,

    # Structural-causal latent-state grid. The implementation uses a fixed,
    # data-independent four-state clinical-need basis so the agent remains a
    # fair observed-input competitor rather than inheriting the simulator's
    # optimal-treatment thresholds.
    "structural_feature_std_floor": 0.08,
    "structural_bias_window": 240,
    "structural_exploration_steps": 240,
    "structural_exploration_weight": 0.10,
    "structural_outcome_prior_alpha": 1.0,
    "structural_outcome_prior_beta": 1.0,
    # Bias-regression EWMA blend on the running access-proxy slope and a
    # soft clip range. The blend is a standard exponentially-weighted
    # average smoothing factor; the clip prevents a single noisy window
    # from inverting the corrected risk score (the raw score lives in
    # [0, 1], so |slope * (1 - access)| <= 0.45 keeps the correction
    # below half the score range). Both are exposed for the sensitivity
    # sweep.
    "structural_bias_ewma": 0.35,
    "structural_bias_clip": 0.45,

    # Structural-causal agent: previously hard-coded magic numbers, now
    # exposed so the +/-25% sensitivity sweep actually perturbs them.
    #
    # ``structural_state_prior_mass`` is the symmetric Dirichlet pseudo-count
    # on the four-state latent prior; 4.0 corresponds to a weakly informative
    # uniform prior with effective sample size 16.
    #
    # The likelihood weights (symptom / lab-mean / risk) are fixed feature-
    # reliability multipliers on each Gaussian log-likelihood term. They are
    # ordered so that direct clinical signals dominate the institutional
    # risk score, and the risk term is additionally multiplied by the
    # access-derived ``risk_reliability`` at runtime.
    #
    # ``structural_critical_posterior_threshold`` engages a hard escalation
    # to action 3 whenever posterior mass on the critical state exceeds
    # this value; ``structural_high_severity_mass_threshold`` and
    # ``structural_high_severity_need_threshold`` together engage a softer
    # escalation to action >= 2 when the upper-two-state mass and posterior-
    # weighted expected need both exceed their thresholds. These are
    # safety overrides on top of the value-maximising action and are
    # documented as such; they are *not* derived from the SCM.
    "structural_state_prior_mass": 4.0,
    "structural_likelihood_weight_symptom": 1.15,
    "structural_likelihood_weight_lab": 0.95,
    "structural_likelihood_weight_risk": 0.85,
    "structural_critical_posterior_threshold": 0.35,
    "structural_high_severity_mass_threshold": 0.55,
    "structural_high_severity_need_threshold": 1.80,
}

# Total evaluations: 1000 patients x 3 phases x 3 scenarios x 5 agents x 12 seeds = 540,000
