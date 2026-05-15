"""Main benchmark runner. Orchestrates all scenarios x phases x agents x seeds."""

from functools import partial
from pathlib import Path

import numpy as np

from v3_health.core.types import AgentObservation
from v3_health.core.config import HEALTHCARE_CONFIG as C
from v3_health.agents.baseline_agent import BaselineAgent
from v3_health.agents.workflow_agent import WorkflowAgent
from v3_health.agents.causal_light_agent import CausalLightAgent
from v3_health.agents.causal_agent import CausalAgent
from v3_health.agents.structural_causal_agent import StructuralCausalAgent
from v3_health.scenarios import (
    treatment_allocation,
    missing_data_bias,
    historical_bias_feedback,
)
from v3_health.evaluation.intervention_auditor import InterventionAuditor
from v3_health.evaluation.fairness_metrics import compute_metrics, compute_phase_comparison
from v3_health.evaluation.statistical_analysis import (
    build_effect_size_rows,
    build_seed_metric_rows,
    build_seed_summary_rows,
    build_tradeoff_rows,
    compute_pairwise_tests,
    compute_seed_level_summary,
    identify_negative_result,
    write_csv,
    write_json,
)


SCENARIOS = {
    "treatment_allocation_confounding": treatment_allocation,
    "missing_data_bias": missing_data_bias,
    "historical_bias_feedback": historical_bias_feedback,
}

# Per-scenario inputs to the post-hoc parent-alignment audit. The parent
# variable comes from each scenario's TRUE_PARENTS["treatment_needed"]; the
# confounder set names ground-truth audit fields the agent should not be
# tracking if it is aligned with treatment need. All three v3_health
# scenarios inject SES-linked bias.
SCENARIO_ALIGNMENT_AUDIT = {
    "treatment_allocation_confounding": ("true_health", ["ses"]),
    "missing_data_bias":               ("true_health", ["ses"]),
    "historical_bias_feedback":        ("true_health", ["ses"]),
}

AGENTS = {
    "baseline": BaselineAgent,
    "workflow": WorkflowAgent,
    "causal_light": CausalLightAgent,
    "stability_filtered": CausalAgent,
    "structural_causal": StructuralCausalAgent,
}

AGENT_LABELS = {
    "baseline": "Baseline",
    "workflow": "Workflow",
    "stability_filtered": "Stability-Filtered",
    "causal_light": "Causal Light",
    "structural_causal": "Structural Causal",
}

AGENT_FAMILIES = {
    "baseline": "policy baseline",
    "workflow": "adaptive workflow heuristic",
    "stability_filtered": "structurally constrained adaptive heuristic",
    "causal_light": "structural-prior ablation",
    "structural_causal": "explicit structural causal model",
}

QUERY_SUPPORT = {
    "baseline": "observational",
    "workflow": "observational",
    "stability_filtered": "observational prediction plus structural exclusions, stability audits, and stratum-conditioned action-success tables",
    "causal_light": "observational under a structural exclusion prior",
    "structural_causal": "observational prediction plus latent-state interventional value estimation",
}

COUNTERFACTUAL_NOTE = (
    "No headline benchmark result should be interpreted as a counterfactual "
    "query. The structural-causal agent estimates interventional action values "
    "over a latent-state model, but no agent performs individual-level abduction."
)


def _primary_claim_agent() -> str:
    if "structural_causal" in AGENTS:
        return "structural_causal"
    if "stability_filtered" in AGENTS:
        return "stability_filtered"
    if "causal_light" in AGENTS:
        return "causal_light"
    return next(iter(AGENTS))


def run_single(scenario_name, scenario_module, agent_name, agent_class, seed, auditor):
    # RNG hygiene: spawn three independent streams from the same seed
    # so that adding a stochastic agent later cannot perturb the patient
    # generation or the outcome resolution streams. ``np.random.SeedSequence``
    # produces decorrelated child seeds via a hash-based mixing function
    # (see Salmon et al., 2011; documented in NumPy's ``SeedSequence``).
    seed_seq = np.random.SeedSequence(seed)
    gen_ss, resolve_ss, agent_ss = seed_seq.spawn(3)
    gen_rng = np.random.default_rng(gen_ss)
    resolve_rng = np.random.default_rng(resolve_ss)
    agent_rng = np.random.default_rng(agent_ss)

    agent = agent_class()
    if hasattr(agent, "set_rng"):
        agent.set_rng(agent_rng)
    n = C["n_patients_per_phase"]
    keep_state = bool(C.get("keep_agent_state_across_phases", True))

    for phase_idx, phase in enumerate(C["phases"]):
        # Headline protocol carries agent state through the full three-phase
        # trajectory. Resetting at each boundary is available only as a
        # secondary phase-conditional sensitivity mode.
        if not keep_state or phase_idx == 0:
            agent.reset()
            if hasattr(agent, "set_rng"):
                agent.set_rng(agent_rng)

        for i in range(n):
            patient_id = seed * 100_000 + phase_idx * n + i
            latent, observed = scenario_module.generate(patient_id, phase, gen_rng)

            obs = AgentObservation(
                patient=observed,
                scenario=scenario_name,
                phase=phase,
            )

            action, reasoning = agent.decide(obs)
            outcome = scenario_module.resolve(action, latent, resolve_rng)

            auditor.log(
                outcome=outcome,
                scenario=scenario_name,
                phase=phase,
                agent_type=agent_name,
                seed=seed,
                agent_reasoning=reasoning,
            )

            agent.update(obs, action, {"success": outcome.success})


def run_full_benchmark(output_dir: str = "results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    auditor = InterventionAuditor(str(output_path))
    active_agents = list(AGENTS.keys())
    agent_labels = {
        agent: C.get("agent_labels", {}).get(
            agent,
            AGENT_LABELS.get(agent, agent.replace("_", " ").title()),
        )
        for agent in active_agents
    }
    primary_claim_agent = _primary_claim_agent()

    total_runs = len(SCENARIOS) * len(AGENTS) * C["n_seeds"]
    current = 0

    for scenario_name, scenario_module in SCENARIOS.items():
        for agent_name, agent_class in AGENTS.items():
            for seed in range(C["n_seeds"]):
                current += 1
                print(f"[{current}/{total_runs}] {scenario_name} | {agent_name} | seed={seed}")
                run_single(scenario_name, scenario_module, agent_name, agent_class, seed, auditor)

    auditor.save("audit_log.jsonl")

    scenario_results: dict = {}
    for scenario_name in SCENARIOS:
        parent_var, confounder_vars = SCENARIO_ALIGNMENT_AUDIT[scenario_name]
        scenario_module = SCENARIOS[scenario_name]
        # Sanity check: declared parent must agree with the scenario's
        # exported TRUE_PARENTS dict.
        declared_parents = scenario_module.TRUE_PARENTS.get("treatment_needed", set())
        assert parent_var in declared_parents, (
            f"{scenario_name}: parent_var '{parent_var}' not in TRUE_PARENTS"
            f"['treatment_needed']={declared_parents}"
        )
        metrics_fn = partial(
            compute_metrics, parent_var=parent_var, confounder_vars=confounder_vars
        )
        phase_fn = partial(
            compute_phase_comparison,
            parent_var=parent_var,
            confounder_vars=confounder_vars,
        )
        scenario_results[scenario_name] = {}
        for agent_name in AGENTS:
            records = auditor.get_records(scenario=scenario_name, agent_type=agent_name)
            scenario_results[scenario_name][agent_name] = {
                "overall": metrics_fn(records),
                "by_phase": phase_fn(records),
                "seed_level": compute_seed_level_summary(
                    records,
                    compute_metrics=metrics_fn,
                    compute_phase_comparison=phase_fn,
                ),
            }

    pairwise_tests = compute_pairwise_tests(scenario_results)
    negative_result = identify_negative_result(
        scenario_results,
        pairwise_tests,
        focal_agent=primary_claim_agent,
        agent_labels=agent_labels,
    )

    payload = {
        "meta": {
            "n_seeds": C["n_seeds"],
            "n_patients_per_phase": C["n_patients_per_phase"],
            "phases": C["phases"],
            "scenarios": list(SCENARIOS.keys()),
            "agents": active_agents,
            "primary_claim_agent": primary_claim_agent,
            "primary_claim_label": agent_labels.get(primary_claim_agent, primary_claim_agent),
            "agent_labels": agent_labels,
            "agent_families": {
                agent: C.get("agent_families", {}).get(
                    agent,
                    AGENT_FAMILIES.get(agent, "unspecified"),
                )
                for agent in active_agents
            },
            "query_support": {
                agent: QUERY_SUPPORT.get(agent, "unspecified")
                for agent in active_agents
            },
            "counterfactual_note": COUNTERFACTUAL_NOTE,
            "confidence_interval": C.get(
                "confidence_interval",
                "95% Student-t CI on n=12 seed means",
            ),
            "significance_test": C.get(
                "significance_test",
                "paired exact sign-flip permutation test over matched seeds",
            ),
            "effect_size": C.get("effect_size", "Cohen's dz over paired seed deltas"),
            "primary_estimand": C.get(
                "primary_estimand",
                "matched-seed drop in success_rate from normal to adversarial "
                "with agent state carried across the full three-phase trajectory",
            ),
            "phase_protocol": (
                "carry_state_across_phases"
                if bool(C.get("keep_agent_state_across_phases", True))
                else "phase_conditional_reset"
            ),
            "phase_protocol_note": (
                "Headline protocol carries agent state from normal through "
                "surface_shift to adversarial, so phase comparisons estimate "
                "within-trajectory adaptation under shift."
            ),
        },
        "scenario_results": scenario_results,
        "pairwise_tests": pairwise_tests,
        "negative_result": negative_result,
    }

    write_json(output_path / "results_summary.json", payload)
    write_json(output_path / "negative_result.json", negative_result or {})
    write_csv(output_path / "seed_metrics.csv", build_seed_metric_rows(scenario_results))
    write_csv(output_path / "seed_summary.csv", build_seed_summary_rows(scenario_results))
    write_csv(output_path / "paired_significance.csv", pairwise_tests)
    write_csv(output_path / "effect_sizes.csv", build_effect_size_rows(pairwise_tests))
    write_csv(output_path / "tradeoff_summary.csv", build_tradeoff_rows(scenario_results))

    print(f"\nDone. Results saved to {output_path}")
    print(f"Total records: {len(auditor.records)}")

    return payload


if __name__ == "__main__":
    run_full_benchmark()
