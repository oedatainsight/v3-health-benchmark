"""Seed-level summaries and paired statistical comparisons for v3_health."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as _sp_stats


def _stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed derived from arbitrary identifiers.

    Replaces the previous ``seed = 313 + len(rows)`` pattern, whose value
    silently shifted whenever a metric or agent was added or reordered.
    Two callers with identical ``parts`` always get the same seed.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")


PHASES = ["normal", "surface_shift", "adversarial"]
# ``action_alignment_diagnostic`` is reported descriptively per slice but
# is *not* an outcome metric: it is a partial-correlation feature-
# alignment diagnostic on a hidden ground-truth column the agents do not
# observe. It is excluded from ``COMPARISON_METRICS_BY_SLICE`` so it does
# not enter the pairwise significance grid.
OVERALL_METRICS = [
    "success_rate",
    "near_optimal_rate",
    "avg_cost",
    "fairness_gap_ses",
    "fairness_gap_ses_max_min",
    "under_treatment_disparity",
    "bias_perpetuation_rate",
    "group_fairness_gap",
    "action_alignment_diagnostic",
    "clip_rate",
]
DEGRADATION_METRICS = [
    "success_rate_drop",
    "fairness_gap_increase",
    "near_optimal_rate_drop",
    "action_alignment_diagnostic_drop",
]
COMPARISON_METRICS_BY_SLICE = {
    "overall": [
        "success_rate",
        "near_optimal_rate",
        "avg_cost",
        "fairness_gap_ses",
    ],
    "normal": [
        "success_rate",
        "near_optimal_rate",
        "avg_cost",
        "fairness_gap_ses",
    ],
    "surface_shift": [
        "success_rate",
        "near_optimal_rate",
        "avg_cost",
        "fairness_gap_ses",
    ],
    "adversarial": [
        "success_rate",
        "near_optimal_rate",
        "avg_cost",
        "fairness_gap_ses",
    ],
    "degradation": [
        "success_rate_drop",
        "fairness_gap_increase",
        "near_optimal_rate_drop",
    ],
}
LOWER_IS_BETTER = {
    "avg_cost",
    "fairness_gap_ses",
    "fairness_gap_ses_max_min",
    "under_treatment_disparity",
    "group_fairness_gap",
    "clip_rate",
    "success_rate_drop",
    "fairness_gap_increase",
    "near_optimal_rate_drop",
    "action_alignment_diagnostic_drop",
}

DEFAULT_AGENT_LABELS = {
    "baseline": "Baseline",
    "workflow": "Workflow",
    "stability_filtered": "Stability-Filtered",
    "causal": "Stability-Filtered (legacy key)",
    "causal_light": "Causal Light",
    "structural_causal": "Structural Causal",
}


def _agent_label(agent: str, agent_labels: dict[str, str] | None = None) -> str:
    if agent_labels and agent in agent_labels:
        return agent_labels[agent]
    if agent in DEFAULT_AGENT_LABELS:
        return DEFAULT_AGENT_LABELS[agent]
    return agent.replace("_", " ").title()


def seed_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    method: str = "student_t",
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> dict:
    """
    Confidence interval for the mean of seed-level metrics.

    With ``n=12`` seeds the percentile bootstrap is known to undercover
    (its tails are pinned to the empirical sample), which is why this
    benchmark defaults to a one-sample Student-t CI:

        mean +/- t_{n-1, 1 - alpha/2} * s / sqrt(n)

    For sensitivity, the BCa bootstrap (``method='bca'``) and the
    percentile bootstrap (``method='percentile'``) are also available;
    they are not the default and are reported only when explicitly
    requested.
    """
    clean = np.asarray([float(v) for v in values if v is not None], dtype=float)
    n = int(clean.size)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "sem": None,
            "ci_lower": None,
            "ci_upper": None,
            "n": 0,
            "method": method,
        }
    if n == 1:
        value = float(clean[0])
        return {
            "mean": value,
            "std": 0.0,
            "sem": 0.0,
            "ci_lower": value,
            "ci_upper": value,
            "n": 1,
            "method": method,
        }

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    sem = float(std / np.sqrt(n))
    alpha = 1.0 - confidence

    if method == "student_t":
        crit = float(_sp_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        half = crit * sem
        ci_lower = mean - half
        ci_upper = mean + half
    elif method == "percentile":
        rng = np.random.default_rng(seed)
        samples = rng.choice(clean, size=(n_bootstrap, n), replace=True)
        means = samples.mean(axis=1)
        ci_lower = float(np.quantile(means, alpha / 2.0))
        ci_upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    elif method == "bca":
        rng = np.random.default_rng(seed)
        samples = rng.choice(clean, size=(n_bootstrap, n), replace=True)
        boot_means = samples.mean(axis=1)
        # bias-correction
        prop_below = float(np.mean(boot_means < mean))
        prop_below = float(np.clip(prop_below, 1.0 / (n_bootstrap + 1), 1.0 - 1.0 / (n_bootstrap + 1)))
        z0 = float(_sp_stats.norm.ppf(prop_below))
        # acceleration via jackknife
        jack = np.array([np.delete(clean, i).mean() for i in range(n)])
        jack_mean = float(jack.mean())
        num = float(np.sum((jack_mean - jack) ** 3))
        denom = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5)
        accel = float(num / denom) if denom > 1e-18 else 0.0
        z_lo = float(_sp_stats.norm.ppf(alpha / 2.0))
        z_hi = float(_sp_stats.norm.ppf(1.0 - alpha / 2.0))
        a_lo = float(_sp_stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo))))
        a_hi = float(_sp_stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi))))
        ci_lower = float(np.quantile(boot_means, a_lo))
        ci_upper = float(np.quantile(boot_means, a_hi))
    else:
        raise ValueError(f"Unknown CI method: {method!r}")

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": n,
        "method": method,
    }


# Backward-compatibility alias. The previous ``bootstrap_mean_ci`` name
# implied a percentile bootstrap, which on n=12 seeds undercovers; new
# code should call ``seed_mean_ci`` directly.
def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> dict:
    return seed_mean_ci(
        values,
        confidence=confidence,
        method="student_t",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def compute_seed_level_summary(
    records: list[dict],
    compute_metrics,
    compute_phase_comparison,
) -> dict:
    seeds = sorted({int(r["seed"]) for r in records})
    per_seed: list[dict] = []

    for seed in seeds:
        seed_records = [r for r in records if int(r["seed"]) == seed]
        per_seed.append({
            "seed": seed,
            "overall": compute_metrics(seed_records),
            "by_phase": compute_phase_comparison(seed_records),
        })

    overall_summary = {
        metric: seed_mean_ci(
            [run["overall"].get(metric) for run in per_seed],
            seed=_stable_seed("overall", metric),
        )
        for metric in OVERALL_METRICS
    }

    by_phase_summary: dict[str, dict] = {}
    for phase in [*PHASES, "degradation"]:
        metrics = OVERALL_METRICS if phase != "degradation" else DEGRADATION_METRICS
        by_phase_summary[phase] = {
            metric: seed_mean_ci(
                [run["by_phase"].get(phase, {}).get(metric) for run in per_seed],
                seed=_stable_seed("phase", phase, metric),
            )
            for metric in metrics
        }

    return {
        "per_seed": per_seed,
        "overall": overall_summary,
        "by_phase": by_phase_summary,
    }


def _seed_metric_map(seed_level: dict, slice_name: str, metric: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for run in seed_level.get("per_seed", []):
        seed = int(run["seed"])
        if slice_name == "overall":
            value = run["overall"].get(metric)
        else:
            value = run["by_phase"].get(slice_name, {}).get(metric)
        if value is not None:
            values[seed] = float(value)
    return values


def _paired_permutation_p_value(diffs: np.ndarray) -> float:
    """Two-sided paired sign-flip permutation p-value.

    For ``n <= 16`` we enumerate all ``2**n`` sign assignments exactly
    (with ``n=12`` seeds this is the exhaustive 4096-permutation case;
    no RNG is involved). For larger ``n`` we draw 20 000 random sign
    flips with a deterministic stream seeded from ``diffs`` so that the
    same input always produces the same Monte-Carlo estimate.
    """
    n = diffs.size
    if n == 0:
        return float("nan")

    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0

    if n <= 16:
        sign_patterns = np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=float)
        null_means = (sign_patterns * diffs).mean(axis=1)
    else:
        rng = np.random.default_rng(_stable_seed("perm", tuple(np.round(diffs, 12).tolist())))
        sign_patterns = rng.choice([-1.0, 1.0], size=(20000, n), replace=True)
        null_means = (sign_patterns * diffs).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= observed - 1e-12))


def _paired_t_p_value(diffs: np.ndarray) -> float:
    """Two-sided paired-t p-value on seed-level deltas."""
    n = int(diffs.size)
    if n < 2:
        return float("nan")
    std = float(diffs.std(ddof=1))
    if std < 1e-18:
        return 1.0 if abs(float(diffs.mean())) < 1e-18 else 0.0
    t_stat = float(diffs.mean()) / (std / np.sqrt(n))
    return float(2.0 * _sp_stats.t.sf(abs(t_stat), df=n - 1))


def _cohens_dz(diffs: np.ndarray) -> float:
    if diffs.size < 2:
        return 0.0
    std = float(diffs.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(diffs.mean() / std)


def _effect_size_label(effect_size: float) -> str:
    magnitude = abs(effect_size)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def _favored_agent(agent_a: str, agent_b: str, metric: str, mean_delta: float) -> str:
    if abs(mean_delta) < 1e-12:
        return "tie"
    if metric in LOWER_IS_BETTER:
        return agent_a if mean_delta < 0 else agent_b
    return agent_a if mean_delta > 0 else agent_b


def compute_pairwise_tests(scenario_results: dict) -> list[dict]:
    rows: list[dict] = []

    for scenario, agent_results in scenario_results.items():
        agents = sorted(agent_results.keys())
        for agent_a, agent_b in itertools.combinations(agents, 2):
            seed_level_a = agent_results[agent_a].get("seed_level", {})
            seed_level_b = agent_results[agent_b].get("seed_level", {})

            for slice_name, metrics in COMPARISON_METRICS_BY_SLICE.items():
                for metric in metrics:
                    values_a = _seed_metric_map(seed_level_a, slice_name, metric)
                    values_b = _seed_metric_map(seed_level_b, slice_name, metric)
                    common_seeds = sorted(set(values_a) & set(values_b))
                    if not common_seeds:
                        continue

                    arr_a = np.asarray([values_a[seed] for seed in common_seeds], dtype=float)
                    arr_b = np.asarray([values_b[seed] for seed in common_seeds], dtype=float)
                    diffs = arr_a - arr_b
                    ci = seed_mean_ci(
                        diffs.tolist(),
                        seed=_stable_seed("pair", scenario, slice_name, metric, agent_a, agent_b),
                    )
                    effect_size = _cohens_dz(diffs)
                    paired_t_p = _paired_t_p_value(diffs)

                    # Reference seed sets so callers can audit when paired
                    # comparisons across metrics for the same agent pair are
                    # built from different subsets of seeds.
                    seeds_overall_a = sorted(
                        _seed_metric_map(seed_level_a, "overall", "success_rate")
                    )
                    seeds_overall_b = sorted(
                        _seed_metric_map(seed_level_b, "overall", "success_rate")
                    )
                    reference_seeds = sorted(set(seeds_overall_a) & set(seeds_overall_b))
                    seed_set_matches_reference = (
                        list(common_seeds) == list(reference_seeds)
                    )

                    rows.append({
                        "scenario": scenario,
                        "slice": slice_name,
                        "metric": metric,
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "n_paired_seeds": len(common_seeds),
                        "paired_seeds": list(common_seeds),
                        "seed_set_matches_reference": seed_set_matches_reference,
                        "mean_a": float(arr_a.mean()),
                        "mean_b": float(arr_b.mean()),
                        "mean_delta": float(diffs.mean()),
                        "ci_lower": ci["ci_lower"],
                        "ci_upper": ci["ci_upper"],
                        "ci_method": ci["method"],
                        "p_value": _paired_permutation_p_value(diffs),
                        "p_value_paired_t": paired_t_p,
                        "effect_size_dz": effect_size,
                        "effect_size_label": _effect_size_label(effect_size),
                        "favored_agent": _favored_agent(agent_a, agent_b, metric, float(diffs.mean())),
                    })

    _annotate_multiple_comparison_corrections(rows)
    return rows


def _holm_adjusted(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979)."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (n - rank) * pvals[idx]
        running_max = max(running_max, scaled)
        adjusted[idx] = float(min(1.0, running_max))
    return adjusted


def _bh_adjusted(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (Benjamini & Hochberg 1995)."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank in reversed(range(n)):
        idx = order[rank]
        scaled = pvals[idx] * n / (rank + 1)
        running_min = min(running_min, scaled)
        adjusted[idx] = float(min(1.0, running_min))
    return adjusted


def _annotate_multiple_comparison_corrections(rows: list[dict]) -> None:
    """Attach Holm and Benjamini-Hochberg adjusted p-values, computed within
    each ``(scenario, slice)`` family. The family choice follows the
    convention that a reader scans one (scenario, slice) panel at a time
    when looking for significant differences across the agent x metric grid;
    correction is applied inside that panel rather than globally, which
    would be over-conservative across orthogonal scenarios."""
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        families[(row["scenario"], row["slice"])].append(idx)

    for indices in families.values():
        perm_pvals = [rows[i]["p_value"] for i in indices]
        t_pvals = [rows[i]["p_value_paired_t"] for i in indices]
        holm_perm = _holm_adjusted(perm_pvals)
        bh_perm = _bh_adjusted(perm_pvals)
        holm_t = _holm_adjusted(t_pvals)
        bh_t = _bh_adjusted(t_pvals)
        for k, i in enumerate(indices):
            rows[i]["family"] = f"{rows[i]['scenario']}::{rows[i]['slice']}"
            rows[i]["family_size"] = len(indices)
            rows[i]["p_value_holm"] = holm_perm[k]
            rows[i]["p_value_bh"] = bh_perm[k]
            rows[i]["p_value_paired_t_holm"] = holm_t[k]
            rows[i]["p_value_paired_t_bh"] = bh_t[k]


def build_seed_metric_rows(scenario_results: dict) -> list[dict]:
    rows: list[dict] = []
    for scenario, agent_results in scenario_results.items():
        for agent, result in agent_results.items():
            for run in result.get("seed_level", {}).get("per_seed", []):
                seed = int(run["seed"])
                for metric in OVERALL_METRICS:
                    value = run["overall"].get(metric)
                    if value is not None:
                        rows.append({
                            "scenario": scenario,
                            "agent": agent,
                            "seed": seed,
                            "slice": "overall",
                            "metric": metric,
                            "value": float(value),
                        })
                for phase in PHASES:
                    for metric in OVERALL_METRICS:
                        value = run["by_phase"].get(phase, {}).get(metric)
                        if value is not None:
                            rows.append({
                                "scenario": scenario,
                                "agent": agent,
                                "seed": seed,
                                "slice": phase,
                                "metric": metric,
                                "value": float(value),
                            })
                for metric in DEGRADATION_METRICS:
                    value = run["by_phase"].get("degradation", {}).get(metric)
                    if value is not None:
                        rows.append({
                            "scenario": scenario,
                            "agent": agent,
                            "seed": seed,
                            "slice": "degradation",
                            "metric": metric,
                            "value": float(value),
                        })
    return rows


def _summary_row(scenario: str, agent: str, slice_name: str, metric: str, summary: dict) -> dict:
    return {
        "scenario": scenario,
        "agent": agent,
        "slice": slice_name,
        "metric": metric,
        "mean": summary.get("mean"),
        "std": summary.get("std"),
        "ci_lower": summary.get("ci_lower"),
        "ci_upper": summary.get("ci_upper"),
        "n": summary.get("n"),
    }


def build_seed_summary_rows(scenario_results: dict) -> list[dict]:
    rows: list[dict] = []
    for scenario, agent_results in scenario_results.items():
        for agent, result in agent_results.items():
            seed_level = result.get("seed_level", {})
            for metric, summary in seed_level.get("overall", {}).items():
                rows.append(_summary_row(scenario, agent, "overall", metric, summary))
            for phase, metrics in seed_level.get("by_phase", {}).items():
                for metric, summary in metrics.items():
                    rows.append(_summary_row(scenario, agent, phase, metric, summary))
    return rows


def build_effect_size_rows(pairwise_tests: list[dict]) -> list[dict]:
    return [
        {
            "scenario": row["scenario"],
            "slice": row["slice"],
            "metric": row["metric"],
            "agent_a": row["agent_a"],
            "agent_b": row["agent_b"],
            "mean_delta": row["mean_delta"],
            "effect_size_dz": row["effect_size_dz"],
            "effect_size_label": row["effect_size_label"],
            "favored_agent": row["favored_agent"],
        }
        for row in pairwise_tests
    ]


def build_tradeoff_rows(scenario_results: dict) -> list[dict]:
    rows: list[dict] = []
    for scenario, agent_results in scenario_results.items():
        for agent, result in agent_results.items():
            overall = result.get("seed_level", {}).get("overall", {})
            degradation = result.get("seed_level", {}).get("by_phase", {}).get("degradation", {})
            rows.append({
                "scenario": scenario,
                "agent": agent,
                "success_rate_mean": overall.get("success_rate", {}).get("mean"),
                "success_rate_ci_lower": overall.get("success_rate", {}).get("ci_lower"),
                "success_rate_ci_upper": overall.get("success_rate", {}).get("ci_upper"),
                "fairness_gap_mean": overall.get("fairness_gap_ses", {}).get("mean"),
                "fairness_gap_ci_lower": overall.get("fairness_gap_ses", {}).get("ci_lower"),
                "fairness_gap_ci_upper": overall.get("fairness_gap_ses", {}).get("ci_upper"),
                "robustness_drop_mean": degradation.get("success_rate_drop", {}).get("mean"),
                "robustness_drop_ci_lower": degradation.get("success_rate_drop", {}).get("ci_lower"),
                "robustness_drop_ci_upper": degradation.get("success_rate_drop", {}).get("ci_upper"),
                "near_optimal_rate_mean": overall.get("near_optimal_rate", {}).get("mean"),
                "near_optimal_rate_ci_lower": overall.get("near_optimal_rate", {}).get("ci_lower"),
                "near_optimal_rate_ci_upper": overall.get("near_optimal_rate", {}).get("ci_upper"),
            })
    return rows


def identify_negative_result(
    scenario_results: dict,
    pairwise_tests: list[dict],
    focal_agent: str = "causal",
    agent_labels: dict[str, str] | None = None,
) -> dict | None:
    pairwise_lookup = {
        (
            row["scenario"],
            row["slice"],
            row["metric"],
            row["agent_a"],
            row["agent_b"],
        ): row
        for row in pairwise_tests
    }
    strongest: dict | None = None
    best_score = -1.0

    def lookup_test(scenario: str, slice_name: str, metric: str, agent_a: str, agent_b: str) -> dict | None:
        direct = pairwise_lookup.get((scenario, slice_name, metric, agent_a, agent_b))
        if direct is not None:
            return direct
        return pairwise_lookup.get((scenario, slice_name, metric, agent_b, agent_a))

    for scenario, agent_results in scenario_results.items():
        if focal_agent not in agent_results:
            continue
        focal = agent_results[focal_agent].get("seed_level", {})
        focal_success = focal.get("overall", {}).get("success_rate", {}).get("mean")
        focal_fairness = focal.get("overall", {}).get("fairness_gap_ses", {}).get("mean")
        focal_robustness = focal.get("by_phase", {}).get("degradation", {}).get("success_rate_drop", {}).get("mean")
        if None in (focal_success, focal_fairness, focal_robustness):
            continue

        focal_label = _agent_label(focal_agent, agent_labels)

        for competitor in [agent for agent in agent_results if agent != focal_agent]:
            other = agent_results[competitor].get("seed_level", {})
            other_success = other.get("overall", {}).get("success_rate", {}).get("mean")
            other_fairness = other.get("overall", {}).get("fairness_gap_ses", {}).get("mean")
            other_robustness = other.get("by_phase", {}).get("degradation", {}).get("success_rate_drop", {}).get("mean")
            if None in (other_success, other_fairness, other_robustness):
                continue

            utility_deficit = max(0.0, other_success - focal_success)
            fairness_deficit = max(0.0, focal_fairness - other_fairness)
            robustness_deficit = max(0.0, focal_robustness - other_robustness)
            score = utility_deficit + fairness_deficit + robustness_deficit
            if score <= best_score or score == 0.0:
                continue

            strengths: list[str] = []
            if other_success > focal_success:
                strengths.append("raw utility")
            if other_fairness < focal_fairness:
                strengths.append("fairness")
            if other_robustness < focal_robustness:
                strengths.append("robustness")

            success_test = lookup_test(scenario, "overall", "success_rate", focal_agent, competitor)
            fairness_test = lookup_test(scenario, "overall", "fairness_gap_ses", focal_agent, competitor)
            robustness_test = lookup_test(scenario, "degradation", "success_rate_drop", focal_agent, competitor)
            competitor_label = _agent_label(competitor, agent_labels)

            strongest = {
                "scenario": scenario,
                "focal_agent": focal_agent,
                "focal_label": focal_label,
                "competitor": competitor,
                "competitor_label": competitor_label,
                "score": score,
                "summary": (
                    f"{focal_label} does not dominate {competitor_label} in "
                    f"{scenario.replace('_', ' ')}; {competitor_label} is stronger "
                    f"on {', '.join(strengths)}."
                ),
                "metrics": {
                    "success_rate": {
                        "focal": focal_success,
                        "competitor": other_success,
                        "difference": focal_success - other_success,
                        "p_value": None if success_test is None else success_test["p_value"],
                    },
                    "fairness_gap_ses": {
                        "focal": focal_fairness,
                        "competitor": other_fairness,
                        "difference": focal_fairness - other_fairness,
                        "p_value": None if fairness_test is None else fairness_test["p_value"],
                    },
                    "success_rate_drop": {
                        "focal": focal_robustness,
                        "competitor": other_robustness,
                        "difference": focal_robustness - other_robustness,
                        "p_value": None if robustness_test is None else robustness_test["p_value"],
                    },
                },
            }
            best_score = score

    return strongest


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialised = {
                k: (",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v)
                for k, v in row.items()
            }
            writer.writerow(serialised)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)