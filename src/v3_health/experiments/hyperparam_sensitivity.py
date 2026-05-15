"""
One-axis hyperparameter sensitivity sweep.

For each tuneable knob in :mod:`v3_health.core.config.AGENT_HYPERPARAMS`
this script perturbs the value by +/-25% (and toggles tuple entries
proportionally), re-runs a *small* benchmark sweep, and reports the
delta in the headline statistics versus the released-config run.

The sweep is deliberately small (``n_patients_per_phase=120``,
``n_seeds=3``) so it can run in a couple of minutes; its purpose is
robustness verification, not headline reporting. A reviewer can grow
``n_patients_per_phase`` and ``n_seeds`` for a fuller table.

The headline statistics tracked are:
    * mean ``success_rate`` of the structural-causal agent (overall),
    * mean ``fairness_gap_ses`` of the structural-causal agent (overall),
    * mean ``action_alignment_diagnostic`` of the structural-causal agent
      (overall, when defined).

Output: ``hyperparam_sensitivity.csv`` in the supplied results dir,
plus a console summary sorted by absolute success-rate impact.
"""

from __future__ import annotations

import csv
import importlib
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# Configurable subset of HYPERPARAMS to sweep. Tuples of action thresholds
# and stratum edges are scaled in place; integers are rounded after scaling.
_KEYS_TO_SWEEP = (
    "stratum_edges",
    "action_thresholds",
    "mnar_bump_workflow_base",
    "mnar_bump_workflow_slope",
    "mnar_bump_causal",
    "low_avail_threshold",
    "mnar_threshold_causal",
    "disagree_spread",
    "red_flag_symptom",
    "red_flag_floor",
    "adapt_blend",
    "stable_threshold",
    "unstable_threshold",
    "drift_normalization",
    "action_success_override_floor",
    "structural_exploration_weight",
)


def _scale(value: Any, factor: float) -> Any:
    if isinstance(value, tuple):
        return tuple(_scale(v, factor) for v in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, int(round(value * factor)))
    if isinstance(value, float):
        return float(value * factor)
    return value


def _run_one(output_dir: Path) -> dict[str, float | None]:
    """Run a small sweep and return mean headline stats across all agents.

    Many hyperparameters affect only one agent family (e.g.,
    ``disagree_spread`` is workflow-only). Reporting the mean across
    agents picks up impact wherever it lands; per-agent breakdown is
    kept in the CSV too.
    """
    # Re-import so module-level constants in agent files pick up the
    # current ``AGENT_HYPERPARAMS`` snapshot.
    for mod_name in (
        "v3_health.agents.causal_agent",
        "v3_health.agents.workflow_agent",
        "v3_health.agents.structural_causal_agent",
        "v3_health.agents.causal_light_agent",
        "v3_health.evaluation.run_benchmark",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    from v3_health.evaluation.run_benchmark import run_full_benchmark

    payload = run_full_benchmark(str(output_dir))
    sr_block = payload["scenario_results"]
    success: list[float] = []
    fairness: list[float] = []
    targeting: list[float] = []
    per_agent: dict[str, list[float]] = {}
    for scenario in sr_block:
        for agent, agent_block in sr_block[scenario].items():
            overall = agent_block.get("overall", {})
            if "success_rate" in overall:
                success.append(float(overall["success_rate"]))
                per_agent.setdefault(agent, []).append(
                    float(overall["success_rate"])
                )
            if "fairness_gap_ses" in overall:
                fairness.append(float(overall["fairness_gap_ses"]))
            if overall.get("action_alignment_diagnostic") is not None:
                targeting.append(float(overall["action_alignment_diagnostic"]))
    out = {
        "success_rate": float(np.mean(success)) if success else None,
        "fairness_gap_ses": float(np.mean(fairness)) if fairness else None,
        "action_alignment_diagnostic": (
            float(np.mean(targeting)) if targeting else None
        ),
    }
    for agent, vals in per_agent.items():
        out[f"success_rate_{agent}"] = float(np.mean(vals))
    return out


def run_sensitivity_sweep(
    output_dir: str = "results/sensitivity",
    *,
    n_patients_per_phase: int = 120,
    n_seeds: int = 3,
    perturbation: float = 0.25,
) -> list[dict]:
    from v3_health.core import config as cfg

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Snapshot the released configuration we'll perturb against.
    baseline_hp = deepcopy(cfg.AGENT_HYPERPARAMS)
    baseline_hc = deepcopy(cfg.HEALTHCARE_CONFIG)

    cfg.HEALTHCARE_CONFIG["n_patients_per_phase"] = n_patients_per_phase
    cfg.HEALTHCARE_CONFIG["n_seeds"] = n_seeds

    rows: list[dict] = []
    print(f"[sensitivity] baseline run (n={n_patients_per_phase}, seeds={n_seeds})")
    with tempfile.TemporaryDirectory() as tmp:
        baseline_stats = _run_one(Path(tmp))
    print(
        f"  baseline: success={baseline_stats['success_rate']}, "
        f"fairness={baseline_stats['fairness_gap_ses']}, "
        f"alignment={baseline_stats['action_alignment_diagnostic']}"
    )
    rows.append(
        {
            "key": "(baseline)",
            "direction": "0",
            "factor": 1.0,
            **baseline_stats,
            "delta_success_rate": 0.0,
            "delta_fairness_gap_ses": 0.0,
            "delta_action_alignment_diagnostic": 0.0,
        }
    )

    for key in _KEYS_TO_SWEEP:
        if key not in baseline_hp:
            continue
        for direction, factor in (("-", 1.0 - perturbation), ("+", 1.0 + perturbation)):
            # Restore baseline before each perturbation.
            cfg.AGENT_HYPERPARAMS.clear()
            cfg.AGENT_HYPERPARAMS.update(deepcopy(baseline_hp))
            cfg.AGENT_HYPERPARAMS[key] = _scale(baseline_hp[key], factor)
            print(
                f"[sensitivity] {key} {direction}{int(perturbation*100)}% "
                f"-> {cfg.AGENT_HYPERPARAMS[key]}"
            )
            with tempfile.TemporaryDirectory() as tmp:
                stats = _run_one(Path(tmp))

            def _delta(metric: str) -> float | None:
                a = stats.get(metric)
                b = baseline_stats.get(metric)
                if a is None or b is None:
                    return None
                return float(a - b)

            rows.append(
                {
                    "key": key,
                    "direction": direction,
                    "factor": factor,
                    **stats,
                    "delta_success_rate": _delta("success_rate"),
                    "delta_fairness_gap_ses": _delta("fairness_gap_ses"),
                    "delta_action_alignment_diagnostic": _delta(
                        "action_alignment_diagnostic"
                    ),
                }
            )

    # Restore baseline state.
    cfg.AGENT_HYPERPARAMS.clear()
    cfg.AGENT_HYPERPARAMS.update(baseline_hp)
    cfg.HEALTHCARE_CONFIG.clear()
    cfg.HEALTHCARE_CONFIG.update(baseline_hc)

    csv_path = out_root / "hyperparam_sensitivity.csv"
    extra_keys = sorted(
        {
            k for r in rows for k in r.keys()
            if k.startswith("success_rate_")
        }
    )
    fieldnames = [
        "key",
        "direction",
        "factor",
        "success_rate",
        "fairness_gap_ses",
        "action_alignment_diagnostic",
        "delta_success_rate",
        "delta_fairness_gap_ses",
        "delta_action_alignment_diagnostic",
        *extra_keys,
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Print top-impact sorted summary.
    impact_rows = [
        r for r in rows
        if r["key"] != "(baseline)" and r["delta_success_rate"] is not None
    ]
    impact_rows.sort(key=lambda r: abs(r["delta_success_rate"]), reverse=True)
    print(f"\n[sensitivity] wrote {csv_path}")
    print("\nTop hyperparameter impacts on mean (across-agent) success rate:")
    print(f"  {'key':<32} {'dir':<3} {'delta_success':>14} {'delta_fairness':>15}")
    for r in impact_rows[:10]:
        print(
            f"  {r['key']:<32} {r['direction']:<3} "
            f"{r['delta_success_rate']:>+14.4f} "
            f"{r['delta_fairness_gap_ses']:>+15.4f}"
        )

    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/sensitivity"
    run_sensitivity_sweep(out)
