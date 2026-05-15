"""Action-alignment diagnostic for v3_health.

For each (scenario, agent, phase) we ask a narrow post-hoc question: do
the logged actions align more strongly with the true parent of treatment
need (``true_health``) or with a known confounder injected by the
simulator (SES in all three v3_health scenarios)?

We answer this with two partial Pearson correlations on the audit log:

    rho_parent = |corr(action, true_health | confounders)|
    rho_conf   = max_c |corr(action, c | true_health, other confounders)|

and report

    action_alignment_diagnostic = rho_parent / (rho_parent + rho_conf)

in [0, 1]. The denominator uses ``max`` over confounders rather than a
mean: a single dominant confounder is the worst-case alignment failure,
and averaging over multiple confounders dilutes it. This is a
descriptive audit built with hidden ground truth that the agents never
observe. It is *not* evidence of causal identification, graph discovery,
or counterfactual reasoning, and it is deliberately excluded from the
pairwise significance grid.
"""

from __future__ import annotations

import numpy as np


def _residualize(target: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """Return the residual of ``target`` after OLS on ``[1, regressors]``."""
    if regressors.ndim == 1:
        regressors = regressors.reshape(-1, 1)
    design = np.column_stack([np.ones(len(target)), regressors])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> float:
    """Partial correlation of ``x`` and ``y`` controlling for ``z``."""
    if z is None or (hasattr(z, "size") and z.size == 0):
        rx, ry = x, y
    else:
        rx = _residualize(x, z)
        ry = _residualize(y, z)
    sx, sy = float(rx.std()), float(ry.std())
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _stack(columns: list[np.ndarray]) -> np.ndarray | None:
    if not columns:
        return None
    return np.column_stack(columns)


def compute_action_alignment_audit(
    records: list[dict],
    parent_var: str,
    confounder_vars: list[str],
) -> dict:
    """Population-level partial-correlation diagnostics for one slice."""
    if not records or not confounder_vars:
        return {
            "n": len(records),
            "partial_corr_action_parent": None,
            "partial_corr_action_confounder": None,
            "partial_corr_action_confounder_per_var": {},
            "action_alignment_diagnostic": None,
        }

    action = np.asarray([float(r["action"]) for r in records])
    parent = np.asarray([float(r[parent_var]) for r in records])
    cols = {c: np.asarray([float(r.get(c, 0.0)) for r in records])
            for c in confounder_vars}

    # Drop confounders with no variation in this slice (e.g. ``group`` is
    # constant in scenarios 1 and 2): partialling on a constant is a no-op
    # but adds rank-deficiency to the design matrix.
    active = [c for c in confounder_vars if float(cols[c].std()) > 1e-9]
    if not active:
        return {
            "n": len(records),
            "partial_corr_action_parent": None,
            "partial_corr_action_confounder": None,
            "partial_corr_action_confounder_per_var": {},
            "action_alignment_diagnostic": None,
        }

    if float(action.std()) < 1e-9 or float(parent.std()) < 1e-9:
        # Agent is constant or all patients had identical health: the
        # metric is undefined.
        return {
            "n": len(records),
            "partial_corr_action_parent": None,
            "partial_corr_action_confounder": None,
            "partial_corr_action_confounder_per_var": {c: None for c in active},
            "action_alignment_diagnostic": None,
        }

    rho_causal = abs(_partial_corr(action, parent, _stack([cols[c] for c in active])))

    per_conf: dict[str, float] = {}
    for c in active:
        controls = [parent] + [cols[o] for o in active if o != c]
        per_conf[c] = abs(_partial_corr(action, cols[c], _stack(controls)))

    # Worst-case (max) aggregation: the diagnostic should flag any single
    # confounder that dominates action alignment, not a population mean
    # that can dilute one strong confounder with several weak ones.
    rho_conf_agg = float(np.max(list(per_conf.values()))) if per_conf else 0.0
    denom = rho_causal + rho_conf_agg
    score = rho_causal / denom if denom > 1e-12 else None

    return {
        "n": len(records),
        "partial_corr_action_parent": float(rho_causal),
        "partial_corr_action_confounder": float(rho_conf_agg),
        "partial_corr_action_confounder_aggregator": "max",
        "partial_corr_action_confounder_per_var": {k: float(v) for k, v in per_conf.items()},
        "action_alignment_diagnostic": float(score) if score is not None else None,
    }


# Legacy alias kept so older notebooks can still import the helper.
compute_causal_targeting = compute_action_alignment_audit
