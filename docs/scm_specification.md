# SCM Specification

The benchmark is synthetic. It is designed to test robustness under known observation-process failures, not to claim empirical calibration to real patient populations.

The canonical parameters for the SCM are in `configs/v3_health_default.yaml` and `configs/scenarios/*.yaml`. The implementation lives in `src/v3_health/core/scm.py` and `src/v3_health/core/outcome_resolver.py`.

## Latent State

For each patient, the simulator samples:

- `S`: socioeconomic status stratum in `{0, 1, 2}`
- `H`: latent health need in `[0, 1]`
- `G`: binary protected-group indicator
- `access`: access-to-care proxy in `[0, 1]`
- `measurement_prob`: lab observation probability
- `O(H)`: optimal treatment level in `{0, 1, 2, 3}`

The agents never observe `H`, `access`, `measurement_prob`, or `O(H)` directly.

## Observed Patient

The agent sees:

- `observed_risk_score`
- lab results with missing values
- demographic proxies
- prior utilization
- presenting complaint severity
- missing-lab count
- protected-group indicator

Scenarios alter the observation process, not the outcome law.

## Treatment-Mismatch Function

Let `A` be the agent action and `O(H)` be the optimal treatment level implied by latent health thresholds.

```text
g(A, H) = |A - O(H)|
```

Domain:

- `A in {0, 1, 2, 3}`
- `H in [0, 1]`

Range:

- `g(A, H) in {0, 1, 2, 3}`

Interpretation:

- `0` means exact treatment match.
- Larger values mean larger mismatch from latent treatment need.

Boundary behavior:

- Actions are expected to be valid treatment levels.
- `O(H)` is deterministic from `optimal_treatment_thresholds` in `configs/v3_health_default.yaml`.

## Symmetric Severity Penalty

Let `under(A, H)` be true when `A < O(H)` and `over(A, H)` be true when `A > O(H)`.

The headline setting uses a symmetric penalty:

```text
u(A, H) = g(A, H) * H if A != O(H)
u(A, H) = 0 otherwise
```

When `outcome_symmetric_severity_penalty` is disabled, the penalty applies only to under-treatment:

```text
u(A, H) = g(A, H) * H if A < O(H)
u(A, H) = 0 otherwise
```

The runtime switch is `outcome_symmetric_severity_penalty` in `configs/v3_health_default.yaml`.

## Outcome Variants

The configured headline variant is `logistic`.

```text
eta = logistic_intercept
      - logistic_gap_penalty * g(A, H)
      - logistic_health_penalty * H
      - I[severity_active] * logistic_undertreat_penalty * g(A, H) * H

P(success) = sigmoid(eta)
```

The appendix/sensitivity variant is `clipped_linear`.

```text
raw = success_probs_by_gap[g(A, H)]
      - health_penalty_weight * H
      - I[severity_active] * under_treatment_severity_penalty * H * g(A, H)

P(success) = clip(raw, 0.05, 0.95)
```

Both variants are implemented in `src/v3_health/core/outcome_resolver.py`. The selected variant is configured by `outcome_model`.

## Scenario Schedules

Exact phase schedules are in:

- `configs/scenarios/s1_treatment_allocation_confounding.yaml`
- `configs/scenarios/s2_mnar_measurement.yaml`
- `configs/scenarios/s3_historical_bias_feedback.yaml`

Generate reviewer-facing tables with:

```bash
PYTHONPATH=src python3 scripts/tables/generate_scenario_spec_table.py \
  --config configs/v3_health_default.yaml \
  --output-dir outputs/tables
```
