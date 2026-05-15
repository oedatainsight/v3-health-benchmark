# Causal Light vs Stability-Filtered Example

This example shows why `causal_light` and `stability_filtered` are separate comparators.

## Shared Patient Observation

```text
observed_risk_score = 0.90
presenting_complaint_severity = 0.15
lab_results = {lab_a: 0.10, lab_b: 0.20}
prior_utilization = 0.95
demographics = high-access proxies
n_missing_labs = 0
```

## Causal Light

`causal_light` excludes demographics and utilization, then computes:

```text
clinical_estimate = mean(symptom, mean(labs))
score = 0.70 * clinical_estimate + 0.30 * observed_risk_score
```

With the high observed risk score, the final score can move upward even though the clinical signals are low. The agent does not learn whether observed risk is stable across phases.

## Stability-Filtered

`stability_filtered` also excludes demographics and utilization, but it tracks whether observed risk remains stable over recent action-conditional windows.

If observed risk has become unstable, it uses:

```text
score = clinical_estimate
```

It can therefore ignore the high risk score for this patient after the benchmark has shown that risk is no longer a reliable action signal. It also keeps per-stratum action-success tables that can softly override the threshold action when enough recent evidence exists.

## Distinction

`causal_light` tests the structural-prior-only ablation: exclude forbidden proxies and use a fixed clinical/risk blend.

`stability_filtered` tests whether adaptive stability auditing and recent outcome feedback improve robustness beyond the structural prior alone.
