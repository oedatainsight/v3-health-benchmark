# Agent Specifications

The canonical agent hyperparameters are in `configs/agents/*.yaml`. The implementations live in `src/v3_health/agents/`.

All agents implement the same interface:

```python
def decide(observation: AgentObservation) -> tuple[int, str]
def update(observation: AgentObservation, action: int, outcome: dict) -> None
def reset() -> None
```

Actions are treatment levels `0` through `3`.

## Baseline

Source:

- `configs/agents/baseline.yaml`
- `src/v3_health/agents/baseline_agent.py`

Decision rule:

- Exclude demographic proxies.
- Use observed risk as the primary signal.
- If missingness exceeds the configured fail-state threshold, use symptoms plus a safety margin.
- If risk and symptoms strongly disagree, use the more severe signal.
- Add a bounded caution bump when recent outcomes fail frequently.

State:

- recent failure count
- recent total count

## Workflow

Source:

- `configs/agents/workflow.yaml`
- `src/v3_health/agents/workflow_agent.py`

Decision rule:

- Read symptoms, labs, and observed risk.
- Aggregate available labs.
- Blend signals using adaptive weights.
- Add missing-data caution when lab availability is low.
- Apply disagreement and red-flag safety overrides.

Update rule:

- Track recent signal values and success outcomes.
- Reweight symptoms, labs, and risk using a correlational quality proxy.

## Causal Light

Source:

- `configs/agents/causal_light.yaml`
- `src/v3_health/agents/causal_light_agent.py`

Decision rule:

- Exclude demographics and utilization.
- Estimate clinical need from symptoms and observed labs.
- Blend clinical estimate with observed risk using the configured risk blend.
- Threshold the score into treatment level.

Update rule:

- No adaptive update beyond base history logging.

## Stability-Filtered

Source:

- `configs/agents/stability_filtered.yaml`
- `src/v3_health/agents/causal_agent.py`

Decision rule:

- Exclude demographic proxies and prior utilization as direct treatment drivers.
- Build clinical estimate from symptoms and labs.
- Use observed risk only when recent action-conditional stability checks permit it.
- Add informative-missingness bump when missingness exceeds the configured threshold.
- Use stratum-conditioned action-success tables as a capped soft override.

Update rule:

- Track feature/outcome records in bounded windows.
- Recompute feature stability across successive windows.
- Track per-stratum, per-action observed success rates.

## Structural Causal

Source:

- `configs/agents/structural_causal.yaml`
- `src/v3_health/agents/structural_causal_agent.py`

Decision rule:

- Estimate an access score from utilization, demographics, and lab availability.
- Adjust observed risk using an access-proxy bias regression.
- Infer posterior mass over fixed latent clinical-need states.
- Choose action by expected learned interventional success value, with early exploration bonus.
- Apply configured high-severity safety escalations.

Update rule:

- Update latent-state prior mass.
- Update Gaussian feature accumulators for symptoms, labs, and adjusted risk.
- Update `E[success | do(action), latent_state]` Beta tables using posterior responsibilities.
- Update risk-bias slope over a running window.

## Phase State

The headline protocol carries agent state through `normal -> surface_shift -> adversarial`. This is configured by `keep_agent_state_across_phases` in `configs/v3_health_default.yaml`.

## Generated Agent Table

Generate reviewer-facing agent parameter tables with:

```bash
PYTHONPATH=src python3 scripts/tables/generate_agent_spec_table.py \
  --config configs/v3_health_default.yaml \
  --output-dir outputs/tables
```
