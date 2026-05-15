from importlib import import_module

__all__ = [
    "BaselineAgent",
    "WorkflowAgent",
    "CausalAgent",
    "CausalLightAgent",
    "StructuralCausalAgent",
]

_EXPORTS = {
    "BaselineAgent": "v3_health.agents.baseline_agent",
    "WorkflowAgent": "v3_health.agents.workflow_agent",
    "CausalAgent": "v3_health.agents.causal_agent",
    "CausalLightAgent": "v3_health.agents.causal_light_agent",
    "StructuralCausalAgent": "v3_health.agents.structural_causal_agent",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
