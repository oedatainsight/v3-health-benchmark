"""Benchmark configuration loader.

The YAML files under ``configs/`` are the source of truth. This module
normalizes them into the legacy dict globals that the benchmark runtime
already consumes.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "v3_health_default.yaml"

_TOP_LEVEL_RUNTIME_SECTIONS = (
    "benchmark",
    "population",
    "health_generation",
    "access",
    "measurement",
    "demographics",
    "clinical_observation",
    "outcome",
    "statistics",
)
_TUPLE_KEYS = {
    "optimal_treatment_thresholds",
    "action_thresholds",
    "baseline_action_thresholds",
    "stratum_edges",
    "structural_state_centers",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _resolve_ref(bundle_dir: Path, ref: str | Path) -> Path:
    candidate = (bundle_dir / ref).resolve()
    if candidate.exists():
        return candidate
    fallback = (DEFAULT_CONFIG_PATH.parent / ref).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(candidate)


def _coerce_value(key: str, value: Any) -> Any:
    if key in _TUPLE_KEYS and isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return {k: _coerce_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_value(key, item) for item in value]
    return value


def _coerce_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(key, value) for key, value in mapping.items()}


def _merge_required_mapping(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    section: str,
    path: Path,
) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"{path}: section {section!r} must be a mapping")
    target.update(_coerce_mapping(source))


def load_config_bundle(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and normalize a benchmark config bundle without mutating globals."""

    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    top = _load_yaml(path)
    bundle_dir = path.parent

    healthcare_config: dict[str, Any] = {}
    for section in _TOP_LEVEL_RUNTIME_SECTIONS:
        if section in top:
            _merge_required_mapping(
                healthcare_config,
                top[section],
                section=section,
                path=path,
            )

    scenarios: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    for entry in top.get("scenarios", []):
        if not isinstance(entry, dict) or "id" not in entry or "path" not in entry:
            raise ValueError(f"{path}: every scenario entry must include id and path")
        scenario_path = _resolve_ref(bundle_dir, entry["path"])
        scenario_doc = _load_yaml(scenario_path)
        scenario_id = str(entry["id"])
        if scenario_doc.get("scenario") != scenario_id:
            raise ValueError(
                f"{scenario_path}: scenario id {scenario_doc.get('scenario')!r} "
                f"does not match bundle id {scenario_id!r}"
            )
        scenario_ids.append(scenario_id)
        scenario_spec = deepcopy(scenario_doc)
        scenario_spec["source_path"] = scenario_path
        scenarios.append(scenario_spec)

        if "parameters" in scenario_doc:
            _merge_required_mapping(
                healthcare_config,
                scenario_doc["parameters"],
                section="parameters",
                path=scenario_path,
            )
        phases = scenario_doc.get("phases", {})
        if not isinstance(phases, dict):
            raise ValueError(f"{scenario_path}: phases must be a mapping")
        for phase_name, phase_doc in phases.items():
            if not isinstance(phase_doc, dict):
                raise ValueError(f"{scenario_path}: phase {phase_name!r} must be a mapping")
            if "parameters" in phase_doc:
                _merge_required_mapping(
                    healthcare_config,
                    phase_doc["parameters"],
                    section=f"phases.{phase_name}.parameters",
                    path=scenario_path,
                )

    agent_hyperparams: dict[str, Any] = {}
    agents: list[dict[str, Any]] = []
    agent_ids: list[str] = []
    for entry in top.get("agents", []):
        if not isinstance(entry, dict) or "id" not in entry or "path" not in entry:
            raise ValueError(f"{path}: every agent entry must include id and path")
        agent_path = _resolve_ref(bundle_dir, entry["path"])
        agent_doc = _load_yaml(agent_path)
        agent_id = str(entry["id"])
        if agent_doc.get("agent") != agent_id:
            raise ValueError(
                f"{agent_path}: agent id {agent_doc.get('agent')!r} "
                f"does not match bundle id {agent_id!r}"
            )
        agent_ids.append(agent_id)
        agent_spec = deepcopy(agent_doc)
        agent_spec["source_path"] = agent_path
        agents.append(agent_spec)
        _merge_required_mapping(
            agent_hyperparams,
            agent_doc.get("hyperparams", {}),
            section="hyperparams",
            path=agent_path,
        )

    healthcare_config["scenarios"] = scenario_ids
    healthcare_config["agents"] = agent_ids
    healthcare_config["agent_labels"] = {
        agent["agent"]: agent.get("label", agent["agent"])
        for agent in agents
    }
    healthcare_config["agent_families"] = {
        agent["agent"]: agent.get("family", "unspecified")
        for agent in agents
    }

    return {
        "healthcare_config": healthcare_config,
        "agent_hyperparams": agent_hyperparams,
        "scenarios": scenarios,
        "agents": agents,
        "metadata": {
            "config_path": path,
            "config_name": top.get("name"),
            "version": top.get("version"),
        },
    }


def apply_config_bundle(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a bundle and update compatibility globals in place."""

    bundle = load_config_bundle(config_path)
    HEALTHCARE_CONFIG.clear()
    HEALTHCARE_CONFIG.update(deepcopy(bundle["healthcare_config"]))
    AGENT_HYPERPARAMS.clear()
    AGENT_HYPERPARAMS.update(deepcopy(bundle["agent_hyperparams"]))
    CONFIG_METADATA.clear()
    CONFIG_METADATA.update(deepcopy(bundle["metadata"]))
    return bundle


def snapshot_config() -> dict[str, Any]:
    return {
        "healthcare_config": deepcopy(HEALTHCARE_CONFIG),
        "agent_hyperparams": deepcopy(AGENT_HYPERPARAMS),
        "metadata": deepcopy(CONFIG_METADATA),
    }


def restore_config(snapshot: dict[str, Any]) -> None:
    HEALTHCARE_CONFIG.clear()
    HEALTHCARE_CONFIG.update(deepcopy(snapshot["healthcare_config"]))
    AGENT_HYPERPARAMS.clear()
    AGENT_HYPERPARAMS.update(deepcopy(snapshot["agent_hyperparams"]))
    CONFIG_METADATA.clear()
    CONFIG_METADATA.update(deepcopy(snapshot.get("metadata", {})))


def reset_to_default_config() -> dict[str, Any]:
    return apply_config_bundle(DEFAULT_CONFIG_PATH)


HEALTHCARE_CONFIG: dict[str, Any] = {}
AGENT_HYPERPARAMS: dict[str, Any] = {}
CONFIG_METADATA: dict[str, Any] = {}
reset_to_default_config()
