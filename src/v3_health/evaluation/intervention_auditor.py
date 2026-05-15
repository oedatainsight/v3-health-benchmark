"""
Intervention Auditor: logs every agent decision and checks whether
the agent acted on true causal parents or spurious correlates.
"""

import json
from dataclasses import asdict
from pathlib import Path

from v3_health.core.types import OutcomeRecord
from v3_health.core.config import HEALTHCARE_CONFIG as _C


class InterventionAuditor:
    def __init__(self, output_dir: str, *, log_reasoning: bool | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        # Reasoning strings are diagnostic and bloat the audit log
        # (~432K rows in the headline run). Off by default; the
        # ``log_reasoning`` config flag can re-enable them.
        self.log_reasoning = (
            bool(_C.get("log_reasoning", False))
            if log_reasoning is None
            else bool(log_reasoning)
        )

    def log(self, outcome: OutcomeRecord, scenario: str, phase: str,
            agent_type: str, seed: int, agent_reasoning: str = ""):
        record = asdict(outcome)
        record.update({
            "scenario": scenario,
            "phase": phase,
            "agent_type": agent_type,
            "seed": seed,
        })
        if self.log_reasoning:
            record["agent_reasoning"] = agent_reasoning
        self.records.append(record)

    def save(self, filename: str = "audit_log.jsonl"):
        path = self.output_dir / filename
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        return path

    def get_records(self, **filters) -> list[dict]:
        results = self.records
        for key, value in filters.items():
            results = [r for r in results if r.get(key) == value]
        return results
