from abc import ABC, abstractmethod

from v3_health.core.types import AgentObservation


class BaseAgent(ABC):
    """Abstract agent interface. All agents implement decide()."""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.history: list[dict] = []

    @abstractmethod
    def decide(self, observation: AgentObservation) -> tuple[int, str]:
        """Return (action 0-3, reasoning)."""
        ...

    def update(self, observation: AgentObservation, action: int, outcome: dict):
        self.history.append({
            "observation": observation,
            "action": action,
            "outcome": outcome,
        })

    def reset(self):
        self.history = []
