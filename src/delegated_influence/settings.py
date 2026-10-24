"""Validated game settings used by every episode construction path."""
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GameSettings:
    rounds: int
    actions_per_round: int
    topology: str = "complete"
    self_pull_reward: float = 0
    messages_off: bool = False
    public_chat: bool = False
    declared_intents: bool = False
    seeded_favours: int = 0
    punish_enabled: bool = False
    human_board: bool = False

    def __post_init__(self):
        for name in ("rounds", "actions_per_round", "seeded_favours"):
            value = getattr(self, name)
            minimum = 0 if name == "seeded_favours" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("messages_off", "public_chat", "declared_intents", "punish_enabled", "human_board"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        reward = self.self_pull_reward
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or reward < 0):
            raise ValueError("self_pull_reward must be a finite non-negative number")
        if self.topology not in ("complete", "ring"):
            raise ValueError(f"unknown topology {self.topology!r}")

    @classmethod
    def from_metadata(cls, metadata):
        return cls(**{name: metadata[name] for name in cls.__dataclass_fields__ if name in metadata})

    def as_dict(self):
        return asdict(self)
