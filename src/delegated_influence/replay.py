"""Deterministic state replay for canonical game events."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from delegated_influence import communication


def initial_state(seat_map: dict[str, str]) -> dict[str, Any]:
    return {
        "scores": {seat: 0 for seat in seat_map},
        "public_actions": 0,
        "private_messages": 0,
        "public_messages": 0,
    }


def apply_event(
    state: dict[str, Any], event: dict[str, Any], self_pull_reward: float = 0,
) -> None:
    """Apply one canonical event to replay state in place."""
    if event.get("forfeit"):
        return
    event_type = event.get("type")
    if event_type == "pull":
        beneficiary = event.get("beneficiary")
        reward = self_pull_reward if beneficiary == event.get("agent") else 1
        state["scores"][beneficiary] = state["scores"].get(beneficiary, 0) + reward
        state["public_actions"] += 1
    elif event_type == "punish":
        agent, target = event.get("agent"), event.get("target")
        state["scores"][agent] = state["scores"].get(agent, 0) - 1
        state["scores"][target] = max(0, state["scores"].get(target, 0) - 1)
        state["public_actions"] += 1
    for _, message in communication.iter_messages([event]):
        state["private_messages" if message["type"] == "message" else "public_messages"] += 1


def events(
    seat_map: dict[str, str], event_rows: Iterable[dict[str, Any]], self_pull_reward: float = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a before/after state step for every event plus the final state."""
    state = initial_state(seat_map)
    steps = []
    for event in event_rows:
        before = deepcopy(state)
        apply_event(state, event, self_pull_reward)
        steps.append({"before": before, "after": deepcopy(state)})
    return steps, state
