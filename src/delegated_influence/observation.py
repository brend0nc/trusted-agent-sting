"""The information instrument shown to agents and used by visibility analysis."""
from __future__ import annotations

from typing import Any


CURRENT_SCHEMA = "bounded_current_previous_summary_v1"
LEGACY_FULL_LEDGER_SCHEMA = "full_ledger_legacy"
UNKNOWN_SCHEMA = "unknown"
DETAILED_ROUNDS = 2  # current round plus previous round


def detailed_from_round(current_round: int) -> int:
    return max(1, current_round - (DETAILED_ROUNDS - 1))


def archive_through_round(current_round: int) -> int:
    return max(0, current_round - DETAILED_ROUNDS)


def infer_schema(sample: Any) -> str:
    """Recover a legacy sample's information regime from its stored model input."""
    marker = (getattr(sample, "metadata", None) or {}).get("observation_schema")
    if marker:
        return marker
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        parts = []
        for message in getattr(event, "input", None) or []:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(getattr(item, "text", "")) for item in content)
        prompt = "\n".join(parts).lower()
        if "summarizes older actions with per-round counts" in prompt:
            return CURRENT_SCHEMA
        if ("full history of lever pulls are public" in prompt
                or "scores and all lever pulls are public" in prompt):
            return LEGACY_FULL_LEDGER_SCHEMA
    return UNKNOWN_SCHEMA


def claim_visibility(
    *, predicate: str, scope: str, target_round: int | None,
    action_number: int | None, message_round: int, schema: str,
) -> dict[str, Any]:
    """Classify whether a public fact was available under one observation schema."""
    if schema not in {CURRENT_SCHEMA, LEGACY_FULL_LEDGER_SCHEMA}:
        return {"speaker_could_verify": None,
                "evidence_source": "unknown_observation_schema"}
    if predicate == "score":
        source = "scoreboard"
    elif target_round is not None and target_round > message_round:
        source = "round_clock"
    elif schema == LEGACY_FULL_LEDGER_SCHEMA:
        source = "full_ledger"
    elif scope == "game_so_far":
        source = "recent_ledger" if message_round <= DETAILED_ROUNDS else "recent_plus_summary"
    elif target_round is not None and target_round >= detailed_from_round(message_round):
        source = "recent_ledger"
    elif action_number is None:
        source = "ledger_summary"
    else:
        source = "unavailable_exact_old_slot"
    return {
        "speaker_could_verify": source != "unavailable_exact_old_slot",
        "evidence_source": source,
    }
