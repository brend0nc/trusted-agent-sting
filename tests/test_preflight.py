"""Offline preflight must exercise every treatment it claims to validate."""
from delegated_influence import preflight, task


def _arena_metadata(*, punish_enabled):
    return task.arena_25(
        seeds=(41,), messages_off=False, punish_enabled=punish_enabled,
    ).dataset[0].metadata


def test_punishment_treatment_reaches_simulation_and_prompt_lint():
    enabled = _arena_metadata(punish_enabled=True)
    disabled = _arena_metadata(punish_enabled=False)
    simulated = preflight.simulate_episode(enabled)

    assert simulated["punish_enabled"] is True
    assert any(event.get("type") == "punish" for event in simulated["events"])
    assert simulated["_prompt_profile"]["calls"] == (
        enabled["rounds"] * enabled["actions_per_round"] * len(enabled["seat_map"])
    )
    assert simulated["_prompt_profile"]["conservative_tokens_per_call"] > 0
    assert (simulated["_prompt_profile"]["conservative_tokens_per_call"]
            >= simulated["_prompt_profile"]["minimum_tokens_per_call"])
    assert preflight.lint_prompt(enabled) == []
    assert preflight.lint_prompt(disabled) == []
    assert preflight._condition_key(enabled) != preflight._condition_key(disabled)
