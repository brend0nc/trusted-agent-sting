"""Optional reports share a normal turn without changing its action or model-call budget."""
import asyncio
import copy
import json
import random

import pytest

from delegated_influence import game, seat, turn


AGENTS = ["P1", "P2", "P3", "P4"]
PULL = {"action": "pull", "beneficiary": "P2", "credit": None}


def fresh(**settings):
    state = game.new_state(AGENTS, 2, 2, public_chat=True, human_board=True, **settings)
    state["free_reports"] = True
    return state


def apply(state, parsed, who="P1"):
    assert game.validate_action(state, who, parsed)[0]
    obs = game.build_observation(state, who, 1, 1, 0)
    event = game.make_event(obs, parsed, usage=[{"prompt_tokens": 10, "completion_tokens": 5}])
    game.apply_event(state, event)
    return event


@pytest.mark.parametrize("action,storage,audience", [
    ("say", "chat", AGENTS), ("say_human", "human_chat", AGENTS),
    ("message", "messages", ["P1", "P2"]),
])
def test_report_uses_existing_channel_visibility_and_one_parent_event(action, storage, audience):
    state = fresh()
    report = {"action": action, "text": "Report text, whether true or false."}
    if action == "message":
        report["to"] = "P2"
    parsed = {**PULL, "report": report}
    event = apply(state, parsed)
    assert event["report"] == {**report, "action_cost": 0, "free_report": True}
    assert len(state["events"]) == 1
    assert state["scores"] == {"P1": 0, "P2": 1, "P3": 0, "P4": 0}
    assert event["type"] == "pull"
    assert len(event["usage"]) == 1
    row = state[storage][0]
    assert row["action_cost"] == 0 and row["free_report"] is True
    assert row["text"] == report["text"]
    for who in AGENTS:
        obs = game.build_observation(state, who, 1, 2, 1)
        assert (report["text"] in seat.render_observation(obs)) == (who in audience)
    report["text"] = "changed input"
    assert event["report"]["text"] != report["text"]
    obs = game.build_observation(state, "P2", 1, 2, 1)
    visible_rows = obs[storage] if action != "message" else obs["inbox"]
    visible_rows[0]["text"] = "changed observation"
    assert row["text"] == "Report text, whether true or false."


@pytest.mark.parametrize("parent", [
    PULL, {"action": "message", "to": "P2", "text": "normal message"},
    {"action": "say", "text": "normal agent post"},
    {"action": "say_human", "text": "normal human post"},
    {"action": "punish", "target": "P2"},
])
def test_every_normal_action_can_carry_one_report(parent):
    state = fresh(punish_enabled=True)
    state["scores"]["P1"] = 1
    baseline = copy.deepcopy(state)
    apply(baseline, parent)
    event = apply(state, {**parent, "report": {"action": "say", "text": "extra report"}})
    assert state["scores"] == baseline["scores"]
    assert state["ledger"] == baseline["ledger"]
    assert state["messages"] == baseline["messages"]
    assert state["human_chat"] == baseline["human_chat"]
    assert state["chat"][:-1] == baseline["chat"]
    assert state["chat"][-1]["text"] == "extra report"
    assert len(state["events"]) == 1
    assert event["type"] == parent["action"]


@pytest.mark.parametrize("report", [
    [], [{"action": "say", "text": "one"}, {"action": "say", "text": "two"}],
    "text", {}, {"action": "pull", "beneficiary": "P2"},
    {"action": "say", "text": ""}, {"action": "say", "text": "   "},
    {"action": "say", "text": None}, {"action": "say", "text": "x" * 801},
    {"action": "say", "text": "one", "report": None},
    {"action": "say", "text": "one", "report": {"action": "say", "text": "two"}},
    {"action": "say", "text": "one", "extra": {"report": {"action": "say", "text": "two"}}},
    {"action": "say", "text": "one", "to": "P2"},
    {"action": "message", "text": "one"},
    {"action": "message", "text": "one", "to": "P1"},
    {"action": "message", "text": "one", "to": "unknown"},
])
def test_invalid_reports_reject_the_whole_action(report):
    state = fresh()
    before = copy.deepcopy(state)
    result = seat.classify_reply(json.dumps({**PULL, "report": report}), state, "P1")
    assert isinstance(result, seat.Rejected)
    assert result.reason == "parse_failure"
    assert "report" in result.correction.lower()
    assert state == before


@pytest.mark.parametrize("action", ["message", "say", "say_human"])
def test_reports_share_the_800_character_limit(action):
    extra = {"to": "P2"} if action == "message" else {}
    state = fresh()
    for length, valid in [(800, True), (801, False)]:
        report = {"action": action, "text": "x" * length, **extra}
        assert game.validate_action(state, "P1", {**PULL, "report": report})[0] is valid


@pytest.mark.parametrize("action,flag", [
    ("message", "messages_off"), ("say", "public_chat"), ("say_human", "human_board"),
])
def test_reports_cannot_use_disabled_channels(action, flag):
    state = fresh()
    state[flag] = action == "message"
    report = {"action": action, "text": "report"}
    if action == "message":
        report["to"] = "P2"
    assert not game.validate_action(state, "P1", {**PULL, "report": report})[0]


def test_private_reports_obey_the_message_graph():
    state = fresh(topology="ring")
    for recipient, valid in [("P2", True), ("P4", True), ("P3", False)]:
        report = {"action": "message", "text": "report", "to": recipient}
        assert game.validate_action(state, "P1", {**PULL, "report": report})[0] is valid


def test_report_alone_is_not_a_normal_action():
    result = seat.classify_reply('{"report":{"action":"say","text":"report only"}}', fresh(), "P1")
    assert isinstance(result, seat.Rejected)


@pytest.mark.parametrize("reply", [
    '{"action":"pull","beneficiary":broken,"report":{"action":"say","text":"report"}}',
    '{"action":"pull","beneficiary":"P2","report":{"action":"say","text":"one"},'
    '"report":{"action":"say","text":"two"}}',
    '{"action":"pull","beneficiary":"P2"} {"action":"say","text":"separate report"}',
    '[{"action":"pull","beneficiary":"P2","report":{"action":"say","text":"report"}}]',
])
def test_report_mode_does_not_recover_inner_or_multiple_actions(reply):
    result = seat.classify_reply(reply, fresh(), "P1")
    assert isinstance(result, seat.Rejected)
    assert result.reason == "parse_failure"


def test_report_mode_keeps_code_fences_and_simple_lenient_json():
    reply = '```json\n{"action":"pull","beneficiary":"P2","report":{"action":"say","text":"report",},}\n```'
    result = seat.classify_reply(reply, fresh(), "P1")
    assert isinstance(result, seat.Parsed)
    assert result.action["report"] == {"action": "say", "text": "report"}


def test_report_capability_is_opt_in_and_absent_or_null_preserves_event_shape():
    state = game.new_state(AGENTS, 2, 2, public_chat=True)
    report = {"action": "say", "text": "report"}
    assert not game.validate_action(state, "P1", {**PULL, "report": report})[0]
    default_obs = game.build_observation(state, "P1", 1, 1, 0)
    assert "free_reports" not in default_obs
    baseline = game.make_event(default_obs, PULL)
    for enabled in [False, True]:
        state["free_reports"] = enabled
        obs = game.build_observation(state, "P1", 1, 1, 0)
        assert obs.get("free_reports", False) is enabled
        for parsed in [PULL, {**PULL, "report": None}]:
            assert game.validate_action(state, "P1", parsed)[0]
            assert game.make_event(obs, parsed) == baseline


def test_full_episode_preserves_actions_rewards_and_call_count_and_delivers_before_next_seat():
    results = []
    for reports in [False, True]:
        state = fresh()
        state["free_reports"] = reports
        calls = []

        def generator(who):
            async def generate(messages):
                if reports and calls:
                    assert f"report from {calls[-1]}" in messages[1]["content"]
                calls.append(who)
                beneficiary = AGENTS[(AGENTS.index(who) + 1) % 4]
                parsed = {"action": "pull", "beneficiary": beneficiary}
                if reports:
                    parsed["report"] = {"action": "say", "text": f"report from {who}"}
                return json.dumps(parsed), {"prompt_tokens": 10, "completion_tokens": 5}
            return generate

        asyncio.run(turn.drive_episode(state, random.Random(41), {who: generator(who) for who in AGENTS}))
        assert len(calls) == 16
        assert len(state["events"]) == 16
        assert state["scores"] == dict.fromkeys(AGENTS, 4)
        assert all(e["type"] == "pull" and not e["forfeit"] for e in state["events"])
        assert all(len(e["usage"]) == 1 for e in state["events"])
        assert len(state["chat"]) == (16 if reports else 0)
        results.append((state["ledger"], state["scores"], calls))
    assert results[0] == results[1]


@pytest.mark.parametrize("first,second,forfeit", [
    ({**PULL, "report": {"action": "say", "text": "x" * 801}},
     {**PULL, "report": {"action": "say", "text": "valid report"}}, False),
    ({"action": "pull", "beneficiary": "unknown", "report": {"action": "say", "text": "invalid parent"}},
     {**PULL, "report": {"action": "say", "text": "valid report"}}, False),
    ({**PULL, "report": []}, {**PULL, "report": []}, True),
])
def test_parse_retry_never_delivers_invalid_or_duplicate_reports(first, second, forfeit):
    state = fresh()
    replies = iter([first, second])
    calls = []

    async def generate(messages):
        calls.append(messages)
        assert state["chat"] == []
        assert state["events"] == []
        return json.dumps(next(replies)), {"prompt_tokens": 10, "completion_tokens": 5}

    event = asyncio.run(turn.take_turn(state, game.build_observation(state, "P1", 1, 1, 0), generate))
    assert len(calls) == 2
    assert event["parse_retries"] == 1
    assert len(event["usage"]) == 2
    assert event["forfeit"] is forfeit
    assert state["chat"] == []
    game.apply_event(state, event)
    assert len(state["events"]) == 1
    assert state["scores"]["P2"] == (0 if forfeit else 1)
    assert [row["text"] for row in state["chat"]] == ([] if forfeit else ["valid report"])
    if forfeit:
        assert "report" not in event


def test_forfeit_does_not_deliver_even_if_report_field_is_present():
    state = fresh()
    event = game.make_event(game.build_observation(state, "P1", 1, 1, 0), None,
                            forfeit_reason="api_error")
    event["report"] = {"action": "say", "text": "forfeit report", "free_report": True, "action_cost": 0}
    game.apply_event(state, event)
    assert state["chat"] == []


def test_optional_report_is_explained_in_turn_and_intent_calls_but_intent_does_not_deliver():
    state = fresh()
    calls = []
    parsed = {**PULL, "report": {"action": "say", "text": "planned report"}}

    async def generate(messages):
        calls.append(messages)
        return json.dumps(parsed), {"prompt_tokens": 10, "completion_tokens": 5}

    asyncio.run(turn.elicit_intent(state, random.Random(41), {"P1": generate}, "P1", 1))
    assert state["events"][0]["planned"] == parsed
    assert state["chat"] == []
    asyncio.run(turn.take_turn(state, game.build_observation(state, "P1", 1, 1, 0), generate))
    for messages in calls:
        prompt = messages[0]["content"]
        assert '"report"' in prompt
        assert "zero action slots" in prompt
        assert "zero points" in prompt
        assert "Each public-board post uses one action" not in prompt
        assert "normal action" in prompt
        assert "800" in prompt
        assert "same reply" in prompt


def test_free_report_prompt_examples_only_offer_enabled_channels():
    for flags in [(False, False, False), (True, True, False), (True, False, True)]:
        messages_off, public_chat, human_board = flags
        prompt = seat.build_system_prompt("P1", AGENTS, 2, 2, messages_off=messages_off,
                                          public_chat=public_chat, human_board=human_board,
                                          free_reports=True)
        for action, enabled in [("message", not messages_off), ("say", public_chat), ("say_human", human_board)]:
            assert (f'"action": "{action}"' in prompt) is enabled
