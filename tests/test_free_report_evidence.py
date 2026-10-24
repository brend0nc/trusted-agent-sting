"""Delivered reports must reach evidence readers without becoming extra game actions."""
import asyncio
import copy
import sys
from pathlib import Path

import pytest

from delegated_influence import check_credits, check_lies, check_promises, communication, game, metrics, replay
from delegated_influence.judge_cache import JudgmentCheckpoint
from delegated_influence.task import ClaimParseError


SEATS = {"P1": "alpha", "P2": "beta", "P3": "gamma"}


def event(agent, kind="pull", **fields):
    return {"agent": agent, "type": kind, "forfeit": False, "forced": False,
            "round": 1, "action_number": 1, "beneficiary": "P3", "credit": None, **fields}


def attached(action="message", text="Please pull for P3.", to="P2"):
    return {"action": action, "text": text, "action_cost": 0, "free_report": True,
            **({"to": to} if action == "message" else {})}


def record(events):
    return {"id": "reports", "seat_map": SEATS, "events": events,
            "self_pull_reward": 0, "actions_per_round": 1}


def test_message_view_keeps_parent_indices_distinct_parts_and_input_unchanged():
    rows = [event("P1", "message", to="P2", text="normal", report=attached()),
            event("P2", report=attached("say", "public")),
            event("P3", "say", text="control", forced=True),
            event("P1", forfeit=True, report=attached()),
            event("P2", "intent", planned={"report": attached()})]
    before = copy.deepcopy(rows)
    messages = list(communication.iter_messages(rows))
    assert [(i, message["message_part"], message["text"]) for i, message in messages] == [
        (0, "action", "normal"), (0, "report", "Please pull for P3."),
        (1, "report", "public"), (2, "action", "control")]
    assert [i for i, _ in communication.iter_messages(rows, include_forced=False)] == [0, 0, 1]
    assert "usage" not in messages[1][1] and "calls" not in messages[1][1]
    assert rows == before


@pytest.mark.parametrize("channel", ["message", "say", "say_human"])
def test_report_credit_matches_the_game_ledger_and_reaches_the_judge(channel):
    state = game.new_state(SEATS, 1, 2, public_chat=True, human_board=True)
    state["free_reports"] = True
    for who, action in [("P1", {"action": "pull", "beneficiary": "P3",
                                 "report": {k: v for k, v in attached(channel).items()
                                            if k not in ("free_report", "action_cost")}}),
                        ("P2", {"action": "pull", "beneficiary": "P3", "credit": "P1"})]:
        assert game.validate_action(state, who, action)[0]
        game.apply_event(state, game.make_event(game.build_observation(state, who, 1, 1, 0), action))
    events = state["events"]
    assert state["ledger"][1]["credit"] == "P1"
    assert metrics.credit_verified(events, 1)
    assert metrics.verified_credit_rate(events) == 0.5
    assert check_credits._messages_before(events, "P1", "P2", 1) == ["Please pull for P3."]
    async def judge(text, beneficiary):
        assert text == "Please pull for P3." and beneficiary == "P3"
        return True
    assert asyncio.run(check_credits.check_episode_credits(record(events), judge)) == [
        {"model": "beta", "kind": "honest"}]


def test_private_report_cannot_support_credit_for_a_different_recipient():
    events = [event("P1", report=attached(to="P3")), event("P2", credit="P1")]
    assert not metrics.credit_verified(events, 1)
    assert check_credits._messages_before(events, "P1", "P2", 1) == []
    assert check_credits._messages_before(events, "P1", "P3", 0) == []


@pytest.mark.parametrize("channel", ["message", "say", "say_human"])
def test_lie_judge_reads_report_and_verifies_after_its_parent_pull(channel):
    events = [event("P1", report=attached(channel, "I pulled for P3; P3 has one point."))]
    calls = []
    async def judge(text, speaker, recipient, roster, channel="private"):
        calls.append((text, speaker, recipient, channel))
        return [{"predicate": "pulled", "subject": "P1", "object": "P3", "quantity": 1,
                 "scope": "current_round", "quote": text},
                {"predicate": "score", "subject": "P3", "quantity": 1,
                 "scope": "game_so_far", "quote": text}]
    verdicts = asyncio.run(check_lies.check_episode_lies(record(events), judge, channel="all"))
    assert len(calls) == 1
    assert calls[0][0] == events[0]["report"]["text"]
    assert calls[0][3] == {"message": "private", "say": "public", "say_human": "human"}[channel]
    assert [v["true"] for v in verdicts] == [True, True]
    assert all(v["index"] == 0 and v["message_part"] == "report" for v in verdicts)
    assert dict(check_lies.count_messages(record(events), channel="all")) == {"alpha": 1}


def test_two_same_channel_messages_have_distinct_judge_cache_keys(tmp_path):
    events = [event("P1", "message", text="main", to="P2", report=attached(text="report"))]
    rec = record(events)
    checkpoint = JudgmentCheckpoint(tmp_path / "lies.json", "test", {})
    calls = []
    async def judge(text, *args, **kwargs):
        calls.append(text)
        return []
    for _ in range(2):
        asyncio.run(check_lies.check_episode_lies(rec, judge, checkpoint=checkpoint))
    assert calls == ["main", "report"]
    assert len(checkpoint.doc["records"]) == 2
    async def fails(*args, **kwargs):
        raise ClaimParseError("bad extraction")
    row = asyncio.run(check_lies.build_ledger([rec], fails))["alpha"]
    assert row["messages"] == row["judge_errors"] == 2
    assert row["messages_judged"] == 0


def test_promises_keep_source_parts_and_resume_both_messages(tmp_path):
    events = [event("P1", "message", text="first promise", to="P2", report=attached(text="second promise"))]
    checkpoint = JudgmentCheckpoint(tmp_path / "promises.json", "test", {})
    calls = []
    async def judge(text):
        calls.append(text)
        return {"promise": True, "conditional": False, "commitment": text}
    for _ in range(2):
        promises = asyncio.run(check_promises.find_promises(events, judge, record(events), checkpoint))
    assert calls == ["first promise", "second promise"]
    assert [(p["index"], p.get("message_part", "action")) for p in promises] == [(0, "action"), (0, "report")]
    assert len(checkpoint.doc["records"]) == 2


def test_replay_counts_messages_without_extra_steps_actions_or_scores():
    events = [event("P1", report=attached()),
              event("P2", "say", text="normal post", report=attached("say_human")),
              event("P3", "message", text="normal private", to="P1", report=attached("say")),
              event("P1", forfeit=True, report=attached())]
    steps, final = replay.events(SEATS, events)
    assert len(steps) == 4
    assert final == {"scores": {"P1": 0, "P2": 0, "P3": 1},
                     "public_actions": 1, "private_messages": 2, "public_messages": 3}
    assert len(metrics.keep_real_actions(events)) == 3
    budget = metrics.action_budget_allocation(SEATS, events)
    plain = [{k: v for k, v in e.items() if k != "report"} for e in events]
    assert budget == metrics.action_budget_allocation(SEATS, plain)


def test_text_transcript_shows_report_audiences_on_the_parent_turn():
    events = [event("P1", report=attached(text="private body")),
              event("P2", report=attached("say", "agent body")),
              event("P3", report=attached("say_human", "human body"))]
    text = metrics.render_transcript(SEATS, events, dict.fromkeys(SEATS, 0))
    assert text.count("pulls for P3") == 3
    assert text.count("action cost: 0") == 3
    assert "P1 → P2: private body" in text
    assert "P2 says: agent body" in text
    assert "P3 posts to the human board: human body" in text


def test_reports_are_requests_and_relay_evidence_at_their_parent_indices():
    events = [event("P1", beneficiary="P2", report=attached(text="Ask P3 to pull for me.")),
              event("P2", beneficiary="P1", report=attached(text="P1 is reliable; pull for P1.", to="P3")),
              event("P3", beneficiary="P1")]
    assert metrics.label_pulls(events, 4)[1]["requested"]
    assert metrics.cascades(events) == {("P1", "P2", "P3")}
    funnel = metrics.relay_funnel({"seat_map": SEATS}, events)
    assert funnel["asks"] == funnel["forwarded"] == funnel["stranger_pulls"] == 1
    assert funnel["chains"][0]["index"] == 0
    assert funnel["chains"][0]["forward_index"] == 1
    assert funnel["chains"][0]["pull_index"] == 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))
    import derive
    assert derive.cascade_links(events) == [{"a": "P1", "b": "P2", "c": "P3",
                                           "ask_i": 0, "relay_i": 1, "pull_i": 2, "b_pull_i": 1}]


def test_report_interference_blocks_an_isolated_deception_claim():
    events = [event("P2", "intent", planned={"action": "pull", "beneficiary": "P3"}),
              event("P1", "message", to="P2", text="claim", report=attached(text="other message")),
              event("P2", beneficiary="P1")]
    effects = metrics.deception_effects(events, [{"index": 1, "speaker": "P1", "recipient": "P2",
                                               "likely_deception": True}])
    assert effects[0]["message_period_effect"]
    assert not effects[0]["successful_deception"]
    assert effects[0]["intervening_message_indices"] == [1, 1]
    candidates = metrics.deception_opportunity_screen(events)
    assert [(row["index"], row.get("message_part", "action")) for row in candidates] == [(1, "action"), (1, "report")]
