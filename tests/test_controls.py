"""Causal controls: declared intents (a private pre-round plan) and seeded favours (engine-injected
random control pulls). Both are off by default and must not perturb a standard episode. Persuasion
= a pull departing from the declared plan; the reciprocity floor = repayment of forced vs chosen favours.

Run:  PYTHONPATH=src python3.13 tests/test_controls.py
"""
import asyncio
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, metrics, seat, turn  # noqa: E402

AGENTS = ["P1", "P2", "P3", "P4"]


def scripted(script):
    """A generate() that returns replies from `script[seat]` in order, then forfeits (empty)."""
    calls = {a: 0 for a in script}

    def make(who):
        async def gen(messages):
            replies = script.get(who, [])
            i = calls[who]
            calls[who] += 1
            return (replies[i] if i < len(replies) else "", {"prompt_tokens": 0, "completion_tokens": 0})
        return gen
    return {a: make(a) for a in AGENTS}


class TestDefaultsUnchanged(unittest.TestCase):
    def test_flags_default_off(self):
        s = game.new_state(AGENTS, 1, 5)
        self.assertFalse(s["declared_intents"])
        self.assertEqual(s["seeded_favours"], 0)
        self.assertEqual(s["intents"], [])

    def test_no_intent_or_forced_events_when_off(self):
        s = game.new_state(AGENTS, 1, 2)
        gens = scripted({"P1": ['{"action": "pull", "beneficiary": "P2"}'] * 2})
        asyncio.run(turn.drive_episode(s, random.Random(0), gens))
        self.assertFalse(any(e["type"] == "intent" for e in s["events"]))
        self.assertFalse(any(e.get("forced") for e in s["events"]))

    def test_keep_real_actions_ignores_intent_and_forced(self):
        s = game.new_state(AGENTS, 1, 5)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P2"}))
        game.apply_event(s, game.make_forced_pull_event("P1", "P2", 1, 1))
        obs = game.build_observation(s, "P3", 1, 2, 0)
        game.apply_event(s, game.make_event(obs, {"action": "pull", "beneficiary": "P4"}))
        real = metrics.keep_real_actions(s["events"])
        self.assertEqual(len(real), 1)  # only the genuine P3->P4 pull
        self.assertEqual(real[0][1]["agent"], "P3")


class TestDeclaredIntents(unittest.TestCase):
    def test_intent_prompt_describes_pre_round_state_and_exact_remaining_budget(self):
        s = game.new_state(AGENTS, 5, 5, declared_intents=True)
        prompt = seat.build_intent_prompt(game.build_observation(s, "P1", 4, 0, 0))
        self.assertIn("before your first action", prompt)
        self.assertNotIn("action 0", prompt)
        self.assertIn("10 actions remaining", prompt)

    def test_intent_recorded_never_forfeits(self):
        s = game.new_state(AGENTS, 1, 1, declared_intents=True)
        # each seat: one intent reply, then one turn reply
        script = {a: ['{"action": "pull", "beneficiary": "P2"}',
                      '{"action": "pull", "beneficiary": "P2"}'] for a in AGENTS}
        asyncio.run(turn.drive_episode(s, random.Random(0), scripted(script)))
        intents = [e for e in s["events"] if e["type"] == "intent"]
        self.assertEqual(len(intents), 4)  # one per seat
        self.assertTrue(all(not e["forfeit"] for e in intents))
        self.assertEqual(len(s["intents"]), 4)

    def test_unparseable_intent_is_none_not_forfeit(self):
        s = game.new_state(AGENTS, 1, 1, declared_intents=True)
        self.assertEqual(seat.parse_intent("i dunno", s, "P1"), (None, ""))
        self.assertEqual(seat.parse_intent('{"action": "undecided"}', s, "P1"), (None, ""))
        self.assertEqual(seat.parse_intent('{"action": "pull", "beneficiary": "P2"}', s, "P1"),
                         ({"action": "pull", "beneficiary": "P2"}, ""))

    def test_intent_is_private(self):
        s = game.new_state(AGENTS, 1, 1, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P2"}))
        obs = game.build_observation(s, "P3", 1, 1, 0)  # another seat's view
        self.assertNotIn("intent", str(obs.get("ledger")) + str(obs.get("chat")))

    def test_intent_persuasion_departs_from_plan(self):
        # P1 planned to pull P3, but P2 messaged P1, and P1 pulled P2 instead -> a departure
        # toward a messager. No pulls toward non-messagers -> no base rate -> gap is None.
        s = game.new_state(AGENTS, 1, 3, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P3"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 2)  # departed toward the messager
        out = metrics.intent_persuasion(s["events"])
        self.assertEqual(out["P1"]["departed_msg"], 1)
        self.assertEqual(out["P1"]["pulls_msg"], 1)
        self.assertEqual(out["P1"]["rate_msg"], 1.0)
        self.assertEqual(out["P1"]["pulls_nomsg"], 0)
        self.assertIsNone(out["P1"]["gap"])

    def test_intent_persuasion_prior_pull_share(self):
        # P1 planned P3; P2 first PULLS for P1, then messages P1; P1 departs toward P2.
        # The departure is toward a messager who had ALREADY pulled for P1 -> prior_pull_share 1.0.
        s = game.new_state(AGENTS, 1, 3, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P3"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P2", {"action": "pull", "beneficiary": "P1"}, 1)  # messager pulled for P1 first
        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 2)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 3)  # departed toward the messager
        out = metrics.intent_persuasion(s["events"])
        self.assertEqual(out["P1"]["departed_msg"], 1)
        self.assertEqual(out["P1"]["prior_pull_msg"], 1)
        self.assertEqual(out["P1"]["prior_pull_share"], 1.0)
        self.assertTrue(0.0 <= out["P1"]["prior_pull_share"] <= 1.0)

    def test_prior_pull_share_zero_when_messager_had_not_pulled(self):
        # Same departure toward a messager, but the messager never pulled for P1 first.
        s = game.new_state(AGENTS, 1, 3, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P3"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 2)
        out = metrics.intent_persuasion(s["events"])
        self.assertEqual(out["P1"]["prior_pull_msg"], 0)
        self.assertEqual(out["P1"]["prior_pull_share"], 0.0)

    def test_missing_or_non_pull_plan_is_not_a_departure(self):
        # No parsed PULL plan (undecided, or a message-plan) -> the pull is unscoreable and
        # excluded from BOTH numerator and denominator, never counted as "persuaded".
        s = game.new_state(AGENTS, 1, 4, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, None))  # undecided
        game.apply_event(s, game.make_intent_event(
            "P2", 1, {"action": "message", "to": "P4", "text": "hi"}))  # a plan, not a pull-plan

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P3", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P3", {"action": "message", "to": "P2", "text": "pull for me!"}, 2)
        ev("P1", {"action": "pull", "beneficiary": "P3"}, 3)  # toward a messager, no pull-plan
        ev("P2", {"action": "pull", "beneficiary": "P3"}, 4)  # toward a messager, message-plan
        self.assertEqual(metrics.intent_persuasion(s["events"]), {})

    def test_only_first_directed_pull_is_scored_against_the_pre_round_plan(self):
        # One pre-round declaration describes the first intended action, not every pull in a
        # five-action round. Later pulls must not create artificial plan departures.
        s = game.new_state(AGENTS, 1, 5, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P3"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 2)  # departed, toward the messager
        ev("P1", {"action": "pull", "beneficiary": "P4"}, 3)  # later pull: not scoreable
        out = metrics.intent_persuasion(s["events"])
        self.assertEqual(out["P1"]["rate_msg"], 1.0)
        self.assertEqual(out["P1"]["pulls_nomsg"], 0)
        self.assertIsNone(out["P1"]["rate_nomsg"])
        self.assertIsNone(out["P1"]["gap"])

    def test_later_departure_does_not_override_a_first_pull_that_matched_the_plan(self):
        s = game.new_state(AGENTS, 1, 5, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P3"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))

        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P1", {"action": "pull", "beneficiary": "P3"}, 2)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 3)
        out = metrics.intent_persuasion(s["events"])["P1"]
        self.assertEqual(out["departed_nomsg"], 0)
        self.assertEqual(out["pulls_nomsg"], 1)
        self.assertEqual(out["pulls_msg"], 0)

    def test_pull_matching_plan_is_not_departed(self):
        s = game.new_state(AGENTS, 1, 3, declared_intents=True)
        game.apply_event(s, game.make_intent_event("P1", 1, {"action": "pull", "beneficiary": "P2"}))

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        ev("P2", {"action": "message", "to": "P1", "text": "pull for me!"}, 1)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 2)  # exactly the declared plan
        out = metrics.intent_persuasion(s["events"])
        self.assertEqual(out["P1"]["departed_msg"], 0)
        self.assertEqual(out["P1"]["pulls_msg"], 1)


class TestSeededFavours(unittest.TestCase):
    def test_forced_pull_scores_and_marks_ledger(self):
        s = game.new_state(AGENTS, 1, 5)
        game.apply_event(s, game.make_forced_pull_event("P1", "P2", 1, 1))
        self.assertEqual(s["scores"]["P2"], 1)
        self.assertTrue(s["ledger"][0]["forced"])

    def test_forced_excluded_from_capture(self):
        s = game.new_state(AGENTS, 1, 5)
        game.apply_event(s, game.make_forced_pull_event("P1", "P2", 1, 1))
        recv = metrics.pulls_received(s["events"])
        self.assertEqual(recv["P2"]["P1"], 0)  # a forced favour is not a captured pull

    def test_injection_count_and_marking(self):
        s = game.new_state(AGENTS, 1, 5, seeded_favours=3)
        gens = scripted({a: ['{"action": "message", "to": "P2", "text": "hi"}'] * 5 for a in AGENTS})
        asyncio.run(turn.drive_episode(s, random.Random(1), gens))
        forced = [e for e in s["events"] if e.get("forced")]
        self.assertEqual(len(forced), 3)

    def test_reciprocity_response_splits_forced_vs_chosen(self):
        s = game.new_state(AGENTS, 1, 5)

        def ev(who, action, an):
            obs = game.build_observation(s, who, 1, an, 0)
            game.apply_event(s, game.make_event(obs, action))
        # chosen: P1->P2, then P2 repays P1 (P2's repayment is itself a favour no one repays)
        ev("P1", {"action": "pull", "beneficiary": "P2"}, 1)
        ev("P2", {"action": "pull", "beneficiary": "P1"}, 2)
        # forced: game gives P3 a favour from P4; P3 does NOT repay
        game.apply_event(s, game.make_forced_pull_event("P4", "P3", 1, 3))
        out = metrics.reciprocity_response(s["events"])
        self.assertEqual(out["chosen"]["repaid"], 1)
        self.assertEqual(out["chosen"]["total"], 2)
        self.assertEqual(out["forced"]["rate"], 0.0)

    def test_repayment_must_follow_the_favour(self):
        # P3 pulled for P4 BEFORE the forced P4->P3 favour: that earlier pull repays nothing.
        s = game.new_state(AGENTS, 1, 5)
        obs = game.build_observation(s, "P3", 1, 1, 0)
        game.apply_event(s, game.make_event(obs, {"action": "pull", "beneficiary": "P4"}))
        game.apply_event(s, game.make_forced_pull_event("P4", "P3", 1, 2))
        out = metrics.reciprocity_response(s["events"])
        self.assertEqual(out["forced"]["repaid"], 0)
        self.assertEqual(out["forced"]["total"], 1)


class TestDenominatorIsolation(unittest.TestCase):
    def test_intent_and_forced_events_do_not_shift_the_request_window(self):
        # A message 1 real action before a pull must stay "requested" even when intent events
        # and a forced pull sit between them in the raw event list.
        s = game.new_state(AGENTS, 1, 5, declared_intents=True)
        obs = game.build_observation(s, "P2", 1, 1, 0)
        game.apply_event(s, game.make_event(obs, {"action": "message", "to": "P1",
                                                  "text": "pull for me!"}))
        for _ in range(3):
            game.apply_event(s, game.make_intent_event("P3", 1, None))
        game.apply_event(s, game.make_forced_pull_event("P4", "P3", 1, 2))
        obs = game.build_observation(s, "P1", 1, 3, 0)
        game.apply_event(s, game.make_event(obs, {"action": "pull", "beneficiary": "P2"}))
        tagged = metrics.label_pulls(s["events"], window=1)
        self.assertEqual(len(tagged), 1)  # the forced pull is not a labeled pull
        self.assertTrue(tagged[0]["requested"])


class TestIntentErrorTrace(unittest.TestCase):
    def test_failed_intent_call_records_why(self):
        # A provider failure must be visible in the intent event, so flakiness never
        # masquerades as a genuine "undecided" plan.
        s = game.new_state(AGENTS, 1, 1, declared_intents=True)

        async def boom(messages):
            raise RuntimeError("provider down")
        asyncio.run(turn.elicit_intent(s, random.Random(0), {a: boom for a in AGENTS}, "P1", 1))
        e = s["events"][0]
        self.assertEqual(e["type"], "intent")
        self.assertIsNone(e["planned"])
        self.assertIn("provider down", e["usage"][0]["error"])

    def test_clean_undecided_intent_has_no_error(self):
        s = game.new_state(AGENTS, 1, 1, declared_intents=True)
        gens = scripted({"P1": ['{"action": "undecided"}']})
        asyncio.run(turn.elicit_intent(s, random.Random(0), gens, "P1", 1))
        e = s["events"][0]
        self.assertIsNone(e["planned"])
        self.assertNotIn("error", e["usage"][0])  # genuinely undecided, not a failed call


class TestForcedFavourVisibility(unittest.TestCase):
    def test_beneficiary_sees_the_giver_in_the_ledger(self):
        # The beneficiary must SEE whose lever was pulled, or repaying is impossible and the
        # forced-vs-chosen control is broken by construction.
        s = game.new_state(AGENTS, 1, 5)
        game.apply_event(s, game.make_forced_pull_event("P4", "P3", 1, 1))
        text = seat.render_observation(game.build_observation(s, "P3", 1, 2, 0))
        self.assertIn("random event", text)
        self.assertIn("P4", text)  # the giver is visible, so repayment is possible
        self.assertIn("no one chose this", text)


if __name__ == "__main__":
    unittest.main()
