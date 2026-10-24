"""Credit on pulls: every pull may name the third participant who arranged it (or none). Credit is
structural — no on/off knob — public, and pays the broker nothing. An omitted credit reads as
"none" (a formatting slip is not a forfeit); analysis later verifies self-reports against the
message log (honest / stolen / hidden — see docs/DESIGN.md "Credit & attribution").

Run:  PYTHONPATH=src python3.13 tests/test_credit.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, metrics, seat, turn  # noqa: E402

AGENTS = ["P1", "P2", "P3", "P4"]


def pull(agent, benef, credit=None, rnd=1):
    return {"forfeit": False, "type": "pull", "agent": agent, "beneficiary": benef,
            "credit": credit, "round": rnd}


def msg(sender, to, rnd=1):
    return {"forfeit": False, "type": "message", "agent": sender, "to": to, "text": "pull for them", "round": rnd}


def fresh_state():
    return game.new_state(AGENTS, 1, 5)


class TestCreditValidation(unittest.TestCase):
    def test_credited_pull_accepted(self):
        ok, _ = game.validate_action(fresh_state(), "P2", {"action": "pull", "beneficiary": "P3", "credit": "P1"})
        self.assertTrue(ok)

    def test_pull_without_credit_still_accepted(self):
        # omitted credit = "my own idea"; never a forfeit
        ok, _ = game.validate_action(fresh_state(), "P2", {"action": "pull", "beneficiary": "P3"})
        self.assertTrue(ok)

    def test_credit_must_be_a_participant(self):
        ok, _ = game.validate_action(fresh_state(), "P2", {"action": "pull", "beneficiary": "P3", "credit": "PX"})
        self.assertFalse(ok)

    def test_cannot_credit_self(self):
        ok, _ = game.validate_action(fresh_state(), "P2", {"action": "pull", "beneficiary": "P3", "credit": "P2"})
        self.assertFalse(ok)

    def test_cannot_credit_the_beneficiary(self):
        ok, _ = game.validate_action(fresh_state(), "P2", {"action": "pull", "beneficiary": "P3", "credit": "P3"})
        self.assertFalse(ok)


class TestCreditEffect(unittest.TestCase):
    def _apply(self, state, who, action, action_number=1):
        obs = game.build_observation(state, who, 1, action_number, 0)
        game.apply_event(state, game.make_event(obs, action))
        return state

    def _apply_credited_pull(self, with_prior_message=True):
        s = fresh_state()
        if with_prior_message:  # P1 actually lobbied P2 -> the credit is structurally verified
            self._apply(s, "P1", {"action": "message", "to": "P2", "text": "pull for P3, they helped us"})
        return self._apply(s, "P2", {"action": "pull", "beneficiary": "P3", "credit": "P1"}, action_number=2)

    def test_broker_gets_no_points(self):
        s = self._apply_credited_pull()
        self.assertEqual(s["scores"]["P3"], 1)  # beneficiary scores
        self.assertEqual(s["scores"]["P1"], 0)  # broker scores nothing
        self.assertEqual(s["scores"]["P2"], 0)  # puller scores nothing

    def test_verified_credit_reaches_the_public_ledger(self):
        s = self._apply_credited_pull(with_prior_message=True)
        self.assertEqual(s["ledger"][-1]["credit"], "P1")

    def test_every_pull_record_keeps_nullable_credit_key(self):
        s = fresh_state()
        self._apply(s, "P2", {"action": "pull", "beneficiary": "P3"})
        raw, public = s["events"][-1], s["ledger"][-1]
        self.assertEqual(
            {"agent": raw["agent"], "beneficiary": raw["beneficiary"],
             "credit": raw["credit"]},
            {"agent": "P2", "beneficiary": "P3", "credit": None},
        )
        self.assertEqual(
            {"puller": public["puller"], "beneficiary": public["beneficiary"],
             "credit": public["credit"]},
            {"puller": "P2", "beneficiary": "P3", "credit": None},
        )

    def test_fabricated_credit_is_not_broadcast(self):
        # A-1: P1 never messaged P2, so the claim is fabricated. The EVENT keeps the claim (for the
        # honesty analysis) but the PUBLIC ledger must not spread it into the game.
        s = self._apply_credited_pull(with_prior_message=False)
        self.assertEqual(s["events"][-1]["credit"], "P1")   # claim recorded privately
        self.assertIsNone(s["ledger"][-1]["credit"])        # not published

    def test_public_request_verifies_credit(self):
        s = game.new_state(AGENTS, 1, 5, messages_off=True, public_chat=True)
        self._apply(s, "P1", {"action": "say", "text": "P2, pull for P3"})
        self._apply(s, "P2", {"action": "pull", "beneficiary": "P3", "credit": "P1"},
                    action_number=2)
        self.assertEqual(s["ledger"][-1]["credit"], "P1")
        self.assertEqual(metrics.verified_credit_rate(s["events"]), 1.0)

    def test_credit_visible_in_observation(self):
        s = self._apply_credited_pull()
        obs = game.build_observation(s, "P3", 1, 3, 0)  # the beneficiary sees who brokered
        self.assertEqual(obs["ledger"][-1]["credit"], "P1")


class TestHistoricalCreditSummary(unittest.TestCase):
    def test_round_one_pull_and_credit_are_summarized_late(self):
        s = game.new_state(AGENTS, 5, 5)
        game.apply_event(s, game.make_event(
            game.build_observation(s, "P1", 1, 1, 0),
            {"action": "message", "to": "P2", "text": "pull for P3"}))
        game.apply_event(s, game.make_event(
            game.build_observation(s, "P2", 1, 2, 0),
            {"action": "pull", "beneficiary": "P3", "credit": "P1"}))
        late = game.build_observation(s, "P4", 5, 1, 0)
        self.assertEqual(late["ledger"], [])
        self.assertEqual(late["ledger_summary"]["pulls"][0]["puller"], "P2")
        self.assertEqual(late["ledger_summary"]["credits"], [{
            "credit": "P1", "puller": "P2", "beneficiary": "P3",
            "round_counts": [{"round": 1, "count": 1}],
        }])
        self.assertEqual(len(s["ledger"]), 1)  # analysis state retains the exact event


class TestCreditPrompt(unittest.TestCase):
    def test_prompt_asks_for_credit_by_default(self):
        p = seat.build_system_prompt("P2", AGENTS, 1, 5).lower()
        self.assertIn("credit", p)
        self.assertIn('"credit": null', p)  # valid JSON null, not the string "null"
        self.assertIn("no quotation marks", p)
        self.assertIn("beneficiary asked you directly", p)
        self.assertIn("third", p)         # strictly third-party
        self.assertIn("published", p)     # the verification rule is stated truthfully

    def test_prompt_does_not_promise_payback(self):
        # The old nudge asserted the very effect we observe; it must never come back.
        p = seat.build_system_prompt("P2", AGENTS, 1, 5).lower()
        self.assertNotIn("goodwill", p)
        self.assertNotIn("pulling for you later", p)

    def test_prompt_silent_on_credit_when_messages_off(self):
        # No one can have asked you without messages, so don't mention credit.
        p = seat.build_system_prompt("P2", AGENTS, 1, 5, messages_off=True)
        self.assertNotIn("credit", p.lower())

    def test_render_shows_broker_in_ledger(self):
        obs = {
            "round": 1, "rounds": 1, "action_number": 2, "actions_per_round": 5, "agent": "P3",
            "agents": AGENTS, "scores": {a: 0 for a in AGENTS}, "inbox": [], "outbox": [],
            "ledger": [{"round": 1, "action_number": 1, "puller": "P2", "beneficiary": "P3", "credit": "P1"}],
        }
        self.assertIn("credit: P1", seat.render_observation(obs))


class TestCreditTurn(unittest.TestCase):
    def test_take_turn_carries_credit_into_the_event(self):
        s = fresh_state()
        obs = game.build_observation(s, "P2", 1, 1, 0)
        seen = {}

        async def gen(messages):
            seen["sys"] = messages[0]["content"]
            return '{"action": "pull", "beneficiary": "P3", "credit": "P1"}', {"prompt_tokens": 0, "completion_tokens": 0}

        ev = asyncio.run(turn.take_turn(s, obs, gen))
        self.assertEqual(ev["credit"], "P1")          # credit flows reply -> event
        self.assertIn("credit", seen["sys"].lower())  # the instruction reached the system prompt

    def test_self_credit_reads_as_own_idea_not_forfeit(self):
        # Models write their own name to mean "my own idea"; that intent is unambiguous, so it
        # normalizes to null instead of burning a retry (seen in the 01_smoke run).
        s = fresh_state()
        outcome = seat.classify_reply('{"action": "pull", "beneficiary": "P3", "credit": "P2"}', s, "P2")
        self.assertIsInstance(outcome, seat.Parsed)
        self.assertIsNone(outcome.action["credit"])

    def test_quoted_null_credit_is_treated_as_json_null(self):
        # Seen in the Luna background validation: a harmless JSON type slip must not cost a
        # second model call.
        s = fresh_state()
        outcome = seat.classify_reply(
            '{"action": "pull", "beneficiary": "P3", "credit": "null"}', s, "P2")
        self.assertIsInstance(outcome, seat.Parsed)
        self.assertIsNone(outcome.action["credit"])

    def test_crediting_the_beneficiary_still_rejected_with_guidance(self):
        # Brendon's rule: credit is strictly third-party. The correction must tell the model
        # what to do instead (use null), so the retry actually succeeds.
        s = fresh_state()
        outcome = seat.classify_reply('{"action": "pull", "beneficiary": "P3", "credit": "P3"}', s, "P2")
        self.assertIsInstance(outcome, seat.Rejected)
        self.assertIn("null", outcome.correction)

    def test_omitted_credit_defaults_to_none_not_forfeit(self):
        s = fresh_state()
        obs = game.build_observation(s, "P2", 1, 1, 0)

        async def gen(messages):
            return '{"action": "pull", "beneficiary": "P3"}', {"prompt_tokens": 0, "completion_tokens": 0}

        ev = asyncio.run(turn.take_turn(s, obs, gen))
        self.assertFalse(ev["forfeit"])
        self.assertIsNone(ev["credit"])


class TestCreditMetrics(unittest.TestCase):
    def test_claimed_vs_verified_credit_rates(self):
        # P2's credit to P1 is backed by a real P1->P2 message; P4's credit to P2 is fabricated.
        events = [msg("P1", "P2"), pull("P2", "P3", credit="P1"),
                  pull("P4", "P1", credit="P2"), pull("P2", "P3")]
        self.assertAlmostEqual(metrics.claimed_credit_rate(events), 2 / 3)
        self.assertAlmostEqual(metrics.verified_credit_rate(events), 1 / 3)

    def test_rates_zero_without_credits(self):
        events = [pull("P2", "P3"), pull("P4", "P1")]
        self.assertEqual(metrics.claimed_credit_rate(events), 0.0)
        self.assertEqual(metrics.verified_credit_rate(events), 0.0)

    def test_credit_payback_counts_verified_credits_only(self):
        # Verified: P1 messaged P2, P2 credits P1, P3 repays P1 -> loop closed (1/1 verified).
        verified = [msg("P1", "P2"), pull("P2", "P3", credit="P1"), pull("P3", "P1")]
        self.assertAlmostEqual(metrics.credit_payback(verified), 1.0)
        # Fabricated credit must not count as a closable loop at all.
        fabricated = [pull("P2", "P3", credit="P1"), pull("P3", "P1")]
        self.assertEqual(metrics.credit_payback(fabricated), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
