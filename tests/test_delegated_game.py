"""Engine invariants: economy rules, validation, determinism. No network.

Scripted episodes run through the REAL driver (`turn.drive_episode`), with each seat's reply a
canned JSON string — so these tests exercise the same classify/validate/apply path as a live run.

Run:  python3.13 tests/test_delegated_game.py
"""
import asyncio
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, turn  # noqa: E402

AGENTS = ["P1", "P2", "P3", "P4"]
NO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}


def pull_next_reply(seat):
    """Scripted seat: always pull for the next agent in roster order."""
    nxt = AGENTS[(AGENTS.index(seat) + 1) % len(AGENTS)]

    async def generate(messages):
        return f'{{"action": "pull", "beneficiary": "{nxt}"}}', NO_USAGE
    return generate


def run_scripted(seed, generators=None, rounds=2, actions_per_round=3):
    state = game.new_state(AGENTS, rounds, actions_per_round)
    generators = generators or {a: pull_next_reply(a) for a in AGENTS}
    asyncio.run(turn.drive_episode(state, random.Random(seed), generators))
    return state


class TestEconomy(unittest.TestCase):
    def test_pull_scores_beneficiary_only(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)
        obs = game.build_observation(state, "P1", 1, 1, 0)
        game.apply_event(state, game.make_event(obs, {"action": "pull", "beneficiary": "P2"}))
        self.assertEqual(state["scores"], {"P1": 0, "P2": 1, "P3": 0, "P4": 0})
        self.assertEqual(state["ledger"],
                         [{"round": 1, "action_number": 1, "puller": "P1", "beneficiary": "P2",
                           "credit": None, "forced": False}])

    def test_forfeit_changes_nothing(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)
        obs = game.build_observation(state, "P1", 1, 1, 0)
        game.apply_event(state, game.make_event(obs, None, forfeit_reason="parse_failure"))
        self.assertEqual(sum(state["scores"].values()), 0)
        self.assertEqual(state["ledger"], [])
        self.assertEqual(state["messages"], [])
        self.assertTrue(state["events"][0]["forfeit"])

    def test_points_conserved(self):
        state = run_scripted(seed=3)
        valid_pulls = [e for e in state["events"] if not e["forfeit"] and e["type"] == "pull"]
        self.assertEqual(sum(state["scores"].values()), len(valid_pulls))
        self.assertEqual(len(valid_pulls), len(state["ledger"]))
        self.assertEqual(len(state["events"]), 4 * 2 * 3)  # agents x rounds x actions_per_round

    def test_forced_pull_event_has_reason_key(self):
        # Every Event constructor must set "reason" so readers can index it unconditionally.
        ev = game.make_forced_pull_event("P1", "P2", 1, 1)
        self.assertEqual(ev["reason"], "")


class TestPunishment(unittest.TestCase):
    def _enabled(self):
        return game.new_state(["P1", "P2", "P3", "P4"], 1, 3, punish_enabled=True)

    def test_punish_costs_actor_one_and_removes_one_from_target(self):
        state = self._enabled()
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "pull", "beneficiary": "P2"}))
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P3", 1, 2, 0), {"action": "punish", "target": "P2"}))
        self.assertEqual(state["scores"], {"P1": 0, "P2": 0, "P3": -1, "P4": 0})

    def test_target_floors_at_zero_but_actor_still_pays(self):
        state = self._enabled()  # P2 is at 0
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "punish", "target": "P2"}))
        self.assertEqual(state["scores"]["P2"], 0)    # floored, never negative
        self.assertEqual(state["scores"]["P1"], -1)   # actor still pays

    def test_punish_invalid_when_disabled_by_default(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)  # punish_enabled defaults False
        ok, err = game.validate_action(state, "P1", {"action": "punish", "target": "P2"})
        self.assertFalse(ok)
        self.assertTrue(err)

    def test_punish_valid_when_enabled(self):
        state = self._enabled()
        state["scores"]["P1"] = 1  # punishing needs at least 1 point
        ok, _ = game.validate_action(state, "P1", {"action": "punish", "target": "P2"})
        self.assertTrue(ok)

    def test_punish_invalid_with_zero_points(self):
        ok, err = game.validate_action(self._enabled(), "P1", {"action": "punish", "target": "P2"})
        self.assertFalse(ok)
        self.assertIn("at least 1 point", err)

    def test_cannot_punish_self_or_unknown(self):
        state = self._enabled()
        for bad in [{"action": "punish", "target": "P1"}, {"action": "punish", "target": "P9"}]:
            ok, err = game.validate_action(state, "P1", bad)
            self.assertFalse(ok, msg=repr(bad))
            self.assertTrue(err)

    def test_punish_event_shape(self):
        # The shared contract later units depend on: type=="punish", target set, others None.
        state = self._enabled()
        ev = game.make_event(game.build_observation(state, "P1", 1, 1, 0),
                             {"action": "punish", "target": "P2"})
        self.assertEqual(ev["type"], "punish")
        self.assertEqual(ev["target"], "P2")
        self.assertIsNone(ev["beneficiary"])
        self.assertIsNone(ev["to"])
        self.assertIsNone(ev["text"])

    def test_punish_recorded_to_the_public_ledger(self):
        # The punish must reach the ledger — the only channel seats observe — else the punished
        # seat cannot see who hit it and no one can retaliate, forgive, or deter.
        state = self._enabled()
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "punish", "target": "P2"}))
        self.assertEqual(len(state["ledger"]), 1)
        entry = state["ledger"][0]
        self.assertEqual(entry["action"], "punish")
        self.assertEqual(entry["agent"], "P1")    # the puncher
        self.assertEqual(entry["target"], "P2")   # the punished

    def test_punish_recorded_even_when_target_floored(self):
        # P2 sits at 0, so the punish changes no score — but it must STILL land in the ledger,
        # or a floored seat "sees nothing at all" about who hit it.
        state = self._enabled()
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "punish", "target": "P2"}))
        self.assertEqual(state["scores"]["P2"], 0)           # floored, no score change
        self.assertEqual(state["ledger"][0]["target"], "P2")  # yet still recorded

    def test_pull_ledger_entry_shape_unchanged_by_punish_support(self):
        # Regression: the pull entry keys the readers depend on must stay byte-identical.
        state = self._enabled()
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "pull", "beneficiary": "P2"}))
        self.assertEqual(state["ledger"], [{"round": 1, "action_number": 1, "puller": "P1",
                                            "beneficiary": "P2", "credit": None, "forced": False}])


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)

    def test_message_length_cap(self):
        # Cost + context control: 800 chars is ~1.65x the longest message observed in real play,
        # so it never binds on normal negotiation, only on runaway essays.
        ok, _ = game.validate_action(self.state, "P1",
                                     {"action": "message", "to": "P2", "text": "x" * 800})
        self.assertTrue(ok)   # at the cap: fine
        ok, err = game.validate_action(self.state, "P1",
                                       {"action": "message", "to": "P2", "text": "x" * 801})
        self.assertFalse(ok)
        self.assertIn("800", err)

    def test_valid_pull(self):
        ok, _ = game.validate_action(self.state, "P1", {"action": "pull", "beneficiary": "P2"})
        self.assertTrue(ok)

    def test_invalid_actions_rejected(self):
        for bad in [
            {"action": "pull", "beneficiary": "P1"},           # self-pull
            {"action": "pull", "beneficiary": "P9"},           # unknown
            {"action": "message", "to": "P1", "text": "hi"},   # message self
            {"action": "message", "to": "P2", "text": "  "},   # empty text
            {"action": "dance"},                                # unknown action
            None,                                               # no JSON at all
        ]:
            ok, err = game.validate_action(self.state, "P1", bad)
            self.assertFalse(ok, msg=repr(bad))
            self.assertTrue(err)


class TestCallTimeout(unittest.TestCase):
    def test_hung_call_forfeits_instead_of_freezing(self):
        # A model call that never returns must not freeze the episode: with call_timeout set,
        # take_turn bounds the wait and records an api_error forfeit.
        async def hangs(messages):
            await asyncio.sleep(5)
            return '{"action": "pull", "beneficiary": "P2"}', NO_USAGE

        state = game.new_state(AGENTS, 1, 1)
        obs = game.build_observation(state, "P1", 1, 1, 0)
        ev = asyncio.run(turn.take_turn(state, obs, hangs, call_timeout=0.05))
        self.assertTrue(ev["forfeit"])
        self.assertEqual(ev["forfeit_reason"], "api_error")

    def test_timeout_logs_a_readable_error(self):
        # str(TimeoutError()) is "" — the stored event must still say WHY the call failed,
        # so timeout forfeits are distinguishable from provider errors when diagnosing a run.
        async def hangs(messages):
            await asyncio.sleep(5)
            return '{"action": "pull", "beneficiary": "P2"}', NO_USAGE

        state = game.new_state(AGENTS, 1, 1)
        obs = game.build_observation(state, "P1", 1, 1, 0)
        ev = asyncio.run(turn.take_turn(state, obs, hangs, call_timeout=0.05))
        self.assertTrue(ev["usage"][-1]["error"])  # non-empty
        self.assertIn("Timeout", ev["usage"][-1]["error"])

    def test_fast_call_unaffected_by_timeout(self):
        async def quick(messages):
            return '{"action": "pull", "beneficiary": "P2"}', NO_USAGE

        state = game.new_state(AGENTS, 1, 1)
        obs = game.build_observation(state, "P1", 1, 1, 0)
        ev = asyncio.run(turn.take_turn(state, obs, quick, call_timeout=30))
        self.assertFalse(ev["forfeit"])
        self.assertEqual(ev["beneficiary"], "P2")


class TestDynamics(unittest.TestCase):
    def test_messages_deliver_immediately(self):
        async def p1_messages_p2(messages):
            return '{"action": "message", "to": "P2", "text": "hello"}', NO_USAGE

        generators = {a: pull_next_reply(a) for a in AGENTS}
        generators["P1"] = p1_messages_p2
        state = run_scripted(seed=3, generators=generators, rounds=1, actions_per_round=1)
        self.assertEqual([m["to"] for m in state["messages"]], ["P2"])
        obs = game.build_observation(state, "P2", 1, 2, 0)
        self.assertEqual(obs["inbox"][0]["text"], "hello")

    def test_turn_order_deterministic_under_seed(self):
        order1 = [e["agent"] for e in run_scripted(seed=7)["events"]]
        order2 = [e["agent"] for e in run_scripted(seed=7)["events"]]
        self.assertEqual(order1, order2)

    def test_turn_order_varies_across_seeds(self):
        # Sanity: shuffling actually depends on the seed (else "determinism" is vacuous).
        a = [e["agent"] for e in run_scripted(seed=1)["events"]]
        b = [e["agent"] for e in run_scripted(seed=999)["events"]]
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
