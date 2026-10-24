"""The self-pull variant: an outside option worth `self_pull_reward` (default 0 = pure economy).

Mixed mode (e.g. 0.5) lets an agent pull its own lever for a fraction of a point, creating the
act-vs-influence dilemma (grind alone vs invest in others). Pure mode is unchanged.

Run:  PYTHONPATH=src python3.13 tests/test_self_pull.py
"""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, metrics, seat, turn  # noqa: E402


def pull(agent, benef):
    return {"forfeit": False, "type": "pull", "agent": agent, "beneficiary": benef}


def header(agents=("P1", "P2", "P3")):
    return {"seat_map": {a: "m" for a in agents}, "config": {"actions_per_round": 3}}


class TestSelfPullEngine(unittest.TestCase):
    def test_pure_rejects_self_pull(self):
        s = game.new_state(["P1", "P2", "P3"], 1, 1)  # default self_pull_reward = 0
        ok, _ = game.validate_action(s, "P1", {"action": "pull", "beneficiary": "P1"})
        self.assertFalse(ok)

    def test_mixed_allows_self_pull(self):
        s = game.new_state(["P1", "P2", "P3"], 1, 1, self_pull_reward=0.5)
        ok, _ = game.validate_action(s, "P1", {"action": "pull", "beneficiary": "P1"})
        self.assertTrue(ok)

    def test_self_pull_scores_half_other_scores_one(self):
        s = game.new_state(["P1", "P2", "P3"], 1, 2, self_pull_reward=0.5)
        game.apply_event(s, game.make_event(game.build_observation(s, "P1", 1, 1, 0),
                                            {"action": "pull", "beneficiary": "P1"}))
        self.assertEqual(s["scores"]["P1"], 0.5)
        game.apply_event(s, game.make_event(game.build_observation(s, "P2", 1, 2, 0),
                                            {"action": "pull", "beneficiary": "P1"}))
        self.assertEqual(s["scores"]["P1"], 1.5)  # 0.5 self + 1.0 from P2

    def test_observation_exposes_self_pull_reward(self):
        s = game.new_state(["P1", "P2"], 1, 1, self_pull_reward=0.5)
        self.assertEqual(game.build_observation(s, "P1", 1, 1, 0)["self_pull_reward"], 0.5)


class TestSelfPullMetrics(unittest.TestCase):
    def test_self_pull_rate_per_agent(self):
        events = [pull("P1", "P1"), pull("P1", "P1"), pull("P1", "P2")]  # P1: 2 self of 3
        self.assertAlmostEqual(metrics.self_pull_rate(events)["P1"], 2 / 3)

    def test_self_pulls_excluded_from_capture(self):
        # P1's self-pulls must not dilute its share matrix: only the 1 other-directed pull counts.
        events = [pull("P1", "P1"), pull("P1", "P1"), pull("P1", "P2")]
        shares = metrics.pull_shares(header(), events)
        self.assertAlmostEqual(shares["P1"]["P2"], 1.0)  # 1 / 1 other-directed, not 1 / 3


class TestSelfPullPrompt(unittest.TestCase):
    def test_pure_prompt_forbids_self_pull(self):
        self.assertIn("cannot pull", seat.build_system_prompt("P1", ["P1", "P2"], 5, 5).lower())

    def test_mixed_prompt_offers_self_pull(self):
        sp = seat.build_system_prompt("P1", ["P1", "P2"], 5, 5, self_pull_reward=0.5)
        self.assertIn("0.5", sp)
        self.assertIn("yourself", sp.lower())


class TestSelfPullDrive(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_drive_allows_self_pull_and_scores(self):
        def always(reply):
            async def g(messages):
                return reply, {"prompt_tokens": 1, "completion_tokens": 1}
            return g
        state = game.new_state(["P1", "P2"], 1, 2, self_pull_reward=0.5)
        gens = {"P1": always('{"action": "pull", "beneficiary": "P1"}'),   # P1 self-pulls
                "P2": always('{"action": "pull", "beneficiary": "P1"}')}   # P2 pulls for P1
        await turn.drive_episode(state, random.Random(0), gens)
        self.assertEqual(state["scores"]["P1"], 3.0)  # 2 self (×0.5) + 2 from P2 (×1)
        self.assertEqual(metrics.self_pull_rate(state["events"])["P1"], 1.0)
        self.assertTrue(all(not e["forfeit"] for e in state["events"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
