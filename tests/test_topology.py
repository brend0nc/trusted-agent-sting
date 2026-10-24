"""Communication topology (D5): messaging is restricted by the comm graph; pulls stay global.

Run:  PYTHONPATH=src python3.13 tests/test_topology.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game  # noqa: E402

SEATS = ["P1", "P2", "P3", "P4"]


class TestTopology(unittest.TestCase):
    def test_default_topology_is_complete(self):
        obs = game.build_observation(game.new_state(["P1", "P2", "P3"], 1, 1), "P1", 1, 1, 0)
        self.assertEqual(set(obs["can_message"]), {"P2", "P3"})

    def test_complete_lets_anyone_message_anyone(self):
        s = game.new_state(SEATS, 1, 1, topology="complete")
        ok, _ = game.validate_action(s, "P1", {"action": "message", "to": "P3", "text": "hi"})
        self.assertTrue(ok)

    def test_ring_restricts_messages_to_neighbors(self):
        # ring P1-P2-P3-P4-P1: P1's neighbors are P2 and P4, not P3.
        s = game.new_state(SEATS, 1, 1, topology="ring")
        ok_nb, _ = game.validate_action(s, "P1", {"action": "message", "to": "P2", "text": "hi"})
        ok_far, err = game.validate_action(s, "P1", {"action": "message", "to": "P3", "text": "hi"})
        self.assertTrue(ok_nb)
        self.assertFalse(ok_far)
        self.assertIn("neighbor", err.lower())

    def test_ring_pulls_stay_global(self):
        # A non-neighbor pull is fine — topology only restricts messaging.
        s = game.new_state(SEATS, 1, 1, topology="ring")
        ok, _ = game.validate_action(s, "P1", {"action": "pull", "beneficiary": "P3"})
        self.assertTrue(ok)

    def test_observation_exposes_reachable_recipients(self):
        s = game.new_state(SEATS, 1, 1, topology="ring")
        self.assertEqual(set(game.build_observation(s, "P1", 1, 1, 0)["can_message"]), {"P2", "P4"})


class TestLedgerVisibility(unittest.TestCase):
    """Recent public actions stay detailed; old public facts are compact, while engine state stays full."""

    def test_ledger_details_only_current_and_previous_round(self):
        s = game.new_state(["P1", "P2", "P3"], 5, 2)
        s["ledger"] = [
            {"round": 1, "action_number": 1, "puller": "P2", "beneficiary": "P1"},
            {"round": 2, "action_number": 1, "puller": "P3", "beneficiary": "P1"},
            {"round": 3, "action_number": 1, "puller": "P2", "beneficiary": "P3"},
        ]
        obs = game.build_observation(s, "P1", 3, 1, 0)  # current round = 3
        self.assertEqual({e["round"] for e in obs["ledger"]}, {2, 3})

    def test_old_pulls_are_summarized_by_pair_and_round(self):
        s = game.new_state(["P1", "P2", "P3"], 5, 2)
        s["ledger"] = [
            {"round": 1, "action_number": 1, "puller": "P2", "beneficiary": "P1"},
            {"round": 1, "action_number": 2, "puller": "P2", "beneficiary": "P1"},
            {"round": 2, "action_number": 1, "puller": "P3", "beneficiary": "P1"},
        ]
        obs = game.build_observation(s, "P1", 3, 1, 0)
        self.assertEqual(obs["ledger_summary"]["through_round"], 1)
        self.assertEqual(obs["ledger_summary"]["pulls"], [{
            "puller": "P2", "beneficiary": "P1",
            "round_counts": [{"round": 1, "count": 2}],
        }])

    def test_observation_does_not_delete_full_engine_ledger(self):
        s = game.new_state(["P1", "P2", "P3"], 5, 2)
        s["ledger"] = [
            {"round": 1, "action_number": 1, "puller": "P2", "beneficiary": "P1"},
            {"round": 2, "action_number": 1, "puller": "P3", "beneficiary": "P1"},
            {"round": 3, "action_number": 1, "puller": "P2", "beneficiary": "P3"},
        ]
        game.build_observation(s, "P1", 3, 1, 0)
        self.assertEqual([e["round"] for e in s["ledger"]], [1, 2, 3])

    def test_active_mutual_round_streak_is_summarized(self):
        s = game.new_state(["P1", "P2", "P3"], 5, 2)
        s["ledger"] = [
            {"round": 1, "action_number": 1, "puller": "P1", "beneficiary": "P2"},
            {"round": 1, "action_number": 2, "puller": "P2", "beneficiary": "P1"},
            {"round": 2, "action_number": 1, "puller": "P1", "beneficiary": "P2"},
            {"round": 2, "action_number": 2, "puller": "P2", "beneficiary": "P1"},
        ]
        obs = game.build_observation(s, "P3", 3, 1, 0)
        self.assertEqual(obs["ledger_summary"]["mutual_streaks"], [{
            "left": "P1", "right": "P2", "rounds": 2, "through_round": 2,
        }])


if __name__ == "__main__":
    unittest.main(verbosity=2)
