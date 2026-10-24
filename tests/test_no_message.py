"""The no-message control (D8): messaging disabled -> pure reciprocity floor, the reciprocity baseline.

`messages_off=True` makes the engine reject message actions (pulls only) and the prompt omit messaging,
so agents can't waste turns on rejected messages.

Run:  PYTHONPATH=src python3.13 tests/test_no_message.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, seat  # noqa: E402


class TestNoMessageEngine(unittest.TestCase):
    def test_messages_off_rejects_message_but_allows_pull(self):
        s = game.new_state(["P1", "P2", "P3"], 1, 1, messages_off=True)
        ok_msg, _ = game.validate_action(s, "P1", {"action": "message", "to": "P2", "text": "hi"})
        ok_pull, _ = game.validate_action(s, "P1", {"action": "pull", "beneficiary": "P2"})
        self.assertFalse(ok_msg)
        self.assertTrue(ok_pull)

    def test_messages_on_by_default(self):
        s = game.new_state(["P1", "P2", "P3"], 1, 1)
        ok, _ = game.validate_action(s, "P1", {"action": "message", "to": "P2", "text": "hi"})
        self.assertTrue(ok)

    def test_observation_exposes_messages_off(self):
        s = game.new_state(["P1", "P2"], 1, 1, messages_off=True)
        self.assertTrue(game.build_observation(s, "P1", 1, 1, 0)["messages_off"])


class TestNoMessagePrompt(unittest.TestCase):
    def test_prompt_omits_messaging_when_off(self):
        sp = seat.build_system_prompt("P1", ["P1", "P2"], 5, 5, messages_off=True).lower()
        self.assertNotIn("message", sp)  # no messaging offered at all
        self.assertIn("pull", sp)

    def test_prompt_includes_messaging_by_default(self):
        self.assertIn("message", seat.build_system_prompt("P1", ["P1", "P2"], 5, 5).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
