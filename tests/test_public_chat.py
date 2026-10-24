"""Public chat: a `say` action posts a statement everyone can see, behind a `public_chat` knob
(off by default). It costs a turn, agents can say anything, and we OBSERVE what they use it for
(e.g. spontaneously reporting non-cooperators) — reporting is watched, not a designed action.

Run:  PYTHONPATH=src python3.13 tests/test_public_chat.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, metrics, seat, turn  # noqa: E402

AGENTS = ["P1", "P2", "P3", "P4"]


def chat_state():
    return game.new_state(AGENTS, 1, 5, public_chat=True)


class TestSayValidation(unittest.TestCase):
    def test_say_accepted_when_chat_on(self):
        ok, _ = game.validate_action(chat_state(), "P1", {"action": "say", "text": "hello all"})
        self.assertTrue(ok)

    def test_say_rejected_when_chat_off(self):
        s = game.new_state(AGENTS, 1, 5)  # default: no public chat
        ok, _ = game.validate_action(s, "P1", {"action": "say", "text": "hello all"})
        self.assertFalse(ok)

    def test_say_requires_text(self):
        ok, _ = game.validate_action(chat_state(), "P1", {"action": "say", "text": "  "})
        self.assertFalse(ok)

    def test_say_over_length_cap_rejected(self):
        ok, err = game.validate_action(chat_state(), "P1", {"action": "say", "text": "x" * 801})
        self.assertFalse(ok)
        self.assertIn("800", err)  # the correction names the limit so the retry succeeds


class TestSayEffect(unittest.TestCase):
    def _say(self, state, who, text, action_number=1):
        obs = game.build_observation(state, who, 1, action_number, 0)
        game.apply_event(state, game.make_event(obs, {"action": "say", "text": text}))

    def test_statement_visible_to_everyone(self):
        s = chat_state()
        self._say(s, "P1", "P3 is not cooperating")
        for agent in AGENTS:  # everyone sees the same public chat
            obs = game.build_observation(s, agent, 1, 2, 0)
            self.assertEqual(obs["chat"][0]["speaker"], "P1")
            self.assertIn("not cooperating", obs["chat"][0]["text"])

    def test_say_scores_nothing_and_burns_the_turn(self):
        s = chat_state()
        self._say(s, "P1", "vote for me")
        self.assertEqual(sum(s["scores"].values()), 0)
        self.assertEqual(len(s["events"]), 1)
        self.assertEqual(s["events"][0]["type"], "say")


class TestChatPrompt(unittest.TestCase):
    def test_prompt_offers_say_when_on(self):
        p = seat.build_system_prompt("P1", AGENTS, 1, 5, public_chat=True).lower()
        self.assertIn('"say"', p)
        self.assertIn("everyone", p)

    def test_prompt_silent_on_say_when_off(self):
        p = seat.build_system_prompt("P1", AGENTS, 1, 5).lower()
        self.assertNotIn('"say"', p)

    def test_chat_only_game_keeps_credit(self):
        # messages off but chat on: agents can still be asked (publicly), so credit stays meaningful.
        p = seat.build_system_prompt("P1", AGENTS, 1, 5, messages_off=True, public_chat=True).lower()
        self.assertIn('"say"', p)
        self.assertIn("credit", p)

    def test_render_shows_public_chat(self):
        s = chat_state()
        obs = game.build_observation(s, "P2", 1, 2, 0)
        obs["chat"] = [{"seq": 0, "round": 1, "action_number": 1, "speaker": "P1", "text": "hear ye"}]
        self.assertIn("hear ye", seat.render_observation(obs))
        self.assertIn("P1", seat.render_observation(obs))


class TestChatTurnAndMetrics(unittest.TestCase):
    def test_take_turn_carries_say_into_the_event(self):
        s = chat_state()
        obs = game.build_observation(s, "P1", 1, 1, 0)

        async def gen(messages):
            return '{"action": "say", "text": "greetings"}', {"prompt_tokens": 0, "completion_tokens": 0}

        ev = asyncio.run(turn.take_turn(s, obs, gen))
        self.assertFalse(ev["forfeit"])
        self.assertEqual(ev["type"], "say")
        self.assertEqual(ev["text"], "greetings")

    def test_public_chat_rate(self):
        events = [
            {"forfeit": False, "type": "say", "agent": "P1", "text": "hi"},
            {"forfeit": False, "type": "pull", "agent": "P1", "beneficiary": "P2"},
            {"forfeit": False, "type": "pull", "agent": "P2", "beneficiary": "P1"},
        ]
        rate = metrics.public_chat_rate(events)
        self.assertAlmostEqual(rate["P1"], 0.5)  # 1 say of 2 actions
        self.assertAlmostEqual(rate["P2"], 0.0)

    def test_transcript_shows_says(self):
        events = [{"round": 1, "action_number": 1, "agent": "P1", "forfeit": False,
                   "type": "say", "text": "join me", "to": None, "beneficiary": None, "credit": None}]
        out = metrics.render_transcript({"P1": "m"}, events, {"P1": 0})
        self.assertIn("P1 says: join me", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
