"""turn.take_turn + turn.drive: the async seat-turn loop and episode driver (model injected).

No network, no Inspect — the model seam is a fake async `generate(messages) -> (content, usage)`.

Run:  PYTHONPATH=src python3.13 tests/test_turn.py
"""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, turn  # noqa: E402


def fresh():
    state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)
    return state, game.build_observation(state, "P1", 1, 1, 0)


def queued(replies, calls=None):
    """Fake async model seam. Each reply is a string, None (no content), or an Exception."""
    replies = list(replies)

    async def generate(messages):
        if calls is not None:
            calls.append(messages)
        r = replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r, {"prompt_tokens": 100, "completion_tokens": 10}
    return generate


def always(reply):
    async def generate(messages):
        return reply, {"prompt_tokens": 10, "completion_tokens": 5}
    return generate


class TestTakeTurn(unittest.IsolatedAsyncioTestCase):
    async def test_valid_first_try(self):
        state, obs = fresh()
        event = await turn.take_turn(state, obs, queued(['{"action": "pull", "beneficiary": "P3"}']))
        self.assertFalse(event["forfeit"])
        self.assertEqual(event["type"], "pull")
        self.assertEqual(event["beneficiary"], "P3")
        self.assertEqual(event["parse_retries"], 0)
        self.assertEqual(len(event["raw_responses"]), 1)

    async def test_reprompt_then_valid(self):
        state, obs = fresh()
        calls = []
        gen = queued(["let me think...", '{"action": "message", "to": "P2", "text": "hi"}'], calls)
        event = await turn.take_turn(state, obs, gen)
        self.assertFalse(event["forfeit"])
        self.assertEqual(event["type"], "message")
        self.assertEqual(event["parse_retries"], 1)
        self.assertEqual(len(event["raw_responses"]), 2)
        self.assertIn("invalid", calls[1][-1]["content"])  # the reprompt named the problem

    async def test_double_failure_forfeits_with_raws(self):
        state, obs = fresh()
        event = await turn.take_turn(state, obs, queued(["nope", "still nope"]))
        self.assertTrue(event["forfeit"])
        self.assertEqual(event["forfeit_reason"], "parse_failure")
        self.assertEqual(event["raw_responses"], ["nope", "still nope"])
        self.assertEqual(len(event["usage"]), 2)

    async def test_self_pull_forfeits_distinctly(self):
        state, obs = fresh()
        sp = '{"action": "pull", "beneficiary": "P1"}'
        event = await turn.take_turn(state, obs, queued([sp, sp]))
        self.assertTrue(event["forfeit"])
        self.assertEqual(event["forfeit_reason"], "invalid_self_pull")

    async def test_empty_forfeits_distinctly(self):
        state, obs = fresh()
        event = await turn.take_turn(state, obs, queued(["", ""]))
        self.assertTrue(event["forfeit"])
        self.assertEqual(event["forfeit_reason"], "empty_completion")

    async def test_api_error_on_exception_forfeits(self):
        state, obs = fresh()
        event = await turn.take_turn(state, obs, queued([RuntimeError("boom")]))
        self.assertTrue(event["forfeit"])
        self.assertEqual(event["forfeit_reason"], "api_error")
        self.assertEqual(event["raw_responses"], [])
        self.assertIn("error", event["usage"][0])

    async def test_api_error_on_none_content_forfeits(self):
        state, obs = fresh()
        event = await turn.take_turn(state, obs, queued([None]))
        self.assertTrue(event["forfeit"])
        self.assertEqual(event["forfeit_reason"], "api_error")

    async def test_out_of_credits_kills_the_run(self):
        # a 402 must abort loudly, not forfeit through the rest of the experiment
        state, obs = fresh()
        err = RuntimeError("Error code: 402 - {'error': {'message': 'Insufficient credits. "
                           "Add more using https://openrouter.ai/settings/credits', 'code': 402}}")
        with self.assertRaises(turn.OutOfCredits):
            await turn.take_turn(state, obs, queued([err]))


class TestDrive(unittest.IsolatedAsyncioTestCase):
    async def test_drive_runs_full_episode(self):
        # 2 seats x 2 rounds x 2 actions_per_round = each seat acts 4 times = 8 events.
        state = game.new_state(["P1", "P2"], 2, 2)
        gens = {
            "P1": always('{"action": "pull", "beneficiary": "P2"}'),
            "P2": always('{"action": "pull", "beneficiary": "P1"}'),
        }
        await turn.drive_episode(state, random.Random(0), gens)
        self.assertEqual(len(state["events"]), 8)
        self.assertEqual(state["scores"]["P1"], 4)  # P2 pulled for P1 all 4 of its turns
        self.assertEqual(state["scores"]["P2"], 4)
        self.assertTrue(all(not e["forfeit"] for e in state["events"]))


class TestElicitIntent(unittest.IsolatedAsyncioTestCase):
    async def test_reason_flows_into_the_event(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)
        gens = {"P1": always(
            '{"action": "pull", "beneficiary": "P2", "reason": "P2 backed me last round"}')}
        await turn.elicit_intent(state, random.Random(0), gens, "P1", 1)
        e = state["events"][0]
        self.assertEqual(e["planned"]["beneficiary"], "P2")
        self.assertIn("backed me", e["reason"])


class TestPersonaThreading(unittest.IsolatedAsyncioTestCase):
    async def test_attacker_persona_reaches_system_prompt(self):
        state, obs = fresh()
        calls = []
        await turn.take_turn(state, obs, queued(['{"action": "pull", "beneficiary": "P2"}'], calls), persona="attacker")
        sys_msg = calls[0][0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIn("playing to win", sys_msg["content"].lower())

    async def test_drive_applies_per_seat_personas(self):
        state = game.new_state(["P1", "P2"], 1, 1)
        calls = {"P1": [], "P2": []}
        gens = {"P1": queued(['{"action": "pull", "beneficiary": "P2"}'], calls["P1"]),
                "P2": queued(['{"action": "pull", "beneficiary": "P1"}'], calls["P2"])}
        await turn.drive_episode(state, random.Random(0), gens, personas={"P1": "attacker"})
        self.assertIn("playing to win", calls["P1"][0][0]["content"].lower())     # P1 = attacker
        self.assertNotIn("playing to win", calls["P2"][0][0]["content"].lower())  # P2 = neutral


if __name__ == "__main__":
    unittest.main(verbosity=2)
