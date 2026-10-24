"""seat.classify_reply: pure per-reply classification. No network, no Inspect.

Run:  PYTHONPATH=src python3.13 tests/test_seat.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import game, seat  # noqa: E402


def fresh_state():
    return game.new_state(["P1", "P2", "P3", "P4"], 1, 1)


class TestClassifyReply(unittest.TestCase):
    def test_valid_pull_is_parsed(self):
        out = seat.classify_reply('{"action": "pull", "beneficiary": "P3"}', fresh_state(), "P1")
        self.assertIsInstance(out, seat.Parsed)
        self.assertEqual(out.action, {"action": "pull", "beneficiary": "P3"})

    def test_empty_is_rejected_empty_completion(self):
        for content in ("", "   \n  "):
            out = seat.classify_reply(content, fresh_state(), "P1")
            self.assertIsInstance(out, seat.Rejected)
            self.assertEqual(out.reason, "empty_completion")

    def test_no_json_is_rejected_parse_failure(self):
        out = seat.classify_reply("let me think about it", fresh_state(), "P1")
        self.assertIsInstance(out, seat.Rejected)
        self.assertEqual(out.reason, "parse_failure")

    def test_self_pull_is_rejected_distinctly(self):
        out = seat.classify_reply('{"action": "pull", "beneficiary": "P1"}', fresh_state(), "P1")
        self.assertIsInstance(out, seat.Rejected)
        self.assertEqual(out.reason, "invalid_self_pull")

    def test_correction_names_the_problem(self):
        out = seat.classify_reply('{"action": "pull", "beneficiary": "ZZ"}', fresh_state(), "P1")
        self.assertIsInstance(out, seat.Rejected)
        self.assertIn("invalid", out.correction.lower())

    def test_valid_message_is_parsed(self):
        out = seat.classify_reply('{"action": "message", "to": "P2", "text": "hi"}', fresh_state(), "P1")
        self.assertIsInstance(out, seat.Parsed)
        self.assertEqual(out.action["action"], "message")

    def test_last_action_object_wins_over_preamble_decoy(self):
        reply = 'Maybe {"action": "pull", "beneficiary": "P2"} or {"action": "pull", "beneficiary": "P3"}'
        out = seat.classify_reply(reply, fresh_state(), "P1")
        self.assertIsInstance(out, seat.Parsed)
        self.assertEqual(out.action["beneficiary"], "P3")


class TestExtractJson(unittest.TestCase):
    def test_variants(self):
        pull = {"action": "pull", "beneficiary": "P2"}
        self.assertEqual(seat.extract_json('{"action": "pull", "beneficiary": "P2"}'), pull)
        self.assertEqual(seat.extract_json('```json\n{"action": "pull", "beneficiary": "P2"}\n```'), pull)
        self.assertEqual(seat.extract_json('I will pull. {"action": "pull", "beneficiary": "P2"} Done.'), pull)
        self.assertIsNone(seat.extract_json("no json here"))
        self.assertIsNone(seat.extract_json("{broken"))

    def test_lenient_dialects(self):
        # common model dialects shouldn't be miscounted as parse-failure forfeits (instrument noise)
        pull = {"action": "pull", "beneficiary": "P3"}
        self.assertEqual(seat.extract_json('{"action": "pull", "beneficiary": "P3",}'), pull)  # trailing comma
        self.assertEqual(seat.extract_json("{'action': 'pull', 'beneficiary': 'P3'}"), pull)  # single quotes
        self.assertEqual(
            seat.extract_json('{"action": "pull", "beneficiary": "P3", "credit": None}'),
            {"action": "pull", "beneficiary": "P3", "credit": None})  # python literal

    def test_lenient_still_prefers_last_action_and_clean_json(self):
        text = 'e.g. {"action":"message","to":"P2","text":"x"}\nchoice: {"action":"pull","beneficiary":"P3",}'
        self.assertEqual(seat.extract_json(text), {"action": "pull", "beneficiary": "P3"})

    def test_mixed_quotes_python_literal(self):
        reply = "{'action': 'say', 'text': \"I don't think R0.3 is right\"}"
        self.assertEqual(seat.extract_json(reply), {"action": "say", "text": "I don't think R0.3 is right"})

    def test_single_object_skips_prose_brackets(self):
        # A stray brace or bracket in prose must not turn a valid action into a parse-failure forfeit.
        pull = {"action": "pull", "beneficiary": "P2", "credit": None}
        self.assertEqual(seat.extract_json('No {drama} here. {"action": "pull", "beneficiary": "P2", "credit": null}',
                                           single_object=True), pull)
        self.assertEqual(seat.extract_json('Entry [R0.3] is odd. {"action": "pull", "beneficiary": "P2", "credit": null}',
                                           single_object=True), pull)

    def test_single_object_still_rejects_ambiguous_replies(self):
        invalid_parent = '{"action": "pull", "beneficiary": P2, "report": {"action": "say", "text": "hi"}}'
        self.assertIsNone(seat.extract_json(invalid_parent, single_object=True))  # inner report never recovered
        two = '{"action": "pull", "beneficiary": "P1"} {"action": "pull", "beneficiary": "P2"}'
        self.assertIsNone(seat.extract_json(two, single_object=True))
        self.assertIsNone(seat.extract_json('[{"action": "pull", "beneficiary": "P2"}]', single_object=True))
        self.assertIsNone(seat.extract_json("{'action': 'pull', 'action': 'say'}", single_object=True))
        unquoted = '{report: {action: "dispute", statement: "R0.3"}} {"action":"pull","beneficiary":"P1"}'
        self.assertIsNone(seat.extract_json(unquoted, single_object=True))  # a malformed dispute is never skipped


class TestParseIntent(unittest.TestCase):
    def test_reason_is_returned_alongside_plan(self):
        reply = '{"action": "pull", "beneficiary": "P3", "reason": "P3 backed me last round"}'
        planned, reason = seat.parse_intent(reply, fresh_state(), "P1")
        self.assertEqual(planned["beneficiary"], "P3")
        self.assertIn("backed me", reason)


class TestObservationView(unittest.TestCase):
    def test_render_shows_scores_ledger_and_inbox(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 2, 2)
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P2", 1, 1, 0), {"action": "pull", "beneficiary": "P1"}))
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P3", 1, 1, 1), {"action": "message", "to": "P1", "text": "help me"}))
        text = seat.render_observation(game.build_observation(state, "P1", 1, 2, 0))
        self.assertIn("P1: 1", text)          # score reflects the pull
        self.assertIn("pulled for P1", text)  # public ledger line
        self.assertIn("help me", text)        # this seat's private inbox

    def test_punished_seat_reads_who_punished_it(self):
        # A punished seat must be able to read who hit it in its observation — the only channel it
        # sees — else it cannot retaliate, forgive, or deter, defeating the punish mechanic.
        state = game.new_state(["P1", "P2", "P3", "P4"], 2, 2, punish_enabled=True)
        game.apply_event(state, game.make_event(
            game.build_observation(state, "P1", 1, 1, 0), {"action": "punish", "target": "P2"}))
        text = seat.render_observation(game.build_observation(state, "P2", 1, 2, 0))
        self.assertIn("P1", text)                  # who punished it
        self.assertIn("punished", text.lower())    # the punish word, in plain English

    def test_attacker_clause_drops_message_coaching_when_messages_off(self):
        # Exp 5 (no-talk control): the attacker must still be pushed to win, but must not be
        # told to use a channel the rules block — that would confound the 04-vs-05 contrast.
        on = seat.build_system_prompt("P1", ["P1", "P2", "P3"], 5, 5, persona="attacker")
        off = seat.build_system_prompt("P1", ["P1", "P2", "P3"], 5, 5, persona="attacker",
                                       messages_off=True)
        self.assertIn("private messages", on)
        self.assertNotIn("message", off.split("playing to win")[-1])  # persona tail: no coaching
        self.assertIn("playing to win", off)  # the win-push itself stays

    def test_system_prompt_states_the_rules(self):
        sp = seat.build_system_prompt("P1", ["P1", "P2", "P3"], 5, 5)
        self.assertIn("P1", sp)
        self.assertIn("JSON", sp)
        self.assertIn("800", sp)  # the message-length cap is stated, not silent
        self.assertNotIn("full history", sp.lower())
        self.assertIn("current and previous round", sp.lower())
        self.assertIn("summar", sp.lower())

    def test_render_compacts_repeated_old_pulls(self):
        state = game.new_state(["P1", "P2", "P3"], 5, 5)
        state["ledger"] = [
            {"round": 1, "action_number": i, "puller": "P1", "beneficiary": "P2",
             "credit": None, "forced": False}
            for i in range(1, 101)
        ]
        text = seat.render_observation(game.build_observation(state, "P3", 5, 1, 0))
        self.assertIn("Earlier public record", text)
        self.assertIn("P1>P2: R1=100", text)
        self.assertNotIn("R1.100", text)  # exact old action lines are not repeated

    def test_punish_action_in_menu_only_when_enabled(self):
        off = seat.build_system_prompt("P1", ["P1", "P2", "P3"], 5, 5)
        on = seat.build_system_prompt("P1", ["P1", "P2", "P3"], 5, 5, punish_enabled=True)
        self.assertNotIn('"punish"', off)
        self.assertIn('"action": "punish"', on)   # the JSON example is documented
        self.assertIn("remove 1", on.lower())      # the effect (pay 1 to remove 1) is stated

    def test_render_flags_ring_message_restriction(self):
        ring = game.new_state(["P1", "P2", "P3", "P4"], 1, 1, topology="ring")
        text = seat.render_observation(game.build_observation(ring, "P1", 1, 1, 0))
        self.assertIn("can only message: P2, P4", text)  # P3 is unreachable on the ring

    def test_render_omits_restriction_when_complete(self):
        full = game.new_state(["P1", "P2", "P3", "P4"], 1, 1)
        text = seat.render_observation(game.build_observation(full, "P1", 1, 1, 0))
        self.assertNotIn("can only message", text)

    def test_deadline_warning_in_last_two_rounds(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 5, 5)
        # round 4 of 5 = two rounds left; 5 actions this round + 5 next = 10 actions in the game
        two_left = seat.render_observation(game.build_observation(state, "P1", 4, 1, 0))
        self.assertIn("2 rounds left", two_left)
        self.assertIn("10 actions", two_left)
        one_left = seat.render_observation(game.build_observation(state, "P1", 5, 1, 0))
        self.assertIn("1 round left", one_left)

    def test_no_deadline_warning_before_last_two_rounds(self):
        state = game.new_state(["P1", "P2", "P3", "P4"], 5, 5)
        early = seat.render_observation(game.build_observation(state, "P1", 3, 1, 0))  # 3 rounds left
        self.assertNotIn("round left", early)


class TestPersonas(unittest.TestCase):
    AGENTS = ["P1", "P2", "P3"]

    def sp(self, persona):
        return seat.build_system_prompt("P1", self.AGENTS, 5, 5, persona=persona)

    def test_neutral_is_clean_base(self):
        low = self.sp("neutral").lower()
        self.assertNotIn("playing to win", low)   # no attacker clause
        self.assertNotIn("skepticism", low)       # no inoculation clause

    def test_attacker_seeks_persuasion_and_permits_bargaining(self):
        low = self.sp("attacker").lower()
        self.assertIn("persuad", low)
        self.assertIn("pull for me", low)  # the bargaining example in the attacker clause

    def test_inoculated_warns_and_cites_ledger(self):
        low = self.sp("inoculated").lower()
        self.assertIn("ledger", low)
        self.assertIn("skepti", low)

    def test_personas_extend_the_base_not_replace_it(self):
        base = seat.build_system_prompt("P1", self.AGENTS, 5, 5)  # default neutral
        self.assertTrue(self.sp("attacker").startswith(base))
        self.assertTrue(self.sp("inoculated").startswith(base))


if __name__ == "__main__":
    unittest.main(verbosity=2)
