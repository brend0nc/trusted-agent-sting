"""Trusted-agent sting: builder, block validation, scripted opening, private note, and outcomes. Offline."""
import copy
import json
import random
import unittest
from types import SimpleNamespace

from delegated_influence import game, protocols, sting, turn


def build(**kwargs):
    return sting.trusted_sting(player_models=("grok-4.3",), seeds=(61, 62, 63, 64, 65, 66), **kwargs)


def sample_for(task, condition, index=0):
    return [s for s in task.dataset if s.metadata["protocol"]["condition"] == condition][index]


def seat_generators(script, calls):
    """script(seat, round, action_number) -> action dict. P4 must never be asked."""
    counters = {}

    def make(seat):
        async def generate(messages):
            if seat == sting.TRUSTED:
                raise AssertionError("the scripted trusted seat must never call a model")
            calls.setdefault(seat, []).append(messages)
            n = counters[seat] = counters.get(seat, 0) + 1
            return json.dumps(script(seat, n)), {"prompt_tokens": 10, "completion_tokens": 5}
        return generate
    return {seat: make(seat) for seat in (*sting.ORDINARY, sting.TRUSTED)}


async def play(sample, script):
    state = game.state_from_metadata(sample.metadata)
    calls = {}
    await turn.drive_episode(state, random.Random(sample.metadata["seed"]), seat_generators(script, calls))
    record = {**sample.metadata, "events": state["events"], "scores": state["scores"]}
    return state, record, calls


class TestBuilder(unittest.TestCase):
    def test_one_game_per_condition_per_seed_with_rotating_roles(self):
        task = build()
        self.assertEqual(len(task.dataset), 6 * len(sting.CONDITIONS))
        orders = {(s.metadata["protocol"]["named"], s.metadata["protocol"]["actual"]) for s in task.dataset}
        self.assertEqual(orders, set(sting.ROLE_ORDERS))
        ally_roles = [sting.roles(s.metadata["protocol"])[s.metadata["protocol"]["ally"]]
                      for s in task.dataset if s.metadata["protocol"]["condition"] == "alliance"]
        self.assertEqual(sorted(ally_roles), sorted(sting.ALLY_ROLES * 2))
        protocols.validate_matched_pairs([task])

    def test_offer_arms_message_every_ordinary_seat_and_controls_do_not(self):
        task = build()
        for condition in sting.CONDITIONS:
            messages = [a for a in sample_for(task, condition).metadata["protocol"]["prelude"]
                        if a["action"] == "message"]
            if condition in sting.OFFER_CONDITIONS:
                self.assertEqual({a["to"] for a in messages}, set(sting.ORDINARY))
                self.assertEqual({a["text"] for a in messages}, {sting.OFFER_TEXTS[condition]})
            else:
                self.assertEqual(messages, [])

    def test_offer_texts_share_the_factual_sentence_and_fit_the_message_limit(self):
        from delegated_influence import communication
        for text in sting.OFFER_TEXTS.values():
            self.assertTrue(text.startswith(sting.FACT))
            self.assertLessEqual(len(text), communication.MAX_MESSAGE_CHARS)

    def test_blocks_reject_a_difference_outside_the_condition(self):
        task = build()
        tampered = copy.deepcopy(task)
        tampered.dataset[1].metadata["self_pull_reward"] = 0
        with self.assertRaisesRegex(ValueError, "differs outside its condition"):
            protocols.validate_matched_pairs([tampered])
        missing = SimpleNamespace(dataset=list(copy.deepcopy(task).dataset)[1:])
        with self.assertRaisesRegex(ValueError, "one game per declared condition"):
            protocols.validate_matched_pairs([missing])
        edited = copy.deepcopy(task)
        edited.dataset[0].metadata["protocol"]["prelude"][3]["text"] = "something else"
        with self.assertRaisesRegex(ValueError, "controlled stimulus"):
            protocols.validate_matched_pairs([edited])

    def test_only_the_alliance_condition_names_an_ally(self):
        protocol = copy.deepcopy(sample_for(build(), "direct").metadata["protocol"])
        protocol["ally"] = "P1"
        with self.assertRaisesRegex(ValueError, "only the alliance condition"):
            sting.state_settings(protocol)


class TestEpisode(unittest.IsolatedAsyncioTestCase):
    async def test_offer_reaches_every_seat_and_p4_never_takes_a_free_turn(self):
        sample = sample_for(build(), "direct")
        state, record, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        opening = [e for e in state["events"] if e["round"] == 0]
        self.assertEqual(len(opening), 6)
        self.assertTrue(all(e["agent"] == "P4" and e["forced"] and e["control_source"] == "scripted"
                            for e in opening))
        self.assertFalse([e for e in state["events"] if e["agent"] == "P4" and e["round"] > 0
                          and e.get("type") != "settlement"])
        for seat in sting.ORDINARY:
            first_prompt = calls[seat][0][1]["content"]
            self.assertIn(sting.OFFER_TEXTS["direct"], first_prompt)
            self.assertEqual(len(calls[seat]), 2 * 3)  # 2 rounds x 3 actions

    async def test_alliance_note_goes_only_to_the_allied_seat(self):
        sample = sample_for(build(), "alliance")
        ally = sample.metadata["protocol"]["ally"]
        _, _, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        for seat in sting.ORDINARY:
            system_prompt = calls[seat][0][0]["content"]
            self.assertEqual(sting.ALLIANCE_NOTE in system_prompt, seat == ally)
            self.assertNotIn(sting.OFFER_TEXTS["direct"], calls[seat][0][1]["content"])

    async def test_a_dispute_settles_the_false_claim_and_pays_the_actual_seat(self):
        sample = sample_for(build(), "hint")
        actual = sample.metadata["protocol"]["actual"]

        def script(seat, n):
            if seat == actual and n == 1:
                return {"action": "message", "to": "P4", "text": "No deal, the ledger says otherwise.",
                        "report": {"action": "dispute", "statement": "R0.3"}}
            return {"action": "pull", "beneficiary": seat}
        state, record, _ = await play(sample, script)
        settlement = next(e for e in state["events"] if e.get("type") == "settlement")
        self.assertEqual(settlement["verdict"], "false")
        self.assertEqual(settlement["disputers"], [actual])
        rows = {row["seat"]: row for row in sting.outcomes([record])}
        self.assertTrue(rows[actual]["disputed"])
        self.assertEqual(rows[actual]["dispute_at"], "R1.1")
        self.assertEqual(rows[actual]["messages_to_p4"][0]["text"], "No deal, the ledger says otherwise.")
        self.assertEqual(rows[actual]["role"], "actual")
        self.assertEqual(rows[actual]["settlement_verdict"], "false")
        others = [row for seat, row in rows.items() if seat != actual]
        self.assertFalse(any(row["disputed"] for row in others))

    async def test_undisputed_claim_pays_p4_and_named_and_counts_pulls_for_p4(self):
        sample = sample_for(build(), "none")
        named = sample.metadata["protocol"]["named"]
        state, record, _ = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        settlement = next(e for e in state["events"] if e.get("type") == "settlement")
        self.assertEqual(settlement["verdict"], "unchallenged")
        self.assertEqual(settlement["changes"], {"P4": 1, named: 1})
        for row in sting.outcomes([record]):
            self.assertEqual(row["pulls_for_p4"], {"1": 3, "2": 3})
            self.assertEqual(row["forfeits"], 0)

    async def test_forfeits_are_counted_apart_from_choices(self):
        sample = sample_for(build(), "benign")
        state, record, _ = await play(sample, lambda seat, n: {"action": "fly"})
        for row in sting.outcomes([record]):
            self.assertIsNone(row["disputed"])
            self.assertEqual(row["dispute_status"], "no_valid_turn")
            self.assertEqual(row["dispute_window_valid_turns"], 0)
            self.assertEqual(row["forfeits"], row["turns"])
            self.assertEqual(row["forfeits_by_round"], {"1": 3, "2": 3})
            self.assertEqual(row["forfeit_reasons"], ["parse_failure"])

    async def test_round_one_forfeits_alone_do_not_look_like_silence(self):
        sample = sample_for(build(), "direct")
        # Invalid in round 1 (3 turns), valid pulls in round 2: no dispute choice was observed.
        state, record, _ = await play(sample, lambda seat, n: {"action": "fly"} if n <= 6
                                      else {"action": "pull", "beneficiary": seat})
        for row in sting.outcomes([record]):
            self.assertEqual(row["dispute_status"], "no_valid_turn")
            self.assertEqual(row["forfeits_by_round"], {"1": 3, "2": 0})


if __name__ == "__main__":
    unittest.main()
