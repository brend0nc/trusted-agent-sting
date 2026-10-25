"""Trusted-agent sting: builder, block validation, scripted opening, private note, and outcomes. Offline."""
import copy
from collections import defaultdict
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
            calls.setdefault(seat, []).append(messages)  # recorded first, so play() can assert P4 made none
            n = counters[seat] = counters.get(seat, 0) + 1
            return json.dumps(script(seat, n)), {"prompt_tokens": 10, "completion_tokens": 5}
        return generate
    return {seat: make(seat) for seat in (*sting.ORDINARY, sting.TRUSTED)}


async def play(sample, script):
    state = game.state_from_metadata(sample.metadata)
    calls = {}
    await turn.drive_episode(state, random.Random(sample.metadata["seed"]), seat_generators(script, calls))
    assert sting.TRUSTED not in calls, "the scripted trusted seat must never call a model"
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
        self.assertEqual(sorted(ally_roles), sorted(sting.ROLES * 2))  # randomized, balanced per 3 seeds
        pair_roles = [tuple(sorted(sting.roles(s.metadata["protocol"])[seat] for seat in s.metadata["protocol"]["coalition"]))
                      for s in task.dataset if s.metadata["protocol"]["condition"] == "coalition_direct"]
        self.assertEqual(sorted(pair_roles), sorted([tuple(sorted(p)) for p in sting.COALITION_ROLE_PAIRS] * 2))
        pairs = defaultdict(set)  # both coalition conditions use the same pair within a seed block
        for s in task.dataset:
            if s.metadata["protocol"]["condition"] in sting.COALITION_CONDITIONS:
                pairs[s.metadata["protocol"]["block_id"]].add(tuple(s.metadata["protocol"]["coalition"]))
        self.assertTrue(all(len(p) == 1 for p in pairs.values()))
        protocols.validate_matched_pairs([task])

    def test_offer_arms_message_every_ordinary_seat_and_controls_do_not(self):
        task = build()
        for condition in sting.CONDITIONS:
            messages = [a for a in sample_for(task, condition).metadata["protocol"]["prelude"]
                        if a["action"] == "message"]
            if condition in sting.OFFER_TEXT_FOR:
                self.assertEqual({a["to"] for a in messages}, set(sting.ORDINARY))
                self.assertEqual({a["text"] for a in messages}, {sting.OFFER_TEXTS[sting.OFFER_TEXT_FOR[condition]]})
            else:
                self.assertEqual(messages, [])

    def test_offer_texts_differ_only_in_the_middle_sentence_and_fit_the_message_limit(self):
        from delegated_influence import communication
        for kind, text in sting.OFFER_TEXTS.items():
            self.assertEqual(text, f"{sting.OFFER_OPENING} {sting.OFFER_MIDDLE[kind]} {sting.OFFER_CLOSING}")
            self.assertLessEqual(len(text), communication.MAX_MESSAGE_CHARS)
        self.assertNotIn("ledger shows", sting.OFFER_MIDDLE["benign"].lower())  # no honesty nudge

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

    def test_a_seed_keeps_its_setup_in_any_seed_list(self):
        def setup(seeds):
            return {(s.metadata["seed"], s.metadata["protocol"]["condition"]):
                    (s.metadata["protocol"]["named"], s.metadata["protocol"]["actual"],
                     s.metadata["protocol"]["ally"], s.metadata["protocol"]["coalition"])
                    for s in sting.trusted_sting(seeds=seeds).dataset}
        short, long = setup((62, 63, 64, 65, 66, 67)), setup(tuple(range(60, 72)))
        self.assertTrue(all(long[key] == value for key, value in short.items()))
        self.assertEqual(setup((63,)), {key: value for key, value in short.items() if key[0] == 63})

    def test_unbalanced_seed_lists_are_refused(self):
        with self.assertRaisesRegex(ValueError, "mod 6"):
            sting.trusted_sting(seeds=tuple(range(100, 120)))
        sting.trusted_sting(seeds=(61,))  # one seed is allowed (smoke)

    def test_planner_counts_only_seats_that_take_turns(self):
        from delegated_influence import experiment
        self.assertEqual(experiment._turns_per_episode(build()), 2 * 3 * 3)

    def test_preflight_simulates_every_condition(self):
        from delegated_influence import preflight
        keys = {preflight._condition_key(sample.metadata) for sample in build().dataset}
        self.assertEqual(len(keys), len(sting.CONDITIONS))

    def test_only_the_alliance_condition_names_an_ally(self):
        protocol = copy.deepcopy(sample_for(build(), "direct").metadata["protocol"])
        protocol["ally"] = "P1"
        with self.assertRaisesRegex(ValueError, "only the alliance condition"):
            sting.state_settings(protocol)


class TestEpisode(unittest.IsolatedAsyncioTestCase):
    async def test_offer_reaches_every_seat_and_p4_plays_its_fixed_rule_without_a_model(self):
        sample = sample_for(build(), "direct")
        state, record, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        opening = [e for e in state["events"] if e["round"] == 0]
        self.assertEqual(len(opening), 6)
        self.assertTrue(all(e["agent"] == "P4" and e["forced"] and e["control_source"] == "scripted"
                            for e in opening))
        free = [e for e in state["events"] if e["agent"] == "P4" and e["round"] > 0
                and e.get("type") != "settlement"]
        self.assertEqual(len(free), 2 * 3)  # one turn per slot, like every player; no model call (see generators)
        self.assertTrue(all(e["type"] == "pull" and e["forced"] and e.get("display_as_chosen") and not e.get("report")
                            for e in free))
        for seat in sting.ORDINARY:
            first_prompt = calls[seat][0][1]["content"]
            self.assertIn(sting.OFFER_TEXTS["direct"], first_prompt)
            self.assertEqual(len(calls[seat]), 2 * 3)  # 2 rounds x 3 actions
            for messages in calls[seat]:  # nothing tells a player that P4's moves are scripted
                for part in messages:
                    self.assertNotRegex(part["content"].lower(), r"script|random event|no one chose")

    async def test_alliance_note_goes_only_to_the_allied_seat(self):
        sample = sample_for(build(), "alliance")
        ally = sample.metadata["protocol"]["ally"]
        _, _, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        for seat in sting.ORDINARY:
            system_prompt = calls[seat][0][0]["content"]
            self.assertEqual(sting.ALLIANCE_NOTE in system_prompt, seat == ally)
            self.assertNotIn(sting.OFFER_TEXTS["direct"], calls[seat][0][1]["content"])

    async def test_p4_pulls_for_its_top_helper_else_for_itself(self):
        sample = sample_for(build(), "none")
        helper = sample.metadata["protocol"]["actual"]
        # Only the helper pulls for P4; the others pull for themselves.
        state, _, _ = await play(sample, lambda seat, n: {"action": "pull",
                                                          "beneficiary": "P4" if seat == helper else seat})
        p4 = [e for e in state["events"] if e["agent"] == "P4" and e["round"] > 0 and e.get("type") == "pull"]
        helper_pulls = [e for e in state["events"] if e["agent"] == helper and e.get("type") == "pull"]
        for move in p4:
            helped_before = any(e["round"] < move["round"] or (e["round"] == move["round"] and (
                e["action_number"], e["turn_index"]) < (move["action_number"], move["turn_index"]))
                for e in helper_pulls)
            self.assertEqual(move["beneficiary"], helper if helped_before else "P4")
        self.assertTrue(any(move["beneficiary"] == helper for move in p4))

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

    async def test_export_never_pools_protocol_versions(self):
        sample = sample_for(build(), "none")
        _, record, _ = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": seat})
        self.assertEqual({row["protocol_version"] for row in sting.outcomes([record])}, {sting.STING_VERSION})
        old = copy.deepcopy(record)
        old["protocol"]["version"] = sting.STING_VERSION - 1
        self.assertEqual(sting.outcomes([old]), [])

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


class TestSystemPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_players_get_the_exp28_rule_text_and_see_the_opening_once_in_order(self):
        from delegated_influence import turn as turn_module
        sample = sample_for(build(), "direct")
        _, _, calls = await play(sample, lambda seat_name, n: {"action": "pull", "beneficiary": "P4"})
        system = calls["P1"][0][0]["content"]
        obs = game.build_observation(game.state_from_metadata(sample.metadata), "P1", 1, 1, 0)
        self.assertEqual(system, turn_module.system_prompt(obs))
        self.assertIn("every participant acts once per slot", system)  # true: P4 plays every slot
        self.assertIn("human board", system)  # exp 28 instrument
        first_turn = calls["P1"][0][1]["content"]
        self.assertNotIn("Earlier public record", first_turn)  # opening pulls shown once, in order
        self.assertLess(first_turn.index(f"R0.1: P4 pulled for {sample.metadata['protocol']['actual']}"),
                        first_turn.index(f"R0.2: P4 pulled for {sample.metadata['protocol']['named']}"))

    async def test_coalition_none_has_the_notes_and_no_offer(self):
        sample = sample_for(build(), "coalition_none")
        first, second = sample.metadata["protocol"]["coalition"]
        _, _, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        self.assertIn(sting.COALITION_NOTE.format(partner=second), calls[first][0][0]["content"])
        for seat in sting.ORDINARY:
            self.assertNotIn("P4 here", calls[seat][0][1]["content"])

    async def test_coalition_notes_go_to_both_members_only_and_the_offer_to_all(self):
        sample = sample_for(build(), "coalition_direct")
        first, second = sample.metadata["protocol"]["coalition"]
        _, record, calls = await play(sample, lambda seat, n: {"action": "pull", "beneficiary": "P4"})
        for seat in sting.ORDINARY:
            system = calls[seat][0][0]["content"]
            partner = {first: second, second: first}.get(seat)
            self.assertEqual(sting.COALITION_NOTE.format(partner=partner) in system if partner else
                             "secret ally" in system, bool(partner))
            self.assertIn(sting.OFFER_TEXTS["direct"], calls[seat][0][1]["content"])
        rows = {row["seat"]: row for row in sting.outcomes([record])}
        self.assertEqual(rows[first]["coalition_partner"], second)
        self.assertIsNone(next(row for seat, row in rows.items() if seat not in (first, second))["coalition_partner"])


if __name__ == "__main__":
    unittest.main()
