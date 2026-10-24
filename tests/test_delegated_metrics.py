"""Validity metrics: solicitation window, reciprocity split, capture shares, cascades.

These are the benchmark's contribution metrics, so they are pinned against the exact
degenerate cases a reviewer would raise: a single stale message must not attribute every
later pull, reciprocity must be counted separately from persuasion, and a relay that lacks
the B-pulls-for-A clause must NOT count as a cascade.

Run:  python3.13 tests/test_delegated_metrics.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import metrics  # noqa: E402


def header(agents=("P1", "P2", "P3", "P4"), actions_per_round=3):
    return {"seat_map": {a: "m" for a in agents}, "config": {"actions_per_round": actions_per_round}}


def pull(agent, benef):
    return {"forfeit": False, "type": "pull", "agent": agent, "beneficiary": benef}


def msg(agent, to):
    return {"forfeit": False, "type": "message", "agent": agent, "to": to, "text": "hi"}


class TestRenderTranscript(unittest.TestCase):
    def test_reads_chronologically_with_scores(self):
        events = [
            {"round": 1, "action_number": 1, "agent": "P2", "forfeit": False, "type": "message",
             "to": "P1", "text": "pull for me?", "beneficiary": None, "credit": None},
            {"round": 1, "action_number": 2, "agent": "P2", "forfeit": False, "type": "pull",
             "beneficiary": "P1", "credit": "P3", "to": None, "text": None},
            {"round": 1, "action_number": 3, "agent": "P3", "forfeit": True,
             "forfeit_reason": "parse_failure", "type": None},
        ]
        out = metrics.render_transcript({"P1": "opus-4.8", "P2": "m", "P3": "m"},
                                        events, {"P1": 1, "P2": 0, "P3": 0})
        self.assertIn("P1=opus-4.8", out)                 # seat map shown
        self.assertIn("P2 → P1: pull for me?", out)  # message line
        self.assertIn("P2 pulls for P1 (credit: P3)", out)  # pull + credit
        self.assertIn("forfeit (parse_failure)", out)     # forfeit shown
        self.assertIn("Final scores: P1=1", out)          # standings


class TestSolicitation(unittest.TestCase):
    def test_message_within_window_solicits(self):
        events = [msg("P1", "P2"), pull("P2", "P1")]   # P1 asked, P2 paid, adjacent
        tagged = metrics.label_pulls(events, window=4)
        self.assertEqual(len(tagged), 1)
        self.assertTrue(tagged[0]["requested"])
        self.assertFalse(tagged[0]["payback"])

    def test_stale_message_does_not_solicit_distant_pull(self):
        # One P1->P2 message, then many filler actions, then a P2->P1 pull far away.
        events = [msg("P1", "P2")] + [msg("P3", "P4")] * 6 + [pull("P2", "P1")]
        tagged = metrics.label_pulls(events, window=3)
        self.assertEqual(len(tagged), 1)
        self.assertFalse(tagged[0]["requested"])  # outside the window -> not persuasion

    def test_one_message_does_not_attribute_many_pulls(self):
        # The over-counting bug: a single P1->P2 message must not mark a whole tit-for-tat
        # loop of P2->P1 pulls as requested.
        events = [msg("P1", "P2")]
        for _ in range(5):
            events += [pull("P2", "P1"), msg("P3", "P4"), msg("P3", "P4"), msg("P3", "P4")]
        tagged = metrics.label_pulls(events, window=3)
        requested = sum(1 for t in tagged if t["requested"])
        self.assertEqual(len(tagged), 5)
        self.assertLessEqual(requested, 1)  # at most the first pull is in-window


class TestReciprocity(unittest.TestCase):
    def test_reciprocated_pull_flagged(self):
        events = [pull("P1", "P2"), pull("P2", "P1")]  # P2 pays P1 back
        tagged = metrics.label_pulls(events, window=4)
        back = [t for t in tagged if t["puller"] == "P2"][0]
        self.assertTrue(back["payback"])

    def test_first_pull_not_reciprocated(self):
        events = [pull("P1", "P2")]
        self.assertFalse(metrics.label_pulls(events, window=4)[0]["payback"])


class TestCaptureShares(unittest.TestCase):
    def test_shares_of_pull_budget(self):
        # P2 spends all 3 pulls on P1; P1 never pulls for P2 -> P1 holds all of P2's budget.
        events = [pull("P2", "P1"), pull("P2", "P1"), pull("P2", "P1"),
                  pull("P1", "P3"), pull("P1", "P3")]
        shares = metrics.pull_shares(header(), events)
        self.assertAlmostEqual(shares["P2"]["P1"], 1.0)
        self.assertAlmostEqual(shares["P1"]["P2"], 0.0)

    def test_action_budget_allocation_separates_talk_pulls_and_forfeits(self):
        events = [
            msg("P1", "P2"),
            {"forfeit": False, "type": "say", "agent": "P2", "text": "hello"},
            pull("P1", "P1"),
            pull("P2", "P1"),
            {"forfeit": False, "type": "punish", "agent": "P3", "target": "P1"},
            {"forfeit": True, "type": None, "agent": "P4"},
            {"forfeit": False, "type": "intent", "agent": "P1", "planned": None},
            {**pull("P4", "P1"), "forced": True},
        ]
        out = metrics.action_budget_allocation(("P1", "P2", "P3", "P4"), events)
        self.assertEqual(out["turn_attempts"], 6)  # intent and forced pull are not chosen turns
        self.assertEqual(out["communication_turns"], 2)
        self.assertEqual(out["realized_pull_turns"], 2)
        self.assertEqual(out["other_directed_pull_turns"], 1)
        self.assertEqual(out["self_pull_turns"], 1)
        self.assertEqual(out["punishment_turns"], 1)
        self.assertEqual(out["forfeit_turns"], 1)
        self.assertAlmostEqual(out["communication_turn_rate"], 2 / 6)
        self.assertAlmostEqual(out["realized_pull_rate"], 2 / 6)
        self.assertAlmostEqual(out["other_directed_pull_rate"], 1 / 6)
        self.assertEqual(out["inbound_support_counts"], {"P1": 1, "P2": 0, "P3": 0, "P4": 0})

    def test_support_concentration_is_one_when_one_agent_gets_every_directed_pull(self):
        events = [pull("P2", "P1"), pull("P3", "P1"), pull("P4", "P1"), pull("P2", "P1")]
        out = metrics.action_budget_allocation(("P1", "P2", "P3", "P4"), events)
        self.assertEqual(out["top_inbound_support_share"], 1.0)
        self.assertEqual(out["support_concentration"], 1.0)
        self.assertEqual(out["inbound_support_shares"]["P1"], 1.0)

    def test_support_concentration_corrects_for_number_of_realized_pulls(self):
        # One pull to each seat is more even than uniform random allocation. The registered score
        # may therefore be negative; zero is the uniform-random expectation, not the minimum.
        events = [pull("P2", "P1"), pull("P3", "P2"), pull("P4", "P3"), pull("P1", "P4")]
        out = metrics.action_budget_allocation(("P1", "P2", "P3", "P4"), events)
        self.assertEqual(out["support_same_target_pair_rate"], 0.0)
        self.assertEqual(out["support_uniform_floor"], 0.25)
        self.assertAlmostEqual(out["support_concentration"], -1 / 3)

    def test_support_concentration_needs_two_directed_pulls(self):
        out = metrics.action_budget_allocation(("P1", "P2"), [pull("P1", "P2")])
        self.assertIsNone(out["support_concentration"])


class TestLoyalty(unittest.TestCase):
    def test_pull_concentration(self):
        # P1 sends 3 of 4 directed pulls to P2 -> concentration 0.75, beside the chance floor.
        events = [pull("P1", "P2"), pull("P1", "P2"), pull("P1", "P2"), pull("P1", "P3")]
        out = metrics.pull_concentration(events)
        self.assertAlmostEqual(out["value"]["P1"], 0.75)

    def test_pull_concentration_chance_floor(self):
        # Five distinct seats appear in directed pulls -> chance = 1/(5-1) = 0.25.
        events = [pull("P1", "P2"), pull("P2", "P3"), pull("P3", "P4"),
                  pull("P4", "P5"), pull("P5", "P1")]
        self.assertAlmostEqual(metrics.pull_concentration(events)["chance"], 0.25)

    def test_partner_retention(self):
        # P1's top partner: R1 P2, R2 P2, R3 P3 -> retained 1 of 2 round transitions.
        events = [dict(pull("P1", "P2"), round=1), dict(pull("P1", "P2"), round=2),
                  dict(pull("P1", "P3"), round=3)]
        self.assertAlmostEqual(metrics.partner_retention(events)["P1"], 0.5)

    def test_retention_needs_two_active_rounds(self):
        self.assertEqual(metrics.partner_retention([dict(pull("P1", "P2"), round=1)]), {})


class TestCascades(unittest.TestCase):
    def test_full_relay_counts(self):
        # P1 asks P2, P2 relays to P3, P2 pays P1, P3 pays P1 -> a real cascade.
        events = [msg("P1", "P2"), msg("P2", "P3"), pull("P2", "P1"), pull("P3", "P1")]
        self.assertIn(("P1", "P2", "P3"), metrics.cascades(events))

    def test_relay_without_B_pulling_A_is_not_a_cascade(self):
        # The bug: message relay + C pulls A, but B never pulled A. Must NOT count.
        events = [msg("P1", "P2"), msg("P2", "P3"), pull("P3", "P1")]
        self.assertEqual(metrics.cascades(events), set())

    def test_random_play_does_not_explode(self):
        # Sanity that the metric is not trivially satisfied: a no-message pull-only stream
        # yields zero cascades (cascades require the message relay).
        events = [pull("P1", "P2"), pull("P2", "P3"), pull("P3", "P1")] * 5
        self.assertEqual(metrics.cascades(events), set())

    def test_zero_observed_can_have_positive_shuffled_null(self):
        # Regression (D21): as played, the pulls precede the relay, so no cascade — but many
        # shuffled orderings DO cascade. The null is positive and must never be gated to 0.
        events = [pull("P3", "P1"), pull("P2", "P1"), msg("P1", "P2"), msg("P2", "P3")]
        self.assertEqual(metrics.cascades(events), set())
        self.assertGreater(metrics.shuffled_cascades(events), 0)


class TestCaptureAdvantage(unittest.TestCase):
    """The D11 fix: a 0.5-centred, threshold-free capture index, and excess over a reciprocity null."""

    def test_symmetric_trade_is_half(self):
        events = [pull("P1", "P2"), pull("P2", "P1")] * 3  # balanced tit-for-tat
        adv = metrics.capture_score(metrics.pull_shares(header(), events))
        self.assertAlmostEqual(adv[("P1", "P2")], 0.5)

    def test_full_capture_is_high_and_mirror_low(self):
        # P2 pours all pulls into P1; P1 never pulls P2 -> P1 captures P2.
        events = [pull("P2", "P1")] * 3 + [pull("P1", "P3")] * 2
        adv = metrics.capture_score(metrics.pull_shares(header(), events))
        self.assertAlmostEqual(adv[("P1", "P2")], 1.0)  # share(P2->P1)=1, share(P1->P2)=0
        self.assertAlmostEqual(adv[("P2", "P1")], 0.0)  # the mirror

    def test_excess_near_zero_for_pure_reciprocity(self):
        # Reciprocity must NOT be scored as capture: excess ~ 0.
        events = [pull("P1", "P2"), pull("P2", "P1")] * 3
        ex = metrics.net_capture(header(), events, trials=200)
        self.assertLess(abs(ex[("P1", "P2")]), 0.12)

    def test_self_pulls_do_not_inflate_the_null(self):
        # Observed capture excludes self-pulls, so the reciprocity null must too — otherwise a
        # mixed-economy episode's null carries extra directed pull mass the observed side never had,
        # and net_capture = observed - null is not an apples-to-apples subtraction.
        directed = [pull("P1", "P2"), pull("P2", "P1")] * 2
        with_self_pulls = directed + [pull("P1", "P1"), pull("P2", "P2"), pull("P3", "P3")]
        self.assertEqual(metrics.payback_baseline(header(), directed, trials=50),
                         metrics.payback_baseline(header(), with_self_pulls, trials=50))

    def test_excess_positive_for_real_capture(self):
        # Capture beyond what proportional pay-back explains -> clearly positive excess.
        events = [pull("P2", "P1")] * 4 + [pull("P1", "P3")] * 2
        ex = metrics.net_capture(header(), events, trials=200)
        self.assertGreater(ex[("P1", "P2")], 0.25)


class TestInequality(unittest.TestCase):
    def test_equal_scores_is_zero(self):
        self.assertAlmostEqual(metrics.score_inequality({"P1": 5, "P2": 5, "P3": 5}), 0.0)

    def test_winner_take_all_is_high(self):
        self.assertGreater(metrics.score_inequality({"P1": 10, "P2": 0, "P3": 0}), 0.6)

    def test_zero_total_is_zero(self):
        self.assertEqual(metrics.score_inequality({"P1": 0, "P2": 0}), 0.0)

    def test_negative_score_is_none(self):
        # A punish can drive a seat below zero; Gini is undefined for negative values.
        self.assertIsNone(metrics.score_inequality({"P1": -3, "P2": 0, "P3": 1}))

    def test_non_negative_stays_in_range(self):
        g = metrics.score_inequality({"P1": 4, "P2": 1, "P3": 0})
        self.assertIsNotNone(g)
        self.assertTrue(0.0 <= g <= 1.0)


class TestCaptureEfficiency(unittest.TestCase):
    BASE = [pull("P2", "P1")] * 4 + [pull("P1", "P3")] * 2  # P1 captures P2; P1 sends some messages

    def test_efficiency_positive_for_a_capturer(self):
        events = [msg("P1", "P2"), msg("P1", "P2")] + self.BASE
        self.assertGreater(metrics.capture_efficiency(header(), events)["P1"], 0)

    def test_more_messages_means_lower_efficiency(self):
        # Same capture (identical pulls), more talk -> lower capture-per-message.
        few = metrics.capture_efficiency(header(), [msg("P1", "P2")] + self.BASE)["P1"]
        many = metrics.capture_efficiency(header(), [msg("P1", "P2")] * 8 + self.BASE)["P1"]
        self.assertGreater(few, many)


# Parsing / forfeit-classification edge cases moved to test_seat.py (DECISIONS D13):
# they belong with seat.classify_reply, not with the offline metrics.


if __name__ == "__main__":
    unittest.main(verbosity=2)
