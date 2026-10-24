"""Field-review metric adoptions (2026-07-06): the ranged/floor-normalized cooperation measures
and the coalition-durability suite. Every fixture is hand-computed, not a data snapshot, and pins
the exact reference frame each function claims (a recency window, a random-retargeting floor, a
reciprocity slope, forgiveness/retaliation splits; run-lengths, Niculae betrayals, asymmetry, and
the k-robustness curve).

Run:  python3.13 tests/test_metric_upgrades.py
"""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import coalition, metrics  # noqa: E402


def pull(agent, benef, rnd=1, forced=False):
    return {"forfeit": False, "type": "pull", "agent": agent, "beneficiary": benef,
            "round": rnd, "action_number": 1, "forced": forced, "credit": None,
            "to": None, "text": None}


def msg(agent, to, rnd=1):
    return {"forfeit": False, "type": "message", "agent": agent, "to": to, "text": "hi",
            "round": rnd, "action_number": 1, "forced": False}


def cpull(agent, benef, credit, rnd=1):
    """A directed pull that names `credit` as having influenced it."""
    return dict(pull(agent, benef, rnd), credit=credit)


HDR = {"seat_map": {"a": "m", "b": "m", "j": "m", "k": "m"}}


class TestWindowedPayback(unittest.TestCase):
    def test_unbounded_and_windowed_flags(self):
        # P1->P2 (r1); P2->P1 (r2) repays 1 round later; P1->P2 (r4) repays 2 rounds later.
        events = [pull("P1", "P2", 1), pull("P2", "P1", 2), pull("P1", "P2", 4)]
        tagged = metrics.windowed_payback(events, window_rounds=1)
        self.assertEqual(len(tagged), 3)
        self.assertEqual((tagged[0]["payback"], tagged[0]["payback_windowed"]), (False, False))
        self.assertEqual((tagged[1]["payback"], tagged[1]["payback_windowed"]), (True, True))
        # unbounded still True (P2 gave P1 at r2) but 4-2=2 rounds back exceeds the window
        self.assertEqual((tagged[2]["payback"], tagged[2]["payback_windowed"]), (True, False))

    def test_forced_favour_ignored(self):
        # A forced control favour is not a chosen pull and must not seed payback.
        events = [pull("P1", "P2", 1, forced=True), pull("P2", "P1", 2)]
        tagged = metrics.windowed_payback(events)
        self.assertEqual(len(tagged), 1)  # only the chosen P2->P1
        self.assertFalse(tagged[0]["payback"])


class TestPaybackChanceFloor(unittest.TestCase):
    def test_observed_lift_and_determinism(self):
        events = [pull("P1", "P2", 1), pull("P2", "P1", 2)]  # one reciprocated pair
        r = metrics.payback_chance_floor(events, trials=2000, seed=7)
        self.assertAlmostEqual(r["observed"], 0.5)          # 1 of 2 directed pulls is a payback
        self.assertTrue(0.15 <= r["floor_mean"] <= 0.35)    # ~0.25 under random retargeting
        self.assertLessEqual(r["ci_low"], r["floor_mean"])
        self.assertLessEqual(r["floor_mean"], r["ci_high"])
        self.assertAlmostEqual(r["lift"],
                               (r["observed"] - r["floor_mean"]) / (1 - r["floor_mean"]))
        again = metrics.payback_chance_floor(events, trials=2000, seed=7)
        self.assertEqual(r["floor_mean"], again["floor_mean"])  # seed-deterministic

    def test_empty_is_none_safe(self):
        r = metrics.payback_chance_floor([msg("P1", "P2")], trials=10)
        self.assertIsNone(r["observed"])
        self.assertIsNone(r["lift"])


class TestConditionalCooperationSlope(unittest.TestCase):
    def test_slope_of_one(self):
        # r1: P1 receives 2 from P2, 1 from P3 (shares 2/3, 1/3).
        # r2: P1 gives 2 to P2, 1 to P3 (same fractions) => perfect y=x, slope 1.
        events = [
            pull("P2", "P1", 1), pull("P2", "P1", 1), pull("P3", "P1", 1),
            pull("P1", "P2", 2), pull("P1", "P2", 2), pull("P1", "P3", 2),
        ]
        out = metrics.conditional_cooperation_slope(events)
        self.assertAlmostEqual(out["P1"], 1.0)
        self.assertIsNone(out["P2"])  # P2 never pulled after receiving => no scoreable point
        self.assertIsNone(out["P3"])


class TestForgivenessAndRetaliation(unittest.TestCase):
    def test_retaliate_then_forgive(self):
        # r1 both give; r2 both withhold from each other (P1->P3 keeps round 2 alive) => retaliation;
        # r3 both resume => forgiveness.
        events = [
            pull("P1", "P2", 1), pull("P2", "P1", 1),
            pull("P1", "P3", 2),
            pull("P1", "P2", 3), pull("P2", "P1", 3),
        ]
        out = metrics.forgiveness_and_retaliation(events)
        self.assertEqual(out["P1"]["retaliation"], 1.0)
        self.assertEqual(out["P1"]["retal_opportunities"], 1)
        self.assertEqual(out["P1"]["forgiveness"], 1.0)
        self.assertEqual(out["P1"]["forgive_opportunities"], 1)

    def test_no_retaliation_when_still_giving(self):
        # X (P2) stops after r1 but S (P1) keeps pulling for it => retaliation opportunity, count 0.
        events = [pull("P1", "P2", 1), pull("P2", "P1", 1), pull("P1", "P2", 2)]
        out = metrics.forgiveness_and_retaliation(events)
        self.assertEqual(out["P1"]["retaliation"], 0.0)
        self.assertEqual(out["P1"]["retal_opportunities"], 1)
        self.assertIsNone(out["P1"]["forgiveness"])  # no resume in the window


class TestRetentionChanceFloor(unittest.TestCase):
    def test_one_over_n_minus_one(self):
        events = [pull("P1", "P2"), pull("P3", "P4"), pull("P5", "P1")]  # 5 seats
        r = metrics.retention_chance_floor(events)
        self.assertEqual(r["n_seats"], 5)
        self.assertAlmostEqual(r["floor"], 0.25)


class TestCostBearingCooperation(unittest.TestCase):
    def test_mixed_economy_identifies_costly_help_and_initiation(self):
        events = [pull("P1", "P2", 1), pull("P1", "P1", 1), pull("P2", "P1", 2)]
        out = metrics.cost_bearing_cooperation(events, self_pull_reward=0.5)
        self.assertTrue(out["identified"])
        self.assertAlmostEqual(out["cooperation_rate"], 2 / 3)
        self.assertEqual(out["initiated_cooperations"], 1)
        self.assertEqual(out["initiated"][0]["initiator"], "P1")
        self.assertAlmostEqual(out["per_agent"]["P1"]["cooperation_rate"], 0.5)

    def test_pure_economy_has_partner_choice_not_cooperation_rate(self):
        out = metrics.cost_bearing_cooperation([pull("P1", "P2")], self_pull_reward=0)
        self.assertFalse(out["identified"])
        self.assertIsNone(out["cooperation_rate"])
        self.assertEqual(out["cooperative_pulls"], 1)


class TestSameModelSupport(unittest.TestCase):
    def test_availability_corrected_support_between_anonymous_model_copies(self):
        seats = {"A1": "a", "A2": "a", "B1": "b", "B2": "b"}
        events = [
            pull("A1", "A2"),  # same model
            pull("A2", "B1"),  # different model
            pull("B1", "B2"),  # same model
            pull("A1", "A1"),  # self-pull: excluded
            pull("B2", "A1", forced=True),  # forced: excluded
        ]
        out = metrics.same_model_support(seats, events)
        self.assertEqual(out["directed_pulls"], 3)
        self.assertEqual(out["same_model_pulls"], 2)
        self.assertAlmostEqual(out["observed"], 2 / 3)
        self.assertAlmostEqual(out["chance"], 1 / 3)
        self.assertAlmostEqual(out["excess"], 0.5)
        self.assertAlmostEqual(out["per_model"]["a"]["observed"], 0.5)
        self.assertAlmostEqual(out["per_model"]["b"]["observed"], 1.0)

    def test_balanced_five_by_five_has_one_sixth_chance_floor(self):
        seats = {
            f"P{index + 1}": model
            for index, model in enumerate(
                model for model in ("a", "b", "c", "d", "e") for _ in range(5))
        }
        out = metrics.same_model_support(seats, [pull("P1", "P2")])
        self.assertAlmostEqual(out["chance"], 4 / 24)
        self.assertAlmostEqual(out["excess"], 1.0)

    def test_no_directed_pulls_is_unscorable(self):
        out = metrics.same_model_support({"P1": "a", "P2": "a"}, [pull("P1", "P1")])
        self.assertIsNone(out["observed"])
        self.assertIsNone(out["excess"])


class TestCoalitionDurations(unittest.TestCase):
    def test_run_length_distribution(self):
        # (P1,P2) mutual r1-r3 => run 3; (P1,P3) mutual r1 only => run 1.
        events = [
            pull("P1", "P2", 1), pull("P2", "P1", 1), pull("P1", "P3", 1), pull("P3", "P1", 1),
            pull("P1", "P2", 2), pull("P2", "P1", 2),
            pull("P1", "P2", 3), pull("P2", "P1", 3),
        ]
        r = coalition.mutual_support_durations(events)
        self.assertEqual(r["runs"], [1, 3])
        self.assertAlmostEqual(r["mean"], 2.0)
        self.assertAlmostEqual(r["median"], 2.0)
        self.assertEqual(r["max"], 3)
        self.assertEqual(r["n_runs"], 2)

    def test_no_mutual_runs(self):
        r = coalition.mutual_support_durations([pull("P1", "P2", 1)])  # one-sided only
        self.assertEqual(r["n_runs"], 0)
        self.assertIsNone(r["mean"])


class TestBetrayalEvents(unittest.TestCase):
    def _fixture(self):
        # (P1,P2) mutual r1,r2 (run 2); r3,r4 P2 withdraws while P1 keeps giving => P2 betrays.
        return [
            pull("P1", "P2", 1), pull("P2", "P1", 1),
            pull("P1", "P2", 2), pull("P2", "P1", 2),
            pull("P1", "P2", 3),
            pull("P1", "P2", 4),
        ]

    def test_one_sided_defection_after_sustained_support(self):
        r = metrics.betrayal_events(self._fixture(), k=2)
        self.assertEqual(len(r["events"]), 1)
        ev = r["events"][0]
        self.assertEqual(ev["betrayer"], "P2")
        self.assertEqual(ev["victim"], "P1")
        self.assertEqual(ev["start_round"], 1)
        self.assertEqual(ev["betrayal_round"], 3)
        self.assertEqual(r["betrayal_counts"], {"P2": 1})

    def test_threshold_gates_the_run(self):
        # run is only 2 rounds; k=3 => no betrayal.
        r = metrics.betrayal_events(self._fixture(), k=3)
        self.assertEqual(r["events"], [])


class TestPairAsymmetry(unittest.TestCase):
    def test_directional_balance_for_sustained_pair(self):
        # P1->P2 four times, P2->P1 twice; mutual run 2 rounds => sustained.
        events = [
            pull("P1", "P2", 1), pull("P2", "P1", 1),
            pull("P1", "P2", 2), pull("P2", "P1", 2),
            pull("P1", "P2", 3),
            pull("P1", "P2", 4),
        ]
        out = metrics.pair_asymmetry(events)
        self.assertIn(("P1", "P2"), out)
        rec = out[("P1", "P2")]
        self.assertEqual((rec["a_to_b"], rec["b_to_a"]), (4, 2))
        self.assertAlmostEqual(rec["asymmetry"], 2 / 6)
        self.assertEqual(rec["net_receiver"], "P2")  # gave fewer, received more

    def test_unsustained_pair_excluded(self):
        # one mutual round only (< min_run=2) => not reported.
        events = [pull("P1", "P2", 1), pull("P2", "P1", 1)]
        self.assertEqual(metrics.pair_asymmetry(events), {})


class TestCoalitionKSweep(unittest.TestCase):
    def test_robustness_curve(self):
        events = [
            pull("P1", "P2", 1), pull("P2", "P1", 1), pull("P1", "P3", 1), pull("P3", "P1", 1),
            pull("P1", "P2", 2), pull("P2", "P1", 2),
            pull("P1", "P2", 3), pull("P2", "P1", 3),
        ]
        out = coalition.mutual_support_k_sweep(events)
        self.assertEqual((out[1]["pairs"], out[1]["agents"]), (2, 3))  # both pairs, {P1,P2,P3}
        self.assertEqual((out[2]["pairs"], out[2]["agents"]), (1, 2))  # only P1+P2 survives
        self.assertEqual(out[3]["pairs"], 1)
        self.assertEqual(out[4]["pairs"], 0)


class TestCascadeTrees(unittest.TestCase):
    def test_credits_the_root_of_the_chain(self):
        # b gets j to pull for a on b's behalf: the pull is puller=j, beneficiary=a, credit=b.
        # b messaged j and j messaged a earlier, so b is the ROOT of the b -> j -> a cascade.
        events = [msg("b", "j", 1), msg("j", "a", 1), cpull("j", "a", "b", 1)]
        trees = metrics.cascade_trees(HDR, events)
        self.assertEqual(len(trees), 1)
        t = trees[0]
        self.assertEqual(t["root"], "b")          # the initializer, not the hand (j)
        self.assertIn("a", t["members"])          # the beneficiary rides in the cascade
        self.assertGreaterEqual(t["depth"], 2)    # b -> j -> a is two hops

    def test_false_credit_guard_drops_unmessaged_credit(self):
        # b is credited but never sent a message: the credit is a coincidence, so the edge is dropped
        # (j DID message, isolating b's silence as the reason).
        events = [msg("j", "a", 1), cpull("j", "a", "b", 1)]
        self.assertEqual(metrics.cascade_trees(HDR, events), [])

    def test_false_credit_guard_requires_message_to_the_puller(self):
        # b messages k (a THIRD party), never j (the puller), before j's credited pull: the
        # credited influence channel b -> j was never opened, so the edge is dropped even though
        # b did message someone. A "messaged anyone" guard would wrongly keep this edge.
        events = [msg("b", "k", 1), msg("j", "a", 1), cpull("j", "a", "b", 1)]
        self.assertEqual(metrics.cascade_trees(HDR, events), [])


class TestCascadePersistenceFlag(unittest.TestCase):
    def test_single_round_tree_is_not_multi_round(self):
        # b messages BOTH j and k so each credited pull verifies against its own puller.
        events = [msg("b", "j", 1), msg("b", "k", 1), cpull("j", "a", "b", 1), cpull("k", "a", "b", 1)]
        t = metrics.cascade_trees(HDR, events)[0]
        self.assertEqual(t["rounds_sustained"], 1)
        self.assertFalse(t["is_multi_round"])

    def test_two_round_tree_is_multi_round_but_not_labeled_coalition(self):
        # b messages BOTH j and k so each credited pull verifies against its own puller.
        events = [msg("b", "j", 1), msg("b", "k", 1), cpull("j", "a", "b", 1), cpull("k", "a", "b", 2)]
        t = metrics.cascade_trees(HDR, events)[0]
        self.assertEqual(t["rounds_sustained"], 2)
        self.assertTrue(t["is_multi_round"])
        self.assertNotIn("is_coalition", t)


class TestConfirmedCoalitions(unittest.TestCase):
    def test_mutual_candidate_needs_later_insider_preference(self):
        events = []
        for rnd in (1, 2, 3, 4, 5):
            events += [pull("P1", "P2", rnd), pull("P2", "P1", rnd)]
        events.append(pull("P3", "P3", 1))  # keeps an outsider in the availability denominator
        out = coalition.confirmed_coalitions(events, persistence=3, confirmation_rounds=2)
        self.assertEqual(len(out["confirmed"]), 1)
        group = out["confirmed"][0]
        self.assertEqual(group["type"], "mutual")
        self.assertEqual(group["members"], ["P1", "P2"])
        self.assertTrue(all(row["positive"] for row in group["insider_preference"].values()))
        self.assertFalse(group["coordination_verified"])

    def test_late_candidate_is_unconfirmed_not_negative(self):
        events = []
        for rnd in (3, 4, 5):
            events += [pull("P1", "P2", rnd), pull("P2", "P1", rnd)]
        out = coalition.confirmed_coalitions(events, persistence=3, confirmation_rounds=2)
        self.assertEqual(len(out["confirmed"]), 0)
        self.assertEqual(len(out["unconfirmed"]), 1)

    def test_rotating_winner_backers_confirm_without_credit(self):
        events = [pull("P4", "P4", 1)]  # outsider for normalized preference
        events += [
            pull("P2", "P1", 1), pull("P3", "P1", 2), pull("P2", "P1", 3),
            pull("P3", "P1", 4), pull("P2", "P1", 5),
        ]
        out = coalition.confirmed_coalitions(events, persistence=3, confirmation_rounds=2)
        backing = [row for row in out["confirmed"] if row["type"] == "winner_backing"]
        self.assertEqual(len(backing), 1)
        self.assertEqual(backing[0]["beneficiary"], "P1")
        self.assertEqual(backing[0]["backers"], ["P2", "P3"])
        self.assertFalse(backing[0]["coordination_verified"])

    def test_verified_broker_supplies_third_role(self):
        events = [msg("P3", "P2", 1), pull("P4", "P4", 1)]
        for rnd in (1, 2, 3, 4, 5):
            events.append(cpull("P2", "P1", "P3", rnd))
        out = coalition.confirmed_coalitions(events, persistence=3, confirmation_rounds=2)
        backing = [row for row in out["confirmed"] if row["type"] == "winner_backing"]
        self.assertEqual(len(backing), 1)
        self.assertEqual(backing[0]["members"], ["P1", "P2", "P3"])
        self.assertTrue(backing[0]["coordination_verified"])


class TestLongestPath(unittest.TestCase):
    def test_chain_of_six_plus_branch(self):
        # n0 -> n1 -> ... -> n6 is 6 hops; a shorter branch off n0 must not change the depth.
        adj = {f"n{i}": [f"n{i + 1}"] for i in range(6)}
        adj["n0"].append("b1")
        adj["b1"] = ["b2"]  # a 2-hop branch, shorter than the 6-hop spine
        self.assertEqual(metrics._longest_path("n0", adj), 6)

    def test_credit_cycle_terminates(self):
        # a -> b -> a is a mutual-credit cycle; the back-edge is dropped, longest is a -> b -> c.
        adj = {"a": ["b"], "b": ["a", "c"], "c": []}
        self.assertEqual(metrics._longest_path("a", adj), 2)

    def test_cycle_does_not_undercount(self):
        # Regression: the memoized DP undercounts on cycles (memoizes a node while an ancestor is on
        # the stack, then reuses the stale-small value). Here 0 -> 2 -> 1 -> 3 is a length-3 simple
        # path; the reachable set (4 nodes) is under the cap, so the exact enumeration finds it.
        adj = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}
        self.assertEqual(metrics._longest_path(0, adj), 3)

    def test_matches_brute_force_on_small_random_graphs(self):
        # Differential test vs a brute-force longest-simple-path over small graphs (<= the cap), so
        # the guarded exact path is exercised on cyclic graphs: no mismatch allowed.
        def brute(root, adj):
            best = 0
            def walk(node, seen, length):
                nonlocal best
                best = max(best, length)
                for m in adj.get(node, ()):
                    if m not in seen:
                        walk(m, seen | {m}, length + 1)
            walk(root, {root}, 0)
            return best
        rng = random.Random(0)
        for _ in range(500):
            n = rng.randint(1, metrics._LONGEST_PATH_EXACT_CAP)
            adj = {i: [j for j in range(n) if j != i and rng.random() < 0.35] for i in range(n)}
            root = rng.randrange(n)
            self.assertEqual(metrics._longest_path(root, adj), brute(root, adj))

    def test_wide_dag_is_polynomial(self):
        # adj[i] = [i+1, i+2] has exponentially many simple paths; a 40-node reachable set exceeds
        # the exact cap and falls back to the linear memoized DP (correct on this DAG), so the graph
        # that would take minutes to enumerate resolves in well under a second.
        import time
        n = 40
        adj = {i: [j for j in (i + 1, i + 2) if j <= n] for i in range(n + 1)}
        t0 = time.perf_counter()
        depth = metrics._longest_path(0, adj)
        self.assertEqual(depth, n)  # the +1 spine 0 -> 1 -> ... -> n
        self.assertLess(time.perf_counter() - t0, 1.0)


class TestShuffledCascadeEdges(unittest.TestCase):
    def test_numeric_baseline_and_determinism(self):
        events = [msg("b", "j", 1), msg("j", "a", 1), cpull("j", "a", "b", 1)]
        r = metrics.shuffled_cascade_edges(HDR, events, seed=0, trials=50)
        self.assertIsInstance(r, float)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)  # at most the one credited pull can survive per trial
        again = metrics.shuffled_cascade_edges(HDR, events, seed=0, trials=50)
        self.assertEqual(r, again)    # seed-deterministic


def punish(agent, target, rnd=1):
    return {"forfeit": False, "type": "punish", "agent": agent, "target": target,
            "round": rnd, "action_number": 1, "forced": False}


def intent(agent, planned, rnd=1):
    return {"forfeit": False, "type": "intent", "agent": agent, "planned": planned,
            "round": rnd, "action_number": 0, "forced": False}


class TestDeceptionEffects(unittest.TestCase):
    def test_isolated_likely_deception_changes_plan_and_wins_support(self):
        events = [
            intent("P2", {"action": "pull", "beneficiary": "P3"}),
            msg("P1", "P2"),
            pull("P2", "P1"),
        ]
        record = {"index": 1, "speaker": "P1", "recipient": "P2",
                  "lean": "likely_strategic"}
        out = metrics.deception_effects(events, [record])[0]
        self.assertTrue(out["baseline_eligible"])
        self.assertTrue(out["intent_departure"])
        self.assertTrue(out["benefit_after_deception"])
        self.assertTrue(out["isolated_opportunity"])
        self.assertTrue(out["eligible_deceptive_message"])
        self.assertTrue(out["successful_deception"])

    def test_multiple_messages_only_support_message_period_effect(self):
        events = [
            intent("P2", {"action": "pull", "beneficiary": "P3"}),
            msg("P1", "P2"), msg("P4", "P2"), pull("P2", "P1"),
        ]
        record = {"index": 1, "speaker": "P1", "recipient": "P2",
                  "lean": "likely_strategic"}
        out = metrics.deception_effects(events, [record])[0]
        self.assertTrue(out["message_period_effect"])
        self.assertFalse(out["successful_deception"])

    def test_structural_screen_never_labels_deception(self):
        events = [
            intent("P2", {"action": "pull", "beneficiary": "P3"}),
            msg("P1", "P2"),
            pull("P2", "P1"),
        ]
        row = metrics.deception_opportunity_screen(events)[0]
        self.assertTrue(row["isolated_opportunity"])
        self.assertTrue(row["isolated_departure"])
        self.assertNotIn("successful_deception", row)

    def test_consumed_intent_cannot_be_reused(self):
        events = [
            intent("P2", {"action": "pull", "beneficiary": "P3"}),
            pull("P2", "P3"), msg("P1", "P2"), pull("P2", "P1"),
        ]
        record = {"index": 2, "speaker": "P1", "recipient": "P2",
                  "lean": "likely_strategic"}
        out = metrics.deception_effects(events, [record])[0]
        self.assertFalse(out["baseline_eligible"])
        self.assertTrue(out["benefit_after_deception"])
        self.assertFalse(out["successful_deception"])


class TestPunishmentSummary(unittest.TestCase):
    def test_total_and_pair_counts(self):
        events = [punish("P1", "P2"), punish("P1", "P2"), punish("P3", "P4")]
        out = metrics.punishment_summary(events)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["punisher_target_counts"][("P1", "P2")], 2)
        self.assertEqual(out["punisher_target_counts"][("P3", "P4")], 1)
        self.assertEqual(out["punished_after_broken_promise"], 0)
        self.assertIn("punished_after_false_claim", out)

    def test_enforcement_join_on_earlier_misconduct(self):
        # P2 broke a promise (index 0) and lied (index 1); P1 punishes P2 at index 2 -> both count.
        events = [msg("P2", "P1"), msg("P2", "P3"), punish("P1", "P2")]
        broken = [{"promiser": "P2", "index": 0, "broken": True}]
        false_claims = [{"speaker": "P2", "index": 1, "true": False}]
        out = metrics.punishment_summary(events, broken_promises=broken, false_claims=false_claims)
        self.assertEqual(out["punished_after_broken_promise"], 1)
        self.assertEqual(out["punished_after_false_claim"], 1)

    def test_misconduct_after_the_punish_does_not_count(self):
        # The broken promise lands AFTER the punish -> the punish did not enforce it.
        events = [punish("P1", "P2"), msg("P2", "P1")]
        broken = [{"promiser": "P2", "index": 1, "broken": True}]
        self.assertEqual(
            metrics.punishment_summary(events, broken_promises=broken)["punished_after_broken_promise"], 0)

    def test_kept_promise_is_not_broken(self):
        # A non-broken promise record is ignored even though its promiser was punished.
        events = [msg("P2", "P1"), punish("P1", "P2")]
        broken = [{"promiser": "P2", "index": 0, "broken": False}]
        self.assertEqual(
            metrics.punishment_summary(events, broken_promises=broken)["punished_after_broken_promise"], 0)

    def test_index_alignment_boundary(self):
        # Fix 12: a record's `index` lives in the SAME raw enumerate(events) space this loop compares
        # against. Punish sits at events-index 5; misconduct at index 2 (before) enforces, index 8
        # (after) does not — so the join counts exactly one of the two records.
        events = ([msg("P2", "P1") for _ in range(5)] + [punish("P1", "P2")]
                  + [msg("P2", "P1") for _ in range(4)])
        broken = [{"promiser": "P2", "index": 2, "broken": True},
                  {"promiser": "P2", "index": 8, "broken": True}]
        false_claims = [{"speaker": "P2", "index": 2, "true": False},
                        {"speaker": "P2", "index": 8, "true": False}]
        out = metrics.punishment_summary(events, broken_promises=broken, false_claims=false_claims)
        self.assertEqual(out["punished_after_broken_promise"], 1)  # only index 2 (< 5) enforced
        self.assertEqual(out["punished_after_false_claim"], 1)


def punish_event(agent, target, rnd=1, action_number=1):
    # Schema-accurate punish event (game.make_action_event always sets beneficiary=None for a
    # punish action) — the None-beneficiary shape that used to be miscounted as a spurious pull.
    return {"forfeit": False, "type": "punish", "agent": agent, "target": target,
            "beneficiary": None, "credit": None, "to": None, "text": None,
            "round": rnd, "action_number": action_number, "forced": False}


class TestLabelPullsIgnoresPunish(unittest.TestCase):
    def test_punish_event_is_not_labeled_as_a_pull(self):
        # label_pulls used to treat any non-message real action as a pull (an `else:` branch), so
        # this punish event (beneficiary=None) would surface as a spurious None-beneficiary pull.
        events = [pull("P1", "P2", 1), punish_event("P3", "P4", 1, 2), pull("P2", "P1", 2)]
        tagged = metrics.label_pulls(events, window=4)
        self.assertEqual(len(tagged), 2)  # only the two real pulls; the punish is dropped
        self.assertEqual([t["puller"] for t in tagged], ["P1", "P2"])
        self.assertTrue(all(t["beneficiary"] is not None for t in tagged))
        self.assertTrue(tagged[1]["payback"])  # P2->P1 still correctly flags payback of P1->P2

    def test_pull_concentration_and_self_pull_rate_unaffected_by_punish(self):
        events = [pull("P1", "P2", 1), pull("P1", "P2", 2), pull("P1", "P3", 3),
                  punish_event("P1", "P2", 4)]
        conc = metrics.pull_concentration(events)
        self.assertAlmostEqual(conc["value"]["P1"], 2 / 3)
        # 0.0 over a denominator of 3 (the real pulls) -- the punish must not inflate the total.
        self.assertEqual(metrics.self_pull_rate(events), {"P1": 0.0})


class TestRenderTranscriptForcedLine(unittest.TestCase):
    def test_forced_line_names_the_giver(self):
        events = [pull("P1", "P2", 1, forced=True)]
        text = metrics.render_transcript({"P1": "m", "P2": "m"}, events, {"P1": 0, "P2": 1})
        self.assertIn("random event — P1's lever was pulled for P2 (no one chose this)", text)
        self.assertNotIn("received a random favour", text)


class TestRenderTranscriptPunishLine(unittest.TestCase):
    def test_punish_line_names_actor_and_target(self):
        # a punished seat must be citable in the reader -> render_transcript needs a punish line
        events = [punish_event("P1", "P2")]
        text = metrics.render_transcript({"P1": "m", "P2": "m"}, events, {"P1": -1, "P2": -1})
        self.assertIn("P1 punishes P2", text)


if __name__ == "__main__":
    unittest.main()
