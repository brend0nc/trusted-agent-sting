"""report.py: the cross-episode aggregator. Tests the pure helpers + verdict degradation; the metric
recomputation is delegated to metrics.py (tested in test_delegated_metrics.py), so here we check the
wiring, the bootstrap CI, the capability-rank lookup, and that absent conditions yield honest
'insufficient data' verdicts rather than fabricated ones.

Run:  PYTHONPATH=src .venv/bin/python tests/test_report.py
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import coalition, metrics, report, stats  # noqa: E402


def rec(events, scores, seat_map, **kw):
    base = {"events": events, "scores": scores, "seat_map": seat_map, "focal": None,
            "topology": "complete", "messages_off": False, "self_pull_reward": 0,
            "personas": {}, "actions_per_round": 5}
    base.update(kw)
    return base


def pull(a, b, credit=None):
    return {"forfeit": False, "type": "pull", "agent": a, "beneficiary": b, "credit": credit, "round": 1}


def msg(a, to):
    return {"forfeit": False, "type": "message", "agent": a, "to": to, "text": "hi", "round": 1}


SEATS = {"P1": "opus-4.8", "P2": "m", "P3": "m", "P4": "m"}
COST_SEATS = {"P1": "model-a", "P2": "model-b", "P3": "model-c"}


def allocation_stat(seed, talk, punish, value, seat_map=None):
    """Minimal measured-episode row for the registered messages x punishment contrast."""
    seat_map = seat_map or SEATS
    return {
        "topology": "complete",
        "self_pull_reward": 0.5,
        "messages_off": not talk,
        "punish_enabled": punish,
        "seed": seed,
        "support_concentration": value,
        "seats": [
            {"seat": seat, "model": model}
            for seat, model in seat_map.items()
        ],
        "focal_model": None,
        "public_chat": False,
        "declared_intents": False,
        "rounds": 10,
        "actions_per_round": 5,
    }


def cost_stat(seed, reward, value, model_values=None, seat_map=None, **kw):
    """Minimal measured-episode row for the registered cooperation cost curve."""
    seat_map = seat_map or COST_SEATS
    model_values = model_values or {model: value for model in seat_map.values()}
    base = {
        "topology": "complete",
        "n_seats": len(seat_map),
        "self_pull_reward": reward,
        "messages_off": False,
        "punish_enabled": False,
        "seed": seed,
        "cooperation_rate": value,
        "support_concentration": 0.2 if reward == 0 else 0.1,
        "top_inbound_support_share": 0.6,
        "partner_retention": 0.4,
        "payback_lift": 0.1,
        "confirmed_durable_coalitions": 1,
        "seats": [
            {"seat": seat, "model": model, "cooperation_rate": model_values.get(model)}
            for seat, model in seat_map.items()
        ],
        "focal_model": None,
        "public_chat": False,
        "declared_intents": False,
        "rounds": 10,
        "actions_per_round": 5,
    }
    base.update(kw)
    return base


def population_stat(seed, population, value, *, assortment=0.4, protocol=2, schema="bounded"):
    """Minimal measured row for the Experiment 21 population-level comparison."""
    if population == "diverse":
        models_ = [f"model-{index}" for index in range(25)]
        by_model = {}
    else:
        models_ = [f"model-{index}" for index in range(5) for _ in range(5)]
        by_model = {
            model: {"directed_pulls": 4, "same_model_pulls": 2,
                    "observed": 0.5, "chance": 1 / 6, "excess": assortment}
            for model in sorted(set(models_))
        }
    return {
        "topology": "complete",
        "n_seats": 25,
        "self_pull_reward": 0.5,
        "messages_off": False,
        "punish_enabled": False,
        "public_chat": False,
        "declared_intents": False,
        "rounds": 10,
        "actions_per_round": 5,
        "observation_schema": schema,
        "gameplay_protocol_version": protocol,
        "seed": seed,
        "support_concentration": value,
        "top_inbound_support_share": 0.2,
        "cooperation_rate": 0.6,
        "confirmed_durable_coalitions": 1,
        "confirmed_durable_mutual": 1,
        "confirmed_durable_winner_backing": 0,
        "same_model_support_excess": assortment if population == "five_by_five" else 0,
        "same_model_support_by_model": by_model,
        "seats": [
            {"seat": f"P{index + 1}", "model": model}
            for index, model in enumerate(models_)
        ],
    }


def intent_reactivity_stat(seed, intents, cooperation, concentration, coalitions=1,
                           protocol=2, seat_map=None):
    seat_map = seat_map or {f"P{index + 1}": f"model-{index}" for index in range(5)}
    return {
        "experiment_config": (
            "23_deception_intent_validation" if intents else "22_cooperation_cost_curve"),
        "topology": "complete",
        "n_seats": 5,
        "self_pull_reward": 0.5,
        "messages_off": False,
        "punish_enabled": False,
        "public_chat": False,
        "declared_intents": intents,
        "rounds": 10,
        "actions_per_round": 5,
        "observation_schema": "bounded",
        "gameplay_protocol_version": protocol,
        "seed": seed,
        "cooperation_rate": cooperation,
        "support_concentration": concentration,
        "confirmed_durable_coalitions": coalitions,
        "seats": [{"seat": seat, "model": model} for seat, model in seat_map.items()],
    }


class TestBootstrap(unittest.TestCase):
    def test_constant_values_give_point_ci(self):
        lo, hi = stats.bootstrap_interval([0.2, 0.2, 0.2], trials=200)
        self.assertAlmostEqual(lo, 0.2)
        self.assertAlmostEqual(hi, 0.2)

    def test_single_value(self):
        self.assertEqual(stats.bootstrap_interval([0.5]), (0.5, 0.5))

    def test_empty_and_all_none(self):
        self.assertEqual(stats.bootstrap_interval([]), (None, None))
        self.assertEqual(stats.bootstrap_interval([None, None]), (None, None))

    def test_holm_uses_the_fixed_family_size(self):
        adjusted = stats.holm_bonferroni([0.02, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(adjusted[0][0], 0.08)
        self.assertFalse(adjusted[0][1])

    def test_ci_brackets_the_mean_and_is_deterministic(self):
        vals = [0.0, 0.1, 0.2, 0.3, 0.4]
        lo, hi = stats.bootstrap_interval(vals, trials=500, seed=1)
        self.assertLessEqual(lo, 0.2)
        self.assertGreaterEqual(hi, 0.2)
        self.assertEqual((lo, hi), stats.bootstrap_interval(vals, trials=500, seed=1))  # seeded


class TestHelpers(unittest.TestCase):
    def test_slope_of_known_line(self):
        self.assertAlmostEqual(report._slope([(0, 0.0), (1, 2.0), (2, 4.0)]), 2.0)

    def test_slope_needs_two_distinct_x(self):
        self.assertIsNone(report._slope([(1, 0.0), (1, 5.0)]))

    def test_rank_known_and_unknown(self):
        fam, idx = report._family_rank("opus-4.8")
        self.assertEqual(fam, "anthropic")
        self.assertEqual(idx, 4)  # fifth rung: sonnet-4.6, sonnet-5, opus-4.6, opus-4.7, opus-4.8
        self.assertEqual(report._family_rank("fable-5")[1], 5)  # the top rung
        self.assertEqual(report._family_rank("mockllm/model"), (None, None))

    def test_condition_tags(self):
        self.assertEqual(report.label_condition(rec([], {}, SEATS)), "complete/pure/msg-on")
        self.assertEqual(report.label_condition(rec([], {}, SEATS, self_pull_reward=0.5,
                                              topology="ring")), "ring/mixed/msg-on")

    def test_condition_tags_controls(self):
        # Exp-13 episodes must never pool into the same cell as exp-3.
        r = rec([], {}, SEATS, declared_intents=True, seeded_favours=5)
        self.assertEqual(report.label_condition(r), "complete/pure/msg-on/intents/seeded")

    def test_condition_tags_punish(self):
        # The arena 2x2 punish arms must not pool: punish_enabled changes the tag on its own.
        off = report.label_condition(rec([], {}, SEATS))
        on = report.label_condition(rec([], {}, SEATS, punish_enabled=True))
        self.assertNotEqual(off, on)
        self.assertEqual(on, "complete/pure/msg-on/punish")


class TestEpisodeMetrics(unittest.TestCase):
    def test_recomputes_from_events_matching_metrics(self):
        events = [msg("P2", "P3"), pull("P2", "P3", credit="P1"), pull("P3", "P1"), pull("P2", "P2")]
        r = rec(events, {"P1": 2, "P2": 0, "P3": 1, "P4": 0}, SEATS, focal="P1")
        em = report.measure_episode(r)
        self.assertEqual(em["claimed_credit_rate"], metrics.claimed_credit_rate(events))
        self.assertEqual(em["verified_credit_rate"], metrics.verified_credit_rate(events))
        self.assertEqual(em["credit_payback"], metrics.credit_payback(events))
        self.assertEqual(em["gini"], metrics.score_inequality(r["scores"]))
        self.assertEqual(em["focal_model"], "opus-4.8")
        self.assertEqual(em["condition"], "complete/pure/msg-on")
        self.assertEqual(em["communication_turns"], 1)
        self.assertEqual(em["realized_pull_turns"], 3)
        self.assertEqual(em["other_directed_pull_turns"], 2)
        self.assertEqual(em["inbound_support_counts"], {"P1": 1, "P2": 0, "P3": 1, "P4": 0})
        self.assertEqual(em["top_inbound_support_share"], 0.5)


class TestDeceptionValidation(unittest.TestCase):
    @staticmethod
    def _episode(episode_id, model="model-a", success=True):
        events = [
            {"forfeit": False, "type": "intent", "agent": "P2", "round": 1,
             "planned": {"action": "pull", "beneficiary": "P3"}},
            msg("P1", "P2"),
            pull("P2", "P1" if success else "P3"),
        ]
        return {
            "id": episode_id,
            "events": events,
            "seat_map": {"P1": model, "P2": "target", "P3": "other"},
            "principal_seats": ["P1"],
        }

    def test_structural_audit_is_an_upper_bound(self):
        audit = report.deception_opportunity_audit([self._episode("e1")])
        self.assertEqual(audit["counts"]["isolated_opportunity"], 1)
        self.assertEqual(audit["counts"]["isolated_departure"], 1)
        self.assertIn("no message has been labeled deceptive", audit["interpretation"])

    def test_registered_analysis_deduplicates_claims_at_the_message(self):
        episodes = [self._episode("e1", "model-a", True),
                    self._episode("e2", "model-b", False)]
        records = [
            {"episode_id": "e1", "event_index": 1, "speaker": "P1", "recipient": "P2",
             "lean": "likely_strategic", "deception_source": "false_claim"},
            {"episode_id": "e1", "index": 1, "promiser": "P1", "promisee": "P2",
             "likely_deception_evidence": True, "deception_source": "broken_promise"},
            {"episode_id": "e2", "event_index": 1, "speaker": "P1", "recipient": "P2",
             "lean": "likely_strategic", "deception_source": "false_claim"},
        ]
        result = report.deception_validation_analysis(
            episodes, records, min_events=2, min_episodes=2, min_principal_models=2,
            trials=200, seed=3)
        self.assertTrue(result["event_gate_passed"])
        self.assertEqual(result["n_unique_likely_deceptive_messages"], 2)
        self.assertEqual(result["n_eligible_message_events"], 2)
        self.assertEqual(result["n_successful_deceptions"], 1)
        self.assertEqual(result["successful_deception_rate"], 0.5)
        self.assertEqual(result["eligible_records"][0]["sources"],
                         ["broken_promise", "false_claim"])

    def test_gate_fails_without_episode_and_model_coverage(self):
        episode = self._episode("e1")
        record = {"episode_id": "e1", "event_index": 1, "speaker": "P1",
                  "recipient": "P2", "lean": "likely_strategic"}
        result = report.deception_validation_analysis([episode], [record], trials=20)
        self.assertFalse(result["event_gate_passed"])
        self.assertEqual(len(result["gate_failures"]), 3)

    def test_old_focal_logs_use_focal_as_the_principal_speaker(self):
        episodes = [self._episode("e1", "model-a"), self._episode("e2", "model-b")]
        records = []
        for episode in episodes:
            episode.pop("principal_seats")
            episode["focal"] = "P1"
            records.append({
                "episode_id": episode["id"], "event_index": 1,
                "speaker": "P1", "recipient": "P2", "lean": "likely_strategic",
            })
        result = report.deception_validation_analysis(
            episodes, records, min_events=2, min_episodes=2, min_principal_models=2,
            trials=20)
        self.assertTrue(result["event_gate_passed"])
        self.assertEqual(result["principal_speaker_models"], ["model-a", "model-b"])

    def test_carries_cascade_tree_scalars(self):
        # P2 messages the puller P3, P3 pulls for P1 crediting P2: a P2 -> P3 -> P1 credited tree.
        events = [msg("P2", "P3"), msg("P3", "P1"), pull("P3", "P1", credit="P2")]
        em = report.measure_episode(rec(events, {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS))
        self.assertEqual(em["cascade_tree_count"], 1)
        self.assertGreaterEqual(em["cascade_max_depth"], 2)   # P2 -> P3 -> P1 is two hops
        self.assertEqual(em["cascade_max_reach"], 3)          # {P1, P2, P3}
        self.assertEqual(em["cascade_multi_round"], 0)        # persistence only; no coalition label

    def test_cascade_tree_scalars_zero_when_no_credited_pull(self):
        em = report.measure_episode(rec([pull("P2", "P1")], {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS))
        self.assertEqual((em["cascade_tree_count"], em["cascade_max_depth"],
                          em["cascade_multi_round"], em["cascade_max_reach"]), (0, 0, 0, 0))


class TestSeatAttribution(unittest.TestCase):
    """A-3: per-model numbers must exist even when no seat is focal (lineup/arena)."""

    def _lineup_episode(self, flip=False):
        seats = {"P1": "opus-4.8", "P2": "gpt-5.5", "P3": "glm-5.2", "P4": "glm-5.2"}
        if flip:  # different seat assignment, same models (the shuffle)
            seats = {"P1": "glm-5.2", "P2": "opus-4.8", "P3": "gpt-5.5", "P4": "glm-5.2"}
        opus = next(k for k, v in seats.items() if v == "opus-4.8")
        gpt = next(k for k, v in seats.items() if v == "gpt-5.5")
        events = [msg(gpt, opus), pull(gpt, opus), pull(gpt, opus), pull(opus, gpt)]
        scores = {s_: 0 for s_ in seats}
        scores[opus], scores[gpt] = 2, 1
        principals = [k for k, v in seats.items() if v != "glm-5.2"]
        return report.measure_episode(rec(events, scores, seats, principal_seats=principals))

    def test_model_leaderboard_without_focal(self):
        rows = report.model_leaderboard([self._lineup_episode(), self._lineup_episode(flip=True)])
        by_model = {r["model"]: r for r in rows}
        self.assertIn("opus-4.8", by_model)
        self.assertEqual(by_model["opus-4.8"]["n_seats"], 2)  # pooled across both episodes
        self.assertGreater(by_model["opus-4.8"]["net_capture"], by_model["gpt-5.5"]["net_capture"])

    def test_per_seat_forfeit_rate_is_visible(self):
        seats = {"P1": "opus-4.8", "P2": "bg", "P3": "bg", "P4": "bg"}
        events = [pull("P2", "P1"),
                  {"forfeit": True, "forfeit_reason": "parse_failure", "agent": "P1",
                   "type": None, "round": 1}]
        em = report.measure_episode(rec(events, {"P1": 1, "P2": 0, "P3": 0, "P4": 0},
                                        seats, focal="P1"))
        focal_seat = next(s_ for s_ in em["seats"] if s_["seat"] == "P1")
        self.assertEqual(focal_seat["forfeit_rate"], 1.0)  # P1's only event was a forfeit

    def test_per_seat_forfeit_rate_ignores_intents_and_forced(self):
        # The twin of the scorer fix: intent/forced events are not attempts, so a seat with one
        # forfeit + one intent still shows forfeit_rate 1.0, not 0.5.
        seats = {"P1": "opus-4.8", "P2": "bg", "P3": "bg", "P4": "bg"}
        events = [{"forfeit": False, "type": "intent", "agent": "P1", "planned": None, "round": 1},
                  {"forfeit": False, "type": "pull", "agent": "P1", "beneficiary": "P2",
                   "credit": None, "forced": True, "round": 1},
                  {"forfeit": True, "forfeit_reason": "parse_failure", "agent": "P1",
                   "type": None, "round": 1}]
        em = report.measure_episode(rec(events, {"P1": 0, "P2": 1, "P3": 0, "P4": 0},
                                        seats, focal="P1"))
        focal_seat = next(s_ for s_ in em["seats"] if s_["seat"] == "P1")
        self.assertEqual(focal_seat["forfeit_rate"], 1.0)

    def test_principal_vs_crowd(self):
        table = report.principal_vs_crowd([self._lineup_episode()])
        self.assertIsNotNone(table)
        self.assertEqual({r["model"] for r in table["principals"]}, {"opus-4.8", "gpt-5.5"})
        self.assertEqual(table["crowd"]["model"], "glm-5.2")

    def test_floor_contrast_needs_both_conditions(self):
        on = report.measure_episode(rec([msg("P2", "P1"), pull("P2", "P1")],
                                        {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS, focal="P1"))
        self.assertIsNone(report.floor_contrast([on]))
        off = report.measure_episode(rec([pull("P2", "P1")],
                                         {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS,
                                         focal="P1", messages_off=True))
        rows = report.floor_contrast([on, off])
        self.assertEqual(rows[0]["model"], "opus-4.8")
        self.assertIn("contrast", rows[0])

    def test_health_checks_flag_seat_spread(self):
        h = report.health_checks([self._lineup_episode(), self._lineup_episode(flip=True)])
        self.assertIn("score_by_seat_label", h)
        self.assertIn("P1", h["score_by_seat_label"])


class TestDescriptive(unittest.TestCase):
    """The report now MEASURES (numbers + CIs), it does not confirm/refute."""

    def test_reciprocity_reports_capture_and_split(self):
        events = [msg("P2", "P1"), pull("P1", "P2"), pull("P2", "P1")]
        em = [report.measure_episode(rec(events, {"P1": 1, "P2": 1, "P3": 0, "P4": 0}, SEATS, focal="P1"))]
        r = report.reciprocity(em)
        self.assertIsNotNone(r["net_capture"])
        lo, hi = r["net_capture_ci"]
        self.assertLessEqual(lo, r["net_capture"])
        self.assertGreaterEqual(hi, r["net_capture"])
        self.assertIn("payback_rate", r)

    def test_capability_ladder_uses_real_family_ranks(self):
        ems = []
        for i, model in enumerate(["sonnet-4.6", "opus-4.8"]):  # two rungs of the anthropic ladder
            seats = {"P1": model, "P2": "bg", "P3": "bg", "P4": "bg"}
            events = [msg("P2", "P1")] + [pull("P2", "P1")] * (i + 1) + [pull("P1", "P3")]
            ems.append(report.measure_episode(rec(events, {"P1": 1, "P2": 0, "P3": 1, "P4": 0},
                                                  seats, focal="P1")))
        row = next(r for r in report.capability_ladder(ems) if r["family"] == "anthropic")
        self.assertEqual(row["n_points"], 2)
        self.assertIsNotNone(row["slope"])
        self.assertEqual(len(row["slope_ci"]), 2)  # a CI, not a verdict

    def test_resistance_present_and_absent(self):
        self.assertIsNone(report.resistance([]))
        naive = report.measure_episode(rec([pull("P1", "P2")], {"P1": 0, "P2": 1, "P3": 0, "P4": 0},
                                           SEATS, focal="P2", personas={"P2": "neutral"}))
        inoc = report.measure_episode(rec([pull("P1", "P2")], {"P1": 0, "P2": 1, "P3": 0, "P4": 0},
                                          SEATS, focal="P2", personas={"P2": "inoculated"}))
        r = report.resistance([naive, inoc])
        self.assertIsNotNone(r)
        self.assertIn("diff", r)
        self.assertEqual(r["by_model"][0]["model"], "m")  # B-6: grouped per target model

    def test_cascades_by_topology(self):
        em = [report.measure_episode(rec([pull("P2", "P1")], {"P1": 1, "P2": 0, "P3": 0, "P4": 0},
                                         SEATS, focal="P1", topology="ring"))]
        rows = report.cascades_by_topology(em)
        self.assertEqual(rows[0]["topology"], "ring")
        self.assertIn("cascades_ci", rows[0])

    # Regression (D21): zero observed cascades as played — the pulls precede the relay — but a
    # positive shuffled null. The old `if casc else 0.0` gate recorded 0.0 here, biasing the
    # pooled null low (the retracted "ring cascades clear the null" claim).
    ZERO_CASC = [pull("P3", "P1"), pull("P2", "P1"), msg("P1", "P2"), msg("P2", "P3")]
    ZERO_SCORES = {"P1": 2, "P2": 0, "P3": 0, "P4": 0}

    def test_zero_cascade_episode_records_true_null(self):
        em = report.measure_episode(rec(self.ZERO_CASC, self.ZERO_SCORES, SEATS))
        self.assertEqual(em["cascades"], 0)
        self.assertGreater(em["cascade_null"], 0)

    def test_cascades_by_topology_reports_paired_diff_ci(self):
        em = [report.measure_episode(rec(self.ZERO_CASC, self.ZERO_SCORES, SEATS, topology="ring"))]
        row = report.cascades_by_topology(em)[0]
        self.assertAlmostEqual(row["diff"], em[0]["cascades"] - em[0]["cascade_null"])
        self.assertLess(row["diff"], 0)  # zero observed, positive null -> negative diff
        self.assertEqual(len(row["diff_ci"]), 2)

    def test_by_condition_reports_paired_diff_ci(self):
        em = [report.measure_episode(rec(self.ZERO_CASC, self.ZERO_SCORES, SEATS))]
        row = report.by_condition(em)[0]
        self.assertAlmostEqual(row["cascade_diff"], em[0]["cascades"] - em[0]["cascade_null"])
        self.assertEqual(len(row["cascade_diff_ci"]), 2)

    def test_by_condition_reports_registered_action_budget_decomposition(self):
        events = [msg("P1", "P2"), pull("P2", "P1"), pull("P3", "P1"), pull("P4", "P2")]
        em = [report.measure_episode(rec(events, {"P1": 2, "P2": 1, "P3": 0, "P4": 0}, SEATS))]
        row = report.by_condition(em)[0]
        self.assertEqual(row["communication_turn_rate"], 0.25)
        self.assertEqual(row["realized_pull_rate"], 0.75)
        self.assertEqual(row["other_directed_pull_rate"], 0.75)
        self.assertEqual(row["top_inbound_support_share"], 2 / 3)
        self.assertAlmostEqual(row["support_concentration"], 1 / 9)
        self.assertAlmostEqual(row["support_concentration_ci"][0], 1 / 9)
        self.assertAlmostEqual(row["support_concentration_ci"][1], 1 / 9)

    def test_action_allocation_factorial_uses_paired_seed_contrasts(self):
        # Seed 1: talk effects 0.2 without punishment and 0.4 with punishment.
        # Seed 2: talk effects -0.1 without punishment and 0.1 with punishment.
        episodes = [
            allocation_stat(1, False, False, 0.0),
            allocation_stat(1, True, False, 0.2),
            allocation_stat(1, False, True, 0.1),
            allocation_stat(1, True, True, 0.5),
            allocation_stat(2, False, False, 0.2),
            allocation_stat(2, True, False, 0.1),
            allocation_stat(2, False, True, 0.3),
            allocation_stat(2, True, True, 0.4),
        ]
        row = report.action_allocation_factorial(episodes, trials=200)[0]
        self.assertEqual(row["n_paired_seeds"], 2)
        self.assertAlmostEqual(row["communication_main_effect"], 0.15)
        self.assertAlmostEqual(row["communication_effect_no_punishment"], 0.05)
        self.assertAlmostEqual(row["communication_effect_with_punishment"], 0.25)
        self.assertAlmostEqual(row["communication_punishment_interaction"], 0.2)
        self.assertAlmostEqual(row["punishment_main_effect"], 0.2)
        expected_arm_means = {
            "talk_no_punishment": 0.15,
            "talk_punishment": 0.45,
            "no_talk_no_punishment": 0.1,
            "no_talk_punishment": 0.2,
        }
        for arm, expected in expected_arm_means.items():
            self.assertAlmostEqual(row["arm_means"][arm], expected)
        lo, hi = row["communication_main_effect_ci"]
        self.assertLessEqual(lo, row["communication_main_effect"])
        self.assertGreaterEqual(hi, row["communication_main_effect"])
        self.assertIn("communication_main_effect_p", row)
        self.assertIn("punishment_main_effect_p", row)

    def test_action_allocation_factorial_excludes_incomplete_seed(self):
        episodes = [
            allocation_stat(1, False, False, 0.0),
            allocation_stat(1, True, False, 0.2),
            allocation_stat(1, False, True, 0.1),
            allocation_stat(1, True, True, 0.5),
            allocation_stat(2, False, False, 0.2),
            allocation_stat(2, True, False, None),
            allocation_stat(2, False, True, 0.3),
            allocation_stat(2, True, True, 0.4),
        ]
        row = report.action_allocation_factorial(episodes)[0]
        self.assertEqual(row["paired_seeds"], [1])
        self.assertEqual(row["incomplete_seeds"], [2])
        self.assertEqual(row["n_paired_seeds"], 1)
        self.assertEqual(row["n_incomplete_seeds"], 1)
        self.assertEqual(row["unscorable_by_arm"]["talk_no_punishment"], 1)
        self.assertFalse(any(row["missing_by_arm"].values()))
        self.assertAlmostEqual(row["communication_main_effect"], 0.3)

    def test_action_allocation_factorial_rejects_duplicate_or_mismatched_pairs(self):
        episodes = [
            allocation_stat(1, False, False, 0.0),
            allocation_stat(1, True, False, 0.2),
            allocation_stat(1, False, True, 0.1),
            allocation_stat(1, True, True, 0.5),
        ]
        excluded = report.action_allocation_factorial(episodes + [episodes[0]])[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("duplicate", excluded["reason"])

        swapped = {"P1": "m", "P2": "opus-4.8", "P3": "m", "P4": "m"}
        mismatched = episodes[:-1] + [allocation_stat(1, True, True, 0.5, swapped)]
        excluded = report.action_allocation_factorial(mismatched)[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("seat assignment", excluded["reason"])

    def test_format_report_prints_registered_factorial_contrast(self):
        events = [pull("P2", "P1"), pull("P3", "P1"), pull("P4", "P2")]
        scores = {"P1": 2, "P2": 1, "P3": 0, "P4": 0}
        episodes = [
            report.measure_episode(rec(
                events, scores, SEATS, seed=1, messages_off=not talk,
                punish_enabled=punish, self_pull_reward=0.5))
            for talk in (False, True)
            for punish in (False, True)
        ]
        out = report.format_report(episodes)
        self.assertIn("registered paired messages x punishment contrast", out)
        self.assertIn("primary communication effect", out)
        self.assertIn("secondary interaction", out)

    def test_cooperation_cost_analysis_uses_paired_mixed_arm_slope(self):
        rewards = (0, 0.25, 0.5, 0.75, 1.0)
        values = {
            1: {0: None, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 1.0: 0.2},
            2: {0: None, 0.25: 0.7, 0.5: 0.6, 0.75: 0.5, 1.0: 0.4},
        }
        episodes = []
        for episode_seed in values:
            for reward in rewards:
                value = values[episode_seed][reward]
                model_values = None if reward == 0 else {
                    "model-a": value,
                    "model-b": value - 0.1,
                    "model-c": value - 0.2,
                }
                episodes.append(cost_stat(
                    episode_seed, reward, value, model_values=model_values))

        row = report.cooperation_cost_analysis(episodes, trials=200, seed=2)[0]
        self.assertEqual(row["paired_seeds"], [1, 2])
        self.assertAlmostEqual(row["primary_slope"], -0.6)
        for reward, expected in {0.25: 0.75, 0.5: 0.6, 0.75: 0.45, 1.0: 0.3}.items():
            self.assertAlmostEqual(row["arm_means"][reward], expected)
        self.assertNotIn(0, row["arm_means"])  # pure is never treated as a cooperation rate
        self.assertEqual(row["pure_baseline"]["n_matched_seeds"], 2)
        self.assertAlmostEqual(row["pure_baseline"]["support_concentration"], 0.2)
        self.assertEqual(len(row["model_rows"]), 3)
        self.assertTrue(all("p_adjusted" in model for model in row["model_rows"]))
        by_model = {model["model"]: model for model in row["model_rows"]}
        self.assertEqual(by_model["model-a"]["ranks"][0.25], 1.0)
        self.assertEqual(by_model["model-c"]["ranks"][1.0], 3.0)
        self.assertAlmostEqual(row["endpoint_rank_correlation"], 1.0)

    def test_cooperation_cost_analysis_excludes_incomplete_seed_as_block(self):
        episodes = [
            cost_stat(seed, reward, value)
            for seed, cells in {
                1: {0: None, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 1.0: 0.2},
                2: {0: None, 0.25: 0.7, 0.5: None, 0.75: 0.5, 1.0: 0.4},
            }.items()
            for reward, value in cells.items()
        ]
        row = report.cooperation_cost_analysis(episodes, trials=100)[0]
        self.assertEqual(row["paired_seeds"], [1])
        self.assertEqual(row["incomplete_seeds"], [2])
        self.assertEqual(row["unscorable_by_reward"][0.5], 1)
        self.assertFalse(any(row["missing_by_reward"].values()))
        self.assertAlmostEqual(row["primary_slope"], -0.8)

    def test_cooperation_cost_analysis_rejects_duplicate_or_mismatched_pairs(self):
        episodes = [
            cost_stat(1, reward, value)
            for reward, value in ((0, None), (0.25, 0.8), (0.5, 0.6),
                                  (0.75, 0.4), (1.0, 0.2))
        ]
        excluded = report.cooperation_cost_analysis(episodes + [episodes[-1]])[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("duplicate", excluded["reason"])

        swapped = {"P1": "model-b", "P2": "model-a", "P3": "model-c"}
        mismatched = episodes[:-1] + [cost_stat(1, 1.0, 0.2, seat_map=swapped)]
        excluded = report.cooperation_cost_analysis(mismatched)[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("seat assignment", excluded["reason"])

    def test_format_report_prints_registered_cooperation_cost_analysis(self):
        seat_map = {"P1": "opus-4.8", "P2": "gpt-5.5"}
        episodes = []
        for reward in (0, 0.25, 0.5, 0.75, 1.0):
            events = [pull("P1", "P2"), pull("P2", "P1")]
            if reward:
                events += [pull("P1", "P1"), pull("P2", "P2")]
            episodes.append(report.measure_episode(rec(
                events, {"P1": 1, "P2": 1}, seat_map, seed=1,
                self_pull_reward=reward, rounds=10)))
        out = report.format_report(episodes)
        self.assertIn("registered paired cooperation cost curve", out)
        self.assertIn("primary cooperation trend", out)
        self.assertIn("partner-choice baseline", out)
        self.assertNotIn("self=0:", out)

    def test_population_composition_analysis_is_shared_seed_population_contrast(self):
        episodes = [
            population_stat(1, "diverse", 0.1),
            population_stat(1, "five_by_five", 0.4, assortment=0.5),
            population_stat(2, "diverse", 0.2),
            population_stat(2, "five_by_five", 0.3, assortment=0.3),
        ]
        row = report.population_composition_analysis(episodes, trials=200, seed=3)[0]
        self.assertEqual(row["paired_seeds"], [1, 2])
        self.assertAlmostEqual(row["arm_means"]["diverse"], 0.15)
        self.assertAlmostEqual(row["arm_means"]["five_by_five"], 0.35)
        self.assertAlmostEqual(row["primary_composition_effect"], 0.2)
        self.assertIn("primary_composition_effect_p", row)
        self.assertAlmostEqual(row["same_model_support_excess"], 0.4)
        self.assertEqual(set(row["same_model_support_by_model"]),
                         {f"model-{index}" for index in range(5)})

    def test_same_model_favoritism_summary_needs_no_diverse_arm(self):
        episodes = [
            population_stat(1, "five_by_five", 0.4, assortment=0.5),
            population_stat(2, "five_by_five", 0.3, assortment=0.3),
            population_stat(3, "five_by_five", 0.2, assortment=0.1),
        ]
        episodes[2]["same_model_support_excess"] = None
        self.assertEqual(report.population_composition_analysis(episodes), [])
        row = report.same_model_favoritism_summary(episodes, trials=200, seed=3)[0]
        self.assertEqual(row["seeds"], [1, 2])
        self.assertEqual(row["n_episodes"], 2)
        self.assertEqual(row["n_unscorable"], 1)
        self.assertAlmostEqual(row["same_model_support_excess"], 0.4)
        lo, hi = row["same_model_support_excess_ci"]
        self.assertLessEqual(lo, 0.4)
        self.assertGreaterEqual(hi, 0.4)
        self.assertEqual(set(row["same_model_support_by_model"]),
                         {f"model-{index}" for index in range(5)})
        excluded = report.same_model_favoritism_summary(episodes + [episodes[0]])[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("duplicate", excluded["reason"])
        talk_off = population_stat(4, "five_by_five", 0.4)
        talk_off["messages_off"] = True
        self.assertEqual(report.same_model_favoritism_summary([talk_off]), [])
        out = report.format_report([
            report.measure_episode(rec(
                [pull("P1", "P2")], {"P1": 1, "P2": 1}, {"P1": "a", "P2": "b"}, seed=1))
        ])
        self.assertNotIn("same-model favoritism", out)

    def test_same_model_favoritism_summary_pools_makeup_campaign_episodes(self):
        episodes = [population_stat(1, "five_by_five", 0.4, assortment=0.5),
                    population_stat(2, "five_by_five", 0.3, assortment=0.3)]
        episodes[1]["experiment_config"] = "21_arena_5x5_makeup"  # same registered experiment, rerun seeds
        row = report.same_model_favoritism_summary(episodes, trials=200, seed=3)[0]
        self.assertEqual(row["seeds"], [1, 2])
        episodes[1]["experiment_config"] = "22_cooperation_cost_curve"
        self.assertEqual(report.same_model_favoritism_summary(episodes, trials=200, seed=3)[0]["seeds"], [1])

    def test_population_composition_analysis_excludes_incomplete_seed_as_block(self):
        episodes = [
            population_stat(1, "diverse", 0.1),
            population_stat(1, "five_by_five", 0.4),
            population_stat(2, "diverse", 0.2),
            population_stat(2, "five_by_five", None),
        ]
        row = report.population_composition_analysis(episodes, trials=100)[0]
        self.assertEqual(row["paired_seeds"], [1])
        self.assertEqual(row["incomplete_seeds"], [2])
        self.assertEqual(row["unscorable_by_population"]["five_by_five"], 1)
        self.assertFalse(any(row["missing_by_population"].values()))

    def test_population_composition_analysis_rejects_duplicates_and_protocol_mismatch(self):
        episodes = [
            population_stat(1, "diverse", 0.1),
            population_stat(1, "five_by_five", 0.4),
        ]
        excluded = report.population_composition_analysis(episodes + [episodes[-1]])[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("duplicate", excluded["reason"])
        mismatched = [episodes[0], population_stat(
            1, "five_by_five", 0.4, protocol=3)]
        self.assertEqual(report.population_composition_analysis(mismatched), [])

    def test_intent_reactivity_is_paired_and_descriptive(self):
        episodes = [
            intent_reactivity_stat(1, False, 0.5, 0.1, 1),
            intent_reactivity_stat(1, True, 0.6, 0.3, 2),
            intent_reactivity_stat(2, False, 0.4, 0.2, 0),
            intent_reactivity_stat(2, True, 0.6, 0.3, 1),
        ]
        row = report.intent_reactivity_analysis(episodes, trials=200, seed=4)[0]
        self.assertEqual(row["paired_seeds"], [1, 2])
        self.assertAlmostEqual(row["intent_minus_natural_cooperation"], 0.15)
        self.assertAlmostEqual(row["intent_minus_natural_support_concentration"], 0.15)
        self.assertAlmostEqual(row["intent_minus_natural_durable_coalitions"], 1.0)
        self.assertIn("not an equivalence test", row["interpretation"])

    def test_intent_reactivity_rejects_changed_seating(self):
        natural = intent_reactivity_stat(1, False, 0.5, 0.1)
        swapped = {"P1": "model-1", "P2": "model-0", **{
            f"P{index + 1}": f"model-{index}" for index in range(2, 5)}}
        intent = intent_reactivity_stat(1, True, 0.6, 0.2, seat_map=swapped)
        excluded = report.intent_reactivity_analysis([natural, intent])[0]
        self.assertEqual(excluded["status"], "excluded")
        self.assertIn("intent-reactivity seating", excluded["reason"])


class TestPaperClaimSummary(unittest.TestCase):
    def test_material_branches_use_holm_and_the_practical_interval(self):
        factorial = [{
            "communication_main_effect": 0.10,
            "communication_main_effect_ci": (0.08, 0.12),
            "communication_main_effect_p": 0.001,
            "punishment_main_effect": 0.07,
            "punishment_main_effect_ci": (0.051, 0.09),
            "punishment_main_effect_p": 0.001,
            "communication_punishment_interaction": 0.01,
            "communication_punishment_interaction_ci": (-0.02, 0.04),
            "communication_punishment_interaction_p": 0.5,
        }]
        cost = [{"primary_slope": -0.10, "primary_slope_ci": (-0.12, -0.08),
                 "primary_p": 0.001}]
        composition = [{"primary_composition_effect": None,
                        "primary_composition_effect_ci": (None, None),
                        "primary_composition_effect_p": None}]
        with (mock.patch.object(report, "action_allocation_factorial", return_value=factorial),
              mock.patch.object(report, "cooperation_cost_analysis", return_value=cost),
              mock.patch.object(report, "population_composition_analysis",
                                return_value=composition)):
            result = report.paper_claim_summary([], trials=20)
        self.assertEqual(result["claim_2"]["status"], "material_supported")
        by_name = {row["name"]: row for row in result["claim_3_mechanisms"]}
        self.assertEqual(by_name["cooperation_cost"]["status"], "material_supported")
        self.assertEqual(by_name["punishment"]["status"], "material_supported")
        self.assertEqual(result["paper_branch"], "communication_plus_named_mechanisms")

    def test_missing_mechanisms_do_not_shrink_the_holm_family(self):
        cost = [{"primary_slope": -0.10, "primary_slope_ci": (-0.12, -0.08),
                 "primary_p": 0.02}]
        with (mock.patch.object(report, "action_allocation_factorial", return_value=[]),
              mock.patch.object(report, "cooperation_cost_analysis", return_value=cost),
              mock.patch.object(report, "population_composition_analysis", return_value=[])):
            result = report.paper_claim_summary([], trials=20)
        cost_row = result["claim_3_mechanisms"][0]
        self.assertAlmostEqual(cost_row["p_adjusted"], 0.08)
        self.assertEqual(cost_row["status"], "not_supported")
        self.assertEqual(result["claim_3_branch"], "drop_mechanism_claim")

    def test_format_report_prints_population_snapshot_without_seat_claim(self):
        diverse = {f"P{index + 1}": f"model-{index}" for index in range(25)}
        repeated = {
            f"P{index + 1}": f"model-{index // 5}"
            for index in range(25)
        }
        events = [pull("P1", "P2"), pull("P2", "P1"), pull("P3", "P1")]
        measured = [
            report.measure_episode(rec(
                events, {seat: 0 for seat in seat_map}, seat_map, seed=1,
                self_pull_reward=0.5, rounds=10, gameplay_protocol_version=2,
                observation_schema="bounded"))
            for seat_map in (diverse, repeated)
        ]
        out = report.format_report(measured)
        self.assertIn("registered five-by-five population snapshot", out)
        self.assertIn("primary population composition effect", out)
        self.assertIn("population-level, never a paired-seat effect", out)

    def test_promise_summary_reads_breach_rate(self):
        seats = {"P1": "opus-4.8", "P2": "bg", "P3": "bg", "P4": "bg"}
        em = [report.measure_episode(rec([pull("P2", "P1")], {"P1": 1, "P2": 0, "P3": 0, "P4": 0},
                                         seats, focal="P1"))]
        self.assertIsNone(report.promise_summary(em, None))
        ledger = {"opus-4.8": {"promise_breach_rate": 0.5, "promises": 4,
                                "headline_promises": 2}}
        ps = report.promise_summary(em, ledger)
        self.assertEqual(ps["rows"][0]["model"], "opus-4.8")
        self.assertEqual(ps["rows"][0]["breach_rate"], 0.5)

    def test_format_report_is_descriptive_not_verdicts(self):
        em = [report.measure_episode(rec([msg("P2", "P1"), pull("P2", "P1")],
                                         {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS, focal="P1"))]
        out = report.format_report(em)
        self.assertIn("secondary support dynamics", out)
        self.assertNotIn("refute", out.lower())

    def test_enforcement_joins_episode_aligned_misconduct(self):
        events = [msg("P2", "P1"),
                  {"forfeit": False, "type": "punish", "agent": "P1", "target": "P2",
                   "round": 1}]
        episode = report.measure_episode(rec(
            events, {"P1": 0, "P2": 0, "P3": 0, "P4": 0}, SEATS, id="episode-1"))
        promises = {"m": {"promise_records": [{
            "episode_id": "episode-1", "promiser": "P2",
            "outcome": "broken_by_choice", "outcome_event_index": 0,
        }]}}
        lies = {"m": {"false_claim_records": [{
            "episode_id": "episode-1", "speaker": "P2", "event_index": 0,
        }]}}
        out = report.enforcement_summary([episode], promises, lies)
        self.assertEqual(out["after_broken_by_choice"], 1)
        self.assertEqual(out["after_false_claim"], 1)


class TestControlsWiring(unittest.TestCase):
    """Exp-13: intent_persuasion + reciprocity_response must reach the report (they had no callers)."""

    # In each of two rounds P1 plans P3, then its first directed pull departs: toward messager P2
    # in round 1 and non-messager P4 in round 2 (gap 0). A forced P4->P3 favour is repaid by P3;
    # no chosen favour is repaid (repayments must FOLLOW favours).
    CTRL_EVENTS = [
        {"forfeit": False, "type": "intent", "agent": "P1",
         "planned": {"action": "pull", "beneficiary": "P3"}, "round": 1},
        msg("P2", "P1"),
        pull("P1", "P2"),
        {"forfeit": False, "type": "intent", "agent": "P1",
         "planned": {"action": "pull", "beneficiary": "P3"}, "round": 2},
        {"forfeit": False, "type": "pull", "agent": "P1", "beneficiary": "P4",
         "credit": None, "round": 2},
        {"forfeit": False, "type": "pull", "agent": "P4", "beneficiary": "P3",
         "credit": None, "forced": True, "round": 1},
        pull("P3", "P4"),
    ]
    CTRL_SCORES = {"P1": 0, "P2": 1, "P3": 1, "P4": 2}

    def _ctrl_episode(self):
        return report.measure_episode(rec(self.CTRL_EVENTS, self.CTRL_SCORES, SEATS,
                                          declared_intents=True, seeded_favours=1))

    def test_measure_episode_populates_controls_fields(self):
        em = self._ctrl_episode()
        self.assertEqual(em["intent_rate_msg"], 1.0)
        self.assertEqual(em["intent_rate_nomsg"], 1.0)
        self.assertEqual(em["intent_gap"], 0.0)
        self.assertEqual(em["repay_forced"], 1.0)
        self.assertEqual(em["repay_chosen"], 0.0)

    def test_controls_fields_none_without_the_controls(self):
        em = report.measure_episode(rec([msg("P2", "P1"), pull("P2", "P1")],
                                        {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS))
        self.assertIsNone(em["intent_gap"])
        self.assertIsNone(em["repay_forced"])

    def test_controls_summary_with_episode_bootstrap_cis(self):
        cs = report.controls_summary([self._ctrl_episode(), self._ctrl_episode()])
        self.assertEqual(cs["n_intent"], 2)
        self.assertEqual(cs["n_seeded"], 2)
        self.assertAlmostEqual(cs["intent_gap"], 0.0)
        self.assertEqual(cs["intent_gap_ci"], (0.0, 0.0))  # constant values -> point CI
        self.assertAlmostEqual(cs["repay_forced"], 1.0)
        self.assertAlmostEqual(cs["repay_chosen"], 0.0)
        self.assertEqual(len(cs["repay_chosen_ci"]), 2)
        self.assertEqual(len(cs["repay_forced_ci"]), 2)

    def test_controls_summary_surfaces_prior_pull_concentration_punishment(self):
        cs = report.controls_summary([self._ctrl_episode(), self._ctrl_episode()])
        self.assertIn("intent_prior_pull_share", cs)
        self.assertIn("concentration_chance", cs)
        self.assertIn("punishment", cs)
        self.assertIn("total", cs["punishment"])

    def test_measure_episode_punishment_fields(self):
        events = [{"forfeit": False, "type": "punish", "agent": "P2", "target": "P1",
                   "beneficiary": None, "credit": None, "round": 1}]
        em = report.measure_episode(rec(events, {"P1": 0, "P2": 0, "P3": 0, "P4": 0}, SEATS))
        self.assertEqual(em["punishment_total"], 1)
        self.assertEqual(em["punished_after_broken_promise"], 0)
        self.assertEqual(em["punished_after_false_claim"], 0)

    def test_controls_summary_surfaces_punishment_without_exp13_controls(self):
        # A punish-only episode (no declared intents / seeded favours) still surfaces punishment.
        events = [{"forfeit": False, "type": "punish", "agent": "P2", "target": "P1",
                   "beneficiary": None, "credit": None, "round": 1}]
        cs = report.controls_summary(
            [report.measure_episode(rec(events, {"P1": 0, "P2": 0, "P3": 0, "P4": 0}, SEATS))])
        self.assertIsNotNone(cs)
        self.assertEqual(cs["punishment"]["total"], 1)

    def test_controls_summary_none_without_controls(self):
        em = [report.measure_episode(rec([msg("P2", "P1"), pull("P2", "P1")],
                                         {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS))]
        self.assertIsNone(report.controls_summary(em))
        self.assertNotIn("declared intents", report.format_report(em))

    def test_format_report_prints_controls_section(self):
        out = report.format_report([self._ctrl_episode()])
        self.assertIn("declared intents + seeded favours", out)
        self.assertIn("gap", out)
        self.assertIn("forced", out)


def estat(model=None, by=None, frm=None, topo="complete", moff=False, persona=None, focal="P1"):
    """A minimal measure_episode-shaped stat carrying only the fields contrast_family reads."""
    return {"focal": focal, "focal_model": model, "by_focal": by, "from_focal": frm,
            "topology": topo, "messages_off": moff, "seats": [],
            "personas": {focal: persona} if (focal and persona) else {}}


class TestMultiplicity(unittest.TestCase):
    """Audit fix 4: the registered contrast family + Benjamini-Hochberg, so ~20 CI-based claims
    get a fixed multiplicity denominator instead of being read one interval at a time."""

    def test_benjamini_hochberg_hand_computed(self):
        # All-equal p's: every adjusted p is alpha (the classic step-up cap), all rejected.
        out = stats.benjamini_hochberg([0.05, 0.05, 0.05, 0.05, 0.05])
        for adj, reject in out:
            self.assertAlmostEqual(adj, 0.05)
            self.assertTrue(reject)
        # Audit-shaped: two strong survivors, one fragile 0.04 (raw-significant but BH lifts it over
        # alpha), two nulls -> reject pattern [T, T, F, F, F] with hand-computed q-values.
        out = stats.benjamini_hochberg([0.001, 0.001, 0.04, 0.9, 0.9])
        adj = [a for a, _ in out]
        self.assertAlmostEqual(adj[0], 0.0025)
        self.assertAlmostEqual(adj[1], 0.0025)
        self.assertAlmostEqual(adj[2], 0.04 * 5 / 3)   # 0.0667 > alpha -> the fragile demotion
        self.assertAlmostEqual(adj[3], 0.9)
        self.assertAlmostEqual(adj[4], 0.9)
        self.assertEqual([r for _, r in out], [True, True, False, False, False])

    def test_benjamini_hochberg_preserves_input_order(self):
        # Shuffled input; adjusted p's must come back aligned to the input positions.
        out = stats.benjamini_hochberg([0.03, 0.005, 0.04, 0.01])
        self.assertEqual([round(a, 4) for a, _ in out], [0.04, 0.02, 0.04, 0.02])

    def test_contrast_bootstrap_p_agrees_with_separation(self):
        # Cleanly separated constant sides -> the diff never crosses 0 -> p == 0, CI a point.
        est, ci, p = stats.contrast_bootstrap([0.10, 0.10, 0.10], [0.02, 0.02, 0.02], trials=500)
        self.assertAlmostEqual(est, 0.08)
        self.assertAlmostEqual(ci[0], 0.08)
        self.assertAlmostEqual(ci[1], 0.08)
        self.assertEqual(p, 0.0)
        # Identical sides -> every replicate diff is 0 -> p == 1 (never significant).
        _, _, p_null = stats.contrast_bootstrap([0.05, 0.05], [0.05, 0.05], trials=500)
        self.assertEqual(p_null, 1.0)
        # Empty side -> undefined.
        self.assertEqual(stats.contrast_bootstrap([]), (None, (None, None), None))

    def test_floor_contrast_diff_carries_bootstrap_ci(self):
        on = report.measure_episode(rec([msg("P2", "P1"), pull("P2", "P1"), pull("P2", "P1")],
                                        {"P1": 2, "P2": 0, "P3": 0, "P4": 0}, SEATS, focal="P1"))
        off = report.measure_episode(rec([pull("P2", "P1")],
                                         {"P1": 1, "P2": 0, "P3": 0, "P4": 0}, SEATS,
                                         focal="P1", messages_off=True))
        row = report.floor_contrast([on, off])[0]
        # The per-model diff gets its own seed-stable interval (not two overlapping absolutes).
        self.assertEqual(len(row["contrast_ci"]), 2)
        exp_contrast, exp_ci, _ = stats.contrast_bootstrap([on["by_focal"]], [off["by_focal"]])
        self.assertAlmostEqual(row["contrast"], exp_contrast)
        self.assertEqual(tuple(row["contrast_ci"]), exp_ci)

    # A synthetic family reproducing the audit verdicts (D21): the gpt-5.5/qwen wedge survive, opus
    # is fragile (raw-significant, BH-demoted), and every ring-vs-complete + resist contrast dies.
    def _audit_family(self):
        gpt_talk, qwen_talk = [0.10, 0.10, 0.10], [0.09, 0.09, 0.09]
        opus_talk = [0.05, 0.04, 0.03, 0.0, -0.01, 0.04]
        opus_floor = [0.0, 0.01, 0.0, -0.01, 0.0, 0.01]
        em = []
        for v in gpt_talk:
            em.append(estat("gpt-5.5", by=v, persona="attacker"))
        for v in [0.02] * 3:
            em.append(estat("gpt-5.5", by=v, persona="attacker", moff=True))
        for v in qwen_talk:
            em.append(estat("qwen3.7-max", by=v, persona="attacker"))
        for v in [0.02] * 3:
            em.append(estat("qwen3.7-max", by=v, persona="attacker", moff=True))
        for v in opus_talk:
            em.append(estat("opus-4.8", by=v, persona="attacker"))
        for v in opus_floor:
            em.append(estat("opus-4.8", by=v, persona="attacker", moff=True))
        # ring pads: each ring list == that model's complete-talk list -> diff is exactly 0 (p=1)
        for model, talk in (("gpt-5.5", gpt_talk), ("qwen3.7-max", qwen_talk), ("opus-4.8", opus_talk)):
            for v in talk:
                em.append(estat(model, by=v, topo="ring", persona="attacker"))
        # resist pads: identical constant naive/inoculated sides -> p=1
        for model in ("glm-5.2", "gpt-5.5", "qwen3.7-max"):
            for v in [0.05] * 3:
                em.append(estat(model, frm=v, persona="neutral"))
            for v in [0.05] * 3:
                em.append(estat(model, frm=v, persona="inoculated"))
        return em

    def test_contrast_family_reproduces_audit_survive_pattern(self):
        rows = report.contrast_family(self._audit_family())
        by = {(r["family"], r["contrast"]): r for r in rows}
        # The wedge: gpt-5.5 and qwen survive BH; opus is raw-significant but demoted over alpha.
        self.assertTrue(by[("wedge", "gpt-5.5")]["survives"])
        self.assertTrue(by[("wedge", "qwen3.7-max")]["survives"])
        opus = by[("wedge", "opus-4.8")]
        self.assertLess(opus["p_raw"], 0.05)          # fragile: raw-significant
        self.assertGreater(opus["p_adj"], 0.05)        # ...but BH lifts it over alpha
        self.assertFalse(opus["survives"])
        # Ring-cascade and resist contrasts all die.
        for key, r in by.items():
            if key[0] in ("ring_vs_complete", "resist_gap"):
                self.assertFalse(r["survives"], key)
        # The multiplicity denominator is the full declared set, adjusted together.
        self.assertEqual(sum(1 for r in rows if r["p_raw"] is not None), 9)

    def test_contrast_family_survives_flag_is_false_when_uncomputable(self):
        # Only the talk side present -> the wedge contrast can't be computed -> dropped from BH.
        rows = report.contrast_family([estat("gpt-5.5", by=0.1, persona="attacker")])
        self.assertEqual(rows, [])


def pr(a, b, r):
    """A round-tagged directed pull (the multi-round fixtures the S14 loyalty/coalition metrics need)."""
    return {"forfeit": False, "type": "pull", "agent": a, "beneficiary": b, "credit": None, "round": r}


# A 3-seat coalition episode: P1<->P2 nominate in r1..3, then confirm with insider support in r4..5.
COAL_SEATS = {"P1": "opus-4.8", "P2": "gpt-5.5", "P3": "glm-5.2"}
COAL_EVENTS = [
    pr("P3", "P1", 1),
    pr("P1", "P2", 1), pr("P2", "P1", 1),
    pr("P1", "P2", 2), pr("P2", "P1", 2),
    pr("P1", "P2", 3), pr("P2", "P1", 3),
    pr("P1", "P3", 3),
    pr("P1", "P2", 4), pr("P2", "P1", 4),
    pr("P1", "P2", 5), pr("P2", "P1", 5),
]
COAL_SCORES = {"P1": 5, "P2": 5, "P3": 2}


def coal_episode():
    return report.measure_episode(rec(COAL_EVENTS, COAL_SCORES, COAL_SEATS, focal="P1",
                                      self_pull_reward=0.5))


class TestMetricUpgradesWiring(unittest.TestCase):
    """S14 upgrades (metrics.py) must reach measure_episode + format_report: both payback rates
    beside the chance floor with the normalized lift, per-model cooperation profile, retention vs
    its floor, and the coalition k-sweep + betrayal count + worst pair."""

    def test_measure_episode_carries_payback_and_retention_upgrades(self):
        em = coal_episode()
        # The unbounded payback_rate key stays (site/derive.py reads it) and equals the floor's obs.
        self.assertIn("payback_rate", em)
        pcf = metrics.payback_chance_floor(COAL_EVENTS)
        self.assertAlmostEqual(em["payback_rate"], pcf["observed"])
        self.assertAlmostEqual(em["payback_floor"], pcf["floor_mean"])
        self.assertAlmostEqual(em["payback_lift"], pcf["lift"])
        wp = metrics.windowed_payback(COAL_EVENTS, window_rounds=1)
        self.assertAlmostEqual(em["payback_windowed_rate"],
                               sum(t["payback_windowed"] for t in wp) / len(wp))
        self.assertEqual(em["retention_floor"], metrics.retention_chance_floor(COAL_EVENTS)["floor"])

    def test_seat_rows_carry_cooperation_fields(self):
        seat = next(s for s in coal_episode()["seats"] if s["seat"] == "P1")
        for key in ("cooperation_rate", "initiated_cooperations", "initiation_share",
                    "retaliation", "forgiveness", "coop_slope"):
            self.assertIn(key, seat)

    def test_measure_episode_carries_coalition_upgrades(self):
        em = coal_episode()
        self.assertEqual(em["mutual_support_sweep"][3]["pairs"], 1)
        self.assertEqual(em["mutual_support_runs"], [5])
        self.assertEqual(em["confirmed_durable_coalitions"], 1)
        self.assertEqual(em["confirmed_durable_mutual"], 1)
        self.assertEqual(em["confirmed_durable_winner_backing"], 0)
        self.assertEqual(em["betrayal_count"], 0)                # no room for the >=2 post-run rounds
        self.assertEqual(em["n_sustained_pairs"], 1)
        self.assertEqual(em["worst_pair_label"], "P1+P2")
        self.assertAlmostEqual(em["worst_pair_asymmetry"], 0.0)  # 3v3, balanced

    def test_measure_episode_is_json_safe(self):
        # records.json (site/derive.py) json.dumps the whole measure_episode dict, minus net_capture.
        payload = {k: v for k, v in coal_episode().items() if k != "net_capture"}
        json.dumps(payload)  # must not raise

    def test_reciprocity_carries_loyalty_numbers_and_keeps_old_keys(self):
        r = report.reciprocity([coal_episode()])
        for key in ("payback_rate", "solicited_not_payback_rate", "net_capture"):  # stability
            self.assertIn(key, r)
        for key in ("payback_windowed_rate", "payback_floor", "payback_lift", "payback_lift_ci",
                    "retention", "retention_floor"):
            self.assertIn(key, r)

    def test_cooperation_by_model_pools_seat_rows(self):
        rows = report.cooperation_by_model([coal_episode()])
        by = {r["model"]: r for r in rows}
        self.assertIn("opus-4.8", by)
        self.assertIn("gpt-5.5", by)
        for key in ("retaliation", "forgiveness", "coop_slope", "retention"):
            self.assertIn(key, by["opus-4.8"])
        self.assertIsNotNone(by["opus-4.8"]["cooperation_rate"])

    def test_coalition_summary_pools_and_none_when_empty(self):
        self.assertIsNone(report.coalition_summary([]))
        c = report.coalition_summary([coal_episode(), coal_episode()])
        self.assertEqual(c["durable"]["confirmed"], 1.0)
        self.assertEqual(c["durable"]["mutual"], 1.0)
        self.assertEqual(c["betrayals"], 0)
        self.assertEqual({r["k"] for r in c["sweep"]}, {1, 2, 3, 4, 5})
        self.assertEqual(c["mutual_support_run_max"], 5)
        self.assertEqual(c["worst_pair"]["label"], "P1+P2")

    def test_format_report_prints_the_new_blocks(self):
        out = report.format_report([coal_episode()])
        self.assertIn("loyalty", out.lower())
        self.assertIn("chance floor", out.lower())
        self.assertIn("windowed", out.lower())
        self.assertIn("cooperation per model", out.lower())
        self.assertIn("coalition", out.lower())
        self.assertIn("betrayal", out.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
