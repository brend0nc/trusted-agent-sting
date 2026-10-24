"""run.py: YAML config -> Inspect tasks + a rough cost estimate. No network (eval_set is live-only).

Run:  PYTHONPATH=src .venv/bin/python tests/test_run.py
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import models, run  # noqa: E402


class TestBuildTasks(unittest.TestCase):
    def test_focal_arms_shard_one_task_per_model(self):
        # Sharding keeps every .eval log file small enough to commit to git (GitHub caps files at
        # 100MB and eval_set writes ONE file per task), and makes resume per-model.
        cfg = {"arms": [{"name": "a", "task": "focal_attack",
                         "models": ["opus-4.8", "gpt-5.4"], "reps": 1, "topologies": ["complete"]}]}
        tasks = run.build_tasks(cfg)
        self.assertEqual([n for n, _ in tasks], ["a-opus-4.8", "a-gpt-5.4"])
        self.assertEqual([t.name for _, t in tasks], ["a-opus-4.8", "a-gpt-5.4"])  # unique for eval_set
        for _, t in tasks:
            self.assertEqual(len(t.dataset), 1)  # 1 model x 1 topology x 1 rep each

    def test_models_all_expands_to_roster(self):
        cfg = {"arms": [{"name": "a", "task": "focal_attack", "models": "all",
                         "reps": 1, "topologies": ["complete"]}]}
        tasks = run.build_tasks(cfg)
        self.assertEqual(len(tasks), len(models.ROSTER))  # one shard per model
        self.assertEqual(sum(len(t.dataset) for _, t in tasks), len(models.ROSTER))

    def test_resist_maps_models_to_targets(self):
        cfg = {"arms": [{"name": "r", "task": "focal_resist", "models": ["sonnet-4.6"],
                         "reps": 1, "inoculations": [False, True], "topologies": ["complete"]}]}
        name, t = run.build_tasks(cfg)[0]
        self.assertEqual(name, "r-sonnet-4.6")
        self.assertEqual(len(t.dataset), 2)  # 1 target x 2 inoculation conditions

    def test_lineup_builds_custom_seat_map(self):
        cfg = {"arms": [{"name": "five", "task": "lineup",
                         "seats": ["opus-4.8", "gpt-5.5", "gemini-pro", "deepseek-v4-pro", "grok-4.3"],
                         "reps": 3}]}
        _, t = run.build_tasks(cfg)[0]
        self.assertEqual(len(t.dataset), 3)  # reps
        sm = t.dataset[0].metadata["seat_map"]
        self.assertEqual(len(sm), 5)
        self.assertEqual(set(sm.values()),
                         {"opus-4.8", "gpt-5.5", "gemini-pro", "deepseek-v4-pro", "grok-4.3"})

    def test_lineup_fill_appends_background_crowd(self):
        cfg = {"arms": [{"name": "recruit", "task": "lineup",
                         "seats": ["opus-4.8", "gpt-5.5"],
                         "fill": {"model": "glm-5.2", "count": 3}, "reps": 1}]}
        _, t = run.build_tasks(cfg)[0]
        md = t.dataset[0].metadata
        sm = md["seat_map"]
        self.assertEqual(len(sm), 5)  # 2 principals + 3 fill
        principals = md["principal_seats"]
        self.assertEqual(len(principals), 2)
        self.assertEqual({sm[p] for p in principals}, {"opus-4.8", "gpt-5.5"})
        crowd = [sm[s] for s in sm if s not in principals]
        self.assertEqual(crowd, ["glm-5.2"] * 3)

    def test_lineup_rejects_unknown_seat_model(self):
        cfg = {"arms": [{"name": "x", "task": "lineup", "seats": ["nope-9000"]}]}
        with self.assertRaises(ValueError):
            run.build_tasks(cfg)

    def test_forfeit_smoke_runs_from_config_with_new_background(self):
        cfg = {"arms": [{"name": "bg", "task": "forfeit_smoke", "reps": 1}]}
        _, t = run.build_tasks(cfg)[0]
        candidates = {s.metadata["seat_map"]["P1"] for s in t.dataset}
        self.assertIn(models.BACKGROUND, candidates)

    def test_unknown_task_fails_fast(self):
        with self.assertRaises(ValueError):
            run.build_tasks({"arms": [{"name": "x", "task": "nope"}]})

    def test_unknown_model_fails_fast(self):
        with self.assertRaises(ValueError):
            run.build_tasks({"arms": [{"name": "x", "task": "focal_attack", "models": ["nope-9000"]}]})


class TestArenaConfigs(unittest.TestCase):
    """The new headline configs must load from disk and expand to the expected arms/seats."""

    def _load(self, name):
        import yaml
        return yaml.safe_load((ROOT / "configs" / name).read_text())

    def test_arena_25_config_is_four_paired_arms_of_25_seats(self):
        cfg = self._load("20_arena_25.yaml")
        tasks = run.build_tasks(cfg)
        self.assertEqual(len(tasks), 4)  # 2x2: talk on/off x punish on/off
        for _, t in tasks:
            self.assertEqual(len(t.dataset[0].metadata["seat_map"]), 25)
            self.assertEqual(t.dataset[0].metadata["self_pull_reward"], 0.5)
        # paired seeds: one fixed seed list reused across all four arms
        seed_lists = {tuple(s.metadata["seed"] for s in t.dataset) for _, t in tasks}
        self.assertEqual(len(seed_lists), 1)
        # the 2x2 spans both toggles
        toggles = {(t.dataset[0].metadata["messages_off"],
                    t.dataset[0].metadata["punish_enabled"]) for _, t in tasks}
        self.assertEqual(toggles, {(False, False), (False, True), (True, False), (True, True)})

    def test_arena_5x5_config_is_25_seats(self):
        cfg = self._load("21_arena_5x5.yaml")
        (_, t), = run.build_tasks(cfg)
        self.assertEqual(len(t.dataset[0].metadata["seat_map"]), 25)
        self.assertEqual(t.dataset[0].metadata["self_pull_reward"], 0.5)

    def test_cooperation_cost_curve_pairs_five_reward_arms(self):
        cfg = self._load("22_cooperation_cost_curve.yaml")
        tasks = run.build_tasks(cfg)
        self.assertEqual(len(tasks), 5)
        self.assertEqual(
            [t.dataset[0].metadata["self_pull_reward"] for _, t in tasks],
            [0, 0.25, 0.5, 0.75, 1.0],
        )
        self.assertTrue(all(len(t.dataset) == 30 for _, t in tasks))
        # Lineup shuffle keys ignore the treatment, so each repetition has the same seating.
        for rep in range(30):
            seatings = {tuple(t.dataset[rep].metadata["seat_map"].items()) for _, t in tasks}
            self.assertEqual(len(seatings), 1)

    def test_deception_intent_validation_matches_cost_curve_half_reward_arm(self):
        cost_cfg = self._load("22_cooperation_cost_curve.yaml")
        intent_cfg = self._load("23_deception_intent_validation.yaml")
        cost_tasks = run.build_tasks(cost_cfg)
        half = next(t for _, t in cost_tasks
                    if t.dataset[0].metadata["self_pull_reward"] == 0.5)
        (_, intent), = run.build_tasks(intent_cfg)
        self.assertTrue(intent.dataset[0].metadata["declared_intents"])
        self.assertEqual(intent.dataset[0].metadata["self_pull_reward"], 0.5)
        self.assertEqual(
            [sample.metadata["seat_map"] for sample in intent.dataset],
            [sample.metadata["seat_map"] for sample in half.dataset],
        )

    def test_preliminary_config_is_ten_agents_exploratory(self):
        cfg = self._load("20_preliminary_10.yaml")
        self.assertEqual(cfg["stage"], "exploratory")  # a pilot, never a paper result
        (_, t), = run.build_tasks(cfg)
        md = t.dataset[0].metadata
        self.assertEqual(len(md["seat_map"]), 10)  # 5 principals + 5 glm crowd
        self.assertEqual(len(md["principal_seats"]), 5)

    def test_background_bakeoff_has_three_paired_ten_agent_arms(self):
        cfg = self._load("02b_background_bakeoff.yaml")
        self.assertEqual(cfg["stage"], "exploratory")
        tasks = run.build_tasks(cfg)
        self.assertEqual([name for name, _ in tasks],
                         ["glm_baseline", "luna_none", "luna_low"])
        for _name, t in tasks:
            self.assertEqual(len(t.dataset), 2)
            self.assertEqual(len(t.dataset[0].metadata["seat_map"]), 10)
        none_md = tasks[1][1].dataset[0].metadata
        self.assertEqual(none_md["model_configs"]["gpt-5.6-luna"],
                         {"reasoning_effort": "none", "max_tokens": 256})
        low_md = tasks[2][1].dataset[0].metadata
        self.assertEqual(low_md["model_configs"]["gpt-5.6-luna"],
                         {"reasoning_effort": "low", "max_tokens": 512})

    def test_luna_background_validation_matches_the_focal_design(self):
        cfg = self._load("02c_luna_background_validation.yaml")
        self.assertEqual(cfg["stage"], "exploratory")
        (name, t), = run.build_tasks(cfg)
        self.assertEqual(name, "luna_background_validation-gpt-5.5")
        self.assertEqual(len(t.dataset), 1)
        md = t.dataset[0].metadata
        self.assertEqual(md["rounds"], 5)
        self.assertEqual(md["actions_per_round"], 3)
        seats = md["seat_map"]
        self.assertEqual(len(seats), 5)
        self.assertEqual(sum(model == models.BACKGROUND for model in seats.values()), 4)

    def test_completed_glm_focal_configs_freeze_their_historical_background(self):
        for config_name in ("01_smoke.yaml", "04_attack_complete.yaml", "05_attack_nomsg.yaml",
                            "06_attack_ring.yaml", "07_attack_mixed.yaml", "08_resist.yaml",
                            "13b_attack_intents.yaml"):
            with self.subTest(config=config_name):
                cfg = self._load(config_name)
                self.assertTrue(all(
                    arm.get("background") == "glm-5.2" for arm in cfg["arms"]
                    if arm["task"] in ("focal_attack", "focal_resist")
                ))


class TestEstimate(unittest.TestCase):
    def test_cost_scales_with_episodes_and_turns(self):
        cfg = {"arms": [{"name": "a", "task": "focal_attack",
                         "models": ["opus-4.8", "gpt-5.4"], "reps": 1, "topologies": ["complete"]}]}
        est = run.estimate_cost(cfg, run.build_tasks(cfg))
        self.assertGreater(est["total_usd"], 0)
        self.assertEqual(sum(a["episodes"] for a in est["arms"]), 2)
        self.assertEqual(est["arms"][0]["turns_per_episode"], 5 * 5 * 5)  # rounds x actions x seats

    def test_cost_block_overrides_defaults(self):
        cfg = {"cost": {"usd_per_mtok": 20.0}, "arms": [{"name": "a", "task": "focal_attack",
               "models": ["opus-4.8"], "reps": 1, "topologies": ["complete"]}]}
        cheap = run.estimate_cost({"arms": cfg["arms"]}, run.build_tasks(cfg))
        pricey = run.estimate_cost(cfg, run.build_tasks(cfg))
        self.assertGreater(pricey["total_usd"], cheap["total_usd"])

    def test_estimate_reports_scheduled_calls(self):
        cfg = {"arms": [{"name": "a", "task": "focal_attack",
                         "models": ["opus-4.8"], "reps": 1, "topologies": ["complete"]}]}
        est = run.estimate_cost(cfg, run.build_tasks(cfg))
        self.assertEqual(est["base_calls"], 5 * 5 * 5)
        self.assertEqual(est["scheduled_calls"], 5 * 5 * 5 * 2)  # includes correction retry
        self.assertIn("not a runtime cap", est["scope"])

    def test_declared_intents_add_one_call_per_seat_per_round(self):
        base = {"arms": [{"name": "a", "task": "focal_attack",
                           "models": ["opus-4.8"], "reps": 1,
                           "topologies": ["complete"]}]}
        controlled = {"arms": [{**base["arms"][0], "declared_intents": True}]}
        ordinary = run.estimate_cost(base, run.build_tasks(base))
        intents = run.estimate_cost(controlled, run.build_tasks(controlled))
        self.assertEqual(ordinary["scheduled_calls"], 250)
        self.assertEqual(intents["scheduled_calls"], 275)  # +5 rounds x 5 one-shot intent calls
        self.assertAlmostEqual(intents["total_usd"] / ordinary["total_usd"], 1.1)


class TestBudgetGuard(unittest.TestCase):
    def _estimate(self):
        cfg = {"arms": [{"name": "a", "task": "focal_attack",
                         "models": ["opus-4.8"], "reps": 1, "topologies": ["complete"]}]}
        return run.estimate_cost(cfg, run.build_tasks(cfg))

    def test_live_run_requires_a_configured_ceiling(self):
        with self.assertRaisesRegex(ValueError, "requires budget.max_estimated_usd"):
            run.enforce_budget({}, self._estimate())

    def test_configured_ceiling_rejects_an_oversized_plan(self):
        with self.assertRaisesRegex(ValueError, "estimated cost"):
            run.enforce_budget({"budget": {"max_estimated_usd": 0.01}}, self._estimate())

    def test_configured_ceiling_allows_a_deliberately_large_plan(self):
        estimate = self._estimate()
        cfg = {"budget": {"max_estimated_usd": estimate["total_usd"] + 1}}
        self.assertIsNone(run.enforce_budget(cfg, estimate))

    def test_bad_budget_config_fails_before_a_run(self):
        for value in (0, -1, True, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.enforce_budget({"budget": {"max_estimated_usd": value}}, self._estimate())

    def test_every_runnable_config_has_room_for_its_dry_run_estimate(self):
        import yaml

        for path in sorted((ROOT / "configs").glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = yaml.safe_load(path.read_text())
                estimate = run.estimate_cost(cfg, run.build_tasks(cfg))
                ceiling = cfg.get("budget", {}).get("max_estimated_usd")
                self.assertIsInstance(ceiling, (int, float))
                self.assertGreaterEqual(ceiling, estimate["total_usd"])


class TestLiveRunArtifacts(unittest.TestCase):
    def test_manifest_precedes_eval_and_completed_logs_are_exported(self):
        cfg = {"log_dir": "logs/x", "retry_attempts": 2, "max_tasks": 1,
               "budget": {"max_estimated_usd": 5}}
        task = SimpleNamespace(dataset=[1], name="arena")
        estimate = {"total_usd": 1, "scheduled_calls": 1, "arms": []}
        frozen = SimpleNamespace(eval_metadata={"run_manifest_id": "launch-1"})
        order = []

        with (
            mock.patch.object(
                run.run_artifacts, "create_run_artifacts",
                side_effect=lambda *a, **kw: (order.append("manifest") or frozen),
            ) as create,
            mock.patch.object(
                run, "eval_set",
                side_effect=lambda *a, **kw: (order.append("eval") or (True, ["log"])),
            ) as evaluate,
            mock.patch.object(
                run.run_artifacts, "finalize_run",
                side_effect=lambda *a, **kw: order.append("export"),
            ) as finalize,
        ):
            success, logs = run.launch_live_run("configs/x.yaml", cfg, [("arena", task)], estimate)

        self.assertTrue(success)
        self.assertEqual(logs, ["log"])
        self.assertEqual(order, ["manifest", "eval", "export"])
        create.assert_called_once()
        self.assertEqual(evaluate.call_args.kwargs["metadata"], frozen.eval_metadata)
        self.assertTrue(evaluate.call_args.kwargs["log_model_api"])
        finalize.assert_called_once_with(frozen, ["log"], True)

    def test_interrupted_eval_gets_a_failure_marker(self):
        cfg = {"budget": {"max_estimated_usd": 5}}
        task = SimpleNamespace(dataset=[1], name="arena")
        estimate = {"total_usd": 1, "scheduled_calls": 1, "arms": []}
        frozen = SimpleNamespace(eval_metadata={"run_manifest_id": "launch-1"})
        with mock.patch.object(run.run_artifacts, "create_run_artifacts", return_value=frozen), \
             mock.patch.object(run, "eval_set", side_effect=RuntimeError("provider failed")), \
             mock.patch.object(run.run_artifacts, "record_interrupted_run") as interrupted:
            with self.assertRaises(RuntimeError):
                run.launch_live_run("configs/x.yaml", cfg, [("arena", task)], estimate)
        interrupted.assert_called_once_with(frozen, "RuntimeError")

    def test_export_failure_is_retryable_and_does_not_mark_paid_eval_interrupted(self):
        cfg = {"budget": {"max_estimated_usd": 5}}
        task = SimpleNamespace(dataset=[1], name="arena")
        estimate = {"total_usd": 1, "scheduled_calls": 1, "arms": []}
        frozen = SimpleNamespace(eval_metadata={"run_manifest_id": "launch-1"})
        with mock.patch.object(run.run_artifacts, "create_run_artifacts", return_value=frozen), \
             mock.patch.object(run, "eval_set", return_value=(True, ["paid-log"])), \
             mock.patch.object(run.run_artifacts, "finalize_run",
                               side_effect=RuntimeError("disk full")), \
             mock.patch.object(run.run_artifacts, "record_interrupted_run") as interrupted, \
             mock.patch.object(run.run_artifacts, "record_finalization_error") as recoverable:
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                run.launch_live_run("configs/x.yaml", cfg, [("arena", task)], estimate)
        interrupted.assert_not_called()
        recoverable.assert_called_once_with(frozen, "RuntimeError", eval_success=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
