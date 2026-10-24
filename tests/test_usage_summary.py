"""scripts/usage_summary.py — the offline token/host-mix aggregator. Pure functions + the per-sample
accumulator, exercised on fake samples so nothing here reads a real .eval log or hits the network.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_usage_summary.py
"""
import sys
import gzip
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "site"))

import usage_summary as us  # noqa: E402
import derive  # noqa: E402
from delegated_influence import episodes, models  # noqa: E402


class FakeCall:
    def __init__(self, response):
        self.response = response


class FakeEvent:
    def __init__(self, event, model=None, response=None):
        self.event = event
        self.model = model
        self.call = FakeCall(response) if response is not None else None


class FakeSample:
    def __init__(self, store, events=None, id="s"):
        self.store = store
        self.events = events or []
        self.id = id


def ev(agent, usage):
    return {"agent": agent, "usage": usage}


class TestEventTokens(unittest.TestCase):
    def test_sums_retries(self):
        e = ev("P1", [{"prompt_tokens": 10, "completion_tokens": 5},
                      {"prompt_tokens": 12, "completion_tokens": 3}])
        for usage in e["usage"]:
            usage.update(cache_read_tokens=0, cache_write_tokens=0)
        self.assertEqual(us.event_tokens(e), (22, 8))

    def test_no_usage_is_not_a_turn(self):
        # forced control pulls (and any call-less event) carry an empty usage list -> None.
        self.assertIsNone(us.event_tokens({"agent": "P1", "usage": []}))
        self.assertIsNone(us.event_tokens({"agent": "P1"}))

    def test_error_call_without_usage_keeps_tokens_unknown(self):
        self.assertEqual(us.event_tokens(ev("P1", [{"error": "timeout"}])), (None, None))

    def test_cache_input_counts_once_and_reasoning_is_an_output_subset(self):
        usage = {"prompt_tokens": 100, "cache_read_tokens": 200, "cache_write_tokens": 50,
                 "completion_tokens": 40, "reasoning_tokens": 30}
        self.assertEqual(us.event_tokens(ev("P1", [usage])), (350, 40))

    def test_reader_and_cli_use_the_same_output_alias_precedence(self):
        for output_fields, expected in [
            ({"completion_tokens": 40, "output_tokens": 99}, 40),
            ({"completion_tokens": 0, "output_tokens": 99}, 0),
            ({"completion_tokens": None, "output_tokens": 99}, 99),
            ({"output_tokens": 99}, 99),
            ({"completion_tokens": -1, "output_tokens": 99}, None),
            ({}, None),
        ]:
            with self.subTest(output_fields=output_fields):
                usage = {"prompt_tokens": 100, "cache_read_tokens": 200,
                         "cache_write_tokens": 50, **output_fields}
                event = ev("P1", [usage])
                reader = derive.agent_totals([{
                    "actor": "P1", "type": "pull", "forced": False, "score_changes": {},
                    "calls": [{"usage": usage}],
                }], ["P1"])["P1"]
                by_model, by_arm, host_mix = us.new_aggregates()
                us.accumulate_record("arm", {"id": "episode", "seat_map": {"P1": "model"},
                                             "events": [event]}, by_model, by_arm, host_mix)
                self.assertEqual(us.event_tokens(event), (350, expected))
                self.assertEqual(reader["output_tokens"], expected or 0)
                self.assertEqual(by_arm["arm"]["completion"], reader["output_tokens"])
                self.assertEqual(reader["unknown_output_calls"], int(expected is None))
                self.assertEqual(by_arm["arm"]["unknown_completion_calls"], reader["unknown_output_calls"])
                self.assertEqual(usage, {"prompt_tokens": 100, "cache_read_tokens": 200,
                                         "cache_write_tokens": 50, **output_fields})

    def test_missing_cache_bucket_cannot_be_assumed_zero(self):
        usage = {"prompt_tokens": 100, "cache_read_tokens": 200, "completion_tokens": 40}
        self.assertEqual(us.event_tokens(ev("P1", [usage])), (None, 40))
        usage["cache_write_tokens"] = None
        self.assertEqual(us.event_tokens(ev("P1", [usage])), (None, 40))


class TestModelKey(unittest.TestCase):
    def test_known_spec_maps_to_roster_key(self):
        self.assertEqual(us.model_key(models.openrouter_spec("glm-5.2")), "glm-5.2")

    def test_duplicate_spec_prefers_current_reasoning_instrument_over_legacy_alias(self):
        spec = models.openrouter_spec("deepseek-v4-pro-reasoning")
        self.assertEqual(us.model_key(spec), "deepseek-v4-pro-reasoning")

    def test_unknown_spec_strips_openrouter_prefix(self):
        self.assertEqual(us.model_key("openrouter/foo/bar"), "foo/bar")

    def test_empty_spec(self):
        self.assertEqual(us.model_key(""), "")


def _sample(events=None, id="s"):
    store = {"Episode:seat_map": {"P1": "opus-4.8", "P2": "glm-5.2"},
             "Episode:events": [
                 ev("P1", [{"prompt_tokens": 100, "completion_tokens": 20}]),
                 ev("P2", [{"prompt_tokens": 200, "completion_tokens": 50}]),
                 {"agent": "P1", "usage": []},  # a forced pull: not a turn
             ]}
    for event in store["Episode:events"]:
        for usage in event["usage"]:
            usage.update(cache_read_tokens=0, cache_write_tokens=0)
    return FakeSample(store, events=events, id=id)


class TestAccumulateSample(unittest.TestCase):
    def test_token_aggregation_by_model_and_arm(self):
        by_model, by_arm, host_mix = us.new_aggregates()
        self.assertTrue(us.accumulate_sample("03_x", _sample(), by_model, by_arm, host_mix))
        self.assertEqual(by_model[(2, "opus-4.8")]["turns"], 1)
        self.assertEqual(by_model[(2, "opus-4.8")]["prompt"], 100)
        self.assertEqual(by_model[(2, "glm-5.2")]["completion"], 50)
        # the forced pull is excluded: 2 real turns, seats=2, one episode of 370 tokens.
        self.assertEqual(by_arm["03_x"]["turns"], 2)
        self.assertEqual(by_arm["03_x"]["episodes"], 1)
        self.assertEqual(by_arm["03_x"]["seats"], {2})
        self.assertEqual(by_arm["03_x"]["ep_tokens"], [370])

    def test_skips_sample_without_stored_episode(self):
        by_model, by_arm, host_mix = us.new_aggregates()
        self.assertFalse(us.accumulate_sample("x", FakeSample({}), by_model, by_arm, host_mix))
        self.assertEqual(len(by_arm), 0)

    def test_host_mix_from_model_events(self):
        by_model, by_arm, host_mix = us.new_aggregates()
        spec = models.openrouter_spec("glm-5.2")
        events = [
            FakeEvent("model", model=spec, response={"provider": "Fireworks"}),
            FakeEvent("model", model=spec, response={"provider": "Fireworks"}),
            FakeEvent("model", model=spec, response={"provider": "Novita"}),
            FakeEvent("sample_init"),                       # not a model event -> ignored
            FakeEvent("model", model=spec, response={}),    # no host field -> ignored
            FakeEvent("model", model=spec, response=None),  # no call -> ignored
        ]
        us.accumulate_sample("03_x", _sample(events=events), by_model, by_arm, host_mix)
        self.assertEqual(host_mix[("glm-5.2", "03_x")], Counter({"Fireworks": 2, "Novita": 1}))

    def test_actual_billing_uses_raw_provider_cost_not_inspect_catalog_cost(self):
        sample = FakeSample({"Episode:seat_map": {"P1": "glm-5.2"},
                             "Episode:events": [ev("P1", [{"prompt_tokens": 100,
                                "completion_tokens": 20, "total_cost": 999}])]},
                            events=[FakeEvent("model", model=models.openrouter_spec("glm-5.2"),
                                    response={"usage": {"cost": .33}})])
        by_model, by_arm, host_mix = us.new_aggregates()
        us.accumulate_sample("legacy", sample, by_model, by_arm, host_mix)
        self.assertEqual(by_arm["legacy"]["reported_cost_usd"], .33)
        self.assertEqual(by_arm["legacy"]["unknown_cost_calls"], 0)

    def test_missing_raw_billing_remains_unknown(self):
        by_model, by_arm, host_mix = us.new_aggregates()
        us.accumulate_sample("legacy", _sample(), by_model, by_arm, host_mix)
        self.assertEqual(by_arm["legacy"]["unknown_cost_calls"], 2)
        self.assertEqual(by_arm["legacy"]["reported_cost_usd"], 0)

    def test_partial_raw_billing_does_not_hide_missing_call_cost(self):
        sample = _sample(events=[FakeEvent("model", model=models.openrouter_spec("opus-4.8"),
                                           response={"usage": {"cost": .12}})])
        by_model, by_arm, host_mix = us.new_aggregates()
        us.accumulate_sample("legacy", sample, by_model, by_arm, host_mix)
        self.assertEqual(by_arm["legacy"]["cost_calls"], 2)
        self.assertEqual(by_arm["legacy"]["unknown_cost_calls"], 1)
        self.assertEqual(by_model[(2, "glm-5.2")]["unknown_cost_calls"], 1)

    def test_stop_reasons_count_each_call_once_and_keep_unknowns(self):
        sample = FakeSample({"Episode:seat_map": {"P1": "m"}, "Episode:events": [
            ev("P1", [
                {"prompt_tokens": 1, "completion_tokens": 1, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "stop_reason": "model_length",
                 "finish_reason": "length", "native_finish_reason": "MAX_TOKENS"},
                {"prompt_tokens": 1, "completion_tokens": 1, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "stop_reason": "stop",
                 "finish_reason": "stop", "native_finish_reason": "STOP"},
                {"prompt_tokens": 1, "completion_tokens": 1, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "stop_reason": None,
                 "finish_reason": "length", "native_finish_reason": "MAX_TOKENS"},
            ])]})
        by_model, by_arm, host_mix = us.new_aggregates()

        us.accumulate_sample("reasons", sample, by_model, by_arm, host_mix)

        arm = by_arm["reasons"]
        self.assertEqual(arm["calls"], 3)
        self.assertEqual(arm["stop_reasons"], Counter({"model_length": 1, "stop": 1}))
        self.assertEqual(arm["unknown_stop_reason_calls"], 1)

    def test_stop_reason_table_reports_counts_and_unknowns(self):
        by_model, by_arm, host_mix = us.new_aggregates()
        by_arm["arm"]["calls"] = 4
        by_arm["arm"]["stop_reasons"].update({"stop": 2, "model_length": 1})
        by_arm["arm"]["unknown_stop_reason_calls"] = 1
        output = io.StringIO()

        with redirect_stdout(output):
            us.print_stop_reason_table(by_arm)

        text = output.getvalue()
        self.assertIn("stop 2", text)
        self.assertIn("model_length 1", text)
        self.assertIn("unknown 1", text)
        self.assertIn("4 calls", text)


def native_export(path):
    first = {"prompt_tokens": 100, "cache_read_tokens": 200, "cache_write_tokens": 50,
             "completion_tokens": 40, "reasoning_tokens": 30, "total_cost": .4,
             "cost_source": "openrouter_response_usage_cost", "served_provider": "host-a",
             "observation_characters": 600}
    retry = {"prompt_tokens": 20, "completion_tokens": 10, "total_cost": None}
    last = {"prompt_tokens": 60, "completion_tokens": 30, "total_cost": .2,
            "cost_source": "openrouter_response_usage_cost", "served_provider": "host-b",
            "observation_characters": 120}
    for usage in (retry, last):
        usage.update(cache_read_tokens=0, cache_write_tokens=0)
    metadata = {"experiment_config": "24_boards", "experiment_arm": "talk",
                "seat_map": {"P1": "model-a", "P2": "model-b"}}
    data = {"schema_version": "native-episode-v1", "status": "complete", "episode_id": "native-one",
            "sample_id": "one", "metadata": metadata, "record": {**metadata, "scores": {},
                "events": [{"agent": "P1", "round": 1, "usage": [first, retry],
                            "forfeit": True, "parse_retries": 1},
                           {"agent": "P2", "round": 2, "usage": [last], "forfeit": False}]}}
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)


class TestNativeSummary(unittest.TestCase):
    def test_native_exports_use_canonical_loader_and_keep_actual_unknown_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.episode.json.gz"
            native_export(path)
            with mock.patch.object(us.episodes, "load_episodes", wraps=episodes.load_episodes) as load:
                by_model, by_arm, host_mix = us.collect_paths([directory, directory])
            self.assertEqual(load.call_count, 1)
            arm = by_arm["24_boards/talk"]
            self.assertEqual(arm["episodes"], 1)
            self.assertEqual(arm["calls"], 3)
            self.assertEqual((arm["prompt"], arm["completion"]), (430, 80))
            self.assertEqual(arm["cache_read_tokens"], 200)
            self.assertEqual(arm["cache_write_tokens"], 50)
            self.assertEqual(arm["reasoning_tokens"], 30)
            self.assertEqual((arm["retries"], arm["forfeits"]), (1, 1))
            self.assertAlmostEqual(arm["reported_cost_usd"], .6)
            self.assertEqual(arm["unknown_cost_calls"], 1)
            self.assertEqual(arm["rounds"][1]["input_tokens"], 370)
            self.assertEqual(arm["rounds"][1]["input_known_calls"], 2)
            self.assertEqual(arm["rounds"][1]["observation_known_calls"], 1)
            self.assertEqual(host_mix[("model-a", "24_boards/talk")], Counter({"host-a": 1}))
            self.assertEqual(by_model[(2, "model-a")]["prompt"], 370)

    def test_missing_history_fields_show_unknown_instead_of_zero(self):
        sample = FakeSample({"Episode:seat_map": {"P1": "m"}, "Episode:events": [
            {"agent": "P1", "round": 4, "forfeit": True, "usage": [{"error": "lost response"}]}]})
        by_model, by_arm, host_mix = us.new_aggregates()
        us.accumulate_sample("old", sample, by_model, by_arm, host_mix)
        self.assertEqual(by_arm["old"]["rounds"][4]["input_known_calls"], 0)
        self.assertEqual(by_arm["old"]["rounds"][4]["observation_known_calls"], 0)
        output = io.StringIO()
        with redirect_stdout(output):
            us.print_context_table(by_arm)
        self.assertIn("unknown", output.getvalue())

    def test_partial_cache_coverage_keeps_subtotals_but_prints_total_and_estimate_unknown(self):
        complete = {"prompt_tokens": 100, "cache_read_tokens": 200, "cache_write_tokens": 50,
                    "completion_tokens": 40, "reasoning_tokens": 30}
        partial = {"prompt_tokens": 10, "cache_read_tokens": 20, "completion_tokens": 2}
        sample = FakeSample({"Episode:seat_map": {"P1": "m"}, "Episode:events": [
            {"agent": "P1", "round": 1, "usage": [complete, partial]}]})
        by_model, by_arm, host_mix = us.new_aggregates()
        us.accumulate_sample("cached", sample, by_model, by_arm, host_mix)
        row = by_arm["cached"]
        self.assertEqual(row["prompt"], 350)
        self.assertEqual(row["unknown_input_calls"], 1)
        self.assertEqual(row["uncached_input_tokens"], 110)
        self.assertEqual(row["cache_read_tokens"], 220)
        self.assertEqual(row["unknown_cache_write_calls"], 1)
        self.assertEqual(row["ep_tokens"], [None])
        self.assertEqual(row["rounds"][1]["input_tokens"], 350)
        self.assertEqual(row["rounds"][1]["input_known_calls"], 1)
        output = io.StringIO()
        with redirect_stdout(output):
            us.print_pricing_table(by_model)
            us.print_arm_table(by_arm, 2)
            us.print_cost_table(by_arm)
        text = output.getvalue()
        self.assertIn("unknown", text)
        self.assertIn("uncached", text)
        self.assertTrue(next(line for line in text.splitlines() if line.startswith("cached")).rstrip().endswith("unknown"))
        self.assertNotIn("cache_write_tokens", partial)  # raw normalized source remains untouched

    def test_cli_accepts_native_exports_and_points_to_incomplete_run_status(self):
        with tempfile.TemporaryDirectory() as directory:
            native_export(Path(directory) / "one.episode.json.gz")
            output = io.StringIO()
            with (mock.patch("sys.argv", ["usage_summary", directory]), redirect_stdout(output)):
                us.main()
            text = output.getvalue()
            self.assertIn("reported subtotal", text)
            self.assertIn("unknown bills", text)
            self.assertIn("mean input/call", text)
            self.assertIn("--status", text)
            self.assertIn("24_boards/talk", text)


class TestExpandPaths(unittest.TestCase):
    def test_cli_expands_inputs_once_and_collects_both_formats_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.eval").write_text("")
            (root / "one.episode.json").write_text("{}")
            paths = [directory, str(root / "*"), directory]
            record = {"id": "native", "seat_map": {"P1": "m"}, "events": [],
                      "metadata": {"experiment_config": "pilot", "experiment_arm": "arm"}}
            with (mock.patch.object(us.episodes, "expand_eval_paths", wraps=episodes.expand_eval_paths) as evals,
                  mock.patch.object(us.episodes, "expand_native_paths", wraps=episodes.expand_native_paths) as native,
                  mock.patch.object(us.episodes, "load_episodes", return_value=[record]) as load,
                  mock.patch.object(us, "read_eval_log", return_value=SimpleNamespace(samples=[_sample()])) as read,
                  mock.patch("sys.argv", ["usage_summary", *paths]), redirect_stdout(io.StringIO()) as output):
                us.main()
            self.assertEqual(evals.call_count, 1)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(load.call_count, 1)
            self.assertIn("1 .eval file(s), 1 native export(s), 2 episodes", output.getvalue())

    def test_dir_recurses_and_dedups(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.eval").write_text("")
            sub = Path(d) / "sub"
            sub.mkdir()
            (sub / "c.eval").write_text("")
            got = episodes.expand_eval_paths([d, d])  # same dir twice must not double-count
            self.assertEqual(len(got), 2)
            self.assertTrue(all(f.suffix == ".eval" for f, _root in got))


if __name__ == "__main__":
    unittest.main(verbosity=2)
