"""Research artifact regression tests: frozen launch inputs and joined turn exports."""
import gzip
import json
import copy
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import artifacts  # noqa: E402


class Dumpable(SimpleNamespace):
    def model_dump(self, mode=None):
        return {key: artifacts._jsonable(value) for key, value in vars(self).items()}


def fake_sample():
    span = SimpleNamespace(
        event="span_begin", type="turn", id="span-1", name="P1 · opus-4.8"
    )
    model = SimpleNamespace(
        event="model",
        span_id="span-1",
        timestamp=datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc),
        completed=datetime(2026, 7, 16, 10, 0, 2, tzinfo=timezone.utc),
        working_time=2.0,
        model="openrouter/anthropic/claude-opus-4.8",
        config=Dumpable(max_tokens=256, reasoning_effort="none", seed=41),
        input=[Dumpable(role="system", content="attachment://system"),
               Dumpable(role="user", content="attachment://observation")],
        output=SimpleNamespace(
            model="anthropic/claude-opus-4.8-20260701",
            completion='{"action":"pull","beneficiary":"P2"}',
            error=None,
            metadata=None,
            usage=Dumpable(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                reasoning_tokens=7,
                input_tokens_cache_read=3,
                input_tokens_cache_write=0,
                total_cost=None,
            ),
        ),
        call=SimpleNamespace(
            time=2.0,
            response={
                "id": "generation-1",
                "model": "anthropic/claude-opus-4.8-20260701",
                "provider": "Anthropic",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.0123,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                    "cost_details": {"upstream_inference_cost": 0.0123},
                },
            },
        ),
        error=None,
        retries=1,
    )
    event = {
        "round": 1, "action_number": 1, "turn_index": 0, "agent": "P1",
        "forfeit": False, "forfeit_reason": None, "parse_retries": 0,
        "raw_responses": ['{"action":"pull","beneficiary":"P2"}'],
        "usage": [{"prompt_tokens": 100, "completion_tokens": 20,
                   "served_model": "anthropic/claude-opus-4.8-20260701"}],
        "type": "pull", "beneficiary": "P2", "credit": None, "to": None,
        "text": None, "target": None, "forced": False, "planned": None, "reason": "",
    }
    return SimpleNamespace(
        id="episode-1", uuid="uuid-1",
        metadata={"self_pull_reward": 0, "seed": 41},
        attachments={"system": "exact system prompt", "observation": "exact visible state"},
        events=[span, model],
        store={
            "Episode:seat_map": {"P1": "opus-4.8", "P2": "gpt-5.6-luna"},
            "Episode:scores": {"P1": 0, "P2": 1},
            "Episode:events": [event],
        },
        started_at=datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 16, 10, 0, 2, tzinfo=timezone.utc),
        total_time=2.0, working_time=2.0,
    )


class TestRunManifest(unittest.TestCase):
    def test_freezes_config_and_only_whitelisted_runtime_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/delegated_influence").mkdir(parents=True)
            (root / "src/delegated_influence/game.py").write_text("VALUE = 1\n")
            (root / "pyproject.toml").write_text("[project]\nname='x'\n")
            (root / "uv.lock").write_text("version = 1\n")
            (root / ".env").write_text("OPENROUTER_API_KEY=must-not-leak\n")
            config = root / "configs/arena.yaml"
            config.parent.mkdir()
            config.write_text("stage: confirmatory\narms: []\n")
            task = SimpleNamespace(name="arena", dataset=[1, 2])

            frozen = artifacts.create_run_artifacts(
                config, {"stage": "confirmatory", "arms": []},
                {"total_usd": 12.0}, [("arena", task)], root / "logs",
                retry_attempts=3, max_tasks=4, source_root=root,
                now=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
            )

            manifest = json.loads(frozen.manifest_path.read_text())
            self.assertEqual(manifest["input_config"]["resolved"]["stage"], "confirmatory")
            self.assertEqual(manifest["launch"]["tasks"][0]["episodes"], 2)
            self.assertTrue(manifest["launch"]["log_model_api"])
            self.assertFalse(manifest["privacy"]["dotenv_copied"])
            self.assertEqual(frozen.config_path.read_text(), config.read_text())
            with tarfile.open(frozen.source_snapshot_path, "r:gz") as archive:
                names = set(archive.getnames())
                self.assertIn("src/delegated_influence/game.py", names)
                self.assertNotIn(".env", names)
            combined = frozen.manifest_path.read_text() + frozen.config_path.read_text()
            self.assertNotIn("must-not-leak", combined)
            self.assertEqual(frozen.manifest_path.stat().st_mode & 0o777, 0o444)


class TestAnalysisExport(unittest.TestCase):
    def test_joins_exact_prompt_cost_reasoning_latency_and_score_state(self):
        episode, records = artifacts.sample_export_records(fake_sample())
        self.assertEqual(episode["event_count"], 1)
        record = records[0]
        self.assertEqual(record["join"]["status"], "matched")
        self.assertEqual(record["state_before"]["scores"], {"P1": 0, "P2": 0})
        self.assertEqual(record["state_after"]["scores"], {"P1": 0, "P2": 1})
        call = record["model_calls"][0]
        self.assertEqual(call["input"][1]["content"], "exact visible state")
        self.assertEqual(call["usage"]["reasoning_tokens"], 7)
        self.assertEqual(call["usage"]["openrouter_cost"], 0.0123)
        self.assertEqual(call["latency_seconds"], 2.0)
        self.assertEqual(call["provider"], "Anthropic")

    def test_raw_usage_fills_explicit_none_reasoning_and_cache_fields(self):
        sample = fake_sample()
        usage = sample.events[1].output.usage
        usage.reasoning_tokens = None
        usage.input_tokens_cache_read = None
        usage.input_tokens_cache_write = None

        _episode, records = artifacts.sample_export_records(sample)
        exported = records[0]["model_calls"][0]["usage"]

        self.assertEqual(exported["input_tokens"], 100)
        self.assertEqual(exported["output_tokens"], 20)
        self.assertEqual(exported["total_tokens"], 120)
        self.assertEqual(exported["reasoning_tokens"], 7)
        self.assertEqual(exported["cache_read_tokens"], 0)
        self.assertEqual(exported["cache_write_tokens"], 0)

    def test_missing_timeout_call_does_not_steal_the_next_turns_model_event(self):
        sample = fake_sample()
        successful = copy.deepcopy(sample.store["Episode:events"][0])
        successful["turn_index"] = 1
        timeout = copy.deepcopy(successful)
        timeout.update({
            "turn_index": 0, "forfeit": True, "forfeit_reason": "api_error", "type": None,
            "beneficiary": None, "usage": [{"prompt_tokens": 0, "completion_tokens": 0,
                                                "error": "TimeoutError"}],
        })
        sample.store["Episode:events"] = [timeout, successful]

        _episode, records = artifacts.sample_export_records(sample)

        self.assertEqual(records[0]["join"]["status"], "missing_model_calls")
        self.assertEqual(records[0]["model_calls"], [])
        self.assertEqual(records[1]["join"]["status"], "matched")
        self.assertEqual(records[1]["model_calls"][0]["response_id"], "generation-1")

    def test_extra_mismatched_call_resynchronizes_at_the_next_matching_call(self):
        sample = fake_sample()
        extra = copy.deepcopy(sample.events[1])
        extra.output.usage.input_tokens = 999
        extra.call.response["usage"]["prompt_tokens"] = 999
        sample.events.insert(1, extra)

        _episode, records = artifacts.sample_export_records(sample)

        self.assertEqual(records[0]["join"]["status"], "usage_mismatch")
        self.assertEqual(records[0]["join"]["skipped_model_calls"], 1)
        self.assertEqual(records[0]["model_calls"][0]["response_id"], "generation-1")

    def test_writes_streamable_compressed_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "run.eval"
            source.write_bytes(b"raw-eval-source")
            log = SimpleNamespace(
                location=str(source),
                eval=Dumpable(eval_id="eval-1", metadata={"run_manifest_id": "launch-1"}),
                samples=[fake_sample()],
            )
            result = artifacts.export_eval_log(log, Path(tmp) / "analysis", source)
            with gzip.open(result["path"], "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual([r["record_type"] for r in rows], ["eval", "episode", "event"])
            self.assertEqual(rows[0]["source_eval_sha256"], artifacts.sha256_file(source))
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["events"], 1)
            self.assertEqual(result["usage"]["reasoning_tokens"], 7)
            self.assertEqual(result["usage"]["openrouter_cost_usd"], 0.0123)
            self.assertEqual(result["usage"]["calls_with_exact_cost"], 1)

    def test_finalize_rehydrates_eval_set_header_before_exporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run.eval"
            source.write_bytes(b"raw-eval-source")
            full = SimpleNamespace(
                location=str(source), eval=Dumpable(eval_id="eval-1"), samples=[fake_sample()]
            )
            header = SimpleNamespace(
                location=str(source), eval=Dumpable(eval_id="eval-1"), samples=None
            )
            frozen = artifacts.RunArtifacts(
                "launch-1", root, root / "manifest.json", root / "run-config.yaml",
                root / "source-snapshot.tar.gz",
            )
            with mock.patch.object(artifacts, "read_eval_log", return_value=full) as read:
                result = artifacts.finalize_run(frozen, [header], True)
            read.assert_called_once_with(str(source))
            self.assertEqual(result["exports"][0]["samples"], 1)
            self.assertEqual(result["exports"][0]["events"], 1)

    def test_successful_finalize_refuses_an_incomplete_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run.eval"
            source.write_bytes(b"raw-eval-source")
            no_episode = SimpleNamespace(id="bad", store={}, metadata={}, events=[])
            full = SimpleNamespace(
                location=str(source), eval=Dumpable(eval_id="eval-1"), samples=[no_episode]
            )
            header = SimpleNamespace(location=str(source), samples=None)
            frozen = artifacts.RunArtifacts(
                "launch-1", root, root / "manifest.json", root / "run-config.yaml",
                root / "source-snapshot.tar.gz",
            )
            with mock.patch.object(artifacts, "read_eval_log", return_value=full):
                with self.assertRaisesRegex(RuntimeError, "export incomplete"):
                    artifacts.finalize_run(frozen, [header], True)
            self.assertFalse((root / "completion.json").exists())

    def test_finalization_error_is_retryable_and_does_not_seal_run_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = artifacts.RunArtifacts(
                "launch-1", root, root / "manifest.json", root / "run-config.yaml",
                root / "source-snapshot.tar.gz",
            )

            artifacts.record_finalization_error(frozen, "RuntimeError", eval_success=True)

            self.assertFalse((root / "completion.json").exists())
            status = json.loads((root / "finalization-error.json").read_text())
            self.assertTrue(status["eval_success"])
            self.assertTrue(status["retryable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
