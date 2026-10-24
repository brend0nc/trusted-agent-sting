"""Provider pinning (drift control) + per-call provenance: the optional ROSTER `provider_pref`
threads into the OpenRouter request's extra_body["provider"], and the generator stamps each call's
usage with the served model version (and host, when the response exposes it). No network.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_provider_pinning.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import models, task  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402


class TestProviderPref(unittest.TestCase):
    def test_glm_not_pinned_yet(self):
        # The mechanism must exist but glm-5.2 stays UNPINNED — pinning is a run-time decision.
        self.assertIsNone(models.provider_pref("glm-5.2"))
        self.assertIsNone(models.provider_pref("opus-4.8"))

    def test_unknown_key_is_none(self):
        self.assertIsNone(models.provider_pref("mockllm/model"))

    def test_pref_threads_into_generate_config_extra_body(self):
        pref = {"order": ["z-ai"], "allow_fallbacks": False}
        original = models.ROSTER["glm-5.2"].get("provider_pref")
        models.ROSTER["glm-5.2"]["provider_pref"] = pref
        try:
            cfg = task.build_generate_config("glm-5.2", 41)
            self.assertEqual(cfg.extra_body, {"provider": pref})
        finally:
            if original is None:
                models.ROSTER["glm-5.2"].pop("provider_pref", None)
            else:
                models.ROSTER["glm-5.2"]["provider_pref"] = original

    def test_no_pref_leaves_extra_body_unset(self):
        cfg = task.build_generate_config("opus-4.8", 41)
        self.assertIsNone(cfg.extra_body)
        # The seed remains fixed; live Opus evidence requires temperature omission.
        self.assertEqual(cfg.seed, 41)
        self.assertIsNone(cfg.temperature)

    def test_raw_spec_seat_has_no_extra_body(self):
        # a non-roster seat value (e.g. mockllm/model) must not crash and must not pin.
        self.assertIsNone(task.build_generate_config("mockllm/model", 7).extra_body)


class TestServedFields(unittest.TestCase):
    def test_records_served_model(self):
        out = ModelOutput.from_content(model="openai/gpt-5.5-20260423", content="ok")
        self.assertEqual(task.served_fields(out)["served_model"], "openai/gpt-5.5-20260423")

    def test_records_served_provider_when_metadata_exposes_it(self):
        out = ModelOutput.from_content(model="z-ai/glm-5.2", content="ok")
        out.metadata = {"provider": "Fireworks"}
        self.assertEqual(task.served_fields(out)["served_provider"], "Fireworks")

    def test_omits_served_provider_when_absent(self):
        out = ModelOutput.from_content(model="z-ai/glm-5.2", content="ok")
        self.assertNotIn("served_provider", task.served_fields(out))


class TestGeneratorStampsProvenance(unittest.IsolatedAsyncioTestCase):
    async def test_usage_carries_served_model(self):
        gen = task.build_generator("mockllm/model", seed=1, seat="P1")
        _content, usage = await gen([{"role": "user", "content": "hi"}])
        self.assertEqual(usage["served_model"], "mockllm")  # mockllm sets out.model = 'mockllm'
        self.assertIn("prompt_tokens", usage)
        self.assertIn("completion_tokens", usage)
        self.assertIn("reasoning_tokens", usage)
        self.assertIn("cache_read_tokens", usage)
        self.assertIn("latency_seconds", usage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
