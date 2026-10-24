"""Explicit unsupported-temperature policy must not alter token or reasoning budgets."""
from types import SimpleNamespace

import pytest

from delegated_influence import dispatch, models, task


NO_TEMPERATURE = (
    "sonnet-5", "opus-4.7", "opus-4.8", "fable-5", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5",
)


def test_gemini_live_diagnostic_policy_changes_only_output_limit():
    instrument = task.sampling_instrument("gemini-flash")
    assert instrument["max_tokens"] == 4096
    assert instrument["temperature"] == task.TEMPERATURE
    assert instrument["reasoning_effort"] is None
    assert task.sampling_instrument("gemini-flash", {"max_tokens": 1024})["max_tokens"] == 1024


@pytest.mark.parametrize("model", NO_TEMPERATURE)
def test_unsupported_temperature_is_omitted_without_changing_other_sampling(model):
    instrument = task.sampling_instrument(model)
    assert instrument["temperature"] is None
    assert instrument["max_tokens"] == 1024
    assert instrument["reasoning_effort"] is None
    assert "reasoning" not in models.MODEL_REGISTRY[model]["tags"]


@pytest.mark.parametrize("model", ("sonnet-4.6", "opus-4.6"))
def test_other_nonreasoning_instruments_keep_their_temperature(model):
    instrument = task.sampling_instrument(model)
    assert instrument["temperature"] == task.TEMPERATURE
    assert instrument["max_tokens"] == 1024


def test_sonnet_manifest_accepts_an_endpoint_without_temperature_support():
    """Offline catalogue shape: omission must reach the compiler and request bound."""
    model = "sonnet-5"
    slug = models.MODEL_REGISTRY[model]["id"]
    catalog = {"data": [{"id": slug, "reasoning": {"supported_efforts": ["high"]}}]}
    endpoints = {"data": {"endpoints": [{
        "tag": "fixture", "provider_name": "Fixture", "status": 0,
        "context_length": 1000000, "max_completion_tokens": 128000,
        "supported_parameters": ["max_tokens", "reasoning"],
        "pricing": {"prompt": "0.000002", "completion": "0.00001"},
    }]}}
    planned = [("sonnet", SimpleNamespace(dataset=[SimpleNamespace(metadata={
        "seat_map": {"P1": model}, "model_configs": {},
    })]))]
    manifest = dispatch.compile_manifest(planned, fetch=lambda url:
                                         catalog if url.endswith("/models") else endpoints)
    bound = dispatch.RequestBound(manifest, model)
    assert bound.instrument["temperature"] is None
    assert bound.output == 1024
    assert manifest["models"][model]["provider_tag"] == "fixture"
