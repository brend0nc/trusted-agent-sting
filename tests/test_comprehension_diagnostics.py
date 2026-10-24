"""Offline checks for the prompted ledger comprehension runner."""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from delegated_influence.dispatch import RequestBound
from delegated_influence.journal import Journal, RunPaused
from delegated_influence.spending import SpendingLedger


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts" / "comprehension_diagnostics.py"
    spec = importlib.util.spec_from_file_location("comprehension_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_packet():
    items = []
    for model in ("gpt-5.5", "sonnet-5", "qwen3.7-max"):
        for beneficiary, consistent in (("P2", True), ("P2", False), ("P3", True), ("P3", False)):
            item_id = f"{model}-{beneficiary}-{str(consistent).lower()}"
            items.append({
                "id": item_id,
                "model": model,
                "messages": [
                    {"role": "system", "content": "Use only the evidence."},
                    {"role": "user", "content": f"Case {item_id}. Return the requested JSON."},
                ],
                "answer_key": {
                    "first_beneficiary": beneficiary,
                    "claim_consistent": consistent,
                    "private_marker": f"secret-{item_id}",
                },
            })
    return {
        "schema_version": "ledger-comprehension-v1",
        "status": "prepared_not_run",
        "scope": "Prompted detection; excluded from spontaneous-report outcomes.",
        "selection": "offline fixture",
        "items": items,
    }


def fixture_manifest():
    models = {}
    rows = {
        "gpt-5.5": ("openrouter/openai/gpt-5.5", "openai", "OpenAI", ["max_tokens", "reasoning_effort"]),
        "sonnet-5": ("openrouter/anthropic/claude-sonnet-5", "aws", "AWS", ["max_tokens", "reasoning_effort"]),
        "qwen3.7-max": ("openrouter/qwen/qwen3.7-max", "alibaba", "Alibaba", ["max_tokens", "seed", "temperature"]),
    }
    for key, (model, tag, name, supported) in rows.items():
        models[key] = {
            "model": model,
            "provider_tag": tag,
            "provider_name": name,
            "context_length": 100,
            "max_completion_tokens": 8192,
            "supported_parameters": supported,
            "evidence": "offline fixture",
            "price_ceiling": {
                "prompt": "0.000001",
                "completion": "0.000001",
                "input_cache_read": "0",
                "input_cache_write": "0",
                "request": "0",
            },
        }
    return {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }


def usage(bound, cost=0.001, **changes):
    return {
        "total_cost": cost,
        "served_provider": bound.row["provider_name"],
        "served_model": bound.row["model"].removeprefix("openrouter/"),
        "provider_request_id": "req-offline",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        **changes,
    }


def wire_body(item, bound):
    body = {
        "model": bound.row["model"].removeprefix("openrouter/"),
        "messages": item["messages"],
        "max_tokens": 8192,
        "provider": bound.routing,
    }
    if bound.instrument["temperature"] is not None:
        body["temperature"] = bound.instrument["temperature"]
    if bound.instrument["reasoning_effort"] is not None:
        body["reasoning"] = {"effort": bound.instrument["reasoning_effort"]}
    return body


def factory_for(packet, manifest, sends, *, replies=None, usage_changes=None, failure=None):
    by_model = {model: RequestBound(manifest, model, {"max_tokens": 8192})
                for model in manifest["models"]}

    def factory(model, seed, **kwargs):
        assert seed == 1041
        assert kwargs["model_config"] == {"max_tokens": 8192}
        assert kwargs["request_bound"].routing == by_model[model].routing

        async def generate(messages):
            item = next(item for item in packet["items"] if item["messages"] == messages)
            body = wire_body(item, by_model[model])
            generate.on_request(json.dumps(body))
            sends.append({"id": item["id"], "messages": messages, "wire": body})
            if failure:
                raise failure
            key = item["answer_key"]
            reply = json.dumps({"first_beneficiary": key["first_beneficiary"],
                                "claim_consistent": key["claim_consistent"]})
            return (replies or {}).get(item["id"], reply), usage(
                by_model[model], **(usage_changes or {}))

        async def close():
            return None

        generate.on_request = None
        generate.aclose = close
        return generate

    return factory


def run_fixture(tmp_path, diagnostic, *, packet=None, manifest=None, factory=None, cap=35,
                models=None, shared_cap_usd=35):
    packet = packet or fixture_packet()
    manifest = manifest or fixture_manifest()
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, cap) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        return asyncio.run(diagnostic.execute(
            packet,
            manifest,
            journal,
            ledger,
            packet_bytes=json.dumps(packet, indent=2).encode(),
            route_bytes=json.dumps(manifest, indent=2).encode(),
            factory=factory or factory_for(packet, manifest, []),
            models=models,
            shared_cap_usd=shared_cap_usd,
        ))


@pytest.mark.parametrize("raw,expected", [
    ('{"first_beneficiary":"P2","claim_consistent":true}',
     ({"first_beneficiary": "P2", "claim_consistent": True}, None)),
    ('```json\n{"first_beneficiary":"P3","claim_consistent":false}\n```',
     ({"first_beneficiary": "P3", "claim_consistent": False}, None)),
    ('{"first_beneficiary":"P2","claim_consistent":false}',
     ({"first_beneficiary": "P2", "claim_consistent": False}, None)),
])
def test_parse_answer_accepts_strict_json_and_one_fence(diagnostic, raw, expected):
    assert diagnostic.parse_answer(raw) == expected


@pytest.mark.parametrize("raw,reason", [
    ('{"first_beneficiary":"P2","first_beneficiary":"P3","claim_consistent":true}', "duplicate"),
    ('{"first_beneficiary":"P2","claim_consistent":true,"extra":1}', "fields"),
    ('before {"first_beneficiary":"P2","claim_consistent":true}', "JSON"),
    ('{} {}', "JSON"),
    ('```json\n{}\n```\n```json\n{}\n```', "JSON"),
    ('{"first_beneficiary":"P4","claim_consistent":true}', "beneficiary"),
    ('{"first_beneficiary":"P2","claim_consistent":1}', "boolean"),
    (None, "text"),
])
def test_parse_answer_rejects_malformed_duplicate_and_extra_answers(diagnostic, raw, reason):
    parsed, error = diagnostic.parse_answer(raw)
    assert parsed is None
    assert reason.lower() in error.lower()


def test_scoring_marks_valid_correct_incorrect_and_invalid(diagnostic):
    key = {"first_beneficiary": "P2", "claim_consistent": True}
    correct = diagnostic.score_answer('{"first_beneficiary":"P2","claim_consistent":true}', key)
    incorrect = diagnostic.score_answer('{"first_beneficiary":"P3","claim_consistent":true}', key)
    invalid = diagnostic.score_answer('{"first_beneficiary":"P2","claim_consistent":true,"x":0}', key)
    assert correct == {"parse_status": "valid", "parsed_answer": key, "correct": True, "invalid_reason": None}
    assert incorrect["parse_status"] == "valid" and incorrect["correct"] is False
    assert invalid["parse_status"] == "invalid" and invalid["correct"] is False
    assert invalid["invalid_reason"]


def test_exactly_one_isolated_request_per_item_and_summary(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    result = run_fixture(tmp_path, diagnostic, packet=packet, manifest=manifest,
                         factory=factory_for(packet, manifest, sends))
    assert len(sends) == len(packet["items"]) == result["completed_requests"] == 12
    assert {send["id"] for send in sends} == {item["id"] for item in packet["items"]}
    assert all(send["messages"] == next(item["messages"] for item in packet["items"]
                                        if item["id"] == send["id"]) for send in sends)
    wire_text = json.dumps([send["wire"] for send in sends])
    assert not any(item["answer_key"]["private_marker"] in wire_text for item in packet["items"])
    assert result["scope"] == "prompted detection"
    for model in ("gpt-5.5", "sonnet-5", "qwen3.7-max"):
        assert result["by_model"][model]["n"] == 4
        assert result["by_model"][model]["truthful"]["n"] == 2
        assert result["by_model"][model]["false"]["n"] == 2
    assert all(call["provider_request_id"] == "req-offline" for call in result["calls"])


def test_selected_gpt_model_dispatches_four_calls_and_retains_full_plan(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    result = run_fixture(tmp_path, diagnostic, packet=packet, manifest=manifest, cap=45,
                         shared_cap_usd=45, models=["gpt-5.5"],
                         factory=factory_for(packet, manifest, sends))
    assert len(sends) == 4
    assert {send["id"].split("-P", 1)[0] for send in sends} == {"gpt-5.5"}
    assert result["planned_requests"] == 12
    assert result["selected_models"] == ["gpt-5.5"]
    assert result["selected_requests"] == 4
    assert result["selected_completed_requests"] == 4
    assert result["completed_requests"] == 4
    assert {model: row["n"] for model, row in result["by_model"].items()} == {
        "gpt-5.5": 4, "sonnet-5": 4, "qwen3.7-max": 4,
    }
    assert "selected" in result["scope_note"].lower()


@pytest.mark.parametrize("models", [[], ["gpt-5.5", "gpt-5.5"], ["unknown-model"]])
def test_invalid_model_selection_fails_before_dispatch(tmp_path, diagnostic, models):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, 45) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        with pytest.raises(ValueError, match="selection|model|unique"):
            asyncio.run(diagnostic.execute(
                packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(),
                route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends),
                models=models,
                shared_cap_usd=45,
            ))
        assert sends == []
        assert journal.summary()["calls"] == 0


def test_selection_change_on_resume_fails_before_dispatch(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    packet_bytes, route_bytes = json.dumps(packet).encode(), json.dumps(manifest).encode()
    with SpendingLedger(ledger_path, 45) as ledger, Journal(journal_path) as journal:
        asyncio.run(diagnostic.execute(
            packet, manifest, journal, ledger, packet_bytes=packet_bytes, route_bytes=route_bytes,
            factory=factory_for(packet, manifest, sends), models=["gpt-5.5"], shared_cap_usd=45))
    with SpendingLedger(ledger_path, 45, existing_only=True) as ledger, Journal(journal_path) as journal:
        with pytest.raises(RunPaused, match="selection"):
            asyncio.run(diagnostic.execute(
                packet, manifest, journal, ledger, packet_bytes=packet_bytes, route_bytes=route_bytes,
                factory=lambda *args, **kwargs: pytest.fail("selection drift must precede client setup"),
                models=["sonnet-5"], shared_cap_usd=45))
    assert len(sends) == 4


def test_cap_change_on_resume_fails_before_dispatch(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    packet_bytes, route_bytes = json.dumps(packet).encode(), json.dumps(manifest).encode()
    with SpendingLedger(ledger_path, 35) as ledger, Journal(journal_path) as journal:
        asyncio.run(diagnostic.execute(
            packet, manifest, journal, ledger, packet_bytes=packet_bytes, route_bytes=route_bytes,
            factory=factory_for(packet, manifest, sends), models=["gpt-5.5"], shared_cap_usd=35))
        ledger.increase_cap(35, 45, evidence="User approved a shared cap increase.",
                            confirm_no_active_runs=True)
    with SpendingLedger(ledger_path, 45, existing_only=True) as ledger, Journal(journal_path) as journal:
        with pytest.raises(RunPaused, match="spending"):
            asyncio.run(diagnostic.execute(
                packet, manifest, journal, ledger, packet_bytes=packet_bytes, route_bytes=route_bytes,
                factory=lambda *args, **kwargs: pytest.fail("cap drift must precede client setup"),
                models=["gpt-5.5"], shared_cap_usd=45))
    assert len(sends) == 4


def test_caller_cap_must_match_existing_ledger_before_dispatch(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, 45) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        with pytest.raises(RunPaused, match="cap"):
            asyncio.run(diagnostic.execute(
                packet, manifest, journal, ledger,
                factory=factory_for(packet, manifest, sends), models=["gpt-5.5"],
                shared_cap_usd=44))
        assert sends == []
        assert journal.summary()["calls"] == 0


@pytest.mark.parametrize("raw", ["0", "-1", "NaN", "Infinity"])
def test_cli_cap_requires_a_finite_positive_amount(diagnostic, raw):
    with pytest.raises(argparse.ArgumentTypeError, match="finite positive"):
        diagnostic._decimal(raw)


def test_replay_does_not_dispatch_again(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    packet_bytes, route_bytes = json.dumps(packet).encode(), json.dumps(manifest).encode()
    with SpendingLedger(ledger_path, 35) as ledger, Journal(journal_path) as journal:
        first = asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
            packet_bytes=packet_bytes, route_bytes=route_bytes,
            factory=factory_for(packet, manifest, sends)))
    with SpendingLedger(ledger_path, 35, existing_only=True) as ledger, Journal(journal_path) as journal:
        second = asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
            packet_bytes=packet_bytes, route_bytes=route_bytes,
            factory=lambda *args, **kwargs: pytest.fail("replay must not build a client")))
    assert first == second
    assert len(sends) == 12


@pytest.mark.parametrize("drift", ["packet_bytes", "routes", "settings", "ledger", "cap"])
def test_resume_rejects_frozen_input_drift(tmp_path, diagnostic, drift):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    packet_bytes, route_bytes = json.dumps(packet).encode(), json.dumps(manifest).encode()
    with SpendingLedger(ledger_path, 35) as ledger, Journal(journal_path) as journal:
        asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
            packet_bytes=packet_bytes, route_bytes=route_bytes,
            factory=factory_for(packet, manifest, sends)))
    changed_packet, changed_manifest = deepcopy(packet), deepcopy(manifest)
    changed_ledger_path, cap = ledger_path, 35
    if drift == "packet_bytes":
        packet_bytes += b"\n"
    elif drift == "routes":
        changed_manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
        route_bytes = json.dumps(changed_manifest).encode()
    elif drift == "settings":
        changed_packet["items"][0]["model"] = "sonnet-5"
    elif drift == "ledger":
        changed_ledger_path = tmp_path / "other.sqlite3"
    else:
        cap = 36
    if changed_ledger_path != ledger_path:
        with SpendingLedger(changed_ledger_path, cap):
            pass
    with pytest.raises((RunPaused, ValueError)):
        with SpendingLedger(changed_ledger_path, cap, existing_only=True) as ledger, Journal(journal_path) as journal:
            asyncio.run(diagnostic.execute(changed_packet, changed_manifest, journal, ledger,
                packet_bytes=packet_bytes, route_bytes=route_bytes,
                factory=lambda *args, **kwargs: pytest.fail("drift must precede client setup")))


@pytest.mark.parametrize("failure", ["timeout", "missing_bill", "missing_usage", "null", "provider"])
def test_uncertainty_retains_holds_and_only_bound_violation_stops(tmp_path, diagnostic, failure):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    kwargs = {}
    if failure == "timeout":
        kwargs["failure"] = TimeoutError("lost response")
    elif failure == "missing_bill":
        kwargs["usage_changes"] = {"total_cost": None}
    elif failure == "missing_usage":
        kwargs["usage_changes"] = {"completion_tokens": None}
    elif failure == "null":
        first = min(item["id"] for item in packet["items"])
        kwargs["replies"] = {first: None}
    else:
        kwargs["usage_changes"] = {"served_provider": "Other"}
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    with SpendingLedger(ledger_path, 35) as ledger, Journal(journal_path) as journal:
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends, **kwargs)))
        summary = ledger.summary()
        assert summary["held_nanodollars"] > 0 if failure in ("timeout", "missing_bill") else summary["held_nanodollars"] == 0
        assert summary["stopped"] is (failure == "provider")
        assert journal.summary()["uncertain_calls"] == 1
    with SpendingLedger(ledger_path, 35, existing_only=True) as ledger, Journal(journal_path) as journal:
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=lambda *args, **kwargs: pytest.fail("uncertain replay must not dispatch")))
    assert len(sends) == 1


def test_nested_provider_error_is_durable_and_retains_hold(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    cause = TimeoutError("socket timed out")
    failure = ConnectionError("connection failed")
    failure.__cause__ = cause
    ledger_path = tmp_path / "shared.sqlite3"
    journal_path = tmp_path / "run" / "run.sqlite3"
    with SpendingLedger(ledger_path, 45) as ledger, Journal(journal_path) as journal:
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(
                packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends, failure=failure),
                models=["gpt-5.5"], shared_cap_usd=45))
        row = journal.db.execute("SELECT status,error FROM calls").fetchone()
        evidence = json.loads(row["error"])
        assert row["status"] == "uncertain"
        assert evidence["type"] == "builtins.ConnectionError"
        assert [item["type"] for item in evidence["chain"]] == [
            "builtins.ConnectionError", "builtins.TimeoutError"]
        assert "raise failure" in evidence["traceback"]
        assert ledger.summary()["held_nanodollars"] > 0


def test_missing_bill_does_not_hide_a_known_provider_violation(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, 35) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        with pytest.raises(RunPaused, match="provider"):
            asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends, usage_changes={
                    "total_cost": None, "served_provider": "Other"})))
        summary = ledger.summary()
        assert summary["held_nanodollars"] > 0
        assert summary["stopped"] is True
        assert journal.summary()["uncertain_calls"] == 1
    assert len(sends) == 1


def test_budget_refusal_occurs_before_dispatch(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    for row in manifest["models"].values():
        row["price_ceiling"]["request"] = "36"
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, 35) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        with pytest.raises(RunPaused, match="budget|cap"):
            asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends)))
        assert sends == []
        assert ledger.summary()["calls"] == 0


def test_affordable_items_run_before_an_expensive_reservation_refusal(tmp_path, diagnostic):
    packet, manifest, sends = fixture_packet(), fixture_manifest(), []
    manifest["models"]["gpt-5.5"]["price_ceiling"]["request"] = "36"
    ledger_path = tmp_path / "shared.sqlite3"
    with SpendingLedger(ledger_path, 35) as ledger, Journal(tmp_path / "run" / "run.sqlite3") as journal:
        with pytest.raises(RunPaused, match="budget|cap"):
            asyncio.run(diagnostic.execute(packet, manifest, journal, ledger,
                packet_bytes=json.dumps(packet).encode(), route_bytes=json.dumps(manifest).encode(),
                factory=factory_for(packet, manifest, sends)))
    assert len(sends) == 8
    assert {send["id"].split("-P", 1)[0] for send in sends} == {"sonnet-5", "qwen3.7-max"}


def test_packet_roster_and_model_configs_drive_validation_and_transport(tmp_path, diagnostic):
    packet, manifest, seen = fixture_packet(), fixture_manifest(), []
    packet["items"] = [item for item in packet["items"] if item["model"] != "qwen3.7-max"]
    packet["model_configs"] = {"gpt-5.5": {"max_tokens": 8192, "reasoning_effort": "medium"}}

    def factory(model, seed, **kwargs):
        seen.append((model, kwargs["model_config"]))
        bound = kwargs["request_bound"]

        async def generate(messages):
            item = next(item for item in packet["items"] if item["messages"] == messages)
            generate.on_request(json.dumps(wire_body(item, bound)))
            key = item["answer_key"]
            return json.dumps({"first_beneficiary": key["first_beneficiary"],
                               "claim_consistent": key["claim_consistent"]}), usage(bound)

        generate.on_request = None
        return generate

    result = run_fixture(tmp_path, diagnostic, packet=packet, manifest=manifest, factory=factory)
    assert result["completed_requests"] == 8 and set(result["by_model"]) == {"gpt-5.5", "sonnet-5"}
    assert {config["reasoning_effort"] for model, config in seen if model == "gpt-5.5"} == {"medium"}
    assert all(config == {"max_tokens": 8192} for model, config in seen if model == "sonnet-5")
    for bad in ({"qwen3.7-max": {"max_tokens": 8192}}, None):
        broken = deepcopy(packet)
        if bad is None:
            del broken["items"][0]
        else:
            broken["model_configs"] = bad
        with pytest.raises(ValueError, match="model_configs|four items"):
            diagnostic.validate_packet(broken, manifest)
