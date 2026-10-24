"""Guarded post-hoc judgments retain bills and responses across interruption."""
import asyncio
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from delegated_influence.journal import Journal, RunPaused
from delegated_influence.spending import SpendingLedger


def manifest():
    return {"schema_version": 1, "verified_at": datetime.now(timezone.utc).isoformat(),
            "models": {"opus-4.8": {
                "model": "openrouter/anthropic/claude-opus-4.8", "provider_tag": "azure/us",
                "provider_name": "Azure", "context_length": 200000,  # bound 200k * (6.25e-6 + 2.5e-5) = 6.25 fits the $20 test caps
                "max_completion_tokens": 8192, "supported_parameters": ["max_tokens"],
                "evidence": "offline test", "price_ceiling": {
                    "prompt": "0.000005", "completion": "0.000025",
                    "input_cache_read": "0.0000005", "input_cache_write": "0.00000625", "request": "0"}}}}


class Provider:
    def __init__(self, bound, response='[]', *, error=None, cost=0.001):
        self.bound, self.response, self.error, self.cost = bound, response, error, cost
        self.closed = False
        self.calls = 0

    async def __call__(self, messages):
        self.calls += 1
        self.on_request(json.dumps({"model": "anthropic/claude-opus-4.8", "messages": messages,
                                    "max_tokens": 4096, "provider": self.bound.routing}))
        if self.error:
            raise self.error
        return self.response, {"total_cost": self.cost, "served_provider": "Azure",
                               "served_model": "anthropic/claude-opus-4.8", "provider_request_id": "test-id",
                               "prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 0,
                               "cache_read_tokens": 0, "cache_write_tokens": 0,
                               "provider_usage": {"cost": self.cost}}

    async def aclose(self):
        self.closed = True


def test_response_replays_without_a_second_bill_or_dispatch(tmp_path):
    native = importlib.import_module("delegated_influence.native_judges")
    providers = []
    def factory(model, seed, *, model_config, request_bound):
        provider = Provider(request_bound)
        providers.append(provider)
        return provider
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native.NativeJudgeSession(journal, ledger, manifest(), factory=factory)
        async def exercise():
            generator = session.generator("opus-4.8", "test-instrument")
            first = await generator("input text", request_key="record-1")
            second = await generator("input text", request_key="record-1")
            return first, second
        first, second = asyncio.run(exercise())
        assert first == second
        assert first[0] == "[]"
        assert ledger.summary()["reported_usd"] == "0.001"
        assert sum(p.calls for p in providers) == 1
        assert all(p.closed for p in providers)
        call_id = journal.db.execute("SELECT id FROM calls").fetchone()[0]
        assert 'input text' in journal.call(call_id)["provider_request_body"]


@pytest.mark.parametrize("failure", ["timeout", "missing_bill"])
def test_uncertain_request_keeps_hold_and_prevents_later_dispatch(tmp_path, failure):
    native = importlib.import_module("delegated_influence.native_judges")
    providers = []
    def factory(model, seed, *, model_config, request_bound):
        provider = Provider(request_bound, error=TimeoutError() if failure == "timeout" else None,
                            cost=None if failure == "missing_bill" else 0.001)
        providers.append(provider)
        return provider
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native.NativeJudgeSession(journal, ledger, manifest(), factory=factory)
        generator = session.generator("opus-4.8", "test-instrument")
        with pytest.raises(RunPaused):
            asyncio.run(generator("first", request_key="record-1"))
        with pytest.raises(RunPaused):
            asyncio.run(generator("second", request_key="record-2"))
        assert ledger.summary()["held_usd"] == "6.25"  # hold reserves the context window: 200k * (6.25e-6 + 2.5e-5)
        assert sum(p.calls for p in providers) == 1
        assert all(p.closed for p in providers)
        if failure == "missing_bill":
            call_id = journal.db.execute("SELECT id FROM calls").fetchone()[0]
            assert journal.call(call_id)["response"] == "[]"


def test_parse_failure_preserves_raw_response_without_rebuying(tmp_path):
    native = importlib.import_module("delegated_influence.native_judges")
    from delegated_influence.task import ClaimParseError, make_lie_judge
    providers = []
    def factory(model, seed, *, model_config, request_bound):
        provider = Provider(request_bound, response="not JSON")
        providers.append(provider)
        return provider
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native.NativeJudgeSession(journal, ledger, manifest(), factory=factory)
        judge = make_lie_judge("opus-4.8", generate=session.generator("opus-4.8", "lie"))
        for _ in range(2):
            with pytest.raises(ClaimParseError):
                asyncio.run(native.call_judge(judge, "record-1", "text", "P1", "P2", ["P1", "P2"]))
        call_id = journal.db.execute("SELECT id FROM calls").fetchone()[0]
        assert journal.call(call_id)["response"] == "not JSON"
        assert sum(p.calls for p in providers) == 1


def test_changed_input_or_route_cannot_reuse_a_journal(tmp_path):
    native = importlib.import_module("delegated_influence.native_judges")
    route = manifest()
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native.NativeJudgeSession(journal, ledger, route, factory=lambda *a, **kw: Provider(kw["request_bound"]))
        generator = session.generator("opus-4.8", "test-instrument")
        asyncio.run(generator("original", request_key="record-1"))
        with pytest.raises(RunPaused, match="input changed|metadata changed"):
            asyncio.run(generator("changed", request_key="record-1"))
        route["models"]["opus-4.8"]["provider_name"] = "Other"
        with pytest.raises(RunPaused, match="changed"):
            native.NativeJudgeSession(journal, ledger, route)


def test_both_frozen_holdouts_complete_offline_with_native_provenance(tmp_path):
    native = importlib.import_module("delegated_influence.native_judges")
    root = Path(__file__).resolve().parents[1]
    def script(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    lie = script("calibrate_lie_judges")
    promise = script("validate_promise_judge")
    source = root / "data/paper_lie_structured_holdout_sample.json"
    labels = root / "data/paper_lie_structured_holdout_labels.json"
    import hashlib
    sample = lie.apply_label_sidecar(json.loads(source.read_bytes()), json.loads(labels.read_bytes()),
                                    hashlib.sha256(source.read_bytes()).hexdigest())
    lie_doc = lie.initialise(sample, ["opus-4.8"], sample_path=source, channel="private")
    records, sample_hash, labels_hash = promise.load_frozen_records()
    promise_doc = promise.initialise(records, sample_hash, labels_hash, promise._judge_spec("opus-4.8"))
    providers = []
    def factory(model, seed, *, model_config, request_bound):
        provider = Provider(request_bound)
        original = provider.__class__.__call__
        class JudgeProvider(Provider):
            async def __call__(self, messages):
                if "breach test" in messages[0]["content"]:
                    self.response = '{"promise": false, "conditional": false}'
                return await original(self, messages)
        provider = JudgeProvider(request_bound)
        providers.append(provider)
        return provider
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native.NativeJudgeSession(journal, ledger, manifest(), factory=factory)
        async def fill():
            await lie.fill(lie_doc, ["opus-4.8"], tmp_path / "lie.json", lie.infer_roster(sample["records"]), session=session)
            await promise.fill(promise_doc, tmp_path / "promise.json", session=session)
        asyncio.run(fill())
        assert lie_doc["structured_summary"]["opus-4.8"]["completed_calls"] == 50
        assert promise_doc["summary"]["completed_calls"] == 20
        assert journal.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 70
        assert ledger.summary()["reported_usd"] == "0.07"
        assert all(p.closed for p in providers)
        assert all(row["output"]["usage"]["journal_call_id"] for row in promise_doc["records"])
        assert all(row["outputs"]["opus-4.8"]["served_provider"] == "Azure" for row in lie_doc["records"])


@pytest.mark.parametrize("supports_seed", [False, True])
def test_real_adapter_preserves_judge_prompt_sampling_and_total_input(tmp_path, monkeypatch, supports_seed):
    import httpx
    from delegated_influence import native_adapter, native_judges, task
    captured, clients = [], []
    async def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "gen-offline", "model": "anthropic/claude-opus-4.8",
            "object": "chat.completion", "created": 1, "provider": "Azure",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "[]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.001,
                      "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20}}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    original = native_adapter.get_model
    def build(*args, **kwargs):
        model = original(*args, api_key="offline-dummy-key", http_client=http, **kwargs)
        clients.append(model)
        return model
    monkeypatch.setattr(native_adapter, "get_model", build)
    routes = manifest()
    if supports_seed:
        routes["models"]["opus-4.8"]["supported_parameters"].append("seed")
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native_judges.NativeJudgeSession(journal, ledger, routes)
        judge = task.make_lie_judge("opus-4.8", include_details=True,
                                   generate=session.generator("opus-4.8", task.LIE_JUDGE_INSTRUMENT))
        result = asyncio.run(native_judges.call_judge(judge, "one", "message text", "P1", "P2", ["P1", "P2"]))
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["prompt_tokens"] == 40
        assert len(captured) == 1
        body = captured[0]
        assert body["messages"] == [{"role": "user", "content": task._lie_claim_prompt("message text", "P1", "P2", ["P1", "P2"])}]
        assert body["max_tokens"] == 4096
        assert "temperature" not in body and "reasoning" not in body
        assert body.get("seed") == (0 if supports_seed else None)
        assert body["provider"]["only"] == ["azure/us"]
        assert body["provider"]["allow_fallbacks"] is False
        assert clients[0].api.client.max_retries == 0 and clients[0].config.max_retries == 0
        assert http.is_closed


@pytest.mark.parametrize("kind", ["lie", "promise"])
def test_bulk_transport_pause_stops_before_later_messages(kind):
    from delegated_influence import check_lies, check_promises
    calls, errors = [], []
    async def unavailable(*args, **kwargs):
        calls.append(args)
        raise RunPaused("provider response unknown")
    events = [{"type": "message", "agent": "P1", "to": "P2", "text": text,
               "forfeit": False, "round": 1, "action_number": i}
              for i, text in enumerate(["first", "second"], 1)]
    with pytest.raises(RunPaused):
        if kind == "lie":
            asyncio.run(check_lies.check_episode_lies({"events": events, "seat_map": {"P1": "a", "P2": "b"}}, unavailable, errors=errors))
        else:
            asyncio.run(check_promises.find_promises(events, unavailable, errors=errors))
    assert len(calls) == 1
    assert errors == []


def test_missing_shared_controls_refuse_before_opening_a_journal(tmp_path):
    from types import SimpleNamespace
    from delegated_influence import native_judges
    args = SimpleNamespace(run_dir=str(tmp_path / "run"), route_manifest=None, ledger=None, shared_cap_usd=None)
    with pytest.raises(SystemExit, match="route-manifest"):
        with native_judges.open_session(args, inputs=[], binding={}):
            pytest.fail("must refuse incomplete spending controls")
    assert not (tmp_path / "run").exists()


def test_opener_rejects_cap_change_without_a_saved_audit_chain(tmp_path):
    from types import SimpleNamespace
    from delegated_influence import native_judges
    route_path = tmp_path / "routes.json"
    route_path.write_text(json.dumps(manifest()))
    ledger_path = tmp_path / "spend.sqlite3"
    with SpendingLedger(ledger_path, 20):
        pass
    args = SimpleNamespace(run_dir=str(tmp_path / "run"), route_manifest=str(route_path),
                           ledger=str(ledger_path), shared_cap_usd=20, keys=None)
    with native_judges.open_session(args, inputs=[], binding={}):
        pass
    import sqlite3
    with sqlite3.connect(ledger_path) as db:
        db.execute("UPDATE spending_policy SET cap_usd='25',cap_nanos='25000000000'")
    args.shared_cap_usd = 25
    with pytest.raises(RunPaused, match="audit chain"):
        with native_judges.open_session(args, inputs=[], binding={}):
            pytest.fail("an unaudited cap change must not open a paid session")


@pytest.mark.parametrize("kind", ["lie", "promise"])
def test_native_bulk_ignores_unbound_external_judgments(kind):
    from delegated_influence import check_lies, check_promises
    class UnboundCache:
        def get(self, *args):
            pytest.fail("native sessions must replay their own raw journal")
        def put(self, *args):
            pytest.fail("native sessions must not write unbound external judgments")
    async def judge(*args, **kwargs):
        return [] if kind == "lie" else {"promise": False}
    judge.native_judge = True
    events = [{"type": "message", "agent": "P1", "to": "P2", "text": "first", "forfeit": False}]
    if kind == "lie":
        result = asyncio.run(check_lies.check_episode_lies(
            {"events": events, "seat_map": {"P1": "a", "P2": "b"}}, judge, checkpoint=UnboundCache()))
    else:
        result = asyncio.run(check_promises.find_promises(events, judge, checkpoint=UnboundCache()))
    assert result == []


def test_unbound_holdout_outputs_are_refused_before_dispatch(tmp_path):
    from delegated_influence import native_judges
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("promise_unbound", root / "scripts/validate_promise_judge.py")
    promise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(promise)
    records, sample_hash, labels_hash = promise.load_frozen_records()
    doc = promise.initialise(records, sample_hash, labels_hash, promise._judge_spec("opus-4.8"))
    doc["records"][0]["output"] = {"promise": False}
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native_judges.NativeJudgeSession(journal, ledger, manifest(),
            factory=lambda *a, **kw: pytest.fail("must refuse cached output before dispatch"))
        with pytest.raises(RunPaused, match="unbound"):
            asyncio.run(promise.fill(doc, tmp_path / "out.json", session=session))
        assert ledger.summary()["calls"] == 0


def test_bulk_input_binding_includes_eval_and_native_exports(tmp_path):
    from delegated_influence import native_judges
    files = [tmp_path / "old.eval", tmp_path / "new.episode.json.gz", tmp_path / "run.sqlite3"]
    for path in files:
        path.write_bytes(b"source bytes")
    assert native_judges.source_inputs([tmp_path]) == sorted(files[:2])


def test_unbound_lie_holdout_output_is_refused_before_dispatch(tmp_path):
    from delegated_influence import native_judges
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("lie_unbound", root / "scripts/calibrate_lie_judges.py")
    lie = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lie)
    doc = {"records": [{"record_id": "one", "outputs": {"opus-4.8": {"claims": []}}}]}
    with Journal(tmp_path / "run.sqlite3") as journal, SpendingLedger(tmp_path / "spend.sqlite3", 20) as ledger:
        session = native_judges.NativeJudgeSession(journal, ledger, manifest(),
            factory=lambda *a, **kw: pytest.fail("must refuse cached output before dispatch"))
        with pytest.raises(RunPaused, match="unbound"):
            asyncio.run(lie.fill(doc, ["opus-4.8"], tmp_path / "out.json", ["P1", "P2"], session=session))
        assert ledger.summary()["calls"] == 0
