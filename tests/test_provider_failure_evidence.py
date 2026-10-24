"""Exercise the installed provider adapter with an offline HTTP transport."""
import asyncio
import json

import httpx
import pytest

from delegated_influence import native_adapter, runtime
from delegated_influence.journal import Journal, RunPaused, unpack


@pytest.mark.parametrize('status,body,retryable', [
    (429, {'error': {'code': 429, 'message': 'rate limited'}}, True),
    (503, {'error': {'code': 503, 'message': 'overloaded'}}, True),
    (502, {'error': {'code': 502, 'message': 'invalid upstream response'}}, False),
    (200, {'id': 'gen-failed', 'error': {'code': 503, 'message': 'overloaded'}}, False),
    (200, {'id': 'gen-failed', 'error': {'code': 500, 'message': 'server error'},
           'usage': {'cost': .01}}, False),
])
def test_adapter_retains_actual_status_and_error_body(tmp_path, monkeypatch, status, body, retryable):
    actual = native_adapter.get_model
    sent = []
    async def scenario():
        async def respond(request):
            sent.append(request.content)
            return httpx.Response(status, json=body, headers={'Retry-After': '61'})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        def model(*args, **kwargs):
            return actual(*args, api_key='offline-dummy-key', http_client=client, **kwargs)
        monkeypatch.setattr(native_adapter, 'get_model', model)
        generate = native_adapter.build_generator('openrouter/openai/gpt-4o-mini', 1,
                                                  model_config={'max_tokens': 100})
        try:
            with pytest.raises(Exception) as failure:
                await generate([{'role': 'user', 'content': 'hello'}])
            exc = failure.value
            assert (runtime.unserved_status(exc) is not None) is retryable
            evidence = json.loads(runtime.exception_evidence(exc))
            assert evidence['provider_response']['status_code'] == status
            assert evidence['provider_response']['body'] == body
            assert runtime.retry_after_seconds(exc) == 61
        finally:
            await generate.aclose()
    asyncio.run(scenario())
    assert len(sent) == 1


def test_rejected_attempt_keeps_wire_and_failure_before_retry(tmp_path, monkeypatch):
    from test_strict_runtime import metadata, response, usage, unserved
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    calls = []
    def factory(seat):
        async def generate(messages):
            calls.append(seat)
            generate.on_request(json.dumps({'attempt': len(calls)}))
            if len(calls) == 1:
                raise unserved(429)
            return response(seat), usage()
        return generate
    with Journal(tmp_path/'run.sqlite3') as journal:
        asyncio.run(runtime.run_episode(journal, 'one', metadata(), factory, reserve_usd=1, budget_usd=10))
        row = journal.db.execute('SELECT * FROM call_attempts').fetchone()
        assert json.loads(unpack(row['wire_request'])) == {'attempt': 1}
        assert '429' in row['error']
        assert row['call_id'] in {r['id'] for r in journal.db.execute('SELECT id FROM calls')}
        assert row['evidence'].startswith('HTTP 429 rejected')


def test_timeout_after_rejection_headers_retains_unknown_bill_and_retry_after(monkeypatch):
    actual = native_adapter.get_model
    entered = asyncio.Event()
    class StalledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.sleep(100)
            yield b'{"usage":{"cost":0.01}}'
    async def scenario():
        async def respond(request):
            return httpx.Response(503, headers={'Retry-After': '61'}, stream=StalledBody())
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        monkeypatch.setattr(native_adapter, 'get_model', lambda *a, **kw:
            actual(*a, api_key='offline-dummy-key', http_client=client, **kw))
        generate = native_adapter.build_generator('openrouter/openai/gpt-4o-mini', 1,
                                                  model_config={'max_tokens': 100})
        try:
            pending = asyncio.create_task(generate([{'role': 'user', 'content': 'hello'}]))
            await asyncio.wait_for(entered.wait(), 2)
            with pytest.raises(TimeoutError) as failure:
                await asyncio.wait_for(pending, .01)
            assert runtime.unserved_status(failure.value) is None
            assert runtime.retry_after_seconds(failure.value) == 61
            evidence = json.loads(runtime.exception_evidence(failure.value))['provider_response']
            assert evidence['status_code'] == 503 and evidence['body_complete'] is False
        finally:
            await generate.aclose()
    asyncio.run(scenario())


def test_error_body_is_durable_before_a_bill_overrun_stops_ledger(tmp_path):
    from test_strict_runtime import metadata, MODEL
    from delegated_influence.spending import SpendingLedger
    from decimal import Decimal
    from types import SimpleNamespace
    body = {'id': 'gen-123', 'usage': {'cost': 2},
            'error': {'code': 503, 'message': 'specific failure evidence'}}
    async def generate(messages):
        exc = RuntimeError('provider error')
        exc.native_response = {'status_code': 200, 'body': body, 'body_complete': True}
        raise exc
    with Journal(tmp_path/'run.sqlite3') as journal, SpendingLedger(tmp_path/'spending.sqlite3', 10) as ledger:
        with pytest.raises(RunPaused):
            asyncio.run(runtime.run_episode(journal, 'one', metadata(), lambda _: generate,
                reserve_usd=1, budget_usd=10, spending=ledger, spending_namespace='test',
                request_bounds={MODEL: SimpleNamespace(maximum_usd=Decimal('1'))}))
        row = journal.db.execute('SELECT * FROM calls').fetchone()
        assert json.loads(row['error'])['provider_response']['body'] == body
        assert row['reported'] == 2
        assert ledger.summary()['stopped'] is True
