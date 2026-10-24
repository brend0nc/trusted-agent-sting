"""Cross-ledger recovery and dispatch stops, using offline generators only."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from delegated_influence import dispatch, run, runtime, task
from delegated_influence.journal import Journal, RunPaused, pack
from delegated_influence.spending import SpendingLedger


MODEL = 'mockllm/model'


def metadata(seed=7):
    return {'seat_map': {'P1': MODEL, 'P2': MODEL}, 'seed': seed,
            'rounds': 1, 'actions_per_round': 2,
            'model_configs': {MODEL: {'max_tokens': 100}},
            'sampling_instruments': {MODEL: task.sampling_instrument(MODEL, {'max_tokens': 100})}}


def usage(cost=.01):
    return {'total_cost': cost, 'served_provider': 'Offline',
            'prompt_tokens': 10, 'completion_tokens': 10,
            'cache_read_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 0}


def response(seat):
    return json.dumps({'action': 'pull', 'beneficiary': 'P2' if seat == 'P1' else 'P1'})


def setup_run(tmp_path, seeds=(7,)):
    manifest = {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
                'models': {MODEL: {
                    'model': MODEL, 'provider_tag': 'offline', 'provider_name': 'Offline',
                    'context_length': 1000, 'max_completion_tokens': 100,  # equals the instrument's max_tokens
                    'supported_parameters': ['max_tokens'], 'evidence': 'offline fixture',
                    # 1000 ctx * (.0001 prompt + .00001 completion) = .11 per request bound.
                    'price_ceiling': {'prompt': '.0001', 'completion': '.00001',
                                      'input_cache_read': '0', 'input_cache_write': '0', 'request': '0'}}}}
    manifest_path = tmp_path / 'routes.json'
    manifest_path.write_text(json.dumps(manifest))
    cfg = {'log_dir': str(tmp_path), 'budget': {'max_estimated_usd': 10},
           'arms': [{'name': 'arm', 'task': 'lineup', 'seats': [MODEL, MODEL]}],
           'native': {'max_concurrent_episodes': len(seeds)},
           'spending': {'ledger': str(tmp_path / 'spending.sqlite3'), 'cap_usd': 10,
                        'manifest': str(manifest_path)}}
    config_path = tmp_path / 'config.yaml'
    # JSON is also valid YAML and keeps this fixture independent of a YAML writer.
    config_path.write_text(json.dumps(cfg))
    tasks = [('arm', SimpleNamespace(dataset=[
        SimpleNamespace(id=str(seed), metadata=metadata(seed)) for seed in seeds]))]
    estimates = {'arms': [{'arm': 'arm', 'usd': len(seeds), 'episodes': len(seeds)}]}
    return cfg, tasks, estimates, config_path


def execute(tmp_path, tasks, estimates, cfg):
    with Journal(tmp_path / 'run.sqlite3') as journal:
        return asyncio.run(runtime.run_tasks(journal, tasks, estimates, cfg))


def resolve_cli(monkeypatch, config_path, *arguments):
    monkeypatch.setattr(run.sys, 'argv', ['run', str(config_path), *arguments])
    run.main()


def test_empty_response_known_overrun_stops_shared_ledger(tmp_path):
    async def generate(messages):
        return None, {**usage(2), 'error': 'provider returned no action'}

    async def execute_episode():
        with Journal(tmp_path / 'run.sqlite3') as journal, \
                SpendingLedger(tmp_path / 'spending.sqlite3', 2) as ledger:
            bound = SimpleNamespace(maximum_usd=Decimal('1'), check_usage=lambda _: None)
            with pytest.raises(RunPaused):
                await runtime.run_episode(journal, 'one', metadata(), lambda _: generate,
                    reserve_usd=1, budget_usd=10, spending=ledger,
                    request_bounds={MODEL: bound}, spending_namespace='test')
            assert journal.summary()['reported_cost_usd'] == 2
            assert journal.summary()['uncertain_calls'] == 1
            assert ledger.summary()['reported_usd'] == '2'
            assert ledger.summary()['stopped'] is True
            assert ledger.summary()['held_usd'] == '0'

        # Another process cannot spend the apparently unused half of the cap.
        with SpendingLedger(tmp_path / 'spending.sqlite3', 2) as ledger:
            with pytest.raises(RunPaused):
                ledger.reserve('another-run', 1)

    asyncio.run(execute_episode())


def test_paused_game_keeps_evidence_while_sibling_finishes(tmp_path, monkeypatch):
    """A lost response pauses its own game only; the sibling completes on the same ledger."""
    cfg, tasks, estimates, _ = setup_run(tmp_path, seeds=(7, 8))
    entered = set()
    both_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    sibling_calls = []
    cleanup_statuses = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            entered.add(seed)
            if len(entered) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 1)
            if seed == 7:
                raise TimeoutError('response lost')
            sibling_calls.append(kwargs['seat'])
            await asyncio.wait_for(cleanup_started.wait(), 1)
            return response(kwargs['seat']), usage()

        async def close():
            if seed == 7:
                cleanup_statuses.append(json.loads((tmp_path / 'status.json').read_text())['episodes'])
                cleanup_started.set()
                await asyncio.sleep(.05)

        generate.aclose = close
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(sibling_calls) > 1, 'the sibling game keeps dispatching after the pause'
    assert cleanup_statuses and cleanup_statuses[0].get('paused') == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['episodes'] == {'paused': 1, 'complete': 1}
        assert journal.summary()['reported_cost_usd'] == pytest.approx(.01 * len(sibling_calls))
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['reported_usd'] == str(Decimal('0.01') * len(sibling_calls))
        assert ledger.summary()['calls_by_status'] == {'held': 1, 'settled': len(sibling_calls)}


def unserved(status):
    import httpx, openai
    request = httpx.Request('POST', 'https://openrouter.ai/api/v1/chat/completions')
    return openai.APIStatusError(f'Error code: {status} - provider returned error',
                                 response=httpx.Response(status, request=request), body=None)


def test_unserved_provider_error_is_retried_without_a_bill(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            if len(attempts) == 1:
                raise unserved(429)
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    execute(tmp_path, tasks, estimates, cfg)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['episodes'] == {'complete': 1}
        assert journal.summary()['uncertain_calls'] == 0
        actions = [row['action'] for row in journal.db.execute('SELECT action FROM resolutions')]
        assert actions == ['confirmed_unbilled']
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['calls_by_status'] == {'settled': len(attempts) - 1}
        assert ledger.summary()['held_usd'] == '0'


def test_error_body_without_completion_keeps_unknown_bill(tmp_path, monkeypatch):
    class OpenAIResponseError(Exception):  # Inspect's class name for a 200 whose body is an error object.
        def __init__(self, code, message):
            super().__init__(f'{code}: {message}'); self.code, self.message = code, message

    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            if len(attempts) == 1:
                raise OpenAIResponseError('server_error', 'The server had an error processing your request')
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused, match='uncertain'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(attempts) == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['uncertain_calls'] == 1
        assert journal.db.execute('SELECT count(*) FROM resolutions').fetchone()[0] == 0
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0.11'


def test_connect_timeout_is_retried_as_unserved(tmp_path, monkeypatch):
    import httpx, openai
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            if len(attempts) == 1:
                request = httpx.Request('POST', 'https://openrouter.ai/api/v1/chat/completions')
                try:
                    raise httpx.ConnectTimeout('timed out', request=request)
                except httpx.ConnectTimeout as inner:
                    raise openai.APITimeoutError(request=request) from inner
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    execute(tmp_path, tasks, estimates, cfg)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['episodes'] == {'complete': 1}
        evidence = [row['evidence'] for row in journal.db.execute('SELECT evidence FROM resolutions')]
        assert evidence and evidence[0].startswith('connection failure ConnectTimeout')


def test_empty_response_without_usage_keeps_unknown_bill(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            if len(attempts) == 1:
                return '', {'served_provider': 'Offline', 'served_model': 'm', 'finish_reason': 'stop',
                            'prompt_tokens': None, 'completion_tokens': None, 'total_cost': None, 'provider_usage': {}}
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused, match='missing provider bill'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(attempts) == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['uncertain_calls'] == 1
        assert journal.db.execute('SELECT count(*) FROM resolutions').fetchone()[0] == 0
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0.11'


def test_queued_games_still_start_after_a_sibling_parks_on_an_uncertain_call(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path, seeds=(7, 8, 9))
    cfg['native']['max_concurrent_episodes'] = 1  # games run one after another
    started = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            started.append(seed)
            if seed == 7:
                raise TimeoutError('response lost')  # billing unknown: this game parks
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    assert sorted(set(started)) == [7, 8, 9], 'later games must still start'
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['episodes'] == {'paused': 1, 'complete': 2}
        assert journal.summary()['uncertain_calls'] == 1


def test_journal_uses_wal_with_a_long_busy_timeout(tmp_path):
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert (tmp_path / 'run.sqlite3').exists()


def test_openrouter_error_code_alone_keeps_unknown_bill(tmp_path, monkeypatch):
    class OpenRouterError(Exception):  # Inspect's class for an OpenRouter error body (code outside 408/500/502/504)
        def __init__(self, response):
            super().__init__(f"Error {response['code']} - {response['message']}"); self.response = response

    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            if len(attempts) == 1:
                raise OpenRouterError({'code': 503, 'message': 'Overloaded', 'metadata': {'error_type': 'provider_overloaded'}})
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused, match='uncertain'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(attempts) == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['uncertain_calls'] == 1
        assert journal.db.execute('SELECT count(*) FROM resolutions').fetchone()[0] == 0


def test_unserved_retry_loop_honours_a_dispatch_stop(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    attempts = []
    stop = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            stop.append(True)  # the supervisor asks for a stop while this call is being retried
            raise unserved(429)
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        with pytest.raises(RunPaused, match='dispatch stopped'):
            asyncio.run(runtime.run_tasks(journal, tasks, estimates, cfg, dispatch_paused=lambda: bool(stop)))
    assert len(attempts) == 1, 'no further attempt once dispatch is stopped'
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0', 'the rejected request was confirmed unbilled before the stop'


def test_exhausted_unserved_retries_pause_with_hold_retained(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    monkeypatch.setattr(runtime, 'UNSERVED_BACKOFF_S', 0)
    monkeypatch.setattr(runtime, 'UNSERVED_RETRIES', 2)
    attempts = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            attempts.append(kwargs['seat'])
            raise unserved(503)
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused, match='uncertain'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(attempts) == 3
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['episodes'] == {'paused': 1}
        assert journal.summary()['uncertain_calls'] == 1
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['calls_by_status'] == {'held': 1}


def test_orphan_hold_has_explicit_cli_resolution(tmp_path, monkeypatch):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    dispatched = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            dispatched.append(messages)
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        with monkeypatch.context() as failure:
            failure.setattr(journal, 'begin_call', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
            with pytest.raises(RunPaused):
                asyncio.run(runtime.run_tasks(journal, tasks, estimates, cfg))
        assert journal.summary()['calls'] == 0 and dispatched == []
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        key = ledger.db.execute('SELECT call_key FROM spending_calls').fetchone()['call_key']
        call_id = key.split(':', 1)[1]
        assert ledger.summary()['calls_by_status'] == {'held': 1}

    resolve_cli(monkeypatch, config_path, '--resolve-reservation', call_id,
                '--confirm-unbilled', '--evidence', 'No journal row exists; request dispatch never began')
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0'
        assert ledger.summary()['unbilled_resolutions'] == 1
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    assert dispatched


def test_missing_bill_can_be_reconciled_without_repeating_response(tmp_path, monkeypatch):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(json.dumps(messages, sort_keys=True))
            return response(kwargs['seat']), usage(None if len(requests) == 1 else .01)
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused, match='missing provider bill'):
        execute(tmp_path, tasks, estimates, cfg)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        call_id = journal.db.execute('SELECT id FROM calls').fetchone()['id']
        recovered = {'response': journal.call(call_id)['response'], 'usage': usage(.02)}
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps(recovered))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Provider invoice reports USD0.02')
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    assert requests.count(requests[0]) == 1
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0'
        assert Decimal(ledger.summary()['reported_usd']) == Decimal('.02') + Decimal('.01') * (len(requests) - 1)


def test_confirmed_unbilled_crash_gap_reconciles_before_retry(tmp_path, monkeypatch):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(messages)
            if len(requests) == 1:
                raise TimeoutError('no response')
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        call_id = journal.db.execute('SELECT id FROM calls').fetchone()['id']
    # The operator decision commits to the journal, then the process loses the ledger write.
    with monkeypatch.context() as failure:
        failure.setattr(SpendingLedger, 'confirm_unbilled', lambda *args, **kwargs:
                        (_ for _ in ()).throw(OSError('ledger unavailable')))
        with pytest.raises(OSError, match='ledger unavailable'):
            resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                        '--confirm-unbilled', '--evidence', 'Provider confirms the request was unbilled')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.db.execute('SELECT status FROM calls WHERE id=?', (call_id,)).fetchone()['status'] == 'retryable'
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['calls_by_status'] == {'held': 1}
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0'
        assert ledger.summary()['unbilled_resolutions'] == 1
        assert Decimal(ledger.summary()['reported_usd']) == Decimal('.01') * (len(requests) - 1)


def test_fresh_verification_preserves_frozen_bounds_and_paid_prefix(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(json.dumps(messages, sort_keys=True))
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        with monkeypatch.context() as failure:
            failure.setattr(journal, 'save_event', lambda *args:
                            (_ for _ in ()).throw(RuntimeError('event write interrupted')))
            with pytest.raises(RuntimeError, match='event write interrupted'):
                asyncio.run(runtime.run_tasks(journal, tasks, estimates, cfg))
        original_id = journal.db.execute('SELECT id FROM episodes').fetchone()['id']
    assert len(requests) == 1

    future = datetime.now(timezone.utc) + timedelta(hours=25)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return future if tz else future.replace(tzinfo=None)

    monkeypatch.setattr(dispatch, 'datetime', Later)
    with pytest.raises(ValueError, match='fresh'):
        execute(tmp_path, tasks, estimates, cfg)
    manifest_path = Path(cfg['spending']['manifest'])
    refreshed = json.loads(manifest_path.read_text())
    refreshed['verified_at'] = future.isoformat()
    refreshed['models'][MODEL]['evidence'] = 'A new public catalogue check confirms the same limits'
    manifest_path.write_text(json.dumps(refreshed))
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    assert requests.count(requests[0]) == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert [row['id'] for row in journal.db.execute('SELECT id FROM episodes')] == [original_id]


def test_programmatic_paid_run_requires_spending_guard(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)
    cfg.pop('spending')
    tasks[0][1].dataset[0].metadata['seat_map'] = {'P1': 'gpt-5.5', 'P2': 'gpt-5.5'}
    monkeypatch.setattr(runtime.native_adapter, 'build_generator', lambda *args, **kwargs:
                        pytest.fail('a paid client must not be built without its spending guard'))
    with pytest.raises(RunPaused, match='spending'):
        execute(tmp_path, tasks, estimates, cfg)


def test_actual_gpt_mapping_obeys_strict_bound_with_offline_http(monkeypatch):
    native = runtime.native_adapter
    actual_get_model = native.get_model
    captured = []
    clients = []
    model_config = {'max_tokens': 123, 'reasoning_effort': 'none'}
    manifest = {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
                'models': {'gpt-5.5': {
                    'model': 'openrouter/openai/gpt-5.5', 'provider_tag': 'openai',
                    'provider_name': 'OpenAI', 'context_length': 10000,
                    'max_completion_tokens': 1000, 'evidence': 'offline endpoint fixture',
                    'reasoning': {'mandatory': False, 'supported_efforts': ['none', 'high']},
                    'supported_parameters': ['max_tokens', 'reasoning', 'seed'],
                    'price_ceiling': {'prompt': '.00001', 'completion': '.00001',
                                      'input_cache_read': '.000001', 'input_cache_write': '0', 'request': '0'}}}}
    bound = dispatch.RequestBound(manifest, 'gpt-5.5', model_config)

    async def execute_request():
        async def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                'id': 'gen-offline', 'model': 'openai/gpt-5.5', 'provider': 'OpenAI',
                'object': 'chat.completion', 'created': 1,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hello'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1000, 'completion_tokens': 10, 'total_tokens': 1010,
                          'cost': .0123, 'prompt_tokens_details': {'cached_tokens': 600},
                          'completion_tokens_details': {'reasoning_tokens': 0}}})

        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        def build(*args, **kwargs):
            model = actual_get_model(*args, api_key='offline-dummy-key', http_client=http, **kwargs)
            clients.append(model.api.client)
            return model

        monkeypatch.setattr(native, 'get_model', build)
        generate = native.build_generator('gpt-5.5', 41, model_config=model_config, request_bound=bound)
        try:
            return await generate([{'role': 'system', 'content': 'Game rules'},
                                   {'role': 'user', 'content': 'Choose an action'}])
        finally:
            await generate.aclose()

    completion, billed = asyncio.run(execute_request())
    assert completion == 'hello' and len(captured) == 1
    body = captured[0]
    bound.check_request(json.dumps(body))
    bound.check_usage(billed)
    limits = [body[key] for key in ('max_tokens', 'max_completion_tokens') if key in body]
    assert limits and all(limit == 123 for limit in limits)
    assert body['seed'] == 41 and body['reasoning'] == {'effort': 'none'}
    assert body['provider'] == bound.routing and 'temperature' not in body
    assert json.loads(billed['provider_request_body']) == body
    assert billed['total_cost'] == .0123 and billed['cost_source'] == 'openrouter_response_usage_cost'
    assert billed['prompt_tokens'] == 400 and billed['cache_read_tokens'] == 600
    assert billed['http_requests'] == 1 and clients[0].max_retries == 0


def test_refreshed_manifest_cannot_change_bound_before_dispatch(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)

    def factory(value, seed, **kwargs):
        async def generate(messages):
            return response(kwargs['seat']), usage()
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    manifest_path = Path(cfg['spending']['manifest'])
    changed = json.loads(manifest_path.read_text())
    changed['verified_at'] = datetime.now(timezone.utc).isoformat()
    changed['models'][MODEL]['price_ceiling']['prompt'] = '.0002'
    manifest_path.write_text(json.dumps(changed))
    monkeypatch.setattr(runtime.native_adapter, 'build_generator', lambda *args, **kwargs:
                        pytest.fail('changed bounds must fail before any provider client is built'))
    with pytest.raises(RunPaused, match='route or bound changed'):
        execute(tmp_path, tasks, estimates, cfg)


def test_null_response_commit_survives_crash_before_any_followup_write(tmp_path, monkeypatch):
    dispatched = []
    bound = SimpleNamespace(maximum_usd=Decimal('1'), check_usage=lambda _: None)

    def factory(seat):
        async def generate(messages):
            dispatched.append(messages)
            return None, {**usage(.25), 'error': 'empty provider response'}
        return generate

    async def crash_at_commit():
        with Journal(tmp_path / 'run.sqlite3') as journal, \
                SpendingLedger(tmp_path / 'spending.sqlite3', 10) as ledger:
            finish = journal.finish_call

            def commit_then_crash(*args, **kwargs):
                finish(*args, **kwargs)
                raise SystemExit('process stopped after committing response evidence')

            monkeypatch.setattr(journal, 'finish_call', commit_then_crash)
            with pytest.raises(RunPaused):
                await runtime.run_episode(journal, 'one', metadata(), factory,
                    reserve_usd=1, budget_usd=10, spending=ledger,
                    request_bounds={MODEL: bound}, spending_namespace='test')
            row = journal.db.execute('SELECT status,error,reported FROM calls').fetchone()
            assert row['status'] == 'uncertain'
            assert row['error'] == 'empty provider response'
            assert row['reported'] == .25
            assert journal.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 0
            assert ledger.summary()['held_usd'] == '1'

    asyncio.run(crash_at_commit())
    assert len(dispatched) == 1

    async def recover():
        with Journal(tmp_path / 'run.sqlite3') as journal, \
                SpendingLedger(tmp_path / 'spending.sqlite3', 10) as ledger:
            with pytest.raises(RunPaused, match='uncertain'):
                await runtime.run_episode(journal, 'one', metadata(), factory,
                    reserve_usd=1, budget_usd=10, spending=ledger,
                    request_bounds={MODEL: bound}, spending_namespace='test')
            assert len(dispatched) == 1
            assert journal.summary()['reported_cost_usd'] == .25
            assert ledger.summary()['held_usd'] == '1'
            row = journal.db.execute('SELECT id,seat FROM calls').fetchone()
            journal.resolve_uncertain(row['id'], evidence='Provider recovered the original action and bill',
                                      response=response(row['seat']), usage=usage(.25))

            def remaining(seat):
                async def generate(messages):
                    dispatched.append(messages)
                    return response(seat), usage()
                return generate

            record = await runtime.run_episode(journal, 'one', metadata(), remaining,
                reserve_usd=1, budget_usd=10, spending=ledger,
                request_bounds={MODEL: bound}, spending_namespace='test')
            assert all(call['response'] is not None for event in record['events'] for call in event['calls'])
            assert ledger.summary()['held_usd'] == '0'
            assert Decimal(ledger.summary()['reported_usd']) == Decimal('.25') + Decimal('.01') * (len(dispatched) - 1)

    asyncio.run(recover())


@pytest.mark.parametrize('old_result', [None, {'response': None, 'usage': usage(.25)}])
def test_historical_complete_null_call_cannot_replay(tmp_path, old_result):
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.start_episode('one', metadata(), 1, 10)
        messages = [{'role': 'user', 'content': 'Choose an action'}]
        call_id, _ = journal.begin_call('one', 'P1', 0, messages)
        # Reproduce a row made by the old two-write state transition.
        journal.db.execute("UPDATE calls SET status='complete',result=?,reported=.25 WHERE id=?",
                           (pack(old_result) if old_result is not None else None, call_id))
        with pytest.raises(RunPaused, match='uncertain|missing|empty'):
            journal.begin_call('one', 'P1', 0, messages)
        assert journal.summary()['uncertain_calls'] == 1
        assert journal.summary()['reported_cost_usd'] == .25
        assert journal.episode('one')['reserved'] == 1
        assert journal.db.execute('SELECT result FROM calls WHERE id=?', (call_id,)).fetchone()['result'] == (
            pack(old_result) if old_result is not None else None)
        with pytest.raises(RunPaused):
            journal.complete('one', {})


def known_bill_before_settlement(tmp_path, billed):
    """Stop at the durable journal write, before shared-ledger settlement."""
    cfg, _, _, config_path = setup_run(tmp_path)
    with Journal(tmp_path / 'run.sqlite3') as journal, \
            SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        journal.bind('spending', {**cfg['spending'],
                                  'manifest': json.loads(Path(cfg['spending']['manifest']).read_text())})
        journal.bind('spending_namespace', 'test')
        journal.start_episode('one', metadata(), 1, 10)
        call_id, _ = journal.begin_call('one', 'P1', 0, [])
        ledger.reserve(f'test:{call_id}', 1)
        journal.finish_call(call_id, None, {**usage(billed), 'error': 'empty response'}, None)
        saved_row = dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone())
    return cfg, config_path, call_id, saved_row


@pytest.mark.parametrize('billed,recovered', [(.5, None), (.5, 0), (.5, .75),
                                            (0, None), (0, .25), (0, False)])
def test_recovery_cannot_drop_or_change_a_known_bill(tmp_path, monkeypatch, billed, recovered):
    cfg, config_path, call_id, saved_row = known_bill_before_settlement(tmp_path, billed)
    recovered_path = tmp_path / 'recovered.json'
    recovered_usage = usage(recovered)
    if recovered is None:
        recovered_usage.pop('total_cost')
    recovered_path.write_text(json.dumps({'response': response('P1'), 'usage': recovered_usage}))
    with pytest.raises(ValueError, match='known|bill'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--response-file', str(recovered_path), '--evidence', 'Candidate recovery evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == saved_row
        assert journal.db.execute('SELECT COUNT(*) FROM resolutions').fetchone()[0] == 0
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '1' and ledger.summary()['reported_usd'] == '0'
        assert ledger.summary()['unbilled_resolutions'] == 0


def test_unbilled_resolution_cannot_erase_positive_journal_bill(tmp_path, monkeypatch):
    cfg, config_path, call_id, saved_row = known_bill_before_settlement(tmp_path, .5)
    with pytest.raises(ValueError, match='known|bill'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--confirm-unbilled', '--evidence', 'Candidate unbilled evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == saved_row
        assert journal.db.execute('SELECT COUNT(*) FROM resolutions').fetchone()[0] == 0
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '1' and ledger.summary()['reported_usd'] == '0'


@pytest.mark.parametrize('billed', [0, .5])
def test_matching_recovered_bill_settles_original_hold(tmp_path, monkeypatch, billed):
    cfg, config_path, call_id, _ = known_bill_before_settlement(tmp_path, billed)
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P1'), 'usage': usage(billed)}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Provider recovered the same bill and response')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.call(call_id)['response'] == response('P1')
        assert journal.call(call_id)['usage']['total_cost'] == billed
        assert journal.summary()['uncertain_calls'] == 0
        assert journal.db.execute('SELECT COUNT(*) FROM resolutions').fetchone()[0] == 1
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0'
        assert Decimal(ledger.summary()['reported_usd']) == Decimal(str(billed))


def test_partial_bill_recovery_preserves_saved_provider_and_nested_usage(tmp_path, monkeypatch):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(json.dumps(messages, sort_keys=True))
            measured = usage(None if len(requests) == 1 else .01)
            measured['provider_usage'] = {'prompt_tokens': 10, 'completion_tokens': 10,
                                          'cost': measured['total_cost']}
            return response(kwargs['seat']), measured
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        call_id = journal.db.execute('SELECT id FROM calls').fetchone()['id']
        saved_response = journal.call(call_id)['response']
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': saved_response,
        'usage': {'total_cost': .02, 'provider_usage': {'cost': .02}}}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Provider recovered the missing invoice amount')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        recovered_usage = journal.call(call_id)['usage']
        assert recovered_usage.get('served_provider') == 'Offline'
        assert recovered_usage.get('prompt_tokens') == 10
        assert recovered_usage['provider_usage'] == {'prompt_tokens': 10, 'completion_tokens': 10, 'cost': .02}
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    assert requests.count(requests[0]) == 1
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['stopped'] is False


@pytest.mark.parametrize('supplied', [
    {'total_cost': .01},
    {**usage(.01), 'served_provider': 'Other'},
    {**usage(.01), 'prompt_tokens': 1001},
    {**usage(.01), 'completion_tokens': 1001},  # beyond the context window
    {**usage(.01), 'provider_usage': {'prompt_tokens': 500}},
    {**usage(.01), 'provider_usage': {'completion_tokens': 500}},
    {**usage(.01), 'provider_usage': {'cost': .02}},
    usage(2),
])
def test_bad_recovery_keeps_original_evidence_and_can_be_corrected(tmp_path, monkeypatch, supplied):
    cfg, config_path, call_id, _ = known_bill_before_settlement(tmp_path, None)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.finish_call(call_id, None, {'total_cost': None}, None)
        before = dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone())
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P1'), 'usage': supplied}))
    with pytest.raises(ValueError, match='recovery|evidence|provider|tokens'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--response-file', str(recovered_path), '--evidence', 'Incomplete candidate evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == before
        assert journal.db.execute('SELECT COUNT(*) FROM resolutions').fetchone()[0] == 0
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '1'
        assert ledger.summary()['reported_usd'] == '0' and ledger.summary()['stopped'] is False
    recovered_path.write_text(json.dumps({'response': response('P1'), 'usage': usage(.01)}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Complete provider evidence')
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['reported_usd'] == '0.01' and ledger.summary()['stopped'] is False


def test_recovery_cannot_replace_saved_nonbilling_evidence(tmp_path, monkeypatch):
    cfg, config_path, call_id, before = known_bill_before_settlement(tmp_path, .01)
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P1'),
        'usage': {**usage(.01), 'prompt_tokens': 20}}))
    with pytest.raises(ValueError, match='conflict|contradict|saved'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--response-file', str(recovered_path), '--evidence', 'Contradictory candidate evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == before
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '1' and ledger.summary()['stopped'] is False


def test_recovery_uses_frozen_sampling_even_after_manifest_expiry(tmp_path, monkeypatch):
    cfg, config_path, call_id, _ = known_bill_before_settlement(tmp_path, .01)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        policy = json.loads(journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()['value'])
        policy['manifest']['verified_at'] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        journal.db.execute("UPDATE meta SET value=? WHERE key='spending'", (json.dumps(policy),))
    monkeypatch.setattr(dispatch.task, 'sampling_instrument', lambda *args, **kwargs:
                        pytest.fail('recovery must use recorded sampling, not the current registry'))
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P1'), 'usage': {'total_cost': .01}}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Provider recovered this old request')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.call(call_id)['usage'].get('served_provider') == 'Offline'
    # Expired metadata cannot authorize a new dispatch.
    with pytest.raises(ValueError, match='fresh'):
        dispatch.RequestBound(policy['manifest'], MODEL)


@pytest.mark.parametrize('missing_provider', [None, ''])
def test_missing_live_usage_evidence_pauses_without_permanent_stop(tmp_path, monkeypatch, missing_provider):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(messages)
            measured = usage()
            if len(requests) == 1:
                measured['served_provider'] = missing_provider
            return response(kwargs['seat']), measured
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['stopped'] is False
        assert ledger.summary()['reported_usd'] == '0.01'
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.summary()['uncertain_calls'] == 1
        call_id = journal.db.execute('SELECT id FROM calls').fetchone()['id']
        saved_response = journal.call(call_id)['response']
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': saved_response,
        'usage': {'total_cost': .01, 'served_provider': 'Offline'}}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                '--response-file', str(recovered_path), '--evidence', 'Provider supplied its missing identity')
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}


def test_verified_live_provider_violation_still_stops_shared_dispatch(tmp_path, monkeypatch):
    cfg, tasks, estimates, _ = setup_run(tmp_path)

    def factory(value, seed, **kwargs):
        async def generate(messages):
            return response(kwargs['seat']), {**usage(), 'served_provider': 'Other'}
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with pytest.raises(RunPaused):
        execute(tmp_path, tasks, estimates, cfg)
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['stopped'] is True
        assert ledger.summary()['reported_usd'] == '0.01'


@pytest.mark.parametrize('patch', [{'provider': {}}, {'max_tokens': 101}, {'seed': 99}, {'temperature': .123}])
def test_recovery_checks_frozen_wire_controls_before_mutation(tmp_path, monkeypatch, patch):
    cfg, config_path, call_id, before = known_bill_before_settlement(tmp_path, .01)
    manifest = json.loads(Path(cfg['spending']['manifest']).read_text())
    bound = dispatch.RequestBound(manifest, MODEL, {'max_tokens': 100})
    body = {'model': MODEL, 'provider': bound.routing, 'max_tokens': 100,
            'temperature': bound.instrument['temperature'], 'messages': []}
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P1'),
        'usage': {'total_cost': .01, 'provider_request_body': json.dumps({**body, **patch})}}))
    with pytest.raises(ValueError, match='recovery|request|sampling'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--response-file', str(recovered_path), '--evidence', 'Candidate request evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == before
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '1' and ledger.summary()['stopped'] is False


def test_recovery_cannot_replace_a_saved_response(tmp_path, monkeypatch):
    cfg, config_path, call_id, _ = known_bill_before_settlement(tmp_path, .01)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.finish_call(call_id, response('P1'), usage(.01), None)
        journal.uncertain(call_id, 'provider identity evidence under review')
        before = dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone())
    recovered_path = tmp_path / 'recovered.json'
    recovered_path.write_text(json.dumps({'response': response('P2'), 'usage': {'total_cost': .01}}))
    with pytest.raises(ValueError, match='saved response'):
        resolve_cli(monkeypatch, config_path, '--resolve-call', call_id,
                    '--response-file', str(recovered_path), '--evidence', 'Candidate response evidence')
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert dict(journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()) == before


def paused_cap_run(tmp_path, monkeypatch, *, launcher=execute):
    cfg, tasks, estimates, config_path = setup_run(tmp_path)
    cfg['spending']['cap_usd'] = .12
    config_path.write_text(json.dumps(cfg))
    requests = []

    def factory(value, seed, **kwargs):
        async def generate(messages):
            requests.append(messages)
            return response(kwargs['seat']), usage(.02)
        return generate

    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with SpendingLedger(cfg['spending']['ledger'], .12) as ledger:
        ledger.reserve('other-run:unknown', Decimal('.01'))
    with pytest.raises(RunPaused, match='cap'):
        launcher(tmp_path, tasks, estimates, cfg)
    assert len(requests) == 1
    return cfg, tasks, estimates, config_path, requests


def test_audited_cap_chain_resumes_same_episode_and_keeps_unknown_hold(tmp_path, monkeypatch):
    cfg, tasks, estimates, _, requests = paused_cap_run(tmp_path, monkeypatch)
    with Journal(tmp_path / 'run.sqlite3') as journal:
        first_episode = journal.db.execute('SELECT id,status FROM episodes').fetchone()
        assert first_episode['status'] == 'paused'
        assert json.loads((tmp_path / 'status.json').read_text())['episodes'] == {'paused': 1}
        original = journal.call(journal.db.execute('SELECT id FROM calls').fetchone()['id'])
    with SpendingLedger(cfg['spending']['ledger'], .12) as ledger:
        ledger.increase_cap(.12, .2, evidence='First test approval', confirm_no_active_runs=True)
        ledger.increase_cap(.2, .4, evidence='Second test approval', confirm_no_active_runs=True)
        audits = ledger.cap_increases()
    cfg['spending']['cap_usd'] = .4
    assert execute(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    assert len(requests) == 4
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert journal.db.execute('SELECT id FROM episodes').fetchone()['id'] == first_episode['id']
        assert journal.call(original['call_id']) == original
        assert json.loads(journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()['value'])['cap_usd'] == .4
        assert json.loads(journal.db.execute("SELECT value FROM meta WHERE key='spending_cap_increases'").fetchone()['value']) == audits
    with SpendingLedger(cfg['spending']['ledger'], .4) as ledger:
        assert ledger.summary()['held_usd'] == '0.01'
        assert ledger.summary()['reported_usd'] == '0.08'
        assert ledger.reserve('other-run:unknown', Decimal('.01'))['dispatch_allowed'] is False


@pytest.mark.parametrize('change', ['missing_audit', 'broken_chain', 'route', 'ledger'])
def test_resume_rejects_unproved_cap_or_changed_frozen_controls(tmp_path, monkeypatch, change):
    cfg, tasks, estimates, _, requests = paused_cap_run(tmp_path, monkeypatch)
    with SpendingLedger(cfg['spending']['ledger'], .12) as ledger:
        if change == 'missing_audit':
            ledger.db.execute("UPDATE spending_policy SET cap_usd='0.4',cap_nanos='400000000' WHERE id=1")
        else:
            ledger.increase_cap(.12, .4, evidence='Test approval', confirm_no_active_runs=True)
        if change == 'broken_chain':
            ledger.db.execute("UPDATE spending_cap_increases SET old_cap_usd='0.2'")
    cfg['spending']['cap_usd'] = .4
    if change == 'route':
        path = Path(cfg['spending']['manifest'])
        changed = json.loads(path.read_text())
        changed['models'][MODEL]['context_length'] = 999
        path.write_text(json.dumps(changed))
    if change == 'ledger':
        cfg['spending']['ledger'] = str(tmp_path / 'other.sqlite3')
    with pytest.raises(RunPaused, match='cap|route|ledger|bound'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(requests) == 1
    with Journal(tmp_path / 'run.sqlite3') as journal:
        assert json.loads(journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()['value'])['cap_usd'] == .12
    assert not (tmp_path / 'other.sqlite3').exists()


def test_cap_increase_does_not_clear_a_provider_stop_on_resume(tmp_path, monkeypatch):
    cfg, tasks, estimates, _, requests = paused_cap_run(tmp_path, monkeypatch)
    with SpendingLedger(cfg['spending']['ledger'], .12) as ledger:
        ledger.stop('Verified provider violation')
        ledger.increase_cap(.12, .4, evidence='Test approval', confirm_no_active_runs=True)
    cfg['spending']['cap_usd'] = .4
    with pytest.raises(RunPaused, match='stopped'):
        execute(tmp_path, tasks, estimates, cfg)
    assert len(requests) == 1
    with SpendingLedger(cfg['spending']['ledger'], .4) as ledger:
        assert ledger.summary()['held_usd'] == '0.01'
        assert ledger.summary()['stopped_reason'] == 'Verified provider violation'


@pytest.mark.parametrize('legacy_binding', [False, True])
def test_launch_accepts_only_audited_cap_change_and_keeps_source_config_bindings(tmp_path, monkeypatch, legacy_binding):
    frozen_calls = []
    version = {'source': 'source-v1', 'python': 'test-python-v1'}

    def freeze(config_path, cfg, *args, **kwargs):
        path = tmp_path / f'manifest-{len(frozen_calls)}.json'
        path.write_text(json.dumps({'source': {'tree_sha256': version['source'], 'git': {'commit': 'test'}},
            'runtime': {'python': version['python'], 'packages': {'test': '1'}},
            'input_config': {'resolved': cfg}}))
        frozen_calls.append(path)
        return SimpleNamespace(manifest_path=path)

    monkeypatch.setattr(run.run_artifacts, 'create_run_artifacts', freeze)
    monkeypatch.setattr(run, 'enforce_budget', lambda *args: None)

    def launch(tmp_path, tasks, estimates, cfg):
        return run.launch_native_run(tmp_path / 'config.yaml', cfg, tasks, estimates)

    cfg, tasks, estimates, _, requests = paused_cap_run(tmp_path, monkeypatch, launcher=launch)
    if legacy_binding:
        with Journal(tmp_path / 'run.sqlite3') as journal:
            journal.db.execute("DELETE FROM meta WHERE key='config'")
        # An external manifest is not a trusted replacement for the missing binding.
        cfg['budget']['max_estimated_usd'] = 20
        original = json.loads(frozen_calls[0].read_text())
        original['input_config']['resolved'] = cfg
        frozen_calls[0].write_text(json.dumps(original))
        previous = len(requests)
        with pytest.raises(RunPaused, match='saved.*config'):
            launch(tmp_path, tasks, estimates, cfg)
        assert len(requests) == previous
        return
    with SpendingLedger(cfg['spending']['ledger'], .12) as ledger:
        ledger.increase_cap(.12, .4, evidence='Test approval', confirm_no_active_runs=True)
    cfg['spending']['cap_usd'] = .4
    assert launch(tmp_path, tasks, estimates, cfg)['episodes'] == {'complete': 1}
    cfg['budget']['max_estimated_usd'] = 20
    with pytest.raises(RunPaused, match='config'):
        launch(tmp_path, tasks, estimates, cfg)
    cfg['budget']['max_estimated_usd'] = 10
    version['source'] = 'source-v2'
    with pytest.raises(RunPaused, match='source'):
        launch(tmp_path, tasks, estimates, cfg)
    version['source'] = 'source-v1'
    version['python'] = 'test-python-v2'
    with pytest.raises(RunPaused, match='installed_runtime'):
        launch(tmp_path, tasks, estimates, cfg)
    assert len(requests) == 4


def test_unknown_call_recovery_still_works_after_audited_cap_increase(tmp_path, monkeypatch):
    cfg, config_path, call_id, _ = known_bill_before_settlement(tmp_path, None)
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        ledger.increase_cap(10, 20, evidence='Test approval', confirm_no_active_runs=True)
        assert ledger.summary()['held_usd'] == '1'
    cfg['spending']['cap_usd'] = 20
    config_path.write_text(json.dumps(cfg))
    recovered = tmp_path / 'recovered.json'
    recovered.write_text(json.dumps({'response': response('P1'), 'usage': {'total_cost': .01}}))
    resolve_cli(monkeypatch, config_path, '--resolve-call', call_id, '--response-file', str(recovered),
                '--evidence', 'Recovered provider response and invoice')
    with SpendingLedger(cfg['spending']['ledger'], 20) as ledger:
        assert ledger.summary()['held_usd'] == '0'
        assert ledger.summary()['reported_usd'] == '0.01'
        assert ledger.summary()['calls'] == 1
