"""Fixed saved-prompt diagnostics: offline billing, provenance, and replay checks."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from delegated_influence import dispatch, game, task
from delegated_influence.journal import Journal, RunPaused, canonical
from delegated_influence.spending import SpendingLedger


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'pilot_diagnostics.py'
    spec = importlib.util.spec_from_file_location('pilot_diagnostics_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_plan(tmp_path):
    model = 'mockllm/model'
    manifest = {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
        'models': {model: {'model': model, 'provider_tag': 'offline', 'provider_name': 'Offline',
            'context_length': 1000, 'max_completion_tokens': 200,  # equals the diagnostic's max_tokens
            'supported_parameters': ['max_tokens', 'seed', 'temperature'], 'evidence': 'offline fixture',
            # 1000 ctx * (.0001 prompt + .00002 completion) = .12 per request bound.
            'price_ceiling': {'prompt': '.0001', 'completion': '.00002', 'input_cache_read': '0',
                              'input_cache_write': '0', 'request': '0'}}}}
    bound = dispatch.RequestBound(manifest, model, {'max_tokens': 100})
    metadata = {'seat_map': {'P1': model, 'P2': model}, 'seed': 7, 'rounds': 1,
        'actions_per_round': 2, 'punish_enabled': True,
        'sampling_instruments': {model: task.sampling_instrument(model, {'max_tokens': 100})},
        'spending_instrument': {model: {'route': bound.row, 'maximum_usd': str(bound.maximum_usd),
                                       'provider_routing': bound.routing}}}
    messages = [{'role': 'system', 'content': 'Original game rules'},
                {'role': 'user', 'content': 'P1 has one point. Choose one action.'}]
    wire = {'model': model, 'messages': messages, 'max_tokens': 100, 'seed': 7,
            'temperature': bound.instrument['temperature'], 'provider': bound.routing}
    source_path = tmp_path / 'source.sqlite3'
    with Journal(source_path) as source:
        source.start_episode('original', metadata, 1, 5)
        state = game.state_from_metadata(metadata)
        first = game.make_forced_pull_event('P2', 'P1', 1, 1)
        game.apply_event(state, first)
        source.save_event('original', 0, first)
        call_id, _ = source.begin_call('original', 'P1', 0, messages)
        source.save_request(call_id, json.dumps(wire))
        source.finish_call(call_id, '{"action":"pull","beneficiary":"P2"}', usage(), None)
        event = game.make_event(game.build_observation(state, 'P1', 1, 2, 0),
                               {'action': 'pull', 'beneficiary': 'P2'})
        event['calls'] = [source.call(call_id)]
        source.save_event('original', 1, event)
    route_path = tmp_path / 'routes.json'
    route_path.write_text(json.dumps(manifest))
    plan = {'schema_version': 'fixed-request-diagnostic-v1', 'diagnostic_cap_usd': 5,
        'shared_ledger': str(tmp_path / 'spending.sqlite3'), 'shared_cap_usd': 1,
        'route_manifest': str(route_path), 'probes': [{
            'id': 'probe-200', 'pair_id': 'original', 'kind': 'output_limit', 'model': model,
            'seed': 7, 'max_tokens': 200, 'model_config': {'max_tokens': 200}, 'messages': messages,
            'source': {'journal': str(source_path), 'call_id': call_id, 'original_output_limit': 100,
                       'input_sha256': hashlib.sha256(canonical(messages).encode()).hexdigest()},
            'validation': {'seat': 'P1', 'metadata': metadata}}]}
    return plan, manifest, wire


def usage(cost=.01, provider='Offline'):
    return {'total_cost': cost, 'served_provider': provider, 'prompt_tokens': 10,
            'completion_tokens': 10, 'reasoning_tokens': 0, 'cache_read_tokens': 0, 'cache_write_tokens': 0}


def factory_for(plan, wire, *, measured=None, reply='{"action":"punish","target":"P2"}',
                attempts=None, fail=None, close_fail=False, before_send=None, wire_patch=None):
    def factory(*args, **kwargs):
        async def generate(messages):
            if before_send:
                before_send()
            generate.on_request(json.dumps({**wire, 'max_tokens': 200, **(wire_patch or {})}))
            if attempts is not None:
                attempts.append(messages)
            if fail:
                raise fail
            return reply, measured if measured is not None else usage()
        async def close():
            if close_fail:
                raise RuntimeError('client cleanup failed')
        generate.aclose = close
        return generate
    return factory


def test_reserve_precedes_send_response_precedes_settle_and_source_state_is_used(tmp_path, diagnostic, monkeypatch):
    plan, manifest, wire = fixture_plan(tmp_path)
    source_hash = hashlib.sha256(Path(plan['probes'][0]['source']['journal']).read_bytes()).hexdigest()
    attempts = []
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        def before_send():
            assert ledger.summary()['held_usd'] == '0.12'
            assert journal.summary()['pending_calls'] == 1
        settle = ledger.settle
        def check_settle(key, cost):
            assert journal.call('probe-200:probe:0')['response'] == '{"action":"punish","target":"P2"}'
            return settle(key, cost)
        monkeypatch.setattr(ledger, 'settle', check_settle)
        result = asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
            factory=factory_for(plan, wire, attempts=attempts, before_send=before_send)))
        assert result['calls'][0]['first_response_valid'] is True
        assert ledger.summary()['reported_usd'] == '0.01'
        assert ledger.summary()['held_usd'] == '0'
        assert len(attempts) == 1
        asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
            factory=lambda *args, **kwargs: pytest.fail('complete diagnostic must not dispatch twice')))
    assert hashlib.sha256(Path(plan['probes'][0]['source']['journal']).read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize('change', ['seed', 'messages', 'source_hash', 'seat', 'metadata'])
def test_plan_checks_original_evidence_before_reserving(tmp_path, diagnostic, change):
    plan, manifest, wire = fixture_plan(tmp_path)
    probe = plan['probes'][0]
    if change == 'seed': probe['seed'] = 8
    if change == 'messages':
        probe['messages'][1]['content'] = 'Changed prompt'
        probe['source']['input_sha256'] = hashlib.sha256(canonical(probe['messages']).encode()).hexdigest()
    if change == 'source_hash': probe['source']['input_sha256'] = 'wrong'
    if change == 'seat': probe['validation']['seat'] = 'P2'
    if change == 'metadata': probe['validation']['metadata']['punish_enabled'] = False
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        with pytest.raises(ValueError):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
                factory=lambda *args, **kwargs: pytest.fail('invalid plan must not make a client')))
        assert ledger.summary()['calls'] == 0


def test_wire_non_output_drift_is_rejected_before_send(tmp_path, diagnostic):
    plan, manifest, wire = fixture_plan(tmp_path)
    sent = []
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
                factory=factory_for(plan, wire, attempts=sent, wire_patch={'temperature': .123})))
        assert sent == []
        assert ledger.summary()['held_usd'] == '0.12'


@pytest.mark.parametrize('failure', ['timeout', 'missing_bill', 'null_response', 'provider_violation'])
def test_uncertainty_holds_cost_and_violations_stop_dispatch(tmp_path, diagnostic, failure):
    plan, manifest, wire = fixture_plan(tmp_path)
    attempts = []
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger, factory=factory_for(plan, wire,
                attempts=attempts, measured=usage(None if failure == 'missing_bill' else .01,
                    'Other' if failure == 'provider_violation' else 'Offline'),
                reply=None if failure == 'null_response' else 'valid reply',
                fail=TimeoutError('response lost') if failure == 'timeout' else None)))
        assert ledger.summary()['held_usd'] == ('0.12' if failure in ('timeout', 'missing_bill') else '0')
        assert ledger.summary()['stopped'] is (failure == 'provider_violation')
        assert journal.episode('probe-200')['status'] == 'paused'
        with pytest.raises(RunPaused):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
                factory=lambda *args, **kwargs: pytest.fail('uncertain probe must not dispatch again')))
        assert len(attempts) == 1


def test_crash_after_response_commit_replays_without_another_call(tmp_path, diagnostic, monkeypatch):
    plan, manifest, wire = fixture_plan(tmp_path)
    attempts = []
    path = tmp_path / 'diagnostic' / 'run.sqlite3'
    with Journal(path) as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        finish = journal.finish_call
        def crash_after_commit(*args):
            finish(*args)
            raise SystemExit('simulated crash after durable response')
        monkeypatch.setattr(journal, 'finish_call', crash_after_commit)
        with pytest.raises(SystemExit):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
                factory=factory_for(plan, wire, attempts=attempts)))
        assert ledger.summary()['held_usd'] == '0.12'
    with Journal(path) as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        result = asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
            factory=lambda *args, **kwargs: pytest.fail('saved response must be replayed')))
        assert result['completed_requests'] == 1
        assert ledger.summary()['reported_usd'] == '0.01'
        assert len(attempts) == 1


def test_cleanup_failure_cannot_hide_completed_diagnostic(tmp_path, diagnostic):
    plan, manifest, wire = fixture_plan(tmp_path)
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        result = asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
            factory=factory_for(plan, wire, close_fail=True)))
        assert result['completed_requests'] == 1
        assert json.loads((journal.path.parent / 'results.json').read_text())['completed_requests'] == 1


def test_plan_and_routes_are_frozen_before_any_new_call(tmp_path, diagnostic):
    plan, manifest, wire = fixture_plan(tmp_path)
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        asyncio.run(diagnostic.execute(plan, manifest, journal, ledger, factory=factory_for(plan, wire)))
        changed = deepcopy(plan)
        changed['diagnostic_cap_usd'] = 6
        with pytest.raises(RunPaused, match='diagnostic_plan'):
            asyncio.run(diagnostic.execute(changed, manifest, journal, ledger))
        changed_manifest = deepcopy(manifest)
        changed_manifest['verified_at'] = datetime.now(timezone.utc).isoformat()
        with pytest.raises(RunPaused, match='diagnostic_routes'):
            asyncio.run(diagnostic.execute(plan, changed_manifest, journal, ledger))


@pytest.mark.parametrize('mismatch', ['ledger', 'cap'])
def test_execute_uses_only_the_declared_shared_ledger(tmp_path, diagnostic, mismatch):
    plan, manifest, wire = fixture_plan(tmp_path)
    path = tmp_path / 'other.sqlite3' if mismatch == 'ledger' else Path(plan['shared_ledger'])
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(path, 2 if mismatch == 'cap' else 1) as ledger:
        with pytest.raises(RunPaused, match='ledger|cap'):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger,
                factory=lambda *args, **kwargs: pytest.fail('wrong shared ledger must not create a provider')))
        assert ledger.summary()['calls'] == 0


def test_failed_client_setup_does_not_create_an_orphan_hold(tmp_path, diagnostic):
    plan, manifest, wire = fixture_plan(tmp_path)
    def broken(*args, **kwargs):
        raise ValueError('invalid client configuration')
    with Journal(tmp_path / 'diagnostic' / 'run.sqlite3') as journal, SpendingLedger(plan['shared_ledger'], 1) as ledger:
        with pytest.raises(ValueError, match='client'):
            asyncio.run(diagnostic.execute(plan, manifest, journal, ledger, factory=broken))
        assert ledger.summary()['calls'] == 0
        assert journal.summary()['calls'] == 0


def test_main_freezes_source_and_environment_before_dispatch(tmp_path, diagnostic, monkeypatch):
    plan, manifest, wire = fixture_plan(tmp_path)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with SpendingLedger(plan['shared_ledger'], 1):
        pass
    calls = []
    source_version = {'hash': 'original-source'}
    async def offline_execute(*args, **kwargs):
        calls.append(True)
    monkeypatch.setattr(diagnostic, 'execute', offline_execute)
    monkeypatch.setattr(diagnostic.artifacts, 'runtime_source_paths', lambda root: [])
    monkeypatch.setattr(diagnostic.artifacts, 'sha256_file', lambda path: source_version['hash'])
    monkeypatch.setattr(diagnostic.artifacts, '_snapshot_source', lambda root, paths, destination: destination.write_bytes(b'offline snapshot'))
    monkeypatch.setattr(diagnostic.sys, 'argv', ['pilot_diagnostics', str(path), '--run-dir', str(tmp_path / 'diagnostic'), '--execute'])
    diagnostic.main()
    source_version['hash'] = 'changed-source'
    with pytest.raises(RunPaused, match='source_files'):
        diagnostic.main()
    source_version['hash'] = 'original-source'
    monkeypatch.setattr(diagnostic.runtime, 'RUNTIME_ENVIRONMENT', {'python': 'changed-version'})
    with pytest.raises(RunPaused, match='runtime_environment'):
        diagnostic.main()
    assert calls == [True]
