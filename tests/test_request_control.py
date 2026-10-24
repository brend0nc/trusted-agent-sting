"""Shared admission and cooldown tests use independent database connections."""
import asyncio

import pytest

from delegated_influence.journal import RunPaused


def controller(path, **kwargs):
    from delegated_influence.request_control import RequestControl
    return RequestControl(path, **kwargs)


def test_shared_route_and_provider_limits_and_spacing(tmp_path):
    now = [100.0]
    options = {'route': {'max_in_flight': 1, 'min_interval_seconds': 2},
               'provider': {'max_in_flight': 2, 'min_interval_seconds': 1}}
    a = controller(tmp_path/'control.sqlite3', policy=options, clock=lambda: now[0])
    b = controller(tmp_path/'control.sqlite3', policy=options, clock=lambda: now[0])
    try:
        assert a.try_acquire('a', 'model1', 'provider') == 0
        assert b.try_acquire('b', 'model1', 'provider') > 0
        assert b.try_acquire('c', 'model2', 'provider') > 0
        now[0] += 1
        assert b.try_acquire('c', 'model2', 'provider') == 0
        now[0] += 2
        assert b.try_acquire('d', 'model3', 'provider') > 0  # provider full
        a.release('a')
        assert b.try_acquire('d', 'model3', 'provider') == 0
        b.release('c')
        b.release('d')
        now[0] += 1
        assert b.try_acquire('b', 'model1', 'provider') == 0
    finally:
        a.close(); b.close()


def test_cooldown_shared_staggered_and_other_providers_continue(tmp_path):
    now = [100.0]
    options = {'route': {'max_in_flight': 2, 'min_interval_seconds': 1},
               'provider': {'max_in_flight': 4, 'min_interval_seconds': 1}}
    a = controller(tmp_path/'control.sqlite3', policy=options, clock=lambda: now[0])
    b = controller(tmp_path/'control.sqlite3', policy=options, clock=lambda: now[0])
    try:
        a.cooldown('model1', 'provider1', 60, evidence='HTTP 429; Retry-After: 60')
        assert b.try_acquire('a', 'model1', 'provider1') >= 60
        assert b.try_acquire('b', 'model2', 'provider1') >= 60
        assert b.try_acquire('c', 'model1', 'provider2') == 0
        now[0] += 60
        assert b.try_acquire('a', 'model1', 'provider1') == 0
        assert b.try_acquire('b', 'model2', 'provider1') >= 1
        assert b.db.execute('SELECT count(*) FROM cooldowns').fetchone()[0] == 1
    finally:
        a.close(); b.close()


def test_wait_checks_stop_without_dispatch_and_keeps_crash_slots(tmp_path):
    async def scenario():
        a = controller(tmp_path/'control.sqlite3')
        a.cooldown('model', 'provider', 3600, evidence='offline overload')
        stop = False
        async def stop_soon():
            nonlocal stop
            await asyncio.sleep(.01)
            stop = True
        stopper = asyncio.create_task(stop_soon())
        try:
            with pytest.raises(RunPaused, match='dispatch stopped'):
                await asyncio.wait_for(a.acquire('a', 'model', 'provider', lambda: stop), 1)
            assert a.db.execute('SELECT count(*) FROM active_requests').fetchone()[0] == 0
            assert a.try_acquire('crash', 'other', 'other') == 0
        finally:
            await stopper
            a.close()
        # Reopening cannot silently discard capacity for an unknown remote call.
        b = controller(tmp_path/'control.sqlite3')
        assert b.db.execute('SELECT call_key FROM active_requests').fetchone()[0] == 'crash'
        b.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('policy', [
    {'route': {'max_in_flight': 0}}, {'provider': {'min_interval_seconds': float('nan')}},
    {'route': {'max_in_flight': True}}, {'unknown': {}},
])
def test_invalid_limits_fail_before_admission(tmp_path, policy):
    with pytest.raises(ValueError):
        controller(tmp_path/'control.sqlite3', policy=policy)


def test_shared_policy_cannot_change_under_another_process(tmp_path):
    a = controller(tmp_path/'control.sqlite3')
    try:
        with pytest.raises(RunPaused, match='policy changed'):
            controller(tmp_path/'control.sqlite3', policy={'route': {'max_in_flight': 8}})
    finally:
        a.close()


def test_runtime_cooldown_wait_does_not_reserve_or_repeat_a_call(tmp_path, monkeypatch):
    from test_strict_runtime import setup_run, response, usage
    from delegated_influence import runtime
    from delegated_influence.journal import Journal
    from delegated_influence.spending import SpendingLedger
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    import httpx
    import openai

    cfg, tasks, estimates, _ = setup_run(tmp_path, seeds=(7, 8))
    cfg['native']['request_limits'] = {'route': {'max_in_flight': 1, 'min_interval_seconds': 0},
                                       'provider': {'max_in_flight': 1, 'min_interval_seconds': 0}}
    calls, stop = [], []
    def factory(value, seed, **kwargs):
        async def generate(messages):
            calls.append(seed)
            request = httpx.Request('POST', 'https://openrouter.ai/api/v1/chat/completions')
            raise openai.APIStatusError('rate limited', response=httpx.Response(429, request=request,
                headers={'Retry-After': format_datetime(datetime.now(timezone.utc) + timedelta(hours=1))}), body=None)
        return generate
    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    async def scenario():
        async def stop_soon():
            await asyncio.sleep(.02)
            stop.append(True)
        stopper = asyncio.create_task(stop_soon())
        with Journal(tmp_path/'run.sqlite3') as journal:
            with pytest.raises(RunPaused, match='dispatch stopped'):
                await asyncio.wait_for(runtime.run_tasks(journal, tasks, estimates, cfg,
                    dispatch_paused=lambda: bool(stop)), 1)
            assert journal.summary()['uncertain_calls'] == 0
            assert journal.db.execute('SELECT count(*) FROM call_attempts').fetchone()[0] == 1
        await stopper
    asyncio.run(scenario())
    assert len(calls) == 1
    with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
        assert ledger.summary()['held_usd'] == '0'


@pytest.mark.parametrize('recovery', ['confirmed_unbilled', 'saved_response', 'retry_reservation'])
def test_crash_slot_recovery_continues_without_duplicate_saved_calls(tmp_path, monkeypatch, recovery):
    from test_strict_runtime import setup_run, response, usage, execute, MODEL
    from delegated_influence import runtime
    from delegated_influence.journal import Journal
    from delegated_influence.spending import SpendingLedger
    import json

    cfg, tasks, estimates, _ = setup_run(tmp_path)
    cfg['native']['request_limits'] = {'route': {'max_in_flight': 1, 'min_interval_seconds': 0},
                                       'provider': {'max_in_flight': 1, 'min_interval_seconds': 0}}
    calls = []
    def factory(value, seed, **kwargs):
        async def generate(messages):
            calls.append(messages)
            if len(calls) == 1 and recovery in ('confirmed_unbilled', 'retry_reservation'):
                raise TimeoutError('lost response')
            return response(kwargs['seat']), usage()
        return generate
    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    with Journal(tmp_path/'run.sqlite3') as journal:
        with monkeypatch.context() as crash:
            if recovery == 'saved_response':
                crash.setattr(journal, 'save_event', lambda *a: (_ for _ in ()).throw(OSError('crash before event')))
            with pytest.raises((RunPaused, OSError)):
                asyncio.run(runtime.run_tasks(journal, tasks, estimates, cfg))
        first = journal.db.execute('SELECT id FROM calls').fetchone()['id']
        namespace = json.loads(journal.db.execute("SELECT value FROM meta WHERE key='spending_namespace'").fetchone()['value'])
        key = f'{namespace}:{first}'
        if recovery in ('confirmed_unbilled', 'retry_reservation'):
            journal.resolve_uncertain(first, confirm_unbilled=True, evidence='Provider confirms no charge')
            with SpendingLedger(cfg['spending']['ledger'], 10) as ledger:
                ledger.confirm_unbilled(key, evidence='Provider confirms no charge')
                if recovery == 'retry_reservation':
                    ledger.reserve(key, .11)  # crash after reserve, before begin_call
    control = controller(tmp_path/'spending.requests.sqlite3', policy=cfg['native']['request_limits'])
    assert control.try_acquire(key, MODEL, 'offline') == 0
    control.close()  # Simulate a process killed before releasing its slot.
    with Journal(tmp_path/'run.sqlite3') as journal:
        summary = asyncio.run(asyncio.wait_for(runtime.run_tasks(journal, tasks, estimates, cfg), 1))
        assert summary['episodes'] == {'complete': 1}
    assert calls.count(calls[0]) == (2 if recovery in ('confirmed_unbilled', 'retry_reservation') else 1)
    control = controller(tmp_path/'spending.requests.sqlite3', policy=cfg['native']['request_limits'])
    assert control.db.execute('SELECT count(*) FROM active_requests').fetchone()[0] == 0
    assert control.db.execute('SELECT count(*) FROM recovered_slots').fetchone()[0] == 1
    control.close()


def test_ledger_symlink_cannot_split_shared_limits(tmp_path, monkeypatch):
    from test_strict_runtime import setup_run, response, usage
    from delegated_influence import runtime
    from delegated_influence.journal import Journal
    from delegated_influence.spending import SpendingLedger
    from pathlib import Path
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    cfg1, tasks1, estimates1, _ = setup_run(first)
    cfg2, tasks2, estimates2, _ = setup_run(second, seeds=(8,))
    with SpendingLedger(cfg1['spending']['ledger'], 10):
        pass
    Path(cfg2['spending']['ledger']).symlink_to(cfg1['spending']['ledger'])
    for cfg in (cfg1, cfg2):
        cfg['native']['request_limits'] = {'route': {'max_in_flight': 1, 'min_interval_seconds': 0},
                                          'provider': {'max_in_flight': 1, 'min_interval_seconds': 0}}
    active, maximum = 0, 0
    def factory(value, seed, **kwargs):
        async def generate(messages):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(.01)
                return response(kwargs['seat']), usage()
            finally:
                active -= 1
        return generate
    monkeypatch.setattr(runtime.native_adapter, 'build_generator', factory)
    async def scenario():
        with Journal(first/'run.sqlite3') as one, Journal(second/'run.sqlite3') as two:
            await asyncio.wait_for(asyncio.gather(runtime.run_tasks(one, tasks1, estimates1, cfg1),
                                                 runtime.run_tasks(two, tasks2, estimates2, cfg2)), 2)
    asyncio.run(scenario())
    assert maximum == 1
    assert not (second/'spending.requests.sqlite3').exists()


def test_orphan_slot_reconciliation_keeps_unresolved_holds_and_other_namespaces(tmp_path):
    from delegated_influence.journal import Journal
    from delegated_influence.spending import SpendingLedger
    control = controller(tmp_path/'control.sqlite3', policy={
        'route': {'max_in_flight': 10, 'min_interval_seconds': 0},
        'provider': {'max_in_flight': 10, 'min_interval_seconds': 0}})
    with Journal(tmp_path/'run.sqlite3') as journal, SpendingLedger(tmp_path/'spending.sqlite3', 10) as ledger:
        for key in ('ours:before-reserve', 'ours:held', 'ours:unbilled', 'other:call'):
            assert control.try_acquire(key, 'model', 'provider') == 0
        ledger.reserve('ours:held', 1)
        ledger.reserve('ours:unbilled', 1)
        ledger.confirm_unbilled('ours:unbilled', evidence='operator checked no dispatch')
        control.reconcile(journal, ledger, 'ours')
        remaining = {r['call_key'] for r in control.db.execute('SELECT call_key FROM active_requests')}
        assert remaining == {'ours:held', 'other:call'}
        assert ledger.summary()['held_usd'] == '1'
    control.close()
