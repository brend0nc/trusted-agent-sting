"""Durable calls must prevent paid prefix replay, including failure paths."""
import asyncio
from decimal import Decimal
import json
from pathlib import Path

import pytest

from delegated_influence.journal import Journal, RunPaused
from delegated_influence.runtime import run_episode


def metadata(seed=7):
    return {"seat_map": {"P1": "mockllm/model", "P2": "mockllm/model"},
            "seed": seed, "rounds": 1, "actions_per_round": 2,
            "sampling_instruments": {"mockllm/model": {"max_tokens": 100}}}


def test_paused_resume_setup_failure_does_not_claim_running(tmp_path, monkeypatch):
    from delegated_influence import runtime
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.start_episode('one', metadata(), 1, 10)
        journal.db.execute("UPDATE episodes SET status='paused' WHERE id='one'")
        def fail(_):
            raise ValueError('invalid restored state')
        monkeypatch.setattr(runtime.game, 'state_from_metadata', fail)
        with pytest.raises(ValueError, match='restored state'):
            asyncio.run(run_episode(journal, 'one', metadata(), lambda _: None, reserve_usd=1, budget_usd=10))
        assert journal.episode('one')['status'] == 'paused'


def test_resume_reuses_completed_calls_but_not_independent_episodes(tmp_path):
    calls = []
    async def generate(messages):
        calls.append(messages)
        return '{"action":"pull","target":"P2"}', {"total_cost": .01}
    async def execute():
        with Journal(tmp_path / 'run.sqlite3') as journal:
            await run_episode(journal, 'one', metadata(), lambda _: generate, reserve_usd=1, budget_usd=10)
            before = len(calls)
            first = journal.episode('one')
            # Force replay to simulate a crash after the last response, before export.
            journal.db.execute("UPDATE episodes SET status='running' WHERE id='one'")
            await run_episode(journal, 'one', metadata(), lambda _: generate, reserve_usd=1, budget_usd=10)
            assert len(calls) == before
            assert journal.episode('one')['record'] == first['record']
            await run_episode(journal, 'two', metadata(8), lambda _: generate, reserve_usd=1, budget_usd=10)
            assert len(calls) > before
            assert journal.summary()['reported_cost_usd'] == pytest.approx(.01 * len(calls))
            assert first['record']['events'][0]['calls'][0]['input'] == calls[0]
    asyncio.run(execute())


def test_uncertain_call_stops_recovery_without_second_request(tmp_path):
    attempts = 0
    async def broken(messages):
        nonlocal attempts
        attempts += 1
        raise TimeoutError('response lost')
    async def execute():
        with Journal(tmp_path / 'run.sqlite3') as journal:
            for _ in range(2):
                with pytest.raises(RunPaused, match='uncertain'):
                    await run_episode(journal, 'one', metadata(), lambda _: broken, reserve_usd=1, budget_usd=10)
            assert attempts == 1
            assert journal.summary()['uncertain_calls'] == 1
    asyncio.run(execute())


def test_nested_provider_error_is_durable_and_retains_spending_hold(tmp_path):
    from delegated_influence.spending import SpendingLedger
    attempts = 0

    async def broken(messages):
        nonlocal attempts
        attempts += 1
        try:
            raise TimeoutError('socket timed out')
        except TimeoutError as exc:
            raise ConnectionError('connection failed') from exc

    async def execute():
        ledger_path = tmp_path / 'shared.sqlite3'
        with SpendingLedger(ledger_path, 5) as spending, Journal(tmp_path / 'run.sqlite3') as journal:
            bound = type('Bound', (), {'maximum_usd': Decimal('.5')})()
            with pytest.raises(RunPaused, match='uncertain'):
                await run_episode(
                    journal, 'one', metadata(), lambda _: broken, reserve_usd=1, budget_usd=10,
                    spending=spending, request_bounds={'mockllm/model': bound},
                    spending_namespace='nested-error')
            row = journal.db.execute("SELECT status,error FROM calls").fetchone()
            evidence = json.loads(row['error'])
            assert row['status'] == 'uncertain'
            assert evidence['type'] == 'builtins.ConnectionError'
            assert [item['type'] for item in evidence['chain']] == [
                'builtins.ConnectionError', 'builtins.TimeoutError']
            assert 'raise ConnectionError' in evidence['traceback']
            assert spending.summary()['held_nanodollars'] > 0
        assert attempts == 1

    asyncio.run(execute())


def test_outer_recording_failure_keeps_provider_context_in_pending_call(tmp_path, monkeypatch):
    async def broken(messages):
        raise ConnectionError('provider disconnected')

    async def execute():
        with Journal(tmp_path / 'run.sqlite3') as journal:
            def fail_uncertain(call_id, error):
                raise OSError('journal update failed')
            monkeypatch.setattr(journal, 'uncertain', fail_uncertain)
            with pytest.raises(RunPaused, match='recording failed'):
                await run_episode(
                    journal, 'one', metadata(), lambda _: broken, reserve_usd=1, budget_usd=10)
            row = journal.db.execute("SELECT status,error FROM calls").fetchone()
            evidence = json.loads(row['error'])
            assert row['status'] == 'uncertain'
            assert [item['type'] for item in evidence['chain']] == [
                'builtins.OSError', 'builtins.ConnectionError']
            assert 'fail_uncertain' in evidence['traceback']

    asyncio.run(execute())


def test_changed_request_and_metadata_fail_closed(tmp_path):
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.start_episode('one', metadata(), 1, 10)
        call_id, cached = journal.begin_call('one', 'P1', 0, [{'role':'user','content':'hello'}])
        journal.finish_call(call_id, 'ok', {'total_cost': 0}, None)
        with pytest.raises(RunPaused, match='input changed'):
            journal.begin_call('one', 'P1', 0, [{'role':'user','content':'changed'}])
        with pytest.raises(RunPaused, match='metadata changed'):
            journal.start_episode('one', metadata(42), 1, 10)


def test_reservations_and_unknown_cost_stop_new_games(tmp_path):
    with Journal(tmp_path / 'run.sqlite3') as journal:
        journal.start_episode('one', metadata(), 6, 10)
        with pytest.raises(RunPaused, match='budget'):
            journal.start_episode('two', metadata(8), 6, 10)
        cid, _ = journal.begin_call('one', 'P1', 0, [])
        journal.finish_call(cid, 'ok', {}, .2)
        assert journal.summary()['unknown_cost_calls'] == 1
        with pytest.raises(RunPaused, match='unknown cost'):
            journal.start_episode('two', metadata(8), 1, 10)
        # An explicit estimate policy permits continuation, without relabelling it as actual.
        journal.start_episode('two', metadata(8), 1, 10, allow_estimated_cost=True)
        assert journal.summary()['estimated_cost_usd'] == .2


def test_native_record_roundtrip_into_canonical_reader(tmp_path):
    from delegated_influence.episodes import load_episodes
    async def generate(messages):
        return '{"action":"pull","target":"P2"}', {'total_cost': 0}
    async def execute():
        with Journal(tmp_path / 'run.sqlite3') as journal:
            await run_episode(journal, 'one', metadata(), lambda _: generate, reserve_usd=1, budget_usd=10)
            journal.export_episode('one', tmp_path / 'one.episode.json')
    asyncio.run(execute())
    rows = load_episodes([tmp_path])
    assert len(rows) == 1
    assert rows[0]['checkpoint_episode_id'] == 'one'
    assert rows[0]['events'][0]['calls'][0]['call_id'].startswith('one:')


def test_native_scheduler_uses_existing_mock_adapter(tmp_path):
    from delegated_influence.runtime import run_tasks
    from types import SimpleNamespace
    from delegated_influence import task
    md = metadata()
    md['model_configs'] = {'mockllm/model': {'max_tokens':100}}
    tasks = [('arm', SimpleNamespace(dataset=[SimpleNamespace(id='sample',metadata=md)]))]
    estimate = {'arms':[{'arm':'arm','usd':1,'episodes':1}]}
    cfg = {'budget': {'max_estimated_usd':10}, 'native': {'allow_estimated_cost':True}}
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            result = await run_tasks(journal,tasks,estimate,cfg)
            assert result['episodes']['complete'] == 1
            assert result['calls'] > 0
    asyncio.run(execute())


def test_unpriced_completed_game_does_not_release_budget(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one', metadata(), 1, 1)
        cid, _ = journal.begin_call('one', 'P1', 0, [])
        journal.finish_call(cid, 'ok', {}, None)
        journal.complete('one', {})
        with pytest.raises(RunPaused, match='unpriced'):
            journal.start_episode('two', metadata(8), 1, 1, allow_estimated_cost=True)


def test_crash_after_saved_response_recovers_without_repeat(tmp_path, monkeypatch):
    paid = []
    async def generate(messages):
        paid.append(messages)
        return '{"action":"pull","beneficiary":"P2"}', {'total_cost': .01}
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            original = journal.save_event
            def crash(*args):
                raise RuntimeError('process died before saving event')
            monkeypatch.setattr(journal,'save_event',crash)
            with pytest.raises(RuntimeError, match='process died'):
                await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,budget_usd=10)
            assert len(paid) == 1
            monkeypatch.setattr(journal,'save_event',original)
            await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,budget_usd=10)
            assert paid.count(paid[0]) == 1
    asyncio.run(execute())


def test_uncertain_resolution_requires_evidence_and_retains_audit(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one',metadata(),1,10)
        cid, _ = journal.begin_call('one','P1',0,[])
        journal.uncertain(cid,'lost connection')
        with pytest.raises(ValueError, match='evidence'):
            journal.resolve_uncertain(cid,evidence='',confirm_unbilled=True)
        journal.resolve_uncertain(cid,evidence='Provider request log confirms no charge',confirm_unbilled=True)
        assert journal.summary()['uncertain_calls'] == 0
        assert journal.begin_call('one','P1',0,[]) == (cid,None)
        assert journal.db.execute('SELECT COUNT(*) FROM resolutions').fetchone()[0] == 1


def test_nonfinite_cost_never_reaches_export(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one',metadata(),1,10)
        cid, _ = journal.begin_call('one','P1',0,[])
        journal.finish_call(cid,'ok',{'total_cost':float('nan')},None)
        assert journal.call(cid)['usage']['total_cost'] is None


def test_accounting_failure_saves_response_and_never_forfeits(tmp_path):
    paid = []
    async def generate(messages):
        paid.append(messages)
        return '{"action":"pull","beneficiary":"P2"}', {'total_cost':.01}
    def broken_estimate(*args):
        raise ValueError('bad accounting input')
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            with pytest.raises(RunPaused,match='recording failed'):
                await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,
                                  budget_usd=10,estimate_call=broken_estimate)
            assert len(paid) == 1
            assert journal.summary()['uncertain_calls'] == 0
            assert journal.episode('one')['status'] == 'paused'
            assert journal.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 0
            await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,budget_usd=10)
            assert paid.count(paid[0]) == 1
    asyncio.run(execute())


def test_cannot_complete_a_game_with_uncertain_calls(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one',metadata(),1,10)
        journal.begin_call('one','P1',0,[])
        with pytest.raises(RunPaused,match='cannot enter analysis'):
            journal.complete('one',{})


def test_saved_response_finishes_accounting_on_resume(tmp_path):
    attempts = []
    async def generate(messages):
        attempts.append(messages)
        return '{"action":"pull","beneficiary":"P2"}', {'prompt_tokens':10,'completion_tokens':5}
    def broken(*args):
        raise ValueError('estimate failed')
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            with pytest.raises(RunPaused):
                await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,
                                  budget_usd=10,estimate_call=broken)
            assert journal.summary()['unpriced_calls'] == 1
            await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,
                              budget_usd=10,estimate_call=lambda *args:.01)
            assert journal.summary()['unpriced_calls'] == 0
            assert attempts.count(attempts[0]) == 1
    asyncio.run(execute())


def test_approved_retry_does_not_reuse_old_unbilled_cost(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one',metadata(),1,10)
        cid,_ = journal.begin_call('one','P1',0,[])
        journal.uncertain(cid,'timeout')
        journal.resolve_uncertain(cid,evidence='provider confirmed first attempt unbilled',confirm_unbilled=True)
        journal.begin_call('one','P1',0,[])
        journal.uncertain(cid,'timeout on approved retry')
        assert journal.summary()['unknown_cost_calls'] == 1
        assert journal.summary()['unpriced_calls'] == 1


def test_planned_identity_survives_native_export(tmp_path, monkeypatch):
    from delegated_influence import experiment, runtime, episodes, report
    from datetime import datetime, timezone
    cfg = {'arms':[{'name':'a','task':'lineup','seats':['opus-4.8','gpt-5.5'],
                    'rounds':1,'actions_per_round':1,'reps':1}], 'budget':{'max_estimated_usd':10}}
    tasks = experiment.build_tasks(cfg,config_name='native-test')
    routes = {'schema_version':1, 'verified_at':datetime.now(timezone.utc).isoformat(), 'models': {}}
    for model, instrument in tasks[0][1].dataset[0].metadata['sampling_instruments'].items():
        routes['models'][model] = {'model':instrument['model_spec'],'provider_tag':'test','provider_name':'Test',
            'context_length':10000,'max_completion_tokens':10000,'evidence':'offline fixture',
            'price_ceiling':dict(prompt='.00001',completion='.00001',input_cache_read='0',input_cache_write='0',request='0'),
            'supported_parameters':['max_tokens','reasoning']}
    (tmp_path/'routes.json').write_text(json.dumps(routes))
    cfg['spending'] = {'ledger':str(tmp_path/'spending.sqlite3'),'cap_usd':10,'manifest':str(tmp_path/'routes.json')}
    estimate = experiment.estimate_cost(cfg,tasks)
    def factory(*args,**kwargs):
        async def generate(messages):
            who = kwargs['seat']
            beneficiary = 'P2' if who == 'P1' else 'P1'
            return json.dumps({'action':'pull','beneficiary':beneficiary}), {
                'total_cost':.01,'served_provider':'Test','prompt_tokens':10,'completion_tokens':10}
        return generate
    monkeypatch.setattr(runtime.native_adapter,'build_generator',factory)
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            await runtime.run_tasks(journal,tasks,estimate,cfg)
    asyncio.run(execute())
    (record,) = episodes.load_episodes([tmp_path])
    sample = tasks[0][1].dataset[0]
    assert record['sample_id'] == str(sample.id)
    assert experiment.episode_identity(record['sample_id'],record['metadata']) == sample.metadata['experiment_identity']
    measured = report.measure_episode(record)
    assert measured['runtime_instrument']['runner'] == 'native'
    assert 'python' not in measured['runtime_instrument']
    assert 'packages' not in measured['runtime_instrument']
    assert measured['runtime_environment'] == runtime.RUNTIME_ENVIRONMENT
    assert record['metadata']['runtime_environment'] == runtime.RUNTIME_ENVIRONMENT
    assert all(row['forfeit_rate'] == 0 for row in measured['seats'])


def test_two_games_can_start_while_first_call_is_pending(tmp_path, monkeypatch):
    from delegated_influence import runtime
    from types import SimpleNamespace
    active = set()
    both_started = asyncio.Event()
    def factory(value,seed,**kwargs):
        async def generate(messages):
            active.add(seed)
            if len(active) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), .5)
            who = kwargs['seat']
            return json.dumps({'action':'pull','beneficiary':'P2' if who=='P1' else 'P1'}), {'total_cost':.01}
        return generate
    monkeypatch.setattr(runtime.native_adapter,'build_generator',factory)
    samples = [SimpleNamespace(id=str(seed),metadata=metadata(seed)) for seed in [7,8]]
    cfg = {'budget':{'max_estimated_usd':10},'native':{'max_concurrent_episodes':2}}
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            await runtime.run_tasks(journal,[('arm',SimpleNamespace(dataset=samples))],
                                    {'arms':[{'arm':'arm','usd':2,'episodes':2}]},cfg)
            assert journal.summary()['episodes']['complete'] == 2
            assert journal.summary()['uncertain_calls'] == 0
    asyncio.run(execute())


def test_reopened_pending_call_becomes_uncertain(tmp_path):
    with Journal(tmp_path/'run.sqlite3') as journal:
        journal.start_episode('one',metadata(),1,10)
        journal.begin_call('one','P1',0,[])
    with Journal(tmp_path/'run.sqlite3') as journal:
        assert journal.summary()['uncertain_calls'] == 1
        with pytest.raises(RunPaused, match='uncertain'):
            journal.start_episode('two',metadata(8),1,10)


def test_resume_stops_all_dispatch_when_another_game_is_uncertain(tmp_path, monkeypatch):
    from delegated_influence import runtime
    from types import SimpleNamespace
    async def unexpected(messages):
        pytest.fail('a paid call must not start before uncertain billing is resolved')
    monkeypatch.setattr(runtime.native_adapter,'build_generator',lambda *a,**k:unexpected)
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            journal.start_episode('uncertain',metadata(99),1,10)
            cid,_ = journal.begin_call('uncertain','P1',0,[])
            journal.uncertain(cid,'lost response')
            with pytest.raises(RunPaused,match='uncertain'):
                await runtime.run_tasks(journal,[('arm',SimpleNamespace(dataset=[]))],
                                        {'arms':[{'arm':'arm','usd':1,'episodes':1}]},
                                        {'budget':{'max_estimated_usd':10}})
    asyncio.run(execute())


def test_client_setup_failure_does_not_create_an_uncertain_charge(tmp_path):
    def missing_credentials(_):
        raise ValueError('API key is absent')
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            with pytest.raises(RunPaused):
                await run_episode(journal,'one',metadata(),missing_credentials,reserve_usd=1,budget_usd=10)
            assert journal.summary()['calls'] == 0
            assert journal.summary()['uncertain_calls'] == 0
    asyncio.run(execute())


def test_status_exists_while_first_response_is_pending(tmp_path):
    async def generate(messages):
        status=json.loads((tmp_path/'status.json').read_text())
        assert status['pending_calls'] >= 1
        return '{"action":"pull","beneficiary":"P2"}', {'total_cost':0}
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,budget_usd=10)
    asyncio.run(execute())


def test_bulk_analysis_keeps_large_call_evidence_on_disk(tmp_path):
    from delegated_influence.episodes import load_episodes
    async def generate(messages):
        return '{"action":"pull","beneficiary":"P2"}', {'total_cost':0, 'provider_request_body':'x'*100000}
    async def execute():
        with Journal(tmp_path/'run.sqlite3') as journal:
            await run_episode(journal,'one',metadata(),lambda _:generate,reserve_usd=1,budget_usd=10)
            journal.export_episode('one',tmp_path/'one.episode.json.gz')
    asyncio.run(execute())
    (compact,)=load_episodes([tmp_path])
    (full,)=load_episodes([tmp_path],include_call_evidence=True)
    assert compact['call_evidence_source'].endswith('one.episode.json.gz')
    assert 'input' not in compact['events'][0]['calls'][0]
    assert 'provider_request_body' not in compact['events'][0]['usage'][0]
    assert full['events'][0]['calls'][0]['input'][0]['role']=='system'
    assert len(json.dumps(compact)) < len(json.dumps(full))/5
    assert compact['scores'] == full['scores']
