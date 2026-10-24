"""Exact shared request reservations. These tests never call a provider."""
from decimal import Decimal
import importlib
import multiprocessing
import sqlite3

import pytest


def spending():
    return importlib.import_module('delegated_influence.spending')


def test_reservation_settlement_and_replay_do_not_charge_twice(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path/'spending.sqlite3',1) as ledger:
        assert ledger.reserve('run-a:call-1',Decimal('.6'))['dispatch_allowed'] is True
        assert ledger.reserve('run-a:call-1',Decimal('.60'))['dispatch_allowed'] is False
        with pytest.raises(module.SpendingError,match='cap'):
            ledger.reserve('run-b:call-1',Decimal('.5'))
        assert ledger.summary()['calls']==1
        ledger.settle('run-a:call-1',Decimal('.2'))
        ledger.settle('run-a:call-1',Decimal('.20'))
        assert ledger.reserve('run-a:call-1',Decimal('.6'))['dispatch_allowed'] is False
        assert ledger.summary()['spent_nanodollars']==200_000_000
        assert ledger.reserve('run-b:call-1',Decimal('.8'))['dispatch_allowed'] is True
        assert ledger.summary()['exposure_nanodollars']==1_000_000_000


def test_unknown_charge_keeps_its_full_bound_across_connections_and_resume(tmp_path):
    module = spending()
    path = tmp_path/'shared.sqlite3'
    with module.SpendingLedger(path,1) as first:
        first.reserve('uncertain',Decimal('.8'))
        first.settle('uncertain',None)
        with module.SpendingLedger(path,1) as second:
            assert second.summary()['held_nanodollars']==800_000_000
            assert second.reserve('uncertain',Decimal('.8'))['dispatch_allowed'] is False
            with pytest.raises(module.SpendingError,match='cap'):
                second.reserve('other-run',Decimal('.3'))
    with module.SpendingLedger(path,1) as resumed:
        assert resumed.summary()['held_nanodollars']==800_000_000
        resumed.settle('uncertain',Decimal('.1'))
        assert resumed.summary()['held_nanodollars']==0
        assert resumed.summary()['spent_nanodollars']==100_000_000


def test_cap_and_reused_bound_are_immutable(tmp_path):
    module = spending()
    path = tmp_path/'shared.sqlite3'
    with module.SpendingLedger(path,1) as ledger:
        ledger.reserve('one',Decimal('.25'))
        with pytest.raises(module.SpendingError,match='bound'):
            ledger.reserve('one',Decimal('.25000000001'))
    with pytest.raises(module.SpendingError,match='cap'):
        module.SpendingLedger(path,2)
    with module.SpendingLedger(path,Decimal('1.00')) as ledger:
        assert ledger.summary()['cap_nanodollars']==1_000_000_000


def test_actual_above_bound_is_saved_and_permanently_stops_dispatch(tmp_path):
    module = spending()
    path = tmp_path/'shared.sqlite3'
    with module.SpendingLedger(path,1) as ledger:
        ledger.reserve('overrun',Decimal('.4'))
        ledger.reserve('already-active',Decimal('.1'))
        with pytest.raises(module.SpendingError,match='bound'):
            ledger.settle('overrun',Decimal('1.2'))
        assert ledger.summary()['stopped'] is True
        assert ledger.summary()['spent_nanodollars']==1_200_000_000
        assert ledger.summary()['held_nanodollars']==100_000_000
        # A stop must not discard later bills from calls that were already active.
        ledger.settle('already-active',Decimal('.05'))
    with module.SpendingLedger(path,1) as resumed:
        assert resumed.summary()['spent_nanodollars']==1_250_000_000
        with pytest.raises(module.SpendingError,match='stopped'):
            resumed.reserve('new-call',0)
        assert resumed.summary()['calls']==2


def test_rounding_cannot_create_room_or_hide_a_bound_overrun(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path/'rounding.sqlite3',Decimal('.0000000029')) as ledger:
        assert ledger.summary()['cap_nanodollars']==2
        ledger.reserve('fraction',Decimal('.0000000011'))
        assert ledger.summary()['held_nanodollars']==2
        with pytest.raises(module.SpendingError,match='cap'):
            ledger.reserve('extra',Decimal('.0000000001'))
        # Both amounts round to two nanodollars, but the true cost still exceeds its bound.
        with pytest.raises(module.SpendingError,match='bound'):
            ledger.settle('fraction',Decimal('.0000000012'))
        assert ledger.summary()['stopped'] is True
        assert ledger.summary()['spent_nanodollars']==2


def test_decimal_boundary_has_no_binary_float_drift(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path/'decimals.sqlite3',.3) as ledger:
        ledger.reserve('one',.1)
        ledger.reserve('two',.2)
        assert ledger.summary()['exposure_nanodollars']==300_000_000
        assert ledger.summary()['available_nanodollars']==0


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, Decimal('-0.1')])
def test_invalid_costs_cannot_change_the_ledger(tmp_path,value):
    module = spending()
    with pytest.raises(module.SpendingError):
        module.SpendingLedger(tmp_path/'invalid-cap.sqlite3',value)
    with module.SpendingLedger(tmp_path/'valid.sqlite3',1) as ledger:
        with pytest.raises(module.SpendingError):
            ledger.reserve('bad',value)
        ledger.reserve('valid',Decimal('.5'))
        with pytest.raises(module.SpendingError):
            ledger.settle('valid',value)
        assert ledger.summary()['calls']==1
        assert ledger.summary()['held_nanodollars']==500_000_000


def test_unbilled_release_requires_explicit_evidence_and_reacquires_capacity(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path/'release.sqlite3',1) as ledger:
        ledger.reserve('one',1)
        with pytest.raises(module.SpendingError,match='evidence'):
            ledger.confirm_unbilled('one',evidence='')
        ledger.confirm_unbilled('one',evidence='Provider request history confirms no bill.')
        assert ledger.summary()['held_nanodollars']==0
        ledger.reserve('two',1)
        with pytest.raises(module.SpendingError,match='cap'):
            ledger.reserve('one',1)
        ledger.settle('two',0)
        assert ledger.reserve('one',1)['dispatch_allowed'] is True
        assert ledger.summary()['unbilled_resolutions']==1


def test_conflicting_settlement_does_not_overwrite_a_saved_bill(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path/'settle.sqlite3',1) as ledger:
        with pytest.raises(module.SpendingError,match='unknown'):
            ledger.settle('absent',Decimal('.1'))
        ledger.reserve('one',Decimal('.5'))
        ledger.settle('one',Decimal('.3'))
        ledger.settle('one',None)
        with pytest.raises(module.SpendingError,match='settled'):
            ledger.settle('one',Decimal('.2'))
        assert ledger.summary()['spent_nanodollars']==300_000_000
        with pytest.raises(module.SpendingError,match='settled'):
            ledger.confirm_unbilled('one',evidence='Cannot undo a saved bill this way.')


def _reserve_worker(path,key,barrier,queue):
    module = spending()
    try:
        with module.SpendingLedger(path,1) as ledger:
            barrier.wait(timeout=10)
            result = ledger.reserve(key,Decimal('.6'))
            queue.put(('ok',result['dispatch_allowed']))
    except module.SpendingError:
        queue.put(('denied',False))


@pytest.mark.parametrize('keys,expected_allowed', [(['one','two'],1), (['same','same'],1)])
def test_processes_share_atomic_capacity_and_duplicate_dispatch_permission(tmp_path,keys,expected_allowed):
    module = spending()
    path = tmp_path/'processes.sqlite3'
    context = multiprocessing.get_context('spawn')
    barrier, queue = context.Barrier(2), context.Queue()
    processes = [context.Process(target=_reserve_worker,args=(str(path),key,barrier,queue)) for key in keys]
    for process in processes:
        process.start()
    outcomes = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode==0
    assert sum(allowed for _,allowed in outcomes)==expected_allowed
    with module.SpendingLedger(path,1) as ledger:
        assert ledger.summary()['held_nanodollars']==600_000_000
        assert ledger.summary()['calls']==1


def test_explicit_stop_is_durable_and_still_accepts_outstanding_bills(tmp_path):
    module = spending()
    path = tmp_path/'stopped.sqlite3'
    with module.SpendingLedger(path,1) as ledger:
        ledger.reserve('active',Decimal('.5'))
        ledger.stop('Provider used an unapproved pricing tier.')
        with pytest.raises(module.SpendingError,match='stopped'):
            ledger.reserve('new',0)
        ledger.settle('active',Decimal('.2'))
        assert ledger.summary()['reported_usd']=='0.2'
    with module.SpendingLedger(path,1) as resumed:
        resumed.stop('Another reason must not erase the first violation.')
        assert resumed.summary()['stopped_reason']=='Provider used an unapproved pricing tier.'
        with pytest.raises(module.SpendingError,match='stopped'):
            resumed.reserve('retry',0)


def test_explicit_cap_increase_keeps_bills_unknown_holds_and_stop(tmp_path):
    module = spending()
    path = tmp_path / 'increase.sqlite3'
    with module.SpendingLedger(path, 25) as ledger:
        ledger.reserve('known', 12)
        ledger.settle('known', Decimal('11.20645'))
        ledger.reserve('unknown', Decimal('3.268608'))
        ledger.settle('unknown', None)
        ledger.stop('Verified provider contract violation')
        before = ledger.summary()
        calls_before = [dict(row) for row in ledger.db.execute('SELECT * FROM spending_calls ORDER BY call_key')]
        audit = ledger.increase_cap(25, 30, evidence='Operator approved five additional test dollars',
                                    confirm_no_active_runs=True)
        after = ledger.summary()
        assert audit['old_cap_usd'] == '25' and audit['new_cap_usd'] == '30'
        assert audit['no_active_runs_confirmed'] == 1 and audit['created']
        assert audit['evidence'] == 'Operator approved five additional test dollars'
        assert ledger.cap_increases() == [audit]
        for key in ('spent_nanodollars', 'held_nanodollars', 'reported_usd', 'stopped', 'stopped_reason', 'calls'):
            assert after[key] == before[key]
        assert after['cap_nanodollars'] == 30_000_000_000
        assert [dict(row) for row in ledger.db.execute('SELECT * FROM spending_calls ORDER BY call_key')] == calls_before
        with pytest.raises(module.SpendingError, match='stopped'):
            ledger.reserve('new', 0)
    with pytest.raises(module.SpendingError, match='cap'):
        module.SpendingLedger(path, 25)
    with module.SpendingLedger(path, 30) as reopened:
        assert reopened.cap_increases() == [audit]
        assert reopened.summary()['held_usd'] == '3.268608'


@pytest.mark.parametrize('expected,new,evidence,confirmed', [
    (1, 1, 'approval', True), (1, .9, 'approval', True), (1, 0, 'approval', True),
    (1, -1, 'approval', True), (1, float('nan'), 'approval', True),
    (1, float('inf'), 'approval', True), (1, True, 'approval', True),
    (True, 2, 'approval', True), (0, 2, 'approval', True),
    (1, 2, '', True), (1, 2, '   ', True), (1, 2, None, True),
    (1, 2, 'approval', False), (1, 2, 'approval', 1),
])
def test_invalid_cap_increase_changes_nothing(tmp_path, expected, new, evidence, confirmed):
    module = spending()
    with module.SpendingLedger(tmp_path / 'invalid-increase.sqlite3', 1) as ledger:
        ledger.reserve('unknown', Decimal('.5'))
        before = ledger.summary()
        with pytest.raises(module.SpendingError):
            ledger.increase_cap(expected, new, evidence=evidence, confirm_no_active_runs=confirmed)
        assert ledger.summary() == before
        assert ledger.cap_increases() == []


def test_cap_increase_and_audit_commit_together(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path / 'atomic-increase.sqlite3', 1) as ledger:
        ledger.db.execute('''CREATE TRIGGER reject_cap_audit BEFORE INSERT ON spending_cap_increases
                             BEGIN SELECT RAISE(ABORT, 'simulated audit failure'); END''')
        with pytest.raises(sqlite3.IntegrityError, match='simulated audit failure'):
            ledger.increase_cap(1, 2, evidence='Test approved', confirm_no_active_runs=True)
        assert ledger.summary()['configured_cap_usd'] == '1'
        assert ledger.cap_increases() == []


def _increase_worker(path, new_cap, barrier, queue):
    module = spending()
    try:
        with module.SpendingLedger(path, 1) as ledger:
            barrier.wait(timeout=10)
            audit = ledger.increase_cap(1, new_cap, evidence='Concurrent test approval',
                                        confirm_no_active_runs=True)
            queue.put(('ok', audit['new_cap_usd']))
    except module.SpendingError:
        queue.put(('denied', None))


def test_concurrent_cap_increases_require_exact_old_cap_match(tmp_path):
    module = spending()
    path = tmp_path / 'concurrent-increase.sqlite3'
    with module.SpendingLedger(path, 1) as ledger:
        ledger.reserve('unknown', Decimal('.5'))
    context = multiprocessing.get_context('spawn')
    barrier, queue = context.Barrier(2), context.Queue()
    processes = [context.Process(target=_increase_worker, args=(str(path), cap, barrier, queue)) for cap in (2, 3)]
    for process in processes:
        process.start()
    outcomes = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    accepted = [new for status, new in outcomes if status == 'ok']
    assert len(accepted) == 1
    with module.SpendingLedger(path, Decimal(accepted[0])) as ledger:
        assert len(ledger.cap_increases()) == 1
        assert ledger.summary()['held_usd'] == '0.5'


def test_cap_increase_preserves_exact_decimal_and_rounding(tmp_path):
    module = spending()
    with module.SpendingLedger(tmp_path / 'precise-increase.sqlite3', Decimal('.0000000019')) as ledger:
        audit = ledger.increase_cap(Decimal('.0000000019'), Decimal('.0000000021'),
                                   evidence='Exact boundary approval', confirm_no_active_runs=True)
        assert audit['new_cap_usd'] == '0.0000000021'
        assert ledger.summary()['cap_nanodollars'] == 2
        ledger.reserve('two-nanos', Decimal('.000000002'))
        with pytest.raises(module.SpendingError, match='cap'):
            ledger.reserve('fraction', Decimal('.00000000001'))


def test_cap_increase_cli_requires_confirmation_and_never_creates_a_ledger(tmp_path, capsys):
    module = spending()
    path = tmp_path / 'cli.sqlite3'
    argv = ['--ledger', str(path), '--expected-cap-usd', '1', '--new-cap-usd', '2', '--evidence', 'Test approval']
    with pytest.raises(SystemExit):
        module.main(argv)
    assert not path.exists()
    with pytest.raises(SystemExit):
        module.main([*argv, '--confirm-no-active-runs'])
    assert not path.exists()
    # An existing unrelated SQLite file is not an initialized spending ledger.
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE unrelated (value TEXT)')
    with pytest.raises(SystemExit):
        module.main([*argv, '--confirm-no-active-runs'])
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('unrelated',)]
    path.unlink()
    with module.SpendingLedger(path, 1) as ledger:
        ledger.reserve('unknown', Decimal('.8'))
    module.main([*argv, '--confirm-no-active-runs'])
    with module.SpendingLedger(path, 2) as ledger:
        assert ledger.summary()['calls'] == 1
        assert ledger.summary()['held_usd'] == '0.8'
        assert ledger.cap_increases()[0]['evidence'] == 'Test approval'
    assert 'No request sent' in capsys.readouterr().out
