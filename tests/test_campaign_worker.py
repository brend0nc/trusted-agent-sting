"""Campaign process and backup checks; provider calls stay offline."""
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest
import yaml

from delegated_influence import campaign_worker


def plan_for(tmp_path, count=2):
    work = tmp_path / 'work'
    configs = []
    for index in range(count):
        name = f'run-{index}'
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump({
            'log_dir': str(work / name),
            'spending': {'ledger': str(work / 'spending.sqlite3'), 'cap_usd': 10,
                         'manifest': str(tmp_path / 'routes.json')},
        }))
        configs.append({'name': name, 'path': str(path)})
    return {'campaign_id': 'offline-test', 'allocation_usd': 10,
            'work_dir': str(work), 'backup_dir': str(tmp_path / 'backup'),
            'configs': configs, 'run_seconds': 15, 'backup_seconds': 1}


def real_children(monkeypatch, script):
    """Change only the child program; keep real OS process scheduling and exit codes."""
    original = subprocess.Popen
    children = []
    deadlines = []
    def start(command, **kwargs):
        assert command[:4] == [sys.executable, '-m', 'delegated_influence.campaign_worker', '--child']
        assert command[7] == '--deadline'
        deadlines.append(float(command[8]))
        assert deadlines[-1] > time.monotonic()
        assert len(set(deadlines)) == 1
        process = original([sys.executable, '-c', textwrap.dedent(script), *command[4:]], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(campaign_worker.subprocess, 'Popen', start)
    return children


def test_all_processes_start_before_wait_and_final_follows_exit(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    children = real_children(monkeypatch, '''
        import sys, time
        from pathlib import Path
        import yaml
        cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
        directory = Path(cfg['log_dir'])
        directory.mkdir(parents=True)
        (directory / 'started.json').write_text('{}')
        deadline = time.monotonic() + 5
        while len(list(directory.parent.glob('*/started.json'))) != 2:
            assert time.monotonic() < deadline, 'supervisor serialized its children'
            time.sleep(.01)
        assert not (directory.parent / 'final.json').exists()
        print('both processes started', flush=True)
        (directory / 'finished.json').write_text('{}')
    ''')
    result = campaign_worker.supervise(plan)
    assert result['child_codes'] == {'run-0': 0, 'run-1': 0}
    assert all(child.poll() == 0 for child in children)
    assert result['forced_termination'] is False
    assert result['snapshot_complete'] is True
    assert result['status'] == 'complete'
    assert json.loads((Path(plan['backup_dir']) / 'final.json').read_text()) == result
    for entry in plan['configs']:
        assert (Path(plan['backup_dir']) / entry['name'] / 'finished.json').exists()
        assert 'both processes started' in (Path(plan['work_dir']) / f"{entry['name']}.log").read_text()


@pytest.mark.parametrize('trigger', ['stop_file', 'deadline'])
def test_native_child_drains_response_without_dispatching_next_request(tmp_path, trigger):
    config = tmp_path / 'mock.yaml'
    config.write_text(yaml.safe_dump({'log_dir': str(tmp_path / 'native'),
                                     'budget': {'max_estimated_usd': 10}}))
    script = '''
        import asyncio, json, sys, time
        from pathlib import Path
        from types import SimpleNamespace
        import yaml
        from delegated_influence import campaign_worker, run, runtime
        config, stop = Path(sys.argv[1]), Path(sys.argv[2])
        trigger = sys.argv[3]
        deadline = time.monotonic() + 2
        calls = []
        async def provider(messages):
            calls.append(messages)
            if trigger == 'stop_file':
                stop.touch()
                await asyncio.sleep(.1)
            else:
                # The parent never writes STOP, for example because a backup blocks it.
                await asyncio.sleep(max(0, deadline - time.monotonic()) + .05)
            return '{"action":"pull","beneficiary":"P2"}', {'total_cost': .01}
        runtime.native_adapter.build_generator = lambda *args, **kw: provider
        md = {'seat_map': {'P1': 'mockllm/model', 'P2': 'mockllm/model'},
              'seed': 7, 'rounds': 1, 'actions_per_round': 2,
              'sampling_instruments': {'mockllm/model': {'max_tokens': 100}}}
        tasks = [('arm', SimpleNamespace(name='arm', dataset=[SimpleNamespace(id='one', metadata=md)]))]
        estimate = {'total_usd': 1, 'arms': [{'arm': 'arm', 'usd': 1, 'episodes': 1}]}
        run.main = lambda argv, **kw: run.launch_native_run(config, yaml.safe_load(config.read_text()), tasks, estimate, **kw)
        try:
            campaign_worker.run_child(config, stop, deadline)
        finally:
            (config.parent / 'requests.json').write_text(json.dumps(calls))
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script), str(config), str(tmp_path / 'STOP'), trigger],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'RunPaused' in result.stderr
    assert len(json.loads((tmp_path / 'requests.json').read_text())) == 1
    assert (tmp_path / 'STOP').exists() == (trigger == 'stop_file')
    with sqlite3.connect(tmp_path / 'native' / 'run.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM calls').fetchone()[0] == 1
        assert db.execute('SELECT result FROM calls').fetchone()[0] is not None
        assert db.execute('SELECT status FROM episodes').fetchone()[0] == 'paused'
    assert len(list((tmp_path / 'native').glob('run-artifacts/*/source-snapshot.tar.gz'))) == 1


def test_backup_copies_committed_sqlite_state_and_excludes_secrets(tmp_path):
    work, backup = tmp_path / 'work', tmp_path / 'backup'
    work.mkdir()
    (work / '.keys').mkdir()
    (work / '.keys' / 'secret.json').write_text('do not copy')
    (work / '.env').write_text('do not copy')
    (work / 'unfinished.episode.json.gz.tmp').write_text('unfinished')
    (work / 'finished.episode.json.gz').write_bytes(b'finished export')
    (work / 'escape.json').symlink_to(work / '.keys' / 'secret.json')
    database = work / 'spending.sqlite3'
    with sqlite3.connect(database) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE evidence (value INTEGER)')
        writer.execute('INSERT INTO evidence VALUES (1)')
        writer.commit()
        writer.execute('INSERT INTO evidence VALUES (2)')  # Not committed yet.
        assert campaign_worker.snapshot(work, backup) is True
        with sqlite3.connect(backup / database.name) as saved:
            assert saved.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            assert saved.execute('SELECT value FROM evidence').fetchall() == [(1,)]
        writer.commit()
        assert campaign_worker.snapshot(work, backup) is True
        with sqlite3.connect(backup / database.name) as saved:
            assert saved.execute('SELECT value FROM evidence').fetchall() == [(1,), (2,)]
    assert {path.name for path in backup.iterdir()} == {'spending.sqlite3', 'finished.episode.json.gz'}


def test_backup_recopies_only_changed_files(tmp_path, monkeypatch):
    work, backup = tmp_path / 'work', tmp_path / 'backup'
    work.mkdir()
    export = work / 'done.episode.json.gz'
    export.write_bytes(b'first')
    copies, original = [], shutil.copyfile
    monkeypatch.setattr(campaign_worker.shutil, 'copyfile', lambda src, dst: copies.append(Path(src).name) or original(src, dst))
    assert campaign_worker.snapshot(work, backup) is True
    assert campaign_worker.snapshot(work, backup) is True
    assert copies == ['done.episode.json.gz']
    export.write_bytes(b'later')  # Same size; only the modification time moves.
    os.utime(export, ns=(export.stat().st_atime_ns, export.stat().st_mtime_ns + 1_000_000))
    assert campaign_worker.snapshot(work, backup) is True
    assert copies == ['done.episode.json.gz'] * 2
    assert (backup / export.name).read_bytes() == b'later'


@pytest.mark.parametrize('change', ['cap', 'ledger', 'log_dir', 'duplicate', 'overlap', 'duration', 'nan'])
def test_invalid_plan_cannot_start_processes(tmp_path, monkeypatch, change):
    plan = plan_for(tmp_path)
    path = Path(plan['configs'][0]['path'])
    cfg = yaml.safe_load(path.read_text())
    if change == 'cap':
        cfg['spending']['cap_usd'] = 11
    elif change == 'ledger':
        cfg['spending']['ledger'] = str(tmp_path / 'different.sqlite3')
    elif change == 'log_dir':
        cfg['log_dir'] = str(tmp_path / 'outside')
    elif change == 'duplicate':
        plan['configs'].append(plan['configs'][0])
    elif change == 'overlap':
        plan['backup_dir'] = str(Path(plan['work_dir']) / 'backups')
    elif change == 'duration':
        plan['run_seconds'] = 24 * 3600
    elif change == 'nan':
        plan['allocation_usd'] = float('nan')
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(campaign_worker.subprocess, 'Popen', lambda *a, **kw: pytest.fail('invalid plan launched'))
    with pytest.raises(ValueError):
        campaign_worker.supervise(plan)


def test_campaign_evidence_blocks_restart(tmp_path, monkeypatch):
    plan = plan_for(tmp_path, count=1)
    real_children(monkeypatch, 'pass')
    campaign_worker.supervise(plan)
    monkeypatch.setattr(campaign_worker.subprocess, 'Popen', lambda *a, **kw: pytest.fail('campaign restarted'))
    with pytest.raises(ValueError, match='reuse'):
        campaign_worker.supervise(plan)


@pytest.mark.parametrize('trigger', ['deadline', 'failure', 'startup_error'])
def test_stop_paths_join_all_children_before_final(tmp_path, monkeypatch, trigger):
    plan = plan_for(tmp_path)
    plan['run_seconds'] = 1
    children = real_children(monkeypatch, f'''
        import sys, time
        from pathlib import Path
        if {trigger!r} == 'failure' and Path(sys.argv[1]).stem == 'run-0':
            sys.exit(3)
        stop = Path(sys.argv[3])
        while not stop.exists():
            assert not (stop.parent / 'final.json').exists()
            time.sleep(.01)
        time.sleep(.1)
        (stop.parent / (Path(sys.argv[1]).stem + '-drained.json')).write_text('{{}}')
    ''')
    if trigger == 'startup_error':
        start = campaign_worker.subprocess.Popen
        def fail_second(command, **kwargs):
            if children:
                raise OSError('cannot create second process')
            return start(command, **kwargs)
        monkeypatch.setattr(campaign_worker.subprocess, 'Popen', fail_second)
    result = campaign_worker.supervise(plan)
    assert all(child.poll() is not None for child in children)
    assert result['forced_termination'] is False
    assert result['status'] != 'complete'
    assert result['stop_reason'] == {'deadline': 'deadline', 'failure': 'child_failure',
                                      'startup_error': 'supervisor_error'}[trigger]
    assert (Path(plan['backup_dir']) / 'final.json').exists()


def test_backup_failure_does_not_abandon_children_or_claim_complete(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    children = real_children(monkeypatch, 'import time; time.sleep(.2)')
    def unavailable(*args):
        raise OSError('backup storage unavailable')
    monkeypatch.setattr(campaign_worker, 'snapshot', unavailable)
    result = campaign_worker.supervise(plan)
    assert all(child.poll() == 0 for child in children)
    assert result['snapshot_complete'] is False
    assert result['status'] != 'complete'


def test_forced_shutdown_keeps_full_request_grace_and_reports_failure(tmp_path, monkeypatch):
    from delegated_influence import task
    plan = plan_for(tmp_path, count=1)
    plan['run_seconds'] = 1
    elapsed = [0.0]
    class UnresponsiveChild:
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            assert elapsed[0] >= plan['run_seconds'] + task.CALL_TIMEOUT_S + 60
            self.returncode = -15
        def wait(self, timeout=None):
            assert self.returncode is not None
            return self.returncode
    monkeypatch.setattr(campaign_worker.subprocess, 'Popen', lambda *a, **kw: UnresponsiveChild())
    monkeypatch.setattr(campaign_worker.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(campaign_worker.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(campaign_worker, 'snapshot', lambda *args, **kwargs: True)
    result = campaign_worker.supervise(plan)
    assert result['forced_termination'] is True
    assert result['status'] != 'complete'
    assert result['child_codes'] == {'run-0': -15}


def test_second_signal_during_final_snapshot_still_writes_final(tmp_path, monkeypatch):
    from delegated_influence import task
    plan = plan_for(tmp_path, count=1)
    real_children(monkeypatch, 'pass')
    aborted = []
    def recorder(*_):  # Stands in for the default handler, which would abort the test session.
        aborted.append('handler outside supervise ran')
    outer = signal.signal(signal.SIGINT, recorder)
    try:
        def interrupt(work, backup, *, max_seconds):
            if max_seconds == task.CALL_TIMEOUT_S + 60:  # The final snapshot.
                signal.raise_signal(signal.SIGINT)
            return True
        monkeypatch.setattr(campaign_worker, 'snapshot', interrupt)
        result = campaign_worker.supervise(plan)
        assert signal.getsignal(signal.SIGINT) is recorder
    finally:
        signal.signal(signal.SIGINT, outer)
    assert not aborted
    assert result['status'] == 'complete'
    assert json.loads((Path(plan['backup_dir']) / 'final.json').read_text()) == result


def test_full_mock_campaign_exports_two_native_journals(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    children = real_children(monkeypatch, '''
        import asyncio, json, sys, time
        from pathlib import Path
        from types import SimpleNamespace
        import yaml
        from delegated_influence import campaign_worker, run, runtime
        config, stop = Path(sys.argv[1]), Path(sys.argv[3])
        cfg = yaml.safe_load(config.read_text())
        cfg.pop('spending')  # This test uses mock providers only.
        cfg['budget'] = {'max_estimated_usd': 10}
        directory = Path(cfg['log_dir'])
        async def provider(messages):
            (directory / 'started.json').write_text('{}')
            until = time.monotonic() + 5
            while len(list(directory.parent.glob('*/started.json'))) != 2:
                assert time.monotonic() < until, 'native processes did not overlap'
                await asyncio.sleep(.01)
            return '{"action":"pull","beneficiary":"P2"}', {'total_cost': .01}
        runtime.native_adapter.build_generator = lambda *args, **kw: provider
        md = {'seat_map': {'P1': 'mockllm/model', 'P2': 'mockllm/model'},
              'seed': 7, 'rounds': 1, 'actions_per_round': 2,
              'sampling_instruments': {'mockllm/model': {'max_tokens': 100}}}
        tasks = [('arm', SimpleNamespace(name='arm', dataset=[SimpleNamespace(id='one', metadata=md)]))]
        estimate = {'total_usd': 1, 'arms': [{'arm': 'arm', 'usd': 1, 'episodes': 1}]}
        run.main = lambda argv, **kw: run.launch_native_run(config, cfg, tasks, estimate, **kw)
        campaign_worker.run_child(config, stop, float(sys.argv[5]))
    ''')
    result = campaign_worker.supervise(plan)
    assert result['status'] == 'complete', [path.read_text() for path in Path(plan['work_dir']).glob('*.log')]
    assert all(child.poll() == 0 for child in children)
    backup = Path(plan['backup_dir'])
    assert len(list(backup.glob('*/episodes/*.episode.json.gz'))) == 2
    assert len(list(backup.glob('*/run-artifacts/*/source-snapshot.tar.gz'))) == 2
    for database in backup.glob('*/run.sqlite3'):
        with sqlite3.connect(database) as saved:
            assert saved.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            assert saved.execute('SELECT status FROM episodes').fetchone() == ('complete',)
