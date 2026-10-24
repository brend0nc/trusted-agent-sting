"""One-shot campaign supervisor. Backups never authorize a restart or release spending."""
import argparse
from contextlib import closing, suppress
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time


def validate_plan(plan):
    """Reject unsafe paths and independent caps before any process can dispatch."""
    import yaml
    def absolute(value):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('campaign paths must be absolute')
        return Path(value).resolve()
    def amount(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    if not isinstance(plan, dict) or not isinstance(plan.get('campaign_id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', plan['campaign_id']):
        raise ValueError('invalid campaign_id')
    allocation = plan.get('allocation_usd')
    if not amount(allocation):
        raise ValueError('allocation_usd must be positive and finite')
    for key, default in (('run_seconds', 23 * 3600), ('backup_seconds', 30)):
        if type(plan.get(key, default)) is not int or plan.get(key, default) <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if plan.get('run_seconds', 23 * 3600) > 23 * 3600:
        raise ValueError('run_seconds must leave time before the Sandbox deadline')
    work, backup = absolute(plan.get('work_dir')), absolute(plan.get('backup_dir'))
    if work.is_relative_to(backup) or backup.is_relative_to(work):
        raise ValueError('work and backup paths must not overlap')
    if not isinstance(plan.get('configs'), list) or not plan['configs']:
        raise ValueError('configs must be a nonempty list')
    names, logs = set(), []
    for entry in plan['configs']:
        if not isinstance(entry, dict):
            raise ValueError('each config entry must be a mapping')
        name = entry.get('name')
        path = absolute(entry.get('path'))
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in names or path.stem != name:
            raise ValueError('config names must be unique file stems')
        names.add(name)
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict):
            raise ValueError('each config must be a mapping')
        log = absolute(cfg.get('log_dir'))
        if log == work or not log.is_relative_to(work) or any(log.is_relative_to(old) or old.is_relative_to(log) for old in logs):
            raise ValueError('log_dir must be distinct and below work_dir')
        logs.append(log)
        spending = cfg.get('spending') or {}
        if absolute(spending.get('ledger')) != work / 'spending.sqlite3' or not amount(spending.get('cap_usd')) or spending['cap_usd'] != allocation:
            raise ValueError('each config must share the exact campaign spending ledger and cap')


def run_child(config, stop_file, deadline):
    """Run the saved native launcher; stop new dispatch at the deadline or once the stop file exists."""
    if not math.isfinite(deadline):
        raise ValueError('child deadline must be finite')
    from delegated_influence import run
    stop = Path(stop_file)
    return run.main([str(config), '--runner', 'native'],
                    dispatch_paused=lambda: time.monotonic() >= deadline or stop.exists())


def snapshot(work_dir, backup_dir, *, max_seconds=5):
    """Publish changed files atomically. The time limit is checked only between files."""
    work, backup = Path(work_dir), Path(backup_dir)
    complete = True
    expires = time.monotonic() + max_seconds
    # Journals first: they hold the evidence, and every pass copies them.
    for source in sorted(work.rglob('*'), key=lambda path: (path.suffix != '.sqlite3', path)):
        relative = source.relative_to(work)
        if (not source.is_file() or source.is_symlink()
                or any(part.startswith('.') for part in relative.parts)
                or source.name == 'final.json'
                or not source.name.endswith(('.sqlite3', '.json', '.json.gz', '.jsonl.gz', '.log', '.tar.gz', '.yaml'))):
            continue
        if source.name in ('source-snapshot.tar.gz', 'run-config.yaml') and not (source.parent / 'manifest.json').exists():
            continue
        temporary = None
        try:
            target = backup / relative
            stat = source.stat()
            if source.suffix != '.sqlite3' and target.exists():
                saved = target.stat()
                if (stat.st_size, stat.st_mtime_ns) == (saved.st_size, saved.st_mtime_ns):
                    continue
            if time.monotonic() >= expires:
                complete = False
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            if source.suffix == '.sqlite3':
                temporary.unlink(missing_ok=True)
                with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as origin, \
                        closing(sqlite3.connect(temporary)) as destination:
                    # One step copies every page under a single read lock; a concurrent commit cannot restart it.
                    origin.backup(destination, pages=-1)
                    destination.execute('PRAGMA journal_mode=DELETE')
                    if destination.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                        raise sqlite3.DatabaseError('invalid backup')
            else:
                shutil.copyfile(source, temporary)
                with suppress(OSError):  # A store without utime support recopies the file next pass.
                    os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            temporary.replace(target)
        except (OSError, sqlite3.Error):
            complete = False
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
    return complete


def supervise(plan):
    """Start every configured process, drain them all, then publish final evidence."""
    validate_plan(plan)
    work, backup = Path(plan['work_dir']), Path(plan['backup_dir'])
    if any(directory.exists() and any(directory.iterdir()) for directory in (work, backup)):
        raise ValueError('campaign evidence exists; cannot reuse work or backup directory')
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'started.json').open('x') as marker:
        json.dump({'campaign_id': plan['campaign_id'], 'allocation_usd': plan['allocation_usd']}, marker)
    children, handles = {}, []
    forced, reason, error = False, 'finished', None
    from delegated_influence import task
    grace = task.CALL_TIMEOUT_S + 60
    stop = work / 'STOP'
    deadline = time.monotonic() + plan.get('run_seconds', 23 * 3600)
    next_backup = 0
    interrupted = False
    def request_stop(*_):
        nonlocal interrupted
        interrupted = True
        with suppress(OSError):
            stop.touch()
    previous_signals = {kind: signal.signal(kind, request_stop) for kind in (signal.SIGINT, signal.SIGTERM)}
    def best_effort_snapshot(*, final=False):
        try:
            return snapshot(work, backup, max_seconds=grace if final else plan.get('backup_seconds', 30))
        except Exception:
            return False
    def periodic_backup():
        nonlocal next_backup
        if time.monotonic() >= next_backup:
            best_effort_snapshot()
            next_backup = time.monotonic() + plan.get('backup_seconds', 30)
    try:
        for entry in plan['configs']:
            output = (work / (entry['name'] + '.log')).open('ab', buffering=0)
            handles.append(output)
            children[entry['name']] = subprocess.Popen([
                sys.executable, '-m', 'delegated_influence.campaign_worker',
                '--child', entry['path'], '--stop-file', str(stop), '--deadline', str(deadline)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while any(child.poll() is None for child in children.values()):
            if interrupted:
                reason = 'signal'
                break
            if any(child.poll() not in (None, 0) for child in children.values()):
                reason = 'child_failure'
                break
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            periodic_backup()
            time.sleep(.1)
    except BaseException as exc:
        reason, error = 'supervisor_error', type(exc).__name__
    finally:
        if any(child.poll() is None for child in children.values()):
            with suppress(OSError):
                stop.touch()
            until = time.monotonic() + grace
            while any(child.poll() is None for child in children.values()) and time.monotonic() < until:
                periodic_backup()
                time.sleep(.1)
            for child in children.values():
                if child.poll() is None:
                    forced = True
                    child.terminate()
            for child in children.values():
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()
    complete = best_effort_snapshot(final=True)
    codes = {name: child.wait() for name, child in children.items()}
    result = {'campaign_id': plan['campaign_id'], 'child_codes': codes,
              'forced_termination': forced, 'snapshot_complete': complete,
              'stop_reason': reason, 'error': error,
              'status': 'complete' if reason == 'finished' and not forced and complete and not error and all(code == 0 for code in codes.values()) else 'stopped'}
    for directory in (work, backup):
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'final.json.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(directory / 'final.json')
    # A second signal during the final snapshot must not abort before final.json exists.
    for kind, previous in previous_signals.items():
        signal.signal(kind, previous)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', nargs='?')
    parser.add_argument('--child')
    parser.add_argument('--stop-file')
    parser.add_argument('--deadline', type=float, help='absolute host monotonic dispatch deadline')
    args = parser.parse_args()
    if args.child:
        if not args.stop_file or args.plan or args.deadline is None or not math.isfinite(args.deadline):
            parser.error('--child requires --stop-file, a finite --deadline, and no plan')
        return run_child(args.child, args.stop_file, args.deadline)
    if not args.plan or args.stop_file or args.deadline is not None:
        parser.error('a plan is required outside child mode')
    result = supervise(json.loads(Path(args.plan).read_text()))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
