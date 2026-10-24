"""Request admission shared by processes on one host and one spending ledger.

SQLite retains route/provider cooldowns and active slots. A crashed owner's slots
never expire automatically: uncertain remote requests need evidence before reuse.
This is an operational limit, not a provider quota or an invoice guarantee.
"""
from contextlib import asynccontextmanager, contextmanager
import asyncio
import math
from pathlib import Path
import sqlite3
import time

from delegated_influence.journal import RunPaused, canonical, unpack, valid_cost


DEFAULT_POLICY = {
    'route': {'max_in_flight': 2, 'min_interval_seconds': 1.0},
    'provider': {'max_in_flight': 4, 'min_interval_seconds': .25},
    'routes': {}, 'providers': {},
}
POLL_SECONDS = .25


def normalize_policy(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        raise ValueError('request limits require route, provider, routes, or providers')
    def limits(row, default):
        if not isinstance(row, dict) or set(row) - set(default):
            raise ValueError('request limits require max_in_flight and min_interval_seconds')
        row = {**default, **row}
        count, interval = row['max_in_flight'], row['min_interval_seconds']
        if type(count) is not int or count < 1:
            raise ValueError('max_in_flight must be a positive integer')
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError('min_interval_seconds must be finite and nonnegative')
        return row
    result = {kind: limits(value.get(kind, {}), DEFAULT_POLICY[kind]) for kind in ('route', 'provider')}
    for kind in ('routes', 'providers'):
        rows = value.get(kind, {})
        if not isinstance(rows, dict) or any(not isinstance(key, str) or not key.strip() for key in rows):
            raise ValueError(f'{kind} must map model or provider names to request limits')
        result[kind] = {key: limits(row, result[kind[:-1]]) for key, row in rows.items()}
    return result


async def interruptible_wait(seconds, stopped=None):
    """Check STOP/deadline during a backoff; do not interrupt a sent request."""
    until = time.monotonic() + seconds
    while True:
        if stopped and stopped():
            raise RunPaused('dispatch stopped; no further requests will be sent')
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, POLL_SECONDS))


@asynccontextmanager
async def admission(control, call_key, model, provider, stopped):
    if control is not None:
        await control.acquire(call_key, model, provider, stopped)
    try:
        await interruptible_wait(0, stopped)
        yield
    finally:
        if control is not None:
            control.release(call_key)


class RequestControl:
    def __init__(self, path, *, policy=None, clock=time.time):
        self.policy, self.clock = normalize_policy(policy), clock
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS buckets (
                    key TEXT PRIMARY KEY, next_at REAL NOT NULL, cooldown_until REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS active_requests (
                    call_key TEXT PRIMARY KEY, route TEXT NOT NULL, provider TEXT NOT NULL, started REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS active_route ON active_requests(route);
                CREATE INDEX IF NOT EXISTS active_provider ON active_requests(provider);
                CREATE TABLE IF NOT EXISTS cooldowns (
                    id INTEGER PRIMARY KEY, route TEXT NOT NULL, provider TEXT NOT NULL,
                    created REAL NOT NULL, until REAL NOT NULL, evidence TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recovered_slots (
                    id INTEGER PRIMARY KEY, call_key TEXT NOT NULL, created REAL NOT NULL, evidence TEXT NOT NULL);
            ''')
            with self.transaction():
                saved = self.db.execute('SELECT value FROM policy WHERE id=1').fetchone()
                if saved and saved['value'] != canonical(self.policy):
                    raise RunPaused('request control policy changed; retain the shared policy')
                self.db.execute('INSERT OR IGNORE INTO policy VALUES (1,?)', (canonical(self.policy),))
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def keys(self, model, provider):
        return canonical(['route', provider, model]), canonical(['provider', provider])

    def try_acquire(self, call_key, model, provider):
        """Return zero on admission, otherwise the minimum wait. Acquire both limits atomically."""
        route, provider_key = self.keys(model, provider)
        limits = (self.policy['routes'].get(model, self.policy['route']),
                  self.policy['providers'].get(provider, self.policy['provider']))
        with self.transaction():
            if self.db.execute('SELECT 1 FROM active_requests WHERE call_key=?', (call_key,)).fetchone():
                raise RunPaused(f'{call_key}: existing request slot needs review before dispatch')
            now, wait = self.clock(), 0
            for key, field, limit in zip((route, provider_key), ('route', 'provider'), limits):
                self.db.execute('INSERT OR IGNORE INTO buckets VALUES (?,0,0)', (key,))
                row = self.db.execute('SELECT * FROM buckets WHERE key=?', (key,)).fetchone()
                active = self.db.execute(f'SELECT count(*) FROM active_requests WHERE {field}=?', (key,)).fetchone()[0]
                wait = max(wait, row['next_at'] - now, row['cooldown_until'] - now,
                           POLL_SECONDS if active >= limit['max_in_flight'] else 0)
            if wait <= 0:
                self.db.execute('INSERT INTO active_requests VALUES (?,?,?,?)', (call_key, route, provider_key, now))
                for key, limit in zip((route, provider_key), limits):
                    self.db.execute('UPDATE buckets SET next_at=? WHERE key=?', (now + limit['min_interval_seconds'], key))
            return max(0, wait)

    async def acquire(self, call_key, model, provider, stopped=None):
        while True:
            await interruptible_wait(0, stopped)
            wait = self.try_acquire(call_key, model, provider)
            if wait == 0:
                return
            await interruptible_wait(min(wait, POLL_SECONDS), stopped)

    def release(self, call_key):
        self.db.execute('DELETE FROM active_requests WHERE call_key=?', (call_key,))

    def reconcile(self, journal, spending, namespace):
        """Recover only this exclusively locked journal's slots from durable evidence."""
        prefix = f'{namespace}:'
        for row in self.db.execute('SELECT call_key FROM active_requests').fetchall():
            key = row['call_key']
            if not key.startswith(prefix):
                continue
            call_id = key[len(prefix):]
            call = journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()
            held = spending.db.execute('SELECT * FROM spending_calls WHERE call_key=?', (key,)).fetchone() if spending else None
            evidence = None
            if call is None and (held is None or held['status'] == 'unbilled'):
                evidence = 'No journal call; dispatch never began. Any orphan reservation is confirmed unbilled.'
            elif call is not None and call['status'] == 'complete':
                result = unpack(call['result']) or {}
                cost = (result.get('usage') or {}).get('total_cost')
                if result.get('response') is not None and valid_cost(cost):
                    if spending is not None:
                        if held is None:
                            raise RunPaused(f'{key}: saved response has no spending reservation')
                        # A crash can occur after the response save but before settlement.
                        spending.settle(key, cost)
                    evidence = 'Complete saved response and provider bill; replay must reuse this response.'
            elif call is not None and call['status'] == 'retryable':
                resolution = journal.db.execute("SELECT evidence FROM resolutions WHERE call_id=? AND action='confirmed_unbilled' ORDER BY created DESC LIMIT 1", (call_id,)).fetchone()
                if resolution and (spending is None or held is not None and held['status'] in ('held', 'unbilled')):
                    # Admission/reservation can precede a crash before begin_call.
                    # retryable proves that the next attempt was never recorded or sent.
                    if spending is not None:
                        spending.confirm_unbilled(key, evidence=resolution['evidence'])
                    evidence = 'Confirmed unbilled: ' + resolution['evidence']
            if evidence is not None:
                with self.transaction():
                    self.db.execute('INSERT INTO recovered_slots (call_key,created,evidence) VALUES (?,?,?)',
                                    (key, self.clock(), evidence))
                    self.release(key)

    def cooldown(self, model, provider, seconds, *, evidence):
        if not math.isfinite(seconds) or seconds < 0 or not evidence:
            raise ValueError('cooldown needs a finite delay and evidence')
        route, provider_key = self.keys(model, provider)
        with self.transaction():
            now = self.clock()
            until = now + seconds
            for key in (route, provider_key):
                self.db.execute('INSERT OR IGNORE INTO buckets VALUES (?,0,0)', (key,))
                self.db.execute('UPDATE buckets SET cooldown_until=MAX(cooldown_until,?) WHERE key=?', (until, key))
            self.db.execute('INSERT INTO cooldowns (route,provider,created,until,evidence) VALUES (?,?,?,?,?)',
                            (route, provider_key, now, until, evidence))

    def close(self):
        self.db.close()
