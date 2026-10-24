"""Durable game calls, event progress, and shared run reservations.

The journal is local to one run directory. One process owns it. Completed calls
can be replayed only inside their original episode and with identical input.
"""
from __future__ import annotations

import fcntl
import gzip
import time
import hashlib
import json
import math
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path


class RunPaused(BaseException):
    """Stop infrastructure failures before the game can turn them into forfeits."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def pack(value):
    return zlib.compress(canonical(value).encode())


def unpack(value):
    return json.loads(zlib.decompress(value)) if value is not None else None


def valid_cost(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def merge_recovery_evidence(saved, supplied, path='usage'):
    """Fill missing facts without replacing known provider evidence."""
    merged = dict(saved)
    for key, value in supplied.items():
        old = saved.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            merged[key] = merge_recovery_evidence(old, value, f'{path}.{key}')
        elif old is None or (path == 'usage' and key == 'served_provider' and old == ''):
            merged[key] = value
        elif canonical(old) != canonical(value):
            # JSON distinguishes bools from numbers. Equivalent integer/float
            # amounts are accepted, but True can never stand for one token.
            if (isinstance(old, (int, float)) and not isinstance(old, bool)
                    and isinstance(value, (int, float)) and not isinstance(value, bool) and old == value):
                continue
            raise ValueError(f'recovery conflicts with saved evidence: {path}.{key}')
    return merged


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise RunPaused('another process owns this run journal') from exc
        self.last_status_at = 0.0
        # WAL lets the supervisor's backup read a large journal without blocking this writer;
        # the busy timeout covers checkpoints and any short lock a reader still takes.
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, metadata BLOB NOT NULL, status TEXT NOT NULL,
                reserved REAL NOT NULL, record BLOB, started TEXT NOT NULL, sample_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, episode TEXT NOT NULL, seat TEXT NOT NULL,
                sequence INTEGER NOT NULL, input_hash TEXT NOT NULL, payload BLOB NOT NULL,
                status TEXT NOT NULL, result BLOB, reported REAL, estimated REAL,
                error TEXT, created TEXT NOT NULL, wire_request BLOB);
            CREATE INDEX IF NOT EXISTS calls_episode ON calls(episode);
            CREATE TABLE IF NOT EXISTS resolutions (
                call_id TEXT NOT NULL, created TEXT NOT NULL, action TEXT NOT NULL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS call_attempts (
                id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, created TEXT NOT NULL,
                status TEXT NOT NULL, result BLOB, reported REAL, estimated REAL,
                error TEXT, wire_request BLOB, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                episode TEXT NOT NULL, sequence INTEGER NOT NULL, payload BLOB NOT NULL,
                PRIMARY KEY (episode, sequence));
        ''')
        # The exclusive process lock proves no prior owner can still send these calls.
        self.db.execute("UPDATE calls SET status='uncertain' WHERE status='pending'")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()
        self.lock.close()

    def bind(self, key, value):
        serialized = canonical(value)
        old = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        if old and old['value'] != serialized:
            raise RunPaused(f'{key} changed; use the saved source/config or a new run directory')
        self.db.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)', (key, serialized))

    def episode(self, episode_id):
        row = self.db.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), 'metadata': unpack(row['metadata']), 'record': unpack(row['record'])}

    def start_episode(self, episode_id, metadata, reserve_usd, budget_usd, *, allow_estimated_cost=False, sample_id=None):
        if not valid_cost(reserve_usd) or not valid_cost(budget_usd) or budget_usd <= 0:
            raise ValueError('reservation and budget must be finite nonnegative amounts; budget must exceed zero')
        sample_id = str(sample_id if sample_id is not None else episode_id)
        old = self.episode(episode_id)
        if old:
            if canonical(old['metadata']) != canonical(metadata) or old['sample_id'] != sample_id:
                raise RunPaused(f'episode {episode_id}: metadata changed')
            return old
        # Unknown billing on a running or completed game needs review before more games start. A
        # paused game keeps its evidence and its ledger hold and does not block the others; the
        # resume path in runtime._run_tasks still refuses to dispatch into a journal with any
        # unresolved uncertain call.
        blocking = self.db.execute('''
            SELECT COALESCE(SUM(CASE WHEN c.status='uncertain' THEN 1 ELSE 0 END),0) uncertain,
                   COALESCE(SUM(CASE WHEN c.reported IS NULL AND c.estimated IS NULL AND c.status!='pending' THEN 1 ELSE 0 END),0) unpriced,
                   COALESCE(SUM(CASE WHEN c.reported IS NULL AND c.status!='pending' THEN 1 ELSE 0 END),0) unknown
            FROM calls c JOIN episodes e ON e.id=c.episode WHERE e.status!='paused' ''').fetchone()
        if blocking['uncertain']:
            raise RunPaused('uncertain calls require operator review before new games')
        if blocking['unpriced']:
            raise RunPaused('unpriced calls: billing or a usable estimate is required before new games')
        if blocking['unknown'] and not allow_estimated_cost:
            raise RunPaused('unknown cost: new games paused; review billing or explicitly allow estimates')
        exposure = self.db.execute('''
            SELECT COALESCE(SUM(CASE WHEN e.status='complete' THEN COALESCE(c.debit,0)
                ELSE MAX(e.reserved,COALESCE(c.debit,0)) END),0)
            FROM episodes e LEFT JOIN (
                SELECT episode, SUM(COALESCE(reported,estimated,0)) debit FROM calls GROUP BY episode
            ) c ON c.episode=e.id
        ''').fetchone()[0]
        if exposure + reserve_usd > budget_usd + 1e-9:
            raise RunPaused(f'budget: ${exposure:.2f} spent/reserved plus ${reserve_usd:.2f} exceeds ${budget_usd:.2f}')
        self.db.execute('INSERT INTO episodes VALUES (?,?,?,?,NULL,?,?)',
                        (episode_id, pack(metadata), 'running', reserve_usd,
                         datetime.now(timezone.utc).isoformat(), sample_id))
        return self.episode(episode_id)

    def needs_dispatch(self, episode_id, seat, sequence):
        row = self.db.execute('SELECT status FROM calls WHERE id=?',
                              (f'{episode_id}:{seat}:{sequence}',)).fetchone()
        return row is None or row['status'] == 'retryable'

    def begin_call(self, episode_id, seat, sequence, messages):
        call_id = f'{episode_id}:{seat}:{sequence}'
        digest = hashlib.sha256(canonical(messages).encode()).hexdigest()
        row = self.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()
        if row:
            if row['input_hash'] != digest:
                raise RunPaused(f'call {call_id}: input changed')
            if row['status'] == 'retryable':
                self.db.execute("UPDATE calls SET status='pending',reported=NULL,estimated=NULL,result=NULL,error=NULL,wire_request=NULL WHERE id=?", (call_id,))
                return call_id, None
            if row['status'] != 'complete':
                raise RunPaused(f'call {call_id}: uncertain response; review before any retry')
            result = unpack(row['result'])
            if not isinstance(result, dict) or result.get('response') is None:
                # Older journals could commit complete before marking an empty
                # response uncertain. Keep the bill and require explicit recovery.
                self.uncertain(call_id, 'completed call has no saved response')
                raise RunPaused(f'call {call_id}: uncertain response; review before any retry')
            return call_id, result
        self.db.execute('INSERT INTO calls (id,episode,seat,sequence,input_hash,payload,status,created) VALUES (?,?,?,?,?,?,?,?)',
                        (call_id, episode_id, seat, sequence, digest, pack(messages), 'pending',
                         datetime.now(timezone.utc).isoformat()))
        self.write_status()
        return call_id, None

    def save_request(self, call_id, body):
        """Commit the exact provider request body before the HTTP client sends it."""
        self.db.execute('UPDATE calls SET wire_request=? WHERE id=?', (pack(body), call_id))

    def finish_call(self, call_id, response, usage, estimate):
        """Commit response evidence and its valid replay state in one transition."""
        reported = usage.get('total_cost')
        reported = reported if valid_cost(reported) else None
        estimate = estimate if valid_cost(estimate) else None
        usage = {**usage, 'total_cost': reported, 'estimated_cost_usd': estimate}
        result = {'response': response, 'usage': usage}
        status = 'uncertain' if response is None else 'complete'
        error = str(usage.get('error') or 'empty provider result') if response is None else None
        self.db.execute('UPDATE calls SET status=?,result=?,reported=?,estimated=?,error=COALESCE(?,error) WHERE id=?',
                        (status, pack(result), reported, estimate, error, call_id))
        return result

    def uncertain(self, call_id, error):
        self.db.execute('UPDATE calls SET status=?,error=? WHERE id=?',
                        ('uncertain', error, call_id))

    def resolve_uncertain(self, call_id, *, evidence, response=None, usage=None, confirm_unbilled=False,
                          validate_recovery=None):
        """Record an operator decision. Never guess whether an interrupted request was charged."""
        row = self.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()
        if not row or row['status'] not in ('pending','uncertain'):
            raise ValueError('call is not uncertain')
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('billing or response evidence is required')
        if confirm_unbilled == (response is not None):
            raise ValueError('supply either a recovered response or confirmation that the request was not billed')
        if row['reported'] is not None:
            if confirm_unbilled:
                if row['reported'] > 0:
                    raise ValueError('a known positive bill cannot be confirmed unbilled')
            elif (not isinstance(usage, dict) or not valid_cost(usage.get('total_cost'))
                  or usage['total_cost'] != row['reported']):
                raise ValueError('a recovered response must preserve its known bill')
        if not confirm_unbilled:
            if usage is not None and not isinstance(usage, dict):
                raise ValueError('recovery usage evidence must be an object')
            saved = unpack(row['result']) or {}
            if saved.get('response') is not None and canonical(saved['response']) != canonical(response):
                raise ValueError('recovery conflicts with the saved response')
            usage = merge_recovery_evidence(saved.get('usage') or {}, usage or {})
            usage['accounting_complete'] = False
            if validate_recovery is not None:
                validate_recovery(usage)
        result = None
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            # begin_call clears the current attempt for a permitted retry. Keep
            # its original response, wire body, error, and bill in this audit row.
            self.db.execute('''INSERT INTO call_attempts
                (call_id,created,status,result,reported,estimated,error,wire_request,evidence)
                VALUES (?,?,?,?,?,?,?,?,?)''', (call_id, datetime.now(timezone.utc).isoformat(),
                row['status'], row['result'], row['reported'], row['estimated'], row['error'], row['wire_request'], evidence))
            self.db.execute('INSERT INTO resolutions VALUES (?,?,?,?)',
                (call_id, datetime.now(timezone.utc).isoformat(),
                 'confirmed_unbilled' if confirm_unbilled else 'recovered_response', evidence))
            if confirm_unbilled:
                self.db.execute("UPDATE calls SET status='retryable',reported=0 WHERE id=?", (call_id,))
            else:
                result = self.finish_call(call_id, response, usage, None)
        self.write_status(force=True)
        return result

    def call(self, call_id):
        row = self.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()
        result = unpack(row['result']) or {}
        return {'call_id': call_id, 'input': unpack(row['payload']),
                'response': result.get('response'), 'usage': result.get('usage', {}),
                'error': row['error'], 'provider_request_body': unpack(row['wire_request'])}

    def save_event(self, episode_id, sequence, event):
        old = self.db.execute('SELECT payload FROM events WHERE episode=? AND sequence=?',
                              (episode_id, sequence)).fetchone()
        if old and canonical(unpack(old['payload'])) != canonical(event):
            raise RunPaused(f'episode {episode_id}: event replay changed at {sequence}')
        self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?)',
                        (episode_id, sequence, pack(event)))

    def complete(self, episode_id, record):
        if self.db.execute("SELECT 1 FROM calls WHERE episode=? AND status!='complete' LIMIT 1",
                           (episode_id,)).fetchone():
            raise RunPaused('episode has uncertain calls and cannot enter analysis')
        self.db.execute('UPDATE episodes SET status=?,record=? WHERE id=?',
                        ('complete', pack(record), episode_id))

    def export_episode(self, episode_id, path):
        episode = self.episode(episode_id)
        if episode['status'] != 'complete':
            raise ValueError('only complete episodes can enter the analysis pool')
        source = self.db.execute("SELECT value FROM meta WHERE key='origin'").fetchone()
        origin = json.loads(source['value']) if source else {}
        result = {'schema_version': 'native-episode-v1', **origin, 'status': 'complete',
                  'episode_id': episode_id, 'sample_id': episode['sample_id'],
                  'started': episode['started'], 'metadata': {**episode['metadata'],
                      **{key: episode['record'][key] for key in ('served_instruments','runtime_instrument')
                         if key in episode['record']}},
                  'record': episode['record']}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(temporary, 'wt') as stream:
            stream.write(canonical(result) + '\n')
        temporary.replace(path)

    def summary(self):
        rows = self.db.execute('''SELECT episode, seat, COUNT(*) calls,
            COALESCE(SUM(reported),0) reported_cost_usd,
            COALESCE(SUM(CASE WHEN reported IS NULL THEN estimated ELSE 0 END),0) estimated_cost_usd,
            SUM(CASE WHEN reported IS NULL AND status!='pending' THEN 1 ELSE 0 END) unknown_cost_calls,
            SUM(CASE WHEN reported IS NULL AND estimated IS NULL AND status!='pending' THEN 1 ELSE 0 END) unpriced_calls,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) pending_calls,
            SUM(CASE WHEN status='uncertain' THEN 1 ELSE 0 END) uncertain_calls
            FROM calls GROUP BY episode,seat''').fetchall()
        fields = ('calls','reported_cost_usd','estimated_cost_usd','unknown_cost_calls','uncertain_calls','unpriced_calls','pending_calls')
        episodes = dict(self.db.execute('SELECT status,COUNT(*) FROM episodes GROUP BY status').fetchall())
        return {**{field: sum(row[field] for row in rows) for field in fields},
                'episodes': episodes, 'by_agent': {f"{row['episode']}:{row['seat']}": dict(row) for row in rows},
                'cost_note': 'Reported subtotal excludes unknown bills. Estimates are not a billing cap.'}

    def write_status(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_status_at < 1:
            return
        self.last_status_at = now
        path = self.path.parent / 'status.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.summary(), indent=2) + '\n')
        temporary.replace(path)
