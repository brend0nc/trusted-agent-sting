"""A shared, durable per-request spending cap, independent of run journals.

All callers sharing a billing cap must use this same local SQLite path. A reserve
transaction both checks capacity and grants exactly one dispatch permission.
Repeated reserve calls return dispatch_allowed=False. They do not grant retries.
Timeouts and unknown bills keep the full hold until settlement or an explicit,
evidenced confirmation that the request was unbilled.

Accounting uses integer nanodollars: cap rounds down, holds and bills round up.
Original decimal amounts and their exact reported total are also retained. Stored
integer strings avoid SQLite's 64-bit sum limit. Each mutation is constant-time
and commits with synchronous=FULL before returning. A bill above its original
bound commits its full amount, permanently stops dispatch, and then raises.
Ordinary opens cannot change the cap. An explicit, audited increase requires
the old cap, operator evidence, and confirmation that no runs remain active.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
import sqlite3

from delegated_influence.journal import RunPaused


NANODOLLARS_PER_USD = 1_000_000_000


class SpendingError(RunPaused):
    """Stop dispatch when a shared spending guarantee cannot be maintained."""


def _amount(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise SpendingError(f'{label} must be a finite nonnegative number')
    amount = value if isinstance(value, Decimal) else Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise SpendingError(f'{label} must be a finite nonnegative number')
    return amount


def _text(amount):
    value = format(amount, 'f')
    return (value.rstrip('0').rstrip('.') if '.' in value else value) if amount else '0'


def _nanos(amount, *, upward=True):
    numerator, denominator = amount.as_integer_ratio()
    scaled = numerator * NANODOLLARS_PER_USD
    return ((scaled + denominator - 1) // denominator) if upward else scaled // denominator


def _usd(nanos):
    whole, fraction = divmod(nanos, NANODOLLARS_PER_USD)
    return f'{whole}.{fraction:09d}'.rstrip('0').rstrip('.')


def _add_reported(left, right):
    left = Decimal(left)
    # Set enough decimal precision for every digit and a carry; never use the
    # process's ambient Decimal precision for accumulated reported bills.
    with localcontext() as context:
        context.prec = max(left.adjusted(), right.adjusted()) - min(
            left.as_tuple().exponent, right.as_tuple().exponent) + 2
        return _text(left + right)


class SpendingLedger:
    def __init__(self, path, cap_usd, *, existing_only=False):
        cap = _amount(cap_usd, 'cap')
        self.path = Path(path)
        if not existing_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.path.resolve().as_uri() + '?mode=rw' if existing_only else self.path
        self.db = sqlite3.connect(target, uri=existing_only, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            if existing_only:
                try:
                    policy = self._policy()
                except sqlite3.OperationalError:
                    raise SpendingError('path is not an initialized spending ledger') from None
                if policy is None:
                    raise SpendingError('path is not an initialized spending ledger')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS spending_policy (
                    id INTEGER PRIMARY KEY CHECK(id=1), cap_usd TEXT NOT NULL,
                    cap_nanos TEXT NOT NULL, spent_nanos TEXT NOT NULL,
                    held_nanos TEXT NOT NULL, reported_usd TEXT NOT NULL,
                    stopped_reason TEXT);
                CREATE TABLE IF NOT EXISTS spending_calls (
                    call_key TEXT PRIMARY KEY, maximum_usd TEXT NOT NULL,
                    maximum_nanos TEXT NOT NULL, status TEXT NOT NULL,
                    reported_usd TEXT, reported_nanos TEXT);
                CREATE TABLE IF NOT EXISTS spending_resolutions (
                    id INTEGER PRIMARY KEY, call_key TEXT NOT NULL,
                    evidence TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS spending_cap_increases (
                    id INTEGER PRIMARY KEY, old_cap_usd TEXT NOT NULL, new_cap_usd TEXT NOT NULL,
                    evidence TEXT NOT NULL, no_active_runs_confirmed INTEGER NOT NULL CHECK(no_active_runs_confirmed=1),
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            ''')
            with self._transaction():
                policy = self._policy()
                if policy is None:
                    self.db.execute('INSERT INTO spending_policy VALUES (1,?,?,?,?,?,NULL)',
                                    (_text(cap), str(_nanos(cap, upward=False)), '0', '0', '0'))
                elif Decimal(policy['cap_usd']) != cap:
                    raise SpendingError('shared spending cap differs; ordinary opens cannot change the cap')
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def _transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _policy(self):
        return self.db.execute('SELECT * FROM spending_policy WHERE id=1').fetchone()

    def _call(self, key):
        if not isinstance(key, str) or not key.strip():
            raise SpendingError('call key must be a nonempty string')
        return self.db.execute('SELECT * FROM spending_calls WHERE call_key=?', (key,)).fetchone()

    def increase_cap(self, expected_cap_usd, new_cap_usd, *, evidence, confirm_no_active_runs=False):
        """Increase capacity once, with an exact old-cap match and durable audit.

        Holds cannot identify active runs: an interrupted request can stay held
        indefinitely. The operator must stop all users of this shared ledger.
        This operation changes no call, charge, hold, or provider-violation stop.
        """
        expected = _amount(expected_cap_usd, 'expected cap')
        new = _amount(new_cap_usd, 'new cap')
        if new <= expected:
            raise SpendingError('new cap must be positive and greater than the expected old cap')
        if not isinstance(evidence, str) or not evidence.strip():
            raise SpendingError('a cap increase requires operator approval evidence')
        if confirm_no_active_runs is not True:
            raise SpendingError('confirm that no runs using this ledger remain active')
        with self._transaction():
            policy = self._policy()
            if Decimal(policy['cap_usd']) != expected:
                raise SpendingError('expected old cap does not match the current shared cap')
            self.db.execute('UPDATE spending_policy SET cap_usd=?,cap_nanos=? WHERE id=1',
                            (_text(new), str(_nanos(new, upward=False))))
            audit_id = self.db.execute('''INSERT INTO spending_cap_increases
                (old_cap_usd,new_cap_usd,evidence,no_active_runs_confirmed) VALUES (?,?,?,1)''',
                (_text(expected), _text(new), evidence)).lastrowid
            audit = dict(self.db.execute('SELECT * FROM spending_cap_increases WHERE id=?', (audit_id,)).fetchone())
        return audit

    def cap_increases(self):
        """Return the ordered audit chain for explicit resume reconciliation."""
        return [dict(row) for row in self.db.execute('SELECT * FROM spending_cap_increases ORDER BY id')]

    def reserve(self, call_key, maximum_usd):
        """Return dispatch_allowed=True once; a repeated key never grants a second send."""
        maximum = _amount(maximum_usd, 'request bound')
        bound = _nanos(maximum)
        with self._transaction():
            old = self._call(call_key)
            policy = self._policy()
            if old is not None and Decimal(old['maximum_usd']) != maximum:
                raise SpendingError(f'{call_key}: reused request bound changed')
            if policy['stopped_reason']:
                raise SpendingError(f"dispatch stopped: {policy['stopped_reason']}")
            if old is not None and old['status'] != 'unbilled':
                return {**dict(old), 'dispatch_allowed': False}
            held = int(policy['held_nanos'])
            exposure = int(policy['spent_nanos']) + held
            if exposure + bound > int(policy['cap_nanos']):
                raise SpendingError(f'{call_key}: request reservation would exceed the shared cap')
            self.db.execute('UPDATE spending_policy SET held_nanos=? WHERE id=1', (str(held + bound),))
            if old is None:
                self.db.execute('INSERT INTO spending_calls VALUES (?,?,?,\'held\',NULL,NULL)',
                                (call_key, _text(maximum), str(bound)))
            else:
                self.db.execute("UPDATE spending_calls SET status='held',reported_usd=NULL,reported_nanos=NULL WHERE call_key=?",
                                (call_key,))
            return {**dict(self._call(call_key)), 'dispatch_allowed': True}

    def settle(self, call_key, reported_usd):
        """Settle once; None keeps the reservation. An overrun commits before raising."""
        reported = None if reported_usd is None else _amount(reported_usd, 'reported cost')
        overrun = False
        with self._transaction():
            row = self._call(call_key)
            if row is None:
                raise SpendingError(f'{call_key}: cannot settle an unknown reservation')
            if reported is None:
                return dict(row)
            if row['status'] == 'settled':
                if Decimal(row['reported_usd']) != reported:
                    raise SpendingError(f'{call_key}: a settled bill cannot be changed')
                return dict(row)
            if row['status'] != 'held':
                raise SpendingError(f'{call_key}: request was confirmed unbilled; reserve it before retry')
            policy = self._policy()
            debit = _nanos(reported)
            overrun = reported > Decimal(row['maximum_usd'])
            reason = policy['stopped_reason'] or (
                f'{call_key}: reported cost exceeded its reserved request bound' if overrun else None)
            self.db.execute('''UPDATE spending_policy SET spent_nanos=?,held_nanos=?,
                               reported_usd=?,stopped_reason=? WHERE id=1''',
                            (str(int(policy['spent_nanos']) + debit),
                             str(int(policy['held_nanos']) - int(row['maximum_nanos'])),
                             _add_reported(policy['reported_usd'], reported), reason))
            self.db.execute("UPDATE spending_calls SET status='settled',reported_usd=?,reported_nanos=? WHERE call_key=?",
                            (_text(reported), str(debit), call_key))
            result = dict(self._call(call_key))
        if overrun:
            raise SpendingError(f'{call_key}: reported cost exceeded its request bound; dispatch permanently stopped')
        return result

    def stop(self, reason):
        """Permanently stop new dispatch without blocking outstanding bill settlement."""
        if not isinstance(reason, str) or not reason.strip():
            raise SpendingError('a dispatch stop requires a reason')
        with self._transaction():
            self.db.execute('UPDATE spending_policy SET stopped_reason=COALESCE(stopped_reason,?) WHERE id=1',
                            (reason,))

    def confirm_unbilled(self, call_key, *, evidence):
        """Release a hold only after an explicit operator decision; retain its evidence."""
        if not isinstance(evidence, str) or not evidence.strip():
            raise SpendingError('confirmation requires evidence that the request was not billed')
        with self._transaction():
            row = self._call(call_key)
            if row is None:
                raise SpendingError(f'{call_key}: unknown reservation')
            if row['status'] == 'settled':
                raise SpendingError(f'{call_key}: cannot release a settled bill')
            if row['status'] == 'unbilled':
                return dict(row)
            policy = self._policy()
            self.db.execute('UPDATE spending_policy SET held_nanos=? WHERE id=1',
                            (str(int(policy['held_nanos']) - int(row['maximum_nanos'])),))
            self.db.execute("UPDATE spending_calls SET status='unbilled',reported_usd='0',reported_nanos='0' WHERE call_key=?",
                            (call_key,))
            self.db.execute('INSERT INTO spending_resolutions (call_key,evidence) VALUES (?,?)',
                            (call_key, evidence))
            return dict(self._call(call_key))

    def summary(self):
        # One read transaction keeps counters and counts consistent across processes.
        with self._transaction():
            policy = self._policy()
            counts = dict(self.db.execute('SELECT status,COUNT(*) FROM spending_calls GROUP BY status'))
            resolutions = self.db.execute('SELECT COUNT(*) FROM spending_resolutions').fetchone()[0]
            cap, spent, held = (int(policy[key]) for key in ('cap_nanos','spent_nanos','held_nanos'))
            amounts = {'cap': cap, 'spent': spent, 'held': held, 'exposure': spent + held,
                       'available': max(0, cap - spent - held)}
            return {**{f'{key}_nanodollars': value for key,value in amounts.items()},
                    **{f'{key}_usd': _usd(value) for key,value in amounts.items()},
                    'configured_cap_usd': policy['cap_usd'], 'reported_usd': policy['reported_usd'],
                    'calls': sum(counts.values()), 'calls_by_status': counts,
                    'unbilled_resolutions': resolutions, 'stopped': bool(policy['stopped_reason']),
                    'stopped_reason': policy['stopped_reason'],
                    'accounting_note': 'Cap rounds down; reservation and spending debits round up to nanodollars.'}

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main(argv=None):
    """Record an approved cap increase. This command never sends model requests."""
    import argparse
    import json
    from decimal import InvalidOperation

    def decimal_arg(value):
        try:
            return Decimal(value)
        except InvalidOperation:
            raise argparse.ArgumentTypeError('amount must be a decimal number') from None

    parser = argparse.ArgumentParser(description='Record an approved shared spending cap increase. Sends no model request.')
    parser.add_argument('--ledger', required=True, help='existing shared spending SQLite file')
    parser.add_argument('--expected-cap-usd', required=True, type=decimal_arg)
    parser.add_argument('--new-cap-usd', required=True, type=decimal_arg)
    parser.add_argument('--evidence', required=True, help='reference to operator approval for this higher cap')
    parser.add_argument('--confirm-no-active-runs', action='store_true',
                        help='confirm all runs using this ledger have stopped; unknown holds are retained')
    args = parser.parse_args(argv)
    if not args.confirm_no_active_runs:
        parser.error('confirm that all runs using this ledger have stopped with --confirm-no-active-runs')
    try:
        with SpendingLedger(args.ledger, args.expected_cap_usd, existing_only=True) as ledger:
            audit = ledger.increase_cap(args.expected_cap_usd, args.new_cap_usd, evidence=args.evidence,
                                        confirm_no_active_runs=args.confirm_no_active_runs)
            print(json.dumps(audit, indent=2))
    except (SpendingError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print('Cap increase recorded. No request sent. All charges, unknown holds, and provider stops are retained.')


if __name__ == '__main__':
    main()
