"""Run fixed-prompt output diagnostics through the native transport and shared spending ledger.

Each probe sends one request. No correction request or game continuation is added.
Results stay outside native episode exports and the behavioral analysis pool.
"""
import argparse
import asyncio
from contextlib import closing
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from delegated_influence import artifacts, game, native_adapter, runtime, seat, task
from delegated_influence.dispatch import MissingUsageEvidence, RequestBound, amount
from delegated_influence.journal import Journal, RunPaused, canonical, unpack
from delegated_influence.spending import SpendingLedger


def source_record(probe):
    """Read original evidence and state without opening or changing its journal."""
    path = Path(probe['source']['journal'])
    path = path if path.is_absolute() else ROOT / path
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        call = db.execute('SELECT * FROM calls WHERE id=?', (probe['source']['call_id'],)).fetchone()
        if call is None:
            raise ValueError('diagnostic source call is missing')
        episode = db.execute('SELECT metadata FROM episodes WHERE id=?', (call['episode'],)).fetchone()
        metadata = unpack(episode['metadata'])
        state = game.state_from_metadata(metadata)
        for row in db.execute('SELECT payload FROM events WHERE episode=? ORDER BY sequence', (call['episode'],)):
            event = unpack(row['payload'])
            if any(item['call_id'] == call['id'] for item in event.get('calls', [])):
                break
            game.apply_event(state, event)
        else:
            # An unfinished final turn has not applied its event. All saved
            # preceding events still define the state at that call.
            if db.execute('SELECT 1 FROM calls WHERE episode=? AND created>? LIMIT 1',
                          (call['episode'], call['created'])).fetchone():
                raise ValueError('cannot locate diagnostic source turn in its event history')
        raw = unpack(call['wire_request'])
        if raw is None:
            raise ValueError('diagnostic source needs its exact provider request')
        return {'metadata': metadata, 'state': state, 'seat': call['seat'],
                'messages': unpack(call['payload']), 'input_hash': call['input_hash'],
                'request': json.loads(raw)}


def without_output_limits(body):
    return {key: value for key, value in body.items() if key not in ('max_tokens', 'max_completion_tokens')}


def validate_plan(plan, manifest, *, sources=None):
    if plan.get('schema_version') != 'fixed-request-diagnostic-v1' or not plan.get('probes'):
        raise ValueError('a nonempty fixed-request diagnostic plan is required')
    ids = [p['id'] for p in plan['probes']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate diagnostic request')
    bounds = {}
    sources = {} if sources is None else sources
    cache = {}
    for probe in plan['probes']:
        if probe['kind'] != 'output_limit':
            raise ValueError('this runner supports output-limit probes only')
        digest = hashlib.sha256(canonical(probe['messages']).encode()).hexdigest()
        if digest != probe['source']['input_sha256']:
            raise ValueError('probe input differs from its frozen source')
        source_key = (probe['source']['journal'], probe['source']['call_id'])
        if source_key not in cache:
            cache[source_key] = source_record(probe)
        source = sources[probe['id']] = cache[source_key]
        if (source['input_hash'] != digest or canonical(source['messages']) != canonical(probe['messages'])
                or canonical(source['metadata']) != canonical(probe['validation']['metadata'])
                or source['seat'] != probe['validation']['seat']
                or source['metadata']['seat_map'][source['seat']] != probe['model']):
            raise ValueError('probe differs from its original source evidence')
        if canonical(probe['seed']) != canonical(source['metadata']['seed']):
            raise ValueError('probe changes the original source seed')
        saved = probe['validation']['metadata']['sampling_instruments'][probe['model']]
        if probe['source']['original_output_limit'] != saved['max_tokens']:
            raise ValueError('probe original output limit differs from its source')
        bound = RequestBound(manifest, probe['model'], probe['model_config'])
        if bound.output != probe['max_tokens']:
            raise ValueError('probe output limit differs from its declared treatment')
        for key in ('model_spec', 'temperature', 'reasoning_effort'):
            if saved[key] != bound.instrument[key]:
                raise ValueError(f'probe changes a non-output setting: {key}')
        controls = probe['validation']['metadata']['spending_instrument'][probe['model']]
        if controls['provider_routing'] != bound.routing:
            raise ValueError('probe route differs from the saved source')
        bounds[probe['id']] = bound
    return bounds


def results(journal, plan, sources=None):
    rows = []
    for probe in plan['probes']:
        saved = journal.episode(probe['id'])
        if not saved or saved['status'] != 'complete':
            continue
        result = saved['record']
        state = (sources[probe['id']] if sources is not None else source_record(probe))['state']
        parsed = seat.classify_reply(result['response'], state, probe['validation']['seat'])
        rows.append({key: probe[key] for key in ('id', 'pair_id', 'model', 'max_tokens', 'source')} | {
            'first_response_valid': isinstance(parsed, seat.Parsed),
            'invalid_reason': parsed.reason if isinstance(parsed, seat.Rejected) else None,
            'response': result['response'], 'usage': result['usage']})
    return {'schema_version': 'fixed-request-diagnostic-results-v1', 'planned_requests': len(plan['probes']),
            'completed_requests': len(rows), 'calls': rows,
            'scope': 'Selected saved prompts; output allowance also bounds reasoning. Not a full-game cost estimate.'}


async def execute(plan, manifest, journal, ledger, *, factory=native_adapter.build_generator):
    if (ledger.path.resolve() != Path(plan['shared_ledger']).resolve()
            or amount(ledger.summary()['configured_cap_usd'], 'ledger cap') != amount(plan['shared_cap_usd'], 'plan cap')):
        raise RunPaused('diagnostic shared ledger or cap differs from the frozen plan')
    sources = {}
    bounds = validate_plan(plan, manifest, sources=sources)
    journal.bind('diagnostic_plan', plan)
    journal.bind('diagnostic_routes', manifest)
    if journal.summary()['uncertain_calls']:
        raise RunPaused('diagnostic has uncertain calls; review before another request')
    namespace = 'diagnostic-' + hashlib.sha256(str(journal.path.resolve()).encode()).hexdigest()[:24]
    for probe in plan['probes']:
        identifier, bound = probe['id'], bounds[probe['id']]
        saved = journal.start_episode(identifier, probe, float(bound.maximum_usd), plan['diagnostic_cap_usd'])
        if saved['status'] == 'complete':
            continue
        key = f'{namespace}:{identifier}:probe:0'
        provider = None
        try:
            if journal.needs_dispatch(identifier, 'probe', 0):
                provider = factory(probe['model'], probe['seed'], model_config=probe['model_config'],
                                   request_bound=bound)
                reservation = ledger.reserve(key, bound.maximum_usd)
                if not reservation['dispatch_allowed']:
                    raise RunPaused('diagnostic reservation exists without a replayable response')
            call_id, result = journal.begin_call(identifier, 'probe', 0, probe['messages'])
            if result is None:
                try:
                    def on_request(body):
                        journal.save_request(call_id, body)
                        bound.check_request(body)
                        if canonical(without_output_limits(json.loads(body))) != canonical(
                                without_output_limits(sources[identifier]['request'])):
                            raise RunPaused('diagnostic provider request changes a non-output source setting')
                    provider.on_request = on_request
                    response, usage = await asyncio.wait_for(provider(probe['messages']), task.CALL_TIMEOUT_S)
                except BaseException as exc:
                    journal.uncertain(call_id, str(exc) or type(exc).__name__)
                    raise RunPaused(f'{call_id}: uncertain diagnostic response') from exc
                result = journal.finish_call(call_id, response, dict(usage or {}), None)
            ledger.settle(key, result['usage'].get('total_cost'))
            if result['usage'].get('total_cost') is None:
                journal.uncertain(call_id, 'missing provider bill')
                raise RunPaused('diagnostic bill missing; spending hold retained')
            try:
                bound.check_usage(result['usage'])
            except MissingUsageEvidence as exc:
                journal.uncertain(call_id, str(exc))
                raise
            except RunPaused as exc:
                ledger.stop(str(exc))
                raise
            if result['response'] is None:
                raise RunPaused('diagnostic returned a null response; review saved evidence')
            journal.complete(identifier, result)
        except BaseException:
            journal.db.execute("UPDATE episodes SET status='paused' WHERE id=?", (identifier,))
            raise
        finally:
            journal.write_status(force=True)
            (journal.path.parent / 'spending-status.json').write_text(json.dumps(ledger.summary(), indent=2) + '\n')
            (journal.path.parent / 'results.json').write_text(json.dumps(results(journal, plan, sources), indent=2) + '\n')
            if provider is not None and hasattr(provider, 'aclose'):
                await asyncio.gather(provider.aclose(), return_exceptions=True)
        print(f"{identifier}: {probe['model']} output={probe['max_tokens']} cost={result['usage']['total_cost']}", flush=True)
    return results(journal, plan, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--keys', help='local dotenv file; never included in snapshots')
    parser.add_argument('--execute', action='store_true', help='send the frozen diagnostic requests')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    manifest = json.loads(Path(plan['route_manifest']).read_text())
    bounds = validate_plan(plan, manifest)
    print(f"Validated {len(bounds)} requests; local allowance ${plan['diagnostic_cap_usd']}; shared cap ${plan['shared_cap_usd']}.")
    if not args.execute:
        return
    if args.keys:
        from dotenv import load_dotenv
        load_dotenv(args.keys, override=False)
    sources = artifacts.runtime_source_paths(ROOT) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): artifacts.sha256_file(p) for p in sources}
    with Journal(Path(args.run_dir) / 'run.sqlite3') as journal:
        journal.bind('source_files', hashes)
        journal.bind('runtime_environment', runtime.RUNTIME_ENVIRONMENT)
        snapshot = journal.path.parent / 'source-snapshot.tar.gz'
        if not snapshot.exists():
            artifacts._snapshot_source(ROOT, sources, snapshot)
        with SpendingLedger(plan['shared_ledger'], plan['shared_cap_usd'], existing_only=True) as ledger:
            asyncio.run(execute(plan, manifest, journal, ledger))


if __name__ == '__main__':
    main()
