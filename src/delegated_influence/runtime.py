"""Native episode scheduling. Inspect is used only at the model adapter seam."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import platform
import random
import json
import math
from email.utils import parsedate_to_datetime
from pathlib import Path
import time
import traceback
import uuid

from delegated_influence import game, task, turn, native_adapter
from delegated_influence.dispatch import MissingUsageEvidence
from delegated_influence.journal import Journal, RunPaused, canonical
from delegated_influence.request_control import RequestControl, admission, interruptible_wait


RUNTIME_INSTRUMENT = {'runner':'native', 'adapter_version':5, 'sdk_retries':0, 'prompt_version':3,
                      'retry_policy_version':3, 'request_control_version':1}
RUNTIME_ENVIRONMENT = {'python': platform.python_version(),
                       'packages': {name: importlib.metadata.version(name)
                                    for name in ('inspect_ai','openai','httpx')}}


# Only actual pre-generation rejection statuses qualify. A 200 error body or
# 502 invalid upstream response can follow processing and retain an unknown bill.
# 529 is Anthropic's dedicated overload status: documented as 'temporarily overloaded', no
# output, no bill — the same claim 429 and 503 make, on a code only Anthropic uses. The
# `processing` test below still applies, so a 529 whose body somehow carried usage stays
# unknown. On 2026-09-11 fable-5 parked two arena games on a bare 529 while fable-5.1 parked
# twenty-one on the streamed 200 form of the SAME error, because its larger token budget made
# the SDK stream. One error, two transports; both must be retryable or neither is.
UNSERVED_STATUS = {429, 503, 529}
# The second exception, and it is deliberately narrow. A streaming Anthropic request that is refused
# for load answers HTTP 200 and puts the refusal in the body as an SSE error event, so the status
# code cannot see it. `overloaded_error` has one documented meaning — the API was busy and
# generated nothing — and the envelope carries no id, usage or choices, so the `processing` test
# below still applies unchanged. On 2026-09-11 this parked 14 of 28 games in one campaign and 7 of
# 32 in another inside the same three minutes, because an invisible rejection cannot be retried.
# Do NOT widen this to other error.type values: `api_error` makes no such claim about billing.
# Unserved failures cost nothing, and a parked game is a total loss, so retry for hours rather than
# minutes: 120 attempts at a 120 s ceiling is about 4 h. A 17-minute deepinfra rate-limit outage on
# 2026-09-10 parked 17 arena games at once under the old 12-attempt limit. The dispatch deadline and
# STOP file are still honoured between attempts.
UNSERVED_RETRIES = 120
UNSERVED_BACKOFF_S = 5
UNSERVED_BACKOFF_MAX_S = 120


def anthropic_overloaded(body):
    """Is this body Anthropic's load refusal? Accepts the JSON envelope and the SSE error event.

    A streamed refusal arrives as text: 'event: error\ndata: {"type":"error","error":
    {"type":"overloaded_error",...}}'. Only that exact error type qualifies.
    """
    candidates = []
    if isinstance(body, dict):
        candidates.append(body)
    elif isinstance(body, str):
        for line in body.splitlines():
            if line.startswith('data:'):
                try:
                    candidates.append(json.loads(line[5:].strip()))
                except ValueError:
                    continue
    for row in candidates:
        error = row.get('error') if isinstance(row, dict) else None
        if isinstance(error, dict) and error.get('type') == 'overloaded_error':
            if any(key in row for key in ('id', 'usage', 'choices')):
                return False
            return True
    return False


def unserved_status(exc):
    """Label of a request the provider rejected before generation; None when billing is unknown."""
    import httpx
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        evidence = provider_response(current)
        if evidence:
            status, body = evidence.get('status_code'), evidence.get('body')
            processing = isinstance(body, dict) and any(key in body for key in ('id', 'usage', 'choices'))
            complete = evidence.get('body_complete') is True
            if status == 200 and complete and not processing and anthropic_overloaded(body):
                return 'anthropic overloaded_error'
            return f'HTTP {status}' if (status in UNSERVED_STATUS and not processing
                                        and complete) else None
        if isinstance(current, (httpx.ConnectTimeout, httpx.ConnectError)):
            return f'connection failure {type(current).__name__}'
        current = current.__cause__
    return None


def provider_response(exc):
    """Use actual HTTP evidence, never an error body's code as the HTTP status."""
    import httpx
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        evidence = getattr(exc, 'native_response', None)
        if evidence:
            return evidence
        response = getattr(exc, 'response', None)
        if isinstance(response, httpx.Response):
            try:
                body = response.json()
            except (ValueError, httpx.ResponseNotRead):
                body = None
            return {'status_code': response.status_code, 'retry_after': response.headers.get('Retry-After'),
                    'body': body, 'body_complete': response.is_stream_consumed}
        exc = exc.__cause__
    return {}


def retry_after_seconds(exc):
    value = provider_response(exc).get('retry_after')
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, AttributeError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


def exception_evidence(exc):
    """Serialize an exception stack and cause chain without frame locals."""
    chain = []
    current = exc
    relation = 'exception'
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        kind = type(current)
        chain.append({'relation': relation, 'type': f'{kind.__module__}.{kind.__qualname__}',
                      'message': str(current) or kind.__name__})
        if current.__cause__ is not None:
            current = current.__cause__
            relation = 'cause'
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            relation = 'context'
        else:
            current = None
    kind = type(exc)
    return canonical({
        'schema_version': 'exception-evidence-v1',
        'type': f'{kind.__module__}.{kind.__qualname__}',
        'message': str(exc) or kind.__name__,
        'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)),
        'chain': chain,
        'provider_response': provider_response(exc),
    })


def open_audited_spending_ledger(saved_policy, options):
    """Open the exact configured cap; require an audit chain for any saved-cap change."""
    from delegated_influence.dispatch import amount
    from delegated_influence.spending import SpendingLedger
    if saved_policy and Path(options['ledger']).resolve() != Path(saved_policy['ledger']).resolve():
        raise RunPaused('spending ledger changed; retain the saved ledger')
    ledger = SpendingLedger(options['ledger'], options['cap_usd'], existing_only=bool(saved_policy))
    try:
        if not saved_policy:
            return ledger, []
        saved_cap = amount(saved_policy['cap_usd'], 'saved cap')
        current_cap = amount(ledger.summary()['configured_cap_usd'], 'current cap')
        if saved_cap == current_cap:
            return ledger, []
        audits = ledger.cap_increases()
        previous = None
        contains_saved_cap = False
        for index, audit in enumerate(audits, 1):
            old, new = (amount(audit[key], key) for key in ('old_cap_usd', 'new_cap_usd'))
            if (audit['id'] != index or new <= old or (previous is not None and old != previous)
                    or audit['no_active_runs_confirmed'] != 1 or not audit['created']
                    or not isinstance(audit['evidence'], str) or not audit['evidence'].strip()):
                raise RunPaused('spending cap audit chain is invalid')
            contains_saved_cap |= old == saved_cap
            previous = new
        if current_cap <= saved_cap or not contains_saved_cap or previous != current_cap:
            raise RunPaused('spending cap change lacks a complete audit chain from the saved cap')
        return ledger, audits
    except BaseException:
        ledger.close()
        raise


def record_spending_cap_increase(journal, saved_policy, cap_usd, audits):
    """Keep frozen request controls and record the accepted ledger audit atomically."""
    old_audits = journal.db.execute("SELECT value FROM meta WHERE key='spending_cap_increases'").fetchone()
    if old_audits:
        accepted = json.loads(old_audits['value'])
        if audits[:len(accepted)] != accepted:
            raise RunPaused('previously accepted spending cap audit changed')
    with journal.db:
        journal.db.execute('BEGIN IMMEDIATE')
        journal.db.execute("UPDATE meta SET value=? WHERE key='spending'",
                           (canonical({**saved_policy, 'cap_usd': cap_usd}),))
        journal.db.execute("INSERT OR REPLACE INTO meta VALUES ('spending_cap_increases',?)", (canonical(audits),))


async def run_episode(journal, episode_id, metadata, generator_factory, *, reserve_usd,
                      budget_usd, allow_estimated_cost=False, estimate_call=None, sample_id=None,
                      call_timeout=task.CALL_TIMEOUT_S, spending=None, request_bounds=None,
                      spending_namespace=None, dispatch_paused=None, on_pause=None, request_control=None):
    # Validate state before creating or reopening an active episode record.
    state = game.state_from_metadata(metadata)
    episode = journal.start_episode(episode_id, metadata, reserve_usd, budget_usd,
                                    allow_estimated_cost=allow_estimated_cost, sample_id=sample_id)
    if episode['status'] == 'complete':
        return episode['record']
    if episode['status'] == 'paused':
        journal.db.execute("UPDATE episodes SET status='running' WHERE id=?", (episode_id,))
        journal.write_status(force=True)
    pending_event_calls = []
    generators = {}
    opened = []
    for seat in metadata['seat_map']:
        # The factory is lazy: replaying a completed prefix does not need a model client.
        def wrap(who):
            sequence = 0
            provider = None
            async def durable_generate(messages):
                nonlocal sequence, provider
                dispatch_needed = journal.needs_dispatch(episode_id, who, sequence)
                if dispatch_needed and dispatch_paused and dispatch_paused():
                    raise RunPaused('dispatch stopped; no further requests will be sent')
                call_key = f'{spending_namespace}:{episode_id}:{who}:{sequence}'
                bound = request_bounds[metadata['seat_map'][who]] if request_bounds else None
                if provider is None and dispatch_needed:
                    # Client setup cannot create a billed request. Fail before journaling dispatch.
                    provider = generator_factory(who)
                    opened.append(provider)
                for attempt in range(UNSERVED_RETRIES + 1):
                    if attempt and dispatch_paused and dispatch_paused():
                        raise RunPaused('dispatch stopped; no further requests will be sent')
                    model_key = metadata['seat_map'][who]
                    provider_key = bound.row['provider_tag'] if bound is not None and request_control is not None else model_key
                    async with admission(request_control if dispatch_needed else None, call_key,
                                         model_key, provider_key, dispatch_paused if dispatch_needed else None):
                        if spending is not None and dispatch_needed:
                            old = journal.db.execute('SELECT status FROM calls WHERE id=?',
                                (f'{episode_id}:{who}:{sequence}',)).fetchone()
                            if old and old['status'] == 'retryable':
                                resolution = journal.db.execute(
                                    "SELECT evidence FROM resolutions WHERE call_id=? AND action='confirmed_unbilled' ORDER BY created DESC LIMIT 1",
                                    (f'{episode_id}:{who}:{sequence}',)).fetchone()
                                if resolution:
                                    spending.confirm_unbilled(call_key, evidence=resolution['evidence'])
                            reservation = spending.reserve(call_key, bound.maximum_usd)
                            if not reservation['dispatch_allowed']:
                                raise RunPaused(f'{call_key}: existing spending hold needs review before dispatch')
                        call_id, result = journal.begin_call(episode_id, who, sequence, messages)
                        if result is not None:
                            break
                        try:
                            provider.on_request = lambda body: journal.save_request(call_id, body)
                            response, usage = await asyncio.wait_for(provider(messages), call_timeout)
                        except BaseException as exc:
                            status = unserved_status(exc)
                            delay = max(retry_after_seconds(exc),
                                min(UNSERVED_BACKOFF_S * 2 ** attempt, UNSERVED_BACKOFF_MAX_S)
                                * random.uniform(1, 1.25))
                            details = provider_response(exc)
                            body = details.get('body')
                            error = body.get('error') if isinstance(body, dict) else None
                            overload = details.get('status_code') in UNSERVED_STATUS or (
                                isinstance(error, dict) and error.get('code') in UNSERVED_STATUS)
                            if request_control is not None and overload:
                                request_control.cooldown(model_key, provider_key, delay, evidence=exception_evidence(exc))
                            if request_control is not None:
                                request_control.release(call_key)
                            if status is not None and attempt < UNSERVED_RETRIES:
                                evidence = f'{status} rejected the request before generation; automatic retry {attempt + 1} of {UNSERVED_RETRIES}'
                                journal.uncertain(call_id, exception_evidence(exc))
                                journal.resolve_uncertain(call_id, confirm_unbilled=True, evidence=evidence)
                                if spending is not None:
                                    spending.confirm_unbilled(call_key, evidence=evidence)
                                await interruptible_wait(delay, dispatch_paused)
                                continue
                            # Preserve any bill/ID that survived in an HTTP error body.
                            if isinstance(body, dict):
                                raw_usage = body.get('usage') if isinstance(body.get('usage'), dict) else {}
                                failed_usage = {'total_cost': raw_usage.get('cost'), 'provider_usage': raw_usage,
                                    'served_provider': body.get('provider'), 'served_model': body.get('model'),
                                    'provider_request_id': body.get('id')}
                                journal.finish_call(call_id, None, failed_usage, None)
                                journal.uncertain(call_id, exception_evidence(exc))
                                if spending is not None:
                                    spending.settle(call_key, failed_usage['total_cost'])
                            journal.uncertain(call_id, exception_evidence(exc))
                            journal.write_status(force=True)
                            raise RunPaused(f'call {call_id}: uncertain provider result ({type(exc).__name__})') from exc
                        if response is None:
                            journal.finish_call(call_id, response, dict(usage or {}), None)
                            journal.write_status(force=True)
                            if spending is not None:
                                spending.settle(call_key, (usage or {}).get('total_cost'))
                            raise RunPaused(f'call {call_id}: uncertain provider result; inspect billing before retry')
                        # Save returned evidence before optional accounting or instrumentation.
                        usage = {**dict(usage or {}), 'accounting_complete': False}
                        result = journal.finish_call(call_id, response, usage, None)
                        break
                sequence += 1
                if spending is not None:
                    # Response evidence is durable before settling the shared ledger.
                    spending.settle(call_key, result['usage'].get('total_cost'))
                    if result['usage'].get('total_cost') is None:
                        journal.uncertain(call_id, 'response saved but provider billing is missing')
                        raise RunPaused(f'{call_id}: missing provider bill; spending hold retained')
                    try:
                        bound.check_usage(result['usage'])
                    except MissingUsageEvidence as exc:
                        journal.uncertain(call_id, str(exc))
                        raise
                    except RunPaused as exc:
                        spending.stop(str(exc))
                        raise
                if not result['usage'].get('accounting_complete'):
                    usage = dict(result['usage'])
                    usage['input_characters'] = sum(len(str(m.get('content', ''))) for m in messages)
                    usage['system_characters'] = sum(len(str(m.get('content', ''))) for m in messages
                                                     if m.get('role') == 'system')
                    usage['observation_characters'] = len(str(messages[1].get('content', ''))) if len(messages) > 1 else 0
                    estimate = estimate_call(who, usage) if estimate_call else None
                    usage['accounting_complete'] = True
                    result = journal.finish_call(call_id, result['response'], usage, estimate)
                pending_event_calls.append(journal.call(call_id))
                return result['response'], result['usage']
            async def generate(messages):
                try:
                    return await durable_generate(messages)
                except RunPaused:
                    raise
                except BaseException as exc:
                    journal.db.execute("UPDATE calls SET status='uncertain',error=? WHERE episode=? AND status='pending'",
                                       (exception_evidence(exc), episode_id))
                    raise RunPaused(f'episode {episode_id}: call recording failed ({type(exc).__name__})') from exc
            return generate
        generators[seat] = wrap(seat)

    def on_event(current, event):
        event['calls'] = list(pending_event_calls)
        pending_event_calls.clear()
        journal.save_event(episode_id, len(current['events']) - 1, event)
        journal.write_status()

    try:
        await turn.drive_episode(state, random.Random(metadata['seed']), generators,
                                 personas=metadata.get('personas'), on_event=on_event)
    except BaseException:
        if on_pause:
            on_pause()
        journal.db.execute("UPDATE episodes SET status='paused' WHERE id=?", (episode_id,))
        journal.write_status(force=True)
        raise
    finally:
        await asyncio.gather(*(generate.aclose() for generate in opened if hasattr(generate, 'aclose')),
                             return_exceptions=True)
    served = {}
    for event in state['events']:
        value = metadata['seat_map'].get(event.get('agent'))
        for call in event.get('calls', []):
            instrument = served.setdefault(value, {'calls':0, 'served_models':{}, 'providers':{},
                                                   'unknown_provider_calls':0})
            instrument['calls'] += 1
            for field, bucket in (('served_model','served_models'), ('served_provider','providers')):
                name = call['usage'].get(field)
                if name:
                    instrument[bucket][name] = instrument[bucket].get(name, 0) + 1
            if not call['usage'].get('served_provider'):
                instrument['unknown_provider_calls'] += 1
    record = {**metadata, 'checkpoint_episode_id': episode_id,
              'topology': metadata.get('topology', 'complete'),
              'messages_off': bool(metadata.get('messages_off')), 'served_instruments':served,
              'events': state['events'], 'scores': state['scores']}
    journal.complete(episode_id, record)
    journal.write_status(force=True)
    return record


async def run_tasks(journal, tasks, estimates, cfg, *, dispatch_paused=None):
    """Schedule games under both planning reservations and shared request bounds."""
    from delegated_influence.dispatch import RequestBound
    options = cfg.get('spending')
    paid = any(not value.startswith('mockllm/') for _, item in tasks for sample in item.dataset
               for value in sample.metadata['seat_map'].values())
    if paid and not options:
        raise RunPaused('paid native runs require spending.ledger, cap_usd, and a fresh route manifest')
    bounds_by_sample = {}
    spending = None
    namespace = None
    saved = None
    if options:
        if set(options) != {'ledger', 'cap_usd', 'manifest'}:
            raise ValueError('spending requires exactly ledger, cap_usd, and manifest')
        manifest = json.loads(Path(options['manifest']).read_text())
        policy = {**options, 'ledger': str(Path(options['ledger']).resolve()), 'manifest': manifest}
        previous = journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()
        if not previous and journal.db.execute('SELECT 1 FROM calls LIMIT 1').fetchone():
            raise RunPaused('existing calls need explicit billing reconciliation before adding a spending ledger')
        if previous:
            saved = json.loads(previous['value'])
            def controls(value):
                return {model: {key: item for key, item in row.items()
                                if key not in {'catalog_model','endpoint','matching_endpoints','evidence'}}
                        for model, row in value['models'].items()}
            if (policy['ledger'] != saved['ledger']
                    or controls(manifest) != controls(saved['manifest'])):
                raise RunPaused('spending route or bound changed; retain frozen bounds and reconcile before changing them')
            # Refresh verification without changing the frozen request identity or replay evidence.
            manifest = {**saved['manifest'], 'verified_at': manifest['verified_at']}
        else:
            journal.bind('spending', policy)
        old_namespace = journal.db.execute("SELECT value FROM meta WHERE key='spending_namespace'").fetchone()
        namespace = json.loads(old_namespace['value']) if old_namespace else uuid.uuid4().hex
        journal.bind('spending_namespace', namespace)
        for arm, item in tasks:
            for sample in item.dataset:
                md = sample.metadata
                bounds_by_sample[(arm, sample.id)] = {value: RequestBound(manifest, value,
                    (md.get('model_configs') or {}).get(value)) for value in set(md['seat_map'].values())}
    from delegated_influence.protocols import validate_live_protocols
    capabilities = cfg.get('protocol_capabilities')
    if isinstance(capabilities, str):
        capabilities = json.loads(Path(capabilities).read_text())
    if capabilities is None and options:
        capabilities = {'schema_version':1, 'endpoints': [
            {'model':row['model'], 'provider':row['provider_tag'],
             'supported_reasoning_efforts': (row.get('reasoning') or {}).get('supported_efforts', []),
             'verified_at':manifest['verified_at'], 'evidence':row['evidence'],
             'temperature_omitted':True, 'max_tokens':row['max_completion_tokens']}
            for row in manifest['models'].values()]}
    if options:
        spending, cap_audits = open_audited_spending_ledger(saved, options)
    request_control = None
    try:
        validate_live_protocols(tasks, capabilities)
        native = cfg.get('native') or {}
        if paid or 'request_limits' in native:
            control_path = (Path(options['ledger']) if options else journal.path).resolve().with_suffix('.requests.sqlite3')
            request_control = RequestControl(control_path, policy=native.get('request_limits'))
            journal.bind('request_control', request_control.policy)
        if options and cap_audits:
            record_spending_cap_increase(journal, saved, options['cap_usd'], cap_audits)
        return await _run_tasks(journal, tasks, estimates, cfg, spending, bounds_by_sample, namespace, dispatch_paused, request_control)
    finally:
        if request_control is not None:
            request_control.close()
        if spending is not None:
            (journal.path.parent / 'spending-status.json').write_text(json.dumps(spending.summary(), indent=2) + '\n')
            spending.close()


async def _run_tasks(journal, tasks, estimates, cfg, spending, bounds_by_sample, namespace, dispatch_paused=None, request_control=None):
    if journal.summary()['uncertain_calls']:
        journal.db.execute("UPDATE episodes SET status='paused' WHERE status='running'")
        journal.write_status(force=True)
        raise RunPaused('uncertain calls require operator review before dispatch')
    if request_control is not None:
        request_control.reconcile(journal, spending, namespace)
    options = cfg.get('native') or {}
    unknown = set(options) - {'max_concurrent_episodes', 'allow_estimated_cost', 'request_limits', 'direct_models'}
    if unknown:
        raise ValueError(f'unknown native fields: {sorted(unknown)}')
    concurrency = options.get('max_concurrent_episodes', 1)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('native.max_concurrent_episodes must be a positive integer')
    allow_estimates = options.get('allow_estimated_cost', False)
    if not isinstance(allow_estimates, bool):
        raise ValueError('native.allow_estimated_cost must be a boolean')
    budget = cfg['budget'].get('max_actual_usd', cfg['budget']['max_estimated_usd'])
    from delegated_influence.costs import usage_estimate
    queue = asyncio.Queue()
    rates = {row['arm']: row['usd'] / max(row['episodes'], 1) for row in estimates['arms']}
    for arm, task_value in tasks:
        for sample in task_value.dataset:
            md = {**sample.metadata, 'runtime_instrument': {**RUNTIME_INSTRUMENT,
                    **({'request_limits': request_control.policy} if request_control else {})},
                  'runtime_environment': RUNTIME_ENVIRONMENT}
            if spending is not None:
                md['spending_instrument'] = {value: {'route': bound.row, 'maximum_usd': str(bound.maximum_usd),
                                                    'provider_routing': bound.routing}
                    for value, bound in bounds_by_sample[(arm, sample.id)].items()}
            identity = hashlib.sha256(canonical({'arm': arm, 'sample_id': sample.id,
                                                'metadata': md}).encode()).hexdigest()[:24]
            queue.put_nowait((arm, sample.id, md, identity))
    pauses = []

    async def worker():
        while not queue.empty():  # A paused game keeps its evidence; other games continue.
            arm, sample_id, md, identity = queue.get_nowait()
            def factory(who):
                value = md['seat_map'][who]
                return native_adapter.build_generator(value, md['seed'], seat=who,
                    model_config=(md.get('model_configs') or {}).get(value),
                    **({'request_bound': bounds_by_sample[(arm, sample_id)][value]} if spending else {}))
            try:
                await run_episode(journal, identity, md, factory, reserve_usd=rates[arm],
                    budget_usd=budget, allow_estimated_cost=allow_estimates, sample_id=sample_id,
                    spending=spending, request_bounds=bounds_by_sample.get((arm, sample_id)),
                    spending_namespace=namespace, request_control=request_control,
                    dispatch_paused=lambda: bool(dispatch_paused and dispatch_paused()),
                    on_pause=lambda: pauses.append('game paused; new dispatch stopped'),
                    estimate_call=lambda who, usage: usage_estimate(cfg.get('cost') or {}, md['seat_map'][who], usage))
                journal.export_episode(identity, journal.path.parent / 'episodes' / f'{identity}.episode.json.gz')
            except RunPaused as exc:
                pauses.append(str(exc))
            finally:
                queue.task_done()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    journal.write_status(force=True)
    if pauses:
        raise RunPaused('; '.join(dict.fromkeys(pauses)))
    return journal.summary()
