"""YAML-configured launcher: native game journals, with legacy Inspect compatibility.

  uv run python -m delegated_influence.run configs/01_smoke.yaml [--dry-run]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from inspect_ai import eval_set

from delegated_influence import artifacts as run_artifacts
from delegated_influence import experiment, preflight

# Compatibility exports for existing callers; experiment.py is the single implementation.
DEFAULT_COST = experiment.DEFAULT_COST
expand_models = experiment.expand_models
build_tasks = experiment.build_tasks
estimate_cost = experiment.estimate_cost
enforce_budget = experiment.enforce_budget


def launch_live_run(config_path, cfg, tasks, estimate):
    """Freeze launch inputs, run Inspect, then export joined turn records.

    The manifest is created before the first API call. A failed/interrupted eval gets a sealed
    completion marker while its raw Inspect log remains resumable. No `.env` content is copied.
    """
    enforce_budget(cfg, estimate)
    log_dir = cfg.get("log_dir", "logs")
    retry_attempts = cfg.get("retry_attempts", 3)
    max_tasks = cfg.get("max_tasks", 4)
    frozen = run_artifacts.create_run_artifacts(
        config_path, cfg, estimate, tasks, log_dir,
        retry_attempts=retry_attempts, max_tasks=max_tasks,
    )
    try:
        success, logs = eval_set(
            [t for _, t in tasks], log_dir=log_dir, max_tasks=max_tasks,
            retry_attempts=retry_attempts, metadata=frozen.eval_metadata,
            # Inspect's default keeps raw API details for only the first few calls per model.
            # Full logging is required to retain served provider and exact OpenRouter cost for
            # every turn. The analysis exporter copies only research fields, not raw reasoning.
            log_model_api=True,
        )
    except BaseException as exc:
        run_artifacts.record_interrupted_run(frozen, type(exc).__name__)
        raise
    try:
        run_artifacts.finalize_run(frozen, logs, success)
    except BaseException as exc:
        run_artifacts.record_finalization_error(
            frozen, type(exc).__name__, eval_success=bool(success))
        raise
    return success, logs



def launch_native_run(config_path, cfg, tasks, estimate, *, dispatch_paused=None):
    """Freeze the launch and resume calls from the native game journal."""
    from delegated_influence.journal import Journal, RunPaused, canonical
    from delegated_influence.runtime import run_tasks
    enforce_budget(cfg, estimate)
    log_dir = Path(cfg.get("log_dir", "logs"))
    if not (log_dir / "run.sqlite3").exists() and any(log_dir.glob("**/*.eval")):
        raise RunPaused("this directory has Inspect logs; use --runner inspect to resume them, "
                        "or choose a new log_dir for a native run")
    with Journal(log_dir / "run.sqlite3") as journal:
        frozen = run_artifacts.create_run_artifacts(
            config_path, cfg, estimate, tasks, log_dir,
            retry_attempts=0, max_tasks=(cfg.get("native") or {}).get("max_concurrent_episodes", 1),
            runner="native",
        )
        manifest = json.loads(frozen.manifest_path.read_text())
        journal.bind("source", manifest["source"]["tree_sha256"])
        # Resume combines cached responses and new requests in one game. Keep its
        # environment fixed even though independent games can share a scientific
        # instrument across dependency updates (see runtime_compatibility_key).
        journal.bind("installed_runtime", {key: manifest["runtime"][key] for key in ("python", "packages")})
        journal.bind("plan", [{"arm": arm, "sample_id": sample.id, "metadata": sample.metadata}
                              for arm, task in tasks for sample in task.dataset])
        journal.bind("cost", cfg.get("cost") or {})
        bind_native_config(journal, cfg)
        journal.db.execute("INSERT OR IGNORE INTO meta VALUES ('origin',?)", (canonical({
            "run_manifest": str(frozen.manifest_path), "commit": manifest["source"]["git"]["commit"]}),))
        try:
            summary = asyncio.run(run_tasks(journal, tasks, estimate, cfg, dispatch_paused=dispatch_paused))
        except BaseException:
            journal.write_status(force=True)
            raise
        print(json.dumps(summary, indent=2))
        return summary


def bind_native_config(journal, cfg):
    """Freeze launch settings; spending validates the separately audited cap."""
    from delegated_influence.journal import RunPaused

    def without_cap(value):
        if not value.get('spending'):
            return value
        spending = {key: item for key, item in value['spending'].items() if key != 'cap_usd'}
        spending['ledger'] = str(Path(spending['ledger']).resolve())
        return {**value, 'spending': spending}

    if (not journal.db.execute("SELECT 1 FROM meta WHERE key='config'").fetchone()
            and journal.db.execute('SELECT 1 FROM calls LIMIT 1').fetchone()):
        # A mutable manifest path cannot prove the original config of a legacy journal.
        raise RunPaused('existing calls need a saved launch config binding before resume; use the original runtime')
    journal.bind('config', without_cap(cfg))


def recovery_validator(journal, ledger, policy, call_id, key):
    """Check operator evidence before either durable accounting record changes."""
    from delegated_influence.dispatch import RequestBound, amount
    from delegated_influence.journal import RunPaused, canonical, valid_cost

    call = journal.db.execute('SELECT * FROM calls WHERE id=?', (call_id,)).fetchone()
    held = ledger.db.execute('SELECT * FROM spending_calls WHERE call_key=?', (key,)).fetchone()
    if not call or not held or held['status'] not in ('held', 'settled'):
        raise ValueError('recovery needs a saved call and an outstanding or settled spending reservation')
    metadata = journal.episode(call['episode'])['metadata']
    try:
        model = metadata['seat_map'][call['seat']]
        instrument = metadata['sampling_instruments'][model]
        controls = (metadata.get('spending_instrument') or {}).get(model)
        route = controls['route'] if controls else policy['manifest']['models'][model]
        bound = RequestBound.from_recorded(route, instrument)
        if controls and (canonical(controls['provider_routing']) != canonical(bound.routing)
                         or amount(controls['maximum_usd'], 'saved bound') != bound.maximum_usd):
            raise ValueError('recovery has inconsistent saved request controls')

        def validate_wire(raw):
            bound.check_request(raw)
            body = json.loads(raw)
            expected_seed = metadata['seed'] if 'seed' in route.get('supported_parameters', []) else None
            if (body.get('seed') != expected_seed
                    or body.get('temperature') != instrument['temperature']):
                raise ValueError('recovery request sampling differs from the saved episode')

        wire = journal.call(call_id)['provider_request_body']
        if wire is not None:
            validate_wire(wire)
    except (KeyError, TypeError, RunPaused) as exc:
        raise ValueError(f'recovery needs valid frozen request evidence: {exc}') from exc

    def validate(usage):
        cost = usage.get('total_cost')
        if not valid_cost(cost):
            raise ValueError('recovery requires a finite nonnegative provider bill')
        reported = amount(cost, 'recovered bill')
        if held['status'] == 'settled' and reported != amount(held['reported_usd'], 'settled bill'):
            raise ValueError('recovery must preserve the settled bill')
        ceiling = min(bound.maximum_usd, amount(held['maximum_usd'], 'saved reservation'))
        if reported > ceiling and call['reported'] is None:
            raise ValueError('recovery bill exceeds the frozen request bound; check the provider evidence')
        try:
            bound.check_usage(usage)
            supplied_wire = usage.get('provider_request_body')
            if supplied_wire is not None:
                if wire is not None and supplied_wire != wire:
                    raise ValueError('recovery conflicts with the saved provider request')
                validate_wire(supplied_wire)
            raw_cost = (usage.get('provider_usage') or {}).get('cost')
            if raw_cost is not None and amount(raw_cost, 'raw provider bill') != reported:
                raise ValueError('recovery bill differs from raw provider evidence')
        except (RunPaused, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f'recovery usage evidence is incomplete or inconsistent: {exc}') from exc
    return validate


def main(argv=None, *, dispatch_paused=None):
    ap = argparse.ArgumentParser(description="Run a delegated-influence config with durable game records.")
    ap.add_argument("config", help="path to a YAML run config")
    ap.add_argument("--dry-run", action="store_true", help="print an order-of-magnitude cost, run nothing")
    ap.add_argument("--runner", choices=["native", "inspect"], default="native",
                    help="native game journal (default), or historical Inspect mode")
    ap.add_argument("--status", action="store_true", help="read native progress without model calls")
    ap.add_argument("--resolve-call", help="record a decision for one uncertain native call; sends no request")
    ap.add_argument("--resolve-reservation", help="release an orphan spending hold for a call ID with no journal row")
    ap.add_argument("--response-file", help="JSON with response and usage recovered from the provider")
    ap.add_argument("--confirm-unbilled", action="store_true", help="confirm the uncertain request was not billed")
    ap.add_argument("--evidence", help="billing/response evidence for the resolution audit record")
    args = ap.parse_args(argv)

    config_path = Path(args.config)
    cfg = experiment.load_config(config_path)
    if args.status:
        print((Path(cfg.get("log_dir", "logs")) / "status.json").read_text())
        return
    if args.resolve_call or args.resolve_reservation:
        from delegated_influence.journal import Journal
        from delegated_influence.runtime import open_audited_spending_ledger
        recovered = json.loads(Path(args.response_file).read_text()) if args.response_file else {}
        with Journal(Path(cfg.get("log_dir", "logs")) / "run.sqlite3") as journal:
            if args.resolve_call and args.resolve_reservation:
                raise ValueError('resolve one call or one orphan reservation at a time')
            saved = journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()
            policy = json.loads(saved['value']) if saved else None
            ledger = open_audited_spending_ledger(policy, cfg.get('spending') or policy)[0] if policy else None
            try:
                call_id = args.resolve_call or args.resolve_reservation
                if ledger:
                    namespace = json.loads(journal.db.execute(
                        "SELECT value FROM meta WHERE key='spending_namespace'").fetchone()['value'])
                    key = f'{namespace}:{call_id}'
                if args.resolve_reservation:
                    if not ledger or not args.confirm_unbilled or args.response_file:
                        raise ValueError('orphan resolution requires a spending ledger and confirmation of no billing')
                    if journal.db.execute('SELECT 1 FROM calls WHERE id=?', (call_id,)).fetchone():
                        raise ValueError('call has journal evidence; use --resolve-call')
                    ledger.confirm_unbilled(key, evidence=args.evidence)
                else:
                    # A known settled bill cannot be released by a contradictory resolution.
                    if ledger and args.confirm_unbilled:
                        held = ledger.db.execute('SELECT status FROM spending_calls WHERE call_key=?', (key,)).fetchone()
                        if held and held['status'] == 'settled':
                            raise ValueError('a settled bill cannot be confirmed unbilled')
                    validate = (recovery_validator(journal, ledger, policy, call_id, key)
                                if ledger and not args.confirm_unbilled else None)
                    result = journal.resolve_uncertain(call_id, evidence=args.evidence,
                        response=recovered.get('response'), usage=recovered.get('usage'),
                        confirm_unbilled=args.confirm_unbilled, validate_recovery=validate)
                    if ledger:
                        if args.confirm_unbilled:
                            ledger.confirm_unbilled(key, evidence=args.evidence)
                        else:
                            ledger.settle(key, result['usage']['total_cost'])
            finally:
                if ledger:
                    (journal.path.parent / 'spending-status.json').write_text(json.dumps(ledger.summary(), indent=2) + '\n')
                    ledger.close()
        print("Decision saved. No request sent. Run the config again to resume.")
        return
    tasks = build_tasks(cfg, config_name=config_path.stem)
    est = estimate_cost(cfg, tasks)

    if args.dry_run:
        print(f"# {args.config}: {len(tasks)} arms")
        for r in est["arms"]:
            print(f"  {r['arm']:<22} {r['episodes']:>4} episodes x ~{r['turns_per_episode']} turns  "
                  f"base ~${r['base_usd']:,.1f}, turn-retry allowance ~${r['usd']:,.1f}")
        print(f"  {'TOTAL':<22} base ~${est['base_usd']:,.1f}; "
              f"turn-retry allowance ~${est['total_usd']:,.1f}  "
              f"({est['base_calls']:,} normal / {est['scheduled_calls']:,} planned calls with allowance)")
        print(f"  pricing: {est['pricing_basis']}; planning estimate only, not a billing cap")
        return

    problems = preflight.check_config(config_path)
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"preflight failed for {config_path}:\n{details}")
    if args.runner == "inspect":
        if any(not value.startswith('mockllm/') for _, item in tasks for sample in item.dataset
               for value in sample.metadata['seat_map'].values()):
            raise ValueError('paid gameplay requires the native spending ledger; Inspect remains for historical logs and offline checks')
        launch_live_run(config_path, cfg, tasks, est)
    else:
        from delegated_influence.journal import RunPaused
        try:
            launch_native_run(config_path, cfg, tasks, est, dispatch_paused=dispatch_paused)
        except RunPaused as exc:
            sys.exit(f"Run paused: {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m delegated_influence.run <config.yaml> [--dry-run]")
    main()
