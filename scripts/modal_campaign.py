"""Prepare or explicitly launch one Modal Sandbox with parallel experiment processes.

Preparation sends no requests. Launch requires a funded parent ledger and scoped Modal auth.
An unknown launch or remote failure retains the entire API allocation. Never auto-restart it.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
ENVIRONMENT = 'delegated-influence'
REMOTE = Path('/app/campaign')
WORK = Path('/work/campaign')
# Route evidence expires after 24 hours; image build and Sandbox start happen after the local check.
ROUTE_MAX_AGE = timedelta(hours=23)


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive number')
    return value


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def copy_source(root, image):
    paths = [root / name for name in ('pyproject.toml', 'uv.lock', 'README.md', 'LICENSE')]
    paths += sorted((root / 'src/delegated_influence').rglob('*.py'))
    for source in paths:
        if source.is_symlink() or any((root / p).is_symlink() for p in source.relative_to(root).parents):
            raise ValueError(f'symlink cannot enter the image: {source.name}')
        target = image / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def file_hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_bundle(directory):
    bundle = json.loads((directory / 'bundle.json').read_text())
    if file_hashes(directory / 'image') != bundle['files']:
        raise ValueError('prepared bundle changed; prepare a new campaign')
    if json.loads((directory / 'image/campaign/plan.json').read_text()) != bundle['plan']:
        raise ValueError('local allocation and remote plan differ')
    return bundle


def check_routes(built, manifest):
    from delegated_influence.dispatch import RequestBound
    verified = datetime.fromisoformat(manifest['verified_at'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) - verified > ROUTE_MAX_AGE:
        raise ValueError('route manifest must stay fresh through image build and Sandbox start; refresh routes')
    checked = set()
    for _, task in built:
        for sample in task.dataset:
            for model in set(sample.metadata['seat_map'].values()):
                settings = (sample.metadata.get('model_configs') or {}).get(model)
                identity = (model, json.dumps(settings, sort_keys=True))
                if identity not in checked:
                    RequestBound(manifest, model, settings)
                    checked.add(identity)


def check_prepared_routes(directory, plan):
    from delegated_influence import experiment
    local = directory / 'image/campaign'
    for row in plan['configs']:
        cfg = experiment.load_config(local / 'configs' / Path(row['path']).name)
        manifest = json.loads((local / 'routes' / (row['name'] + '.json')).read_text())
        check_routes(experiment.build_tasks(cfg, config_name=row['name']), manifest)


def prepare(paths, destination, *, campaign_id, allocation=None, compute_budget=None, workers=8):
    import yaml
    from delegated_influence import experiment, preflight

    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}', campaign_id):
        raise ValueError('campaign id must use lowercase letters, digits, and hyphens')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if allocation is not None:
        positive(allocation, 'API allocation')
    if compute_budget is not None:
        positive(compute_budget, 'Modal compute budget')
    paths = [Path(p).resolve() for p in paths]
    if len({p.stem for p in paths}) != len(paths) or not paths:
        raise ValueError('select distinct config names')
    if any(p.stem.startswith('23_') for p in paths):
        raise ValueError('experiment 23 requires its separate 13b gate; omit it from this campaign')
    destination.mkdir(parents=True, exist_ok=False)
    image = destination / 'image'
    copy_source(ROOT, image)
    local = image / 'campaign'
    (local / 'configs').mkdir(parents=True)
    (local / 'routes').mkdir()
    rows = []
    for path in paths:
        cfg = experiment.load_config(path)
        problems = preflight.check_config(path)
        if problems:
            raise ValueError(f'{path.name}: {problems}')
        built = experiment.build_tasks(cfg, config_name=path.stem)
        original_ids = [experiment.episode_identity(s.id, s.metadata) for _, t in built for s in t.dataset]
        manifest_path = Path(cfg['spending']['manifest'])
        manifest = json.loads(manifest_path.read_text())
        check_routes(built, manifest)
        estimate = experiment.estimate_cost(cfg, built)
        cfg = deepcopy(cfg)
        # The plan stays launch-blocked if either allowance is missing.
        cap = allocation if allocation is not None else cfg['spending']['cap_usd']
        cfg['log_dir'] = str(WORK / path.stem)
        cfg['native'] = {**(cfg.get('native') or {}), 'max_concurrent_episodes': workers, 'allow_estimated_cost': False}
        cfg['spending'] = {'ledger': str(WORK / 'spending.sqlite3'), 'cap_usd': cap,
                           'manifest': str(REMOTE / 'routes' / (path.stem + '.json'))}
        cfg['budget'] = {'max_estimated_usd': estimate['total_usd'], 'max_actual_usd': cap}
        rebuilt = experiment.build_tasks(cfg, config_name=path.stem)
        assert original_ids == [experiment.episode_identity(s.id, s.metadata) for _, t in rebuilt for s in t.dataset]
        (local / 'configs' / path.name).write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(local / 'routes' / (path.stem + '.json'), manifest)
        rows.append({'name': path.stem, 'path': str(REMOTE / 'configs' / path.name),
                     'episodes': len(original_ids), 'estimate': estimate,
                     'source_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plan = {'campaign_id': campaign_id, 'allocation_usd': allocation, 'compute_budget_usd': compute_budget,
            'work_dir': str(WORK), 'backup_dir': '/backups/' + campaign_id,
            'configs': rows, 'run_seconds': 23 * 3600, 'backup_seconds': 30,
            'workers_per_experiment': workers, 'ready_for_launch': allocation is not None and compute_budget is not None,
            'budget_note': 'API allocation is reserved in the parent ledger. Modal compute budget is a planning allowance, not an invoice cap.'}
    write_json(local / 'plan.json', plan)
    write_json(destination / 'bundle.json', {'files': file_hashes(image), 'plan': plan}, exclusive=True)
    return plan


def dispatch_reserved(directory, plan, parent_path, parent_cap, create):
    from delegated_influence.spending import SpendingLedger

    allocation = positive(plan.get('allocation_usd'), 'API allocation')
    positive(plan.get('compute_budget_usd'), 'Modal compute budget')
    intent = directory / 'launch-intent.json'
    if intent.exists():
        raise ValueError('campaign already has a launch attempt; inspect its saved state')
    key = 'modal-campaign:' + plan['campaign_id']
    with SpendingLedger(parent_path, parent_cap, existing_only=True) as ledger:
        reservation = ledger.reserve(key, allocation)
        if not reservation['dispatch_allowed']:
            raise ValueError('campaign already has a retained allocation; do not create another Sandbox')
        write_json(intent, {'campaign_id': plan['campaign_id'], 'allocation_key': key,
                           'allocation_usd': allocation, 'parent_ledger': str(Path(parent_path).resolve()),
                           'state': 'creation_outcome_unknown_until_receipt'}, exclusive=True)
        # Any exception keeps the full parent reservation and the launch intent.
        sandbox = create()
        write_json(directory / 'launch-receipt.json', {'sandbox_id': sandbox.object_id,
                   'allocation_key': key, 'state': 'launched'}, exclusive=True)
        sandbox.detach()
        return sandbox.object_id


def require_scoped_modal():
    if os.environ.get('MODAL_ENVIRONMENT') != ENVIRONMENT or os.environ.get('MODAL_CONFIG_PATH') != '/nonexistent/modal.toml':
        raise ValueError('run through scripts/modal.sh to use the scoped Modal environment')
    if not all(os.environ.get(k) for k in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET')):
        raise ValueError('scoped Modal token is not unlocked')


def build_image(image_dir):
    import modal
    return (modal.Image.debian_slim(python_version='3.13')
            .apt_install('git')
            .uv_sync(uv_project_dir=str(image_dir), frozen=True)
            .add_local_dir(image_dir, '/app', copy=True)
            .env({'PYTHONPATH': '/app/src'}))


def accept(directory):
    """Build the campaign image and prove Volume persistence. No model key, no reservation."""
    plan = verify_bundle(directory)['plan']
    require_scoped_modal()
    import modal
    app = modal.App.lookup('delegated-influence-main', environment_name=ENVIRONMENT, create_if_missing=True)
    volume = modal.Volume.from_name('di-results-' + plan['campaign_id'], environment_name=ENVIRONMENT,
                                    create_if_missing=True, version=1)
    marker = f"accept-{plan['campaign_id']}.txt"
    # No login shell: Debian's /etc/profile would reset PATH and hide the uv virtualenv.
    script = ("python -c 'import sys; print(sys.executable)'"
              " && python -c 'import delegated_influence.campaign_worker, delegated_influence.run; print(\"import ok\")'"
              " && git --version"
              " && python -c 'import json; print(json.load(open(\"/app/campaign/plan.json\"))[\"campaign_id\"])'"
              f" && echo accepted > /backups/{marker}")
    sandbox = modal.Sandbox.create('bash', '-c', script, app=app, image=build_image(directory / 'image'),
                                   cpu=(4, 4), memory=(8192, 8192), timeout=1800, workdir='/app', volumes={'/backups': volume})
    sandbox.wait(raise_on_termination=False)
    output = sandbox.stdout.read() + sandbox.stderr.read()
    persisted = None
    if sandbox.returncode == 0:
        for _ in range(12):  # The Volume commits shortly after the Sandbox exits.
            try:
                persisted = b''.join(volume.read_file(marker)).decode().strip()
                break
            except Exception:
                time.sleep(5)
    receipt = {'campaign_id': plan['campaign_id'], 'sandbox_id': sandbox.object_id, 'returncode': sandbox.returncode,
               'output': output[-2000:], 'volume_marker': persisted, 'accepted': sandbox.returncode == 0 and persisted == 'accepted'}
    write_json(directory / 'accept-receipt.json', receipt)
    if not receipt['accepted']:
        raise ValueError('remote acceptance failed; see accept-receipt.json')
    return receipt


def launch(directory, parent_path, parent_cap):
    bundle = verify_bundle(directory)
    plan = bundle['plan']
    if not plan['ready_for_launch']:
        raise ValueError('both main API and Modal compute allowances must be set before launch')
    check_prepared_routes(directory, plan)
    require_scoped_modal()
    from dotenv import dotenv_values
    saved = dotenv_values(ROOT / '.keys')
    key = os.environ.get('OPENROUTER_API_KEY') or saved.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is missing')
    # A config may route some seats straight to their provider (native.direct_models); those keys
    # travel the same way as the OpenRouter one and are absent when nothing needs them.
    secrets = {'OPENROUTER_API_KEY': key}
    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        value = os.environ.get(name) or saved.get(name)
        if value:
            secrets[name] = value
    import modal
    image = build_image(directory / 'image')

    def create():
        app = modal.App.lookup('delegated-influence-main', environment_name=ENVIRONMENT, create_if_missing=True)
        volume = modal.Volume.from_name('di-results-' + plan['campaign_id'], environment_name=ENVIRONMENT,
                                        create_if_missing=True, version=1)
        return modal.Sandbox.create('python', '-m', 'delegated_influence.campaign_worker', str(REMOTE / 'plan.json'),
            app=app, name=plan['campaign_id'], image=image, cpu=(4, 4), memory=(8192, 8192), timeout=86400, workdir='/app',
            volumes={'/backups': volume}, secrets=[modal.Secret.from_dict(secrets)])

    return dispatch_reserved(directory, plan, parent_path, parent_cap, create)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('configs', nargs='+', type=Path)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--campaign-id', required=True)
    prep.add_argument('--api-allocation-usd', type=float)
    prep.add_argument('--modal-compute-budget-usd', type=float)
    prep.add_argument('--workers', type=int, default=8)
    check = sub.add_parser('accept')
    check.add_argument('directory', type=Path)
    live = sub.add_parser('launch')
    live.add_argument('directory', type=Path)
    live.add_argument('--parent-ledger', type=Path, required=True)
    live.add_argument('--parent-cap-usd', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.configs, args.out, campaign_id=args.campaign_id,
                         allocation=args.api_allocation_usd, compute_budget=args.modal_compute_budget_usd,
                         workers=args.workers)
        print(json.dumps(result, indent=2))
    elif args.command == 'accept':
        print(json.dumps(accept(args.directory.resolve()), indent=2))
    else:
        print(launch(args.directory.resolve(), args.parent_ledger, args.parent_cap_usd))


if __name__ == '__main__':
    main()
