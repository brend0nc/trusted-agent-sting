"""Campaign launch must not copy credentials or spend the same allocation twice."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from delegated_influence.spending import SpendingError, SpendingLedger


def module():
    path = Path(__file__).parents[1] / 'scripts/modal_campaign.py'
    assert path.exists(), 'Modal campaign launcher is missing'
    spec = importlib.util.spec_from_file_location('modal_campaign', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_source_bundle_excludes_keys_and_unrelated_artifacts(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in ('pyproject.toml', 'uv.lock', 'README.md', 'LICENSE'):
        (root / name).write_text(name)
    src = root / 'src/delegated_influence'
    src.mkdir(parents=True)
    (src / '__init__.py').write_text('')
    (src / 'worker.py').write_text('pass')
    (src / 'credentials.txt').write_text('SECRET')
    (root / '.keys').write_text('SECRET')
    (root / 'UtqedEcK.txt').write_text('SECRET')
    (root / 'data').mkdir()
    (root / 'data/private.txt').write_text('SECRET')
    dest = tmp_path / 'image'
    module().copy_source(root, dest)
    files = {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    assert files == {'pyproject.toml', 'uv.lock', 'README.md', 'LICENSE',
                     'src/delegated_influence/__init__.py', 'src/delegated_influence/worker.py'}


def test_source_bundle_rejects_symlink_escape(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (tmp_path / 'secret').write_text('SECRET')
    (root / 'pyproject.toml').symlink_to(tmp_path / 'secret')
    with pytest.raises(ValueError, match='symlink'):
        module().copy_source(root, tmp_path / 'image')


def test_source_bundle_allows_symlinked_ancestor_but_rejects_symlinked_package(tmp_path):
    real = tmp_path / 'real'
    (real / 'repo/src/delegated_influence').mkdir(parents=True)
    for name in ('pyproject.toml', 'uv.lock', 'README.md', 'LICENSE'):
        (real / 'repo' / name).write_text(name)
    (real / 'repo/src/delegated_influence/__init__.py').write_text('')
    (tmp_path / 'link').symlink_to(real)
    root = tmp_path / 'link/repo'
    module().copy_source(root, tmp_path / 'image')
    assert (tmp_path / 'image/src/delegated_influence/__init__.py').exists()
    package = root / 'src/delegated_influence'
    shutil.rmtree(package)
    (tmp_path / 'outside').mkdir()
    (tmp_path / 'outside/secret.py').write_text('SECRET')
    package.symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError, match='symlink'):
        module().copy_source(root, tmp_path / 'image-2')


def test_remote_creation_failure_retains_allocation_and_blocks_repeat(tmp_path):
    launch = module()
    ledger_path = tmp_path / 'spending.sqlite3'
    with SpendingLedger(ledger_path, 45) as ledger:
        ledger.reserve('earlier-unknown', 3)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    plan = {'campaign_id': 'test', 'allocation_usd': 10, 'compute_budget_usd': 5}
    calls = []

    def create():
        calls.append('sent')
        raise TimeoutError('response unknown')

    with pytest.raises(TimeoutError):
        launch.dispatch_reserved(prepared, plan, ledger_path, 45, create)
    with SpendingLedger(ledger_path, 45) as ledger:
        assert ledger.summary()['held_usd'] == '13'
    with pytest.raises(ValueError, match='already'):
        launch.dispatch_reserved(prepared, plan, ledger_path, 45, create)
    assert calls == ['sent']


def test_insufficient_cap_blocks_remote_creation(tmp_path):
    launch = module()
    ledger_path = tmp_path / 'spending.sqlite3'
    with SpendingLedger(ledger_path, 45):
        pass
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    calls = []
    plan = {'campaign_id': 'large', 'allocation_usd': 100, 'compute_budget_usd': 5}
    with pytest.raises(SpendingError, match='exceed'):
        launch.dispatch_reserved(prepared, plan, ledger_path, 45, lambda: calls.append('sent'))
    assert not calls


@pytest.mark.parametrize('amount', [None, 0, -1, float('nan'), True])
def test_missing_or_invalid_allowance_blocks_remote_creation(tmp_path, amount):
    calls = []
    with pytest.raises(ValueError):
        module().dispatch_reserved(tmp_path, {'campaign_id': 'test', 'allocation_usd': amount,
                                  'compute_budget_usd': 5}, tmp_path/'not-created', 45,
                                  lambda: calls.append('sent'))
    assert not calls


def test_bundle_hash_detects_changed_launch_input(tmp_path):
    launch = module()
    image = tmp_path / 'image'
    image.mkdir()
    (image / 'config.yaml').write_text('seats: [one]')
    hashes = launch.file_hashes(image)
    (tmp_path / 'bundle.json').write_text(json.dumps({'files': hashes}))
    (image / 'config.yaml').write_text('seats: [two]')
    with pytest.raises(ValueError, match='changed'):
        launch.verify_bundle(tmp_path)


def test_parent_allocation_must_equal_remote_plan(tmp_path):
    launch = module()
    image = tmp_path / 'image'
    (image / 'campaign').mkdir(parents=True)
    (image / 'campaign/plan.json').write_text(json.dumps({'allocation_usd': 100}))
    (tmp_path / 'bundle.json').write_text(json.dumps({
        'files': launch.file_hashes(image), 'plan': {'allocation_usd': 1}}))
    with pytest.raises(ValueError, match='plan'):
        launch.verify_bundle(tmp_path)


def test_stale_route_is_rejected_before_remote_creation():
    sample = SimpleNamespace(metadata={'seat_map': {'P1': 'gpt-5.5'}})
    with pytest.raises(ValueError, match='fresh'):
        module().check_routes([('arm', SimpleNamespace(dataset=[sample]))],
                              {'schema_version': 1, 'verified_at': '2000-01-01T00:00:00Z'})


def test_route_near_expiry_is_rejected_before_remote_creation():
    sample = SimpleNamespace(metadata={'seat_map': {'P1': 'gpt-5.5'}})
    verified_at = (datetime.now(timezone.utc) - timedelta(hours=23, minutes=30)).isoformat()
    with pytest.raises(ValueError, match='fresh'):
        module().check_routes([('arm', SimpleNamespace(dataset=[sample]))],
                              {'schema_version': 1, 'verified_at': verified_at})


def test_accept_requires_verified_bundle_and_scoped_auth_before_modal(tmp_path, monkeypatch):
    launch = module()
    monkeypatch.setattr(launch, 'verify_bundle', lambda _: {'plan': {'campaign_id': 'test'}})
    monkeypatch.delenv('MODAL_ENVIRONMENT', raising=False)
    with pytest.raises(ValueError, match='modal.sh'):
        launch.accept(tmp_path)
    assert not (tmp_path / 'accept-receipt.json').exists()


def test_launch_checks_routes_before_auth_or_dispatch(tmp_path, monkeypatch):
    launch = module()
    monkeypatch.setattr(launch, 'verify_bundle', lambda _: {'plan': {'ready_for_launch': True}})
    def stale(*_):
        raise ValueError('stale routes')
    monkeypatch.setattr(launch, 'check_prepared_routes', stale)
    with pytest.raises(ValueError, match='stale routes'):
        launch.launch(tmp_path, tmp_path / 'unused-ledger', 45)
    assert not (tmp_path / 'unused-ledger').exists()
