"""Campaign status summary stays correct with missing backups and finished campaigns."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / 'scripts/campaign_status.py'
    spec = importlib.util.spec_from_file_location('campaign_status', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_summary_reports_progress_costs_and_final_state():
    status = {'episodes': {'complete': 7, 'running': 16}, 'calls': 900, 'reported_cost_usd': 12.3456,
              'uncertain_calls': 1, 'pending_calls': 16}
    row = module().summarize('di-main-20260909-20', status, {'status': 'stopped', 'stop_reason': 'deadline'})
    assert row == {'campaign': 'di-main-20260909-20', 'complete': 7, 'running': 16, 'paused': 0, 'calls': 900,
                   'reported_usd': 12.35, 'uncertain': 1, 'pending': 16, 'final': 'stopped',
                   'stop_reason': 'deadline', 'backup_seen': True}


def test_summary_marks_campaign_without_backup():
    row = module().summarize('di-main-20260909-25', None, None)
    assert row['backup_seen'] is False
    assert row['complete'] == 0 and row['final'] is None


def test_fetch_refuses_unscoped_environment(monkeypatch):
    monkeypatch.delenv('MODAL_ENVIRONMENT', raising=False)
    try:
        module().fetch('di-main-20260909-25', '25_matched_reasoning')
    except ValueError as exc:
        assert 'modal.sh' in str(exc)
    else:
        raise AssertionError('fetch ran without the scoped environment')
