"""Read campaign progress from result Volumes. Sends no model request and changes nothing remote.

  scripts/modal.sh uv run --frozen --with modal==1.5.5 --with 'cbor2<6' \
    python scripts/campaign_status.py di-main-20260909-20 di-main-20260909-21 ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'delegated-influence'


def summarize(campaign, status, final):
    """One row per experiment from its backed-up status.json and the campaign final.json."""
    episodes = (status or {}).get('episodes') or {}
    return {'campaign': campaign,
            'complete': episodes.get('complete', 0), 'running': episodes.get('running', 0),
            'paused': episodes.get('paused', 0), 'calls': (status or {}).get('calls', 0),
            'reported_usd': round((status or {}).get('reported_cost_usd', 0), 2),
            'uncertain': (status or {}).get('uncertain_calls', 0), 'pending': (status or {}).get('pending_calls', 0),
            'final': (final or {}).get('status'), 'stop_reason': (final or {}).get('stop_reason'),
            'backup_seen': status is not None}


def read_json(volume, path):
    try:
        return json.loads(b''.join(volume.read_file(path)).decode())
    except Exception:
        return None


def fetch(campaign, experiment):
    if os.environ.get('MODAL_ENVIRONMENT') != ENVIRONMENT:
        raise ValueError('run through scripts/modal.sh to use the scoped Modal environment')
    import modal
    volume = modal.Volume.from_name('di-results-' + campaign, environment_name=ENVIRONMENT, version=1)
    return (read_json(volume, f'{campaign}/{experiment}/status.json'), read_json(volume, f'{campaign}/final.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', nargs='+', help='campaign ids; each result Volume is di-results-<id>')
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts/main-launch/2026-09-09',
                        help='directory holding <campaign>/bundle/image/campaign/plan.json')
    parser.add_argument('--history', type=Path, default=ROOT / 'artifacts/main-downloads/status')
    args = parser.parse_args()
    rows = []
    for campaign in args.campaigns:
        plan = json.loads((args.root / campaign / 'bundle/image/campaign/plan.json').read_text())
        for entry in plan['configs']:
            status, final = fetch(campaign, entry['name'])
            rows.append({**summarize(campaign, status, final), 'experiment': entry['name'], 'planned': entry['episodes']})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    args.history.mkdir(parents=True, exist_ok=True)
    (args.history / f'{stamp}.json').write_text(json.dumps(rows, indent=2) + '\n')
    header = f"{'campaign':<22} {'done/plan':>9} {'run':>4} {'paused':>6} {'calls':>7} {'usd':>9} {'unc':>4} {'final':<9} note"
    print(header)
    for row in rows:
        note = '' if row['backup_seen'] else 'no backup yet'
        print(f"{row['campaign']:<22} {row['complete']:>4}/{row['planned']:<4} {row['running']:>4} {row['paused']:>6} "
              f"{row['calls']:>7} {row['reported_usd']:>9} {row['uncertain']:>4} {str(row['final'] or '-'):<9} {note}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
