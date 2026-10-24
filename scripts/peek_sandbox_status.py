"""Read a campaign's live status.json inside its Sandbox (the Volume copy goes stale once the journal outgrows the 5 s backup budget).

Usage: scripts/modal.sh uv run --frozen --with modal==1.5.5 --with 'cbor2<6' python scripts/peek_sandbox_status.py <sandbox id> <experiment name>
"""
import sys, modal
sandbox_id, experiment = sys.argv[1], sys.argv[2]
sb = modal.Sandbox.from_id(sandbox_id)
cmd = ("date -u; python3 -c \"import json;s=json.load(open('/work/campaign/%s/status.json'));"
       "print('live calls',s['calls'],'pending',s['pending_calls'],'uncertain',s['uncertain_calls'],'usd',round(s['reported_cost_usd'],2),s['episodes'])\"; "
       "ls -la --time-style=+%%H:%%M:%%S /work/campaign/%s/status.json /work/campaign/%s/run.sqlite3" % (experiment, experiment, experiment))
p = sb.exec("bash", "-c", cmd)
print(p.stdout.read()); err = p.stderr.read()
if err: print("ERR:", err[:300])
