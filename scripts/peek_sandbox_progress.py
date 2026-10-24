"""Per-game progress for a live campaign, read inside its Sandbox. Downloads nothing.

Reads the live status.json (already written by the worker) for per-episode call counts and
the journal's small `episodes` table (120 rows, WAL read-only) for each game's status, arm
and seed. The 900 MB+ `calls` table is never scanned.

Usage: scripts/modal.sh uv run --frozen --with modal==1.5.5 --with 'cbor2<6' \
         python scripts/peek_sandbox_progress.py <sandbox id> <experiment name>
"""
import sys, modal

sandbox_id, experiment = sys.argv[1], sys.argv[2]

REMOTE = r'''
import json, sqlite3, zlib, collections, datetime, os
exp = "__EXPERIMENT__"
base = "/work/campaign/" + exp
s = json.load(open(base + "/status.json"))
per = collections.Counter()
for key, row in s["by_agent"].items():
    per[key.rsplit(":", 1)[0]] += row["calls"]
con = sqlite3.connect("file:" + base + "/run.sqlite3?mode=ro", uri=True, timeout=60)
con.execute("PRAGMA query_only=ON")
rows = con.execute("SELECT id, status, metadata, started FROM episodes").fetchall()
out = []
for eid, status, meta, started in rows:
    m = json.loads(zlib.decompress(meta))
    total = m.get("rounds", 0) * m.get("actions_per_round", 0) * len(m.get("seat_map", {}))
    out.append((eid, status, m.get("arm") or m.get("name"), m.get("seed"), started, per.get(eid, 0), total))
con.close()
print("NOW", datetime.datetime.now(datetime.timezone.utc).isoformat())
print("TOTALS calls=%d pending=%d uncertain=%d usd=%.2f episodes=%s" % (
    s["calls"], s["pending_calls"], s["uncertain_calls"], s["reported_cost_usd"], s["episodes"]))
print("EPISODES", len(out))
for r in sorted(out, key=lambda r: (r[1], -r[5])):
    print("EP\t%s\t%s\t%s\t%s\t%s\t%d\t%d" % r)
st = os.stat(base + "/run.sqlite3")
print("JOURNAL_BYTES", st.st_size)
'''

p = modal.Sandbox.from_id(sandbox_id).exec("bash", "-c",
    "python3 - <<'PYEOF'\n" + REMOTE.replace("__EXPERIMENT__", experiment) + "\nPYEOF")
print(p.stdout.read())
err = p.stderr.read()
if err:
    print("ERR:", err[:2000])
