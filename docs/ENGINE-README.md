# delegated-influence

A competitive language-model society for measuring who gains influence over other agents' scarce
actions when speaking also costs an action.

Each agent has scarce turns. It may communicate, punish when enabled, pull for another agent to
give them one point, or—in mixed-economy runs—pull for itself for a smaller reward. Every action is
recorded in a complete internal event ledger; agents receive a bounded, neutral view of that ledger.

The action-allocation network is the main research object. Three constructs help explain it:

1. **Cooperation:** did an agent help another when a rewarding selfish action was available?
2. **Coalition:** did repeated support survive a later-round test showing preference for insiders
   over available outsiders?
3. **Deception:** separate ledger contradiction or broken-by-choice promise from likely deception,
   later benefit, intent-to-action effect, and enforcement.

Pure-economy pulls remain useful for partner choice, reciprocity, and loyalty, but do not identify a
cooperation rate. Repeated mutual pulls remain useful pair-persistence data, but are not coalition
labels without later confirmation. Benefit after a deceptive event remains descriptive unless an
unused pre-message intent baseline shows changed behavior.

Start here:

- [`AGENTS.md`](AGENTS.md) — required onboarding and wiki-update routine.
- [`wiki/WORKING.md`](wiki/WORKING.md) — current status and next gates only.
- [`wiki/pages/analyses/2026-07-20-narrative-experiment-map.md`](wiki/pages/analyses/2026-07-20-narrative-experiment-map.md) — paper narrative, preregistration status, and experiment ownership.
- [`wiki/pages/entities/metrics.md`](wiki/pages/entities/metrics.md) — canonical definition → code → output contract.
- [`wiki/CLAIMS.md`](wiki/CLAIMS.md) — result claims and evidence status.

## Status

New runs use a native game journal through `run.py` and `configs/`. Inspect supplies request mapping and historical log support.
Main collection launched on 9 September. Experiment 22 has 150 completed games; Experiment 21 has 29 of 30.
Experiments 24–27 have completed exports and analysis. Human validation of whistleblowing labels remains incomplete.
Experiment 20 is still running. See the [saved analysis](artifacts/main-analysis/2026-09-10/README.md) and
[current recovery record](artifacts/campaign-recovery/2026-09-10/README.md) for evidence and operational limits.
The new runtime adds shared provider limits and conservative retry recovery. Existing campaigns retain their frozen source.

## Built on Inspect

This benchmark runs on [Inspect](https://inspect.aisi.org.uk/) (AISI's evals framework).
One episode = one Inspect `Sample`; a custom solver drives the game loop, calling each seated
model via `get_model("openrouter/...")`; game state lives in the Inspect `Store`. The only Inspect
scorer is the forfeit rate; every metric is recomputed offline from the stored events. The pure
engine (`game.py`) and metrics (`metrics.py`, `coalition.py`, `stats.py`) stay framework-agnostic
and offline-testable.

## Setup

The eval stack (Inspect) needs a **standard (GIL) CPython 3.13** — the free-threaded build has no
wheels for inspect-ai's Rust deps. The pure core (`game`/`metrics`/`seat`/`turn`) is stdlib-only and
runs on either interpreter; only `task.py` and live runs need the venv.

```bash
uv venv --python 3.13 .venv                          # standard CPython 3.13 (gitignored)
uv pip install --python .venv/bin/python -e '.[dev]'
export OPENROUTER_API_KEY=...                          # or a local .env (gitignored); https://openrouter.ai/keys
```

## Run

```bash
# offline pipeline check — no network or cost:
PYTHONPATH=src .venv/bin/python -m pytest -q
# price a run before spending, then launch it (resumable):
uv run python -m delegated_influence.run configs/01_smoke.yaml --dry-run
uv run python -m delegated_influence.run configs/01_smoke.yaml
# aggregate the logs into the descriptive report (paper pool by default, as on the site;
# smoke and other exploratory runs need --pool all):
uv run python -m delegated_influence.report logs/01_smoke --pool all
# after headline runs, apply the frozen paper-level claim table:
uv run python -m delegated_influence.report logs/20_arena_25 logs/21_arena_5x5 logs/22_cooperation_cost_curve --paper-claims
```

Each experiment has one YAML config. New runs use the native scheduler by default.
Use `--runner inspect` to resume historical Inspect runs. Use a new `log_dir` when switching runners;
the launcher refuses to restart old Inspect games as new native games in the same directory.

The native runner saves each request before dispatch and each response before game scoring.
After a restart, it replays saved responses inside the same game. It rejects changed prompts,
plans, source files, or installed runtime versions. Independent games never share model responses.
A lost response pauses the run for billing review. It does not become an agent forfeit or an automatic retry.

```bash
# Read progress and cost coverage without model calls:
uv run python -m delegated_influence.run configs/01_smoke.yaml --status
# Save current public route limits before a paid launch:
uv run python -m delegated_influence.dispatch CONFIG --output routes.json
```

Live runs require planning budgets and `spending: {ledger: PATH, cap_usd: AMOUNT, manifest: ROUTES_JSON}`.
The shared ledger reserves a conservative request charge before dispatch. Concurrent configs must use the same
ledger and cap. Unknown bills keep their reservations. Pauses stop further requests, and incomplete games stay outside analysis.
Provider errors, other clients, judge commands, and credit fees remain outside an invoice guarantee.
See [spending controls and recovery](docs/native-runtime.md) for exact limits.

Set `native.max_concurrent_episodes` to control concurrent games explicitly (default: 1).
Missing provider costs pause paid native runs. The planning estimate option cannot bypass the request ledger.
The legacy blended price remains clearly labeled.
Per-model input/output/cache rates can replace it. See [native runtime and pricing](docs/native-runtime.md).

Every launch freezes its YAML, source snapshot, Git state, and installed package versions under
`run-artifacts/`. Native runs store exact calls and incremental events in `run.sqlite3`, live status in
`status.json`, and complete portable games in `episodes/*.episode.json.gz`. Reports and the reader accept
both these exports and old `.eval` files. Preserve the journal with the run directory for recovery.
No environment variables or credentials enter these artifacts.

Historical Inspect logs remain readable. Paid gameplay now requires the native runner.
Historical exports can be rebuilt without model calls:

```bash
uv run python -m delegated_influence.artifacts logs/<experiment>
```

[Configs 24 and 25](docs/controlled-experiments.md) define exploratory whistleblowing and matched reasoning experiments.
[The cost pilot](docs/cost-pilot.md) keeps full game lengths and reports measured coverage before extrapolation.

## Running on RunPod (concurrent experiments)

Clone the repo under the persistent volume (so `logs/` survives restarts), set
`OPENROUTER_API_KEY` as a pod env var, then:

```bash
bash scripts/runpod_setup.sh          # uv + venv + deps + tests + dry-run every config
bash scripts/launch.sh configs/02_background_check.yaml configs/03_five_principals.yaml
tail -f logs/launcher/03_five_principals.out
```

`launch.sh` runs each config as its own disconnect-proof process. All experiments share one
OpenRouter account (shared rate limits): start with 2–3 concurrent and watch `forfeit_rate` —
API contention shows up there first.

**Sharing results:** run logs are committed to git (big arms are sharded one task per model, so
no single `.eval` file approaches GitHub's 100MB limit). After a run finishes on the pod:

```bash
git add logs/ && git commit -m "logs: <experiment>" && git push
```

then `git pull` on the laptop and run the report/judges over `logs/<experiment>` as usual.

## Layout

```
src/delegated_influence/   game.py (pure engine), metrics.py (offline metrics), models.py (roster)
tests/                     standalone unittest files (run with python3.13 directly, or pytest)
scripts/                   pipeline + analysis
data/raw/                  episode logs (gitignored; released at publication)
docs/                      research-review + reframe HTML docs
wiki/                      the project's persistent brain (LLM-owned, gitignored)
```

## The project brain

`wiki/` is the per-project knowledge base. Start at `wiki/SCHEMA.md` and `wiki/WORKING.md`.
`wiki/DECISIONS.md` records every design decision and why. Root `AGENTS.md` requires agents to
update the canonical wiki owner after every meaningful change. The wiki is currently host-local and
gitignored; it onboards agents in this workspace, but a safe versioned subset is still required for
researchers using a fresh clone.
