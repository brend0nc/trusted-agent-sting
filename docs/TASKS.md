# Sprint task list: trusted-agent sting on the delegated-influence engine (track 3.2)

Started Sat 24 Oct, 12:00 SAST. Brendon chose a variation of delegated influence that answers the 3.2 questions. This list replaces the two earlier lists (the old repo's `docs/sprint/TASKS.md`, and the Liar's Bar list in `~/pandora/liars-bar-starter-copy/`). We change it as we go. Each task has a "done when" line. Only Brendon ticks a box.

- **Hard deadline:** Mon 26 Oct, 13:59 SAST (Sun 23:59 AoE).
- **Internal target:** submit by Sun 25 Oct, 22:00 SAST.
- **Times below are proposals.**
- **Paid steps** are marked 💲. They need Brendon's go. Claude or Codex give him the exact command, and he runs it.

**The 3.2 questions we must answer** (sprint page, Resources tab):
1. A trusted agent offers coordination or collusion to a group of agents. Which of them reciprocate?
2. Do agents accept, reject or become suspicious, depending on how the trusted agent approaches them?
3. Practical start: keep an explicit alliance condition as a positive control. One scripted player offers the same pact to each ordinary agent. Compare a direct offer, an indirect hint and a benign cooperation request. Record verbal acceptance and repeated preferential play.

---

## Phase 0: Lock the design (target Sat 14:00)

- [ ] **0.1 Codex revision 7 for the delegated-influence engine.** Codex is working on it in fork `01a16d34`. It covers: the pact in our game, an offer to each ordinary agent, preferential play over rounds, the alliance positive control, and the engine changes with file:line references.
  Done when: the reply is in the research log, and Claude has checked its code claims.
- [ ] **0.2 Brendon decides the open choices.** These include: the pact, the targets, the arms, the outcomes, the positive control, the models, the sample and the budget.
  Done when: each choice has one answer, written here.
- [ ] **0.3 Write `docs/PREREG.md`, then freeze it.** Include hypotheses, outcomes, coding rules, exclusions, the gate, the analysis and the stop rule.
  Done when: Brendon commits it **before** the pilot.

## Phase 1: Setup (target Sat 14:00)

- [ ] **1.1 New repo with the preexisting engine.** Claude copied the engine from delegated-influence commit `7e44c638` (2 Oct, before the sprint) with `git archive`. It holds src, tests, configs, the site source, pyproject, uv.lock, LICENSE and .gitignore, with no artifacts. The README records the source.
  Done when: Brendon checks it and makes the **first commit with only the copied engine** (`src tests configs site scripts pyproject.toml uv.lock LICENSE .gitignore docs/ENGINE-README.md`). Then `README.md` and `docs/TASKS.md` go in a second commit.
- [ ] **1.2 Install and tests.** Claude ran `uv sync --extra dev --extra site` and the full suite, which took 3 minutes and cost nothing. Claude also copied `scripts/` from the same commit, because some tests import it.
  **Result (Sat 12:10):** 1,454 passed, 19 failed, 2 skipped. All 19 failures are `FileNotFoundError` for old data that the copy leaves out on purpose: judge-calibration samples in `data/`, old labeling files in `artifacts/`, and `site/data/summary.json`. None of them is a game, seat, protocol or runner test.
  Done when: Brendon chooses to delete those tests (and the judge modules they cover) or to copy the data in. Claude recommends deleting them, because the sting does not use the old judges.
- [ ] **1.3 API key and route check.** Brendon adds his OpenRouter key the way the engine expects (`docs/ENGINE-README.md`). Then one live call to the chosen model. No game has run since 10 September. 💲 (under $0.10)

## Phase 2: Build (lean build, chosen Sat ~13:00)

Rules: `docs/design/lean-sting.md`. Claude built the items below on Sat 24 Oct. They are unstaged sprint changes plus new files; Brendon commits them after the first commit.
- [ ] **2.1 Sting module** `src/delegated_influence/sting.py`. It holds the builder, the scripted P4 opening, the offer texts, the alliance note, the block checks and the per-seat export. It is registered as task `trusted_sting`.
- [ ] **2.2 Hooks in the preexisting engine:**
  - `protocols.py`: the sting settings, P4's opening, P4 left out of free turns, and the block check.
  - Private-note plumbing in `game.py`, `seat.py` and `turn.py`.
  - The builder registration in `experiment.py`.
- [ ] **2.3 Configs:** `configs/sting_smoke.yaml` (1 seed, 5 games) and `configs/sting_pilot.yaml` (6 seeds, 30 games). The Grok 4.3 settings are copied from exp 28.
- [ ] **2.4 Tests:** 10 new tests in `tests/test_sting.py`, all passing.
- [ ] **2.5 Offline end-to-end check.** The real native runner played 5 mock games. P4 made 0 model calls, and the export gave 15 seat rows. The dry-run plans are: smoke about $0.80 base ($1.50 with the retry allowance); pilot about $4.50 base ($9.00 with the allowance).
- [ ] **2.6 Bug found and fixed during the check:** the planner silently drops a config key named `models`. The builder argument is now `player_models`.
- [ ] **2.7 Code review by Codex.** The first review found 4 issues, and Claude fixed all 4.
  - Spending: Grok's 1M-token context makes the ledger reserve $15 per request, so the $3 and $10 caps would pause at the first call. Now there is one shared ledger cap of **$75 (needs Brendon's approval)**, with 4 parallel games. The $75 ledger is the only hard ceiling. `budget.max_actual_usd` ($3 smoke, $10 pilot) only stops new games from starting, as the second review showed (`journal.py:152`).
  - Parser: a malformed dispute with unquoted keys was skipped. It now rejects the reply, and there is a test for it.
  - Export: a seat with no valid turn before settlement looked like "did not dispute". It now has `dispute_status: no_valid_turn`, plus forfeits by round, and there are 2 tests.
  - One cap for both configs, so moving from smoke to pilot needs no cap change.
  The second review (Codex) gave a **conditional go**: launch the smoke once Brendon approves $75 as the hard spending ceiling. The expected smoke spend is about $1.
  Done when (Phase 2): the Codex findings are closed and the full suite passes.

## Phase 3: Experiments

Brendon runs every paid command himself, from `~/pandora/trusted-agent-sting`, after the first commit.

- [ ] **E0 Live smoke.** 💲 (about $1; run limit $3; shared ledger cap $75)
  1. Set the key in your shell: `export OPENROUTER_API_KEY=...`
  2. Save the route limits (free): `uv run python -m delegated_influence.dispatch configs/sting_smoke.yaml --output logs/sting_smoke/routes.json`
  3. Run: `uv run python -m delegated_influence.run configs/sting_smoke.yaml`
  4. Export: `uv run python -m delegated_influence.sting logs/sting_smoke --out logs/sting_smoke/rows.jsonl`
  Done when: 5 games complete, few forfeits, and Brendon reads at least two transcripts.
- [ ] **E1 Positive-control pilot.** 💲 (cap $10) The same 4 steps with `sting_pilot`.
  Gate: the alliance seat pulls for P4 and does not dispute in most alliance games, the offers reach all three seats, and at least 95% of turns are valid.
- [ ] **Checkpoint.** If the gate passes, write the main config. If the gate fails, stop and report the failed control.
- [ ] **E2 Main run.** 💲 (size and cap set after the pilot's real cost)

## Phase 4: Analysis (target Sun 12:00)

- [ ] **4.1 Code the raw replies** as accept, refuse or unclear, plus a suspicion flag, without the condition labels. Codex codes 20% on its own.
- [ ] **4.2 Analysis script,** with the positive control and seed-level intervals.
- [ ] **4.3 One headline figure or table.**
- [ ] **4.4 Transcript evidence** from the reader, for each claim.

## Phase 5: Report and submit (target Sun 22:00)

- [ ] **5.1 Draft on the Apart template**, with the Limitations and Dual-Use / Ethical appendix.
- [ ] **5.2 Disclose the preexisting engine** in the README and the report.
- [ ] **5.3 Public repo.** Brendon creates it on GitHub before he submits.
- [ ] **5.4 Review pass,** then submit.
