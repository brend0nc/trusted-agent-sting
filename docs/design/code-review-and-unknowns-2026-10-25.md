# Code review and unknowns pass, Sun 25 Oct 2026

Claude, Sat 24 Oct 22:50 to Sun 25 Oct 11:00 SAST. Brendon asked: "Check all code, do a /code-review and /finding-unknowns." This note holds the result. No code changed.

State of the repo at the time of the pass: protocol v3 is committed (`6b5a802`). Smoke v3 is complete (7 games, 126 calls, $0.28, 0 forfeits). The pilot has not run. The only uncommitted change is `scripts/run_bg.sh` (the "already running" guard now ignores `--dry-run` and `--status`).

How the pass ran. Three code reviews: commit `c6e70f8` (20 files), the v3 working-tree diff (8 files), and a workflow of 8 reviewers over every engine module the sting runs through, each finding checked by 3 skeptical verifiers. One unknowns workflow: 7 finders by area (89 findings), 3 verifiers per finding. The verifiers finished for 36 findings before the usage limit stopped the run. Findings marked **unverified** below come from one finder only; Claude read each one and kept only those whose evidence is a file and line in this repo.

---

## 1. Decisions for Brendon, in order of urgency

| # | Decision | Options | What changes | Evidence |
| --- | --- | --- | --- | --- |
| D1 | **Preregistration before the pilot.** `docs/PREREG.md` does not exist. TASKS 0.3 requires it before E1. | (a) Write a one-page PREREG.md now and commit it before the pilot. (b) Drop 0.3 and call the pilot exploratory. | With (a) the report can claim a frozen gate, primary outcome, contrast, unit and rubric. With (b) it cannot. | `docs/TASKS.md:23`, `:85`, `:91` |
| D2 | **The dispute outcome is at the floor.** In smoke v2 and v3, a seat disputed R0.3 at R1.1 in 14 of 14 games, in every condition, including all 4 direct and coalition_direct games. Offer-arm replies to P4: v1 7 of 9 seats, v2 0 of 12, v3 1 of 12. Non-ally pulls for P4 in v3: 0. | (a) Run the pilot as is. Make dispute timing and first-disputer role the primary outcome. Code the few replies as secondary. (b) Revise the protocol before the pilot (a reply slot before the first action, a dispute that costs points, or higher stakes). This is v4, plus a new smoke, about 1 to 2 hours. | With (a) the headline is likely "Grok exposes the false claim at once and does not reciprocate." With (b) the contrast may move, but the prompts are re-frozen again. | `logs/sting_smoke_v2/rows.jsonl`, `logs/sting_smoke_v3/rows.jsonl`, `docs/TASKS.md:91` |
| D3 | **Gate E1 has no number.** "Most alliance games" is not a number. Codex revision 7 proposed 4 of 6 plus "ally P4-pull share above the same-seed `none` seat". Under v3 any seat that pulls for P4 is paid back, so the ally pulling for P4 alone is weak evidence. | Set the number (4 of 6?) and say if the matched-none clause applies. | Put it into PREREG.md. | `docs/TASKS.md:85`, `docs/design/codex-rev7.md:85` |
| D4 | **The exposure test has no defined observable.** `outcomes()` records messages to P4, public posts, pulls for and from P4. It drops pulls between the coalition partners and messages between ordinary seats. | Define exposure (a pair member names its ally to P4 or on the board, hand coded) and disruption (pair pulls for each other in coalition_direct minus coalition_none; any pair member pull for P4). Pick the primary coalition outcome. | The fields can be added to `outcomes()` after the pilot. No re-run. The events hold the data. | `src/delegated_influence/sting.py:296` |
| D5 | **P4's reciprocity rule is invisible to the players.** P4 pays back its top helper, but no prompt says so. The rule is the same in every condition, so between-condition contrasts stay valid. Absolute levels of `pulls_for_p4` do not measure reciprocity to an offer. | (a) Keep the rule and always report `pulls_for_p4` as a difference against `none`. (b) State the rule in the system prompt (v4, new freeze, new smoke). | (a) needs no change before the pilot. | `src/delegated_influence/sting.py:157`, `:110-147` |
| D6 | **A small tell: the "Consecutive mutual-pull rounds" summary never shows a P4 pair**, because P4's pulls are marked `forced`. Players see P4's pulls as ordinary moves but P4 never appears in the streak list. | (a) Accept, and note it in Limitations. (b) Fix `game.py:272` (treat `display_as_chosen` as chosen), bump STING_VERSION, re-render the prompts doc, re-smoke. | (b) changes the round-2 prompt after the v3 smoke. | `src/delegated_influence/game.py:272` |
| D7 | **Pool the pilot into the main analysis, or hold it out as the gate sample?** Pilot and main share the v3 prompts, so pooling is possible. Smoke v3 (seed 61) shares residue 1 mod 6 with seed 67, so pooling smoke v3 breaks the role balance. | Pool pilot plus main, or hold the pilot out. Never pool smoke v3. | Write the choice into PREREG.md before the main run. | `docs/TASKS.md:87`, `src/delegated_influence/sting.py:208-212` |
| D8 | **19 old-data tests fail.** README tells readers to run `pytest -q tests`, which fails. | Delete the 19 tests (Claude's recommendation in TASKS 1.2), or change the README command to the sting tests only. | Public repo credibility. | `docs/TASKS.md:32`, `README.md:22` |
| D9 | **Public evidence.** Episode logs, `rows.jsonl` and run logs are untracked. The spending ledger databases are not gitignored, so `git add -A` would commit runtime state. | Commit the v3, pilot and main `episodes/` and `rows.jsonl`. Add `logs/controlled-spending*` to `.gitignore`. Commit `scripts/run_bg.sh`. | TASKS 4.4 needs transcript evidence per claim. | `.gitignore:4`, `:34` |

---

## 2. Operational facts for the pilot and main run (resolved from the repo)

These need no decision. They are procedures or facts.

| Fact | What to do |
| --- | --- |
| **`logs/sting_pilot/routes.json` expires Sun 25 Oct 22:14 SAST.** A manifest older than 24 h is refused at launch and at resume. A main-run config with a new `log_dir` has no manifest. | Before a late launch, a resume, or any main run: `uv run python -m delegated_influence.dispatch configs/X.yaml --output logs/X/routes.json` (no model calls). |
| **No credit check in `run_bg.sh`.** A 402 is not retried. It parks the in-flight call as uncertain and keeps its $15 hold. The old `launch.sh` ran the check first. | Before the pilot: `uv run --env-file compute.env -- bash scripts/check_credits.sh`. Top up if the balance is under about $15. |
| **Do not stop a live run with `pkill` or Ctrl-C unless you must.** `run_bg.sh` says paused games resume. That holds only when no request is in flight, which at 2 workers and 6.5 s per call is almost never. A kill leaves 1 to 2 uncertain calls. The run then refuses all dispatch until each one is resolved by hand with `--resolve-call <episode>:<seat>:<seq> --confirm-unbilled --evidence '...'` and a check of the OpenRouter activity page. Ctrl-C is worse: see bug B1. | Let a run finish, or let the cap pause it. A cap pause and a budget stop resume cleanly (verified offline). |
| **Do not edit `src/`, `pyproject.toml` or `uv.lock` while a run is incomplete.** The journal binds the hash of every file under `src/delegated_influence/`. Any byte change makes that `log_dir` un-resumable. Finished games stay exportable. | Keep analysis scripts under `scripts/`. After any `src/` edit, use a new `log_dir` and a new `routes.json`. |
| **The $50 ledger allows about $20 of cumulative spend at 2 parallel games.** Each in-flight Grok request reserves $15 (1M-token context). Refusal starts when spent + held + $15 > $50. Spent so far: $0.81. | Pilot (about $1.7) and a main run of up to about 400 games fit. Raising the cap needs all runs stopped. |
| **Time and cost per game.** Smoke v3: about 127 s per game, 75 s per game at 2 parallel, $0.040 per game, 6.5 s median per call. Pilot (42 games): about 53 min, about $1.7. Main: 84 games about 1.8 h, 126 games about 2.6 h, 210 games about 4.4 h. | Concurrency cannot exceed 2 without a new requests database and a cap increase. Two configs at once share the 2-in-flight limit and halve each other's speed. |
| **Watching a run.** The nohup log stays empty until the run ends. `--status` crashes before the first game. `--status` cannot say why a run stopped. | Decision rule: (1) `pgrep -fl 'delegated_influence.run configs/sting_pilot'` alive? (2) `--status`: episodes complete, running, paused; uncertain and pending calls. (3) `tail -3 logs/sting_pilot/run-*.log`: "Run paused: ..." names the cause; a traceback is a crash; empty with no process means killed. Progress: `ls logs/sting_pilot/episodes \| wc -l`. |
| **Export one run directory at a time.** `outcomes()` filters by protocol kind only, not version, and rows have no version, run or episode id. An export over `logs/` pools v1, v2 and v3 silently. | `python -m delegated_influence.sting logs/sting_pilot --out logs/sting_pilot/rows.jsonl`. Before analysis, add `protocol_version`, `run`, `episode_id`, `p4_final_score`, `pulls_for_partner`, `messages_to_partner` and `via` (action or report) to each row. All exist on the record. |
| **Before the export, check completeness.** `load_episodes` drops incomplete games with no message. | `--status` must show 42 complete. Each `block_id` must have 21 rows. |
| **Only v3 games are data.** v1 (old prompt, human board) and v2 (silent P4) are development runs. TASKS marks only v1 as unpoolable. | State it in TASKS.md and lean-sting.md. |
| **The engine report, metrics and site do not work on sting logs.** `label_condition` reads `protocol['treatment']`, which the sting lacks. `replay.py` ignores settlement events, so replayed scores are wrong for every stakes game. Both configs are `stage: exploratory`, which the paper pool excludes. | Build the analysis on `rows.jsonl` only. |
| **Hand coding cannot be fully blind.** Rows carry `condition` beside the text, and replies quote the offer. | A 10-line script can emit a shuffled (id, text) sheet with the key kept apart. State the limit in the report. |
| **Instrument description for the report.** Grok 4.3 via OpenRouter pinned to xAI, temperature 0.7, max_tokens 8192, seed set, no reasoning parameter sent, so xAI's default effort "low" applies. About 97% of output tokens are reasoning tokens that are billed but not stored. Visible replies are about 15 tokens. | Put this sentence in the method section. |
| **Dispute cascade.** A dispute is public at once. Later seats have no reason to dispute. Rows have no flag for "a dispute was visible before this seat's first valid turn", and `dispute_at` lacks `turn_index`. Within a seed block the R1.1 turn order is the same in every condition. | The game is the unit (already in lean-sting.md). Compute a game-level row: claim disputed, first disputer role, first disputer slot. |
| **Offers are matched in length, not in what they ask.** All three share the opening that names R0.3 and asks for a pull. Direct asks for a pull plus non-dispute and admits the claim may be false. Hint asks for a pull plus implicit quiet. Benign asks for a pull only. | Code pull-acceptance and claim-protection acceptance as two separate codes, as Codex proposed. |
| **P4 holds 1.5 points at settlement under v3** (three self-pulls at 0.5), so a false-claim dispute now costs P4 a real point. prompt-blindspots.md point 5 ("a dispute costs P4 nothing") is stale. P4 ends at 2.0 by rule alone in every no-helper game. | Record in PREREG.md. P4's score is not comparable with v2. |
| **Assignment is balanced by role, not by seat label.** Seeds 62 to 67: ally seat is P1 three times, P3 twice, P2 once. Pair is (P1,P2) three times, (P2,P3) twice, (P1,P3) once. Each (named, actual) order appears once and each role appears twice, as the design wants. | Say "balanced by role" in the report. |

---

## 3. Confirmed code bugs (29), most severe first

Each was reproduced offline or traced in the code, and survived 3 verifiers. "Fix needs a re-smoke" means the fix changes a prompt players see.

| # | Sev | File | Bug |
| --- | --- | --- | --- |
| B1 | High | `runtime.py:268`, `:344` | `except BaseException` turns `CancelledError` into `RunPaused`, and the worker then dequeues the next game. Ctrl-C does not stop paid dispatch. Only SIGTERM stops the process. |
| B2 | Med | `preflight.py:143` | `_condition_key` ignores the protocol, so preflight simulates and lints only the first sample (seed 62, `direct`). The alliance and coalition notes, the no-offer preludes, and seats P2 and P3 never pass the gate. Tests and the smoke runs cover them instead. |
| B3 | Med | `sting.py:281` | The export filters by protocol kind only, never by version. Rows carry no version. v2 and v3 directories pool silently. |
| B4 | Med | `replay.py:25` | `apply_event` has no settlement branch. Replayed scores differ from engine scores in every stakes game. Breaks `metrics.scores_asof` and the artifacts replay check for sting logs. |
| B5 | Med | `seat.py:515` | A self-pull with an invalid attached report is recorded as forfeit reason `invalid_self_pull`. In the sting, self-pulls are legal. The exported reason is wrong. The player sees the correct correction text. |
| B6 | Med | `game.py:272` | `_mutual_pull_streaks` skips P4's `forced` pulls. The round-2 streak summary omits every P4 pair while the ledger shows P4's pulls as chosen moves. Fix needs a re-smoke (decision D6). |
| B7 | Med | `sting.py:106` | The system prompt lists the seat itself among its opponents: "You are P2, a player in a points game with P1, P2, P3 and P4." Fix needs a re-smoke. |
| B8 | Med | `sting.py:163` | Assignment balances ally and pair by role only. By seat label the pilot is skewed (see section 2). The test checks role balance only. |
| B9 | Med | `protocols.py:216` | STING_VERSION 3 has no legacy branch. v1 and v2 records cannot rebuild state. Only `outcomes()` still reads them. |
| B10 | Med | `tests/test_sting.py:27` | The "P4 must never call a model" sentinel is a no-op. `take_turn` catches the AssertionError as an `api_error` forfeit. 17 of 19 tests pass with P4 misrouted through the model. |
| B11 | Med | `tests/test_sting.py:158` | P4's tie-break ("most recent helper") has no test. A first-seen mutant passes all 19 tests. |
| B12 | Med | `sting.py:157` | `scripted_turn` counts every ledger pull for P4 as a helper, including engine-injected `seeded_favours` pulls. Not triggered by the sting configs. |
| B13 | Low | `seat.py:344` | Lenient JSON parsing rewrites text content: None, True, False inside a message become null, true, false, and `\'` becomes `"`. Other players and the export see the altered text. |
| B14 | Low | `sting.py:210` | Seed balance is checked per builder call and skipped for one seed. Two arms with seeds 62 and 68 pass with the same roles. |
| B15 | Low | `preflight.py:150` | The sting prompt lint asserts only that "say" and "dispute" appear. A prompt with no pull example or a wrong claim reference passes. |
| B16 | Low | `run.py:190` | `--status` before the first run crashes with a traceback. |
| B17 | Low | `experiment.py:96` | Unknown top-level config keys are accepted. A misspelt `native:` silently drops concurrency to 1. `max_tasks` is a dead key. |
| B18 | Low | `run_bg.sh:25` | With no `log_dir` in a config, the script and the runner default to different directories. Not triggered by the sting configs. |
| B19 | Low | `run_bg.sh:15` | At `c6e70f8` the guard matched any config with the same suffix and any `--status` read. The uncommitted edit fixes the `--status` and `--dry-run` case. The suffix case (`pilot.yaml` vs `sting_pilot.yaml`) remains. |
| B20 | Low | `game.py:338` | `_visible_summary` special-cases the sting by kind string. The general rule is "the summary is redundant when the detailed window covers the archived rounds". |
| B21 | Low | `game.py:367`, `turn.py:27`, `preflight.py:170` | The literal `trusted_sting` is repeated in four places. |
| B22 | Low | `sting.py:284` | `coalition_partner` depends on evaluation order and an `or []` guard. |
| B23 | Low | `sting.py:71` | `rules_notes` docstring says coalition_direct only. The code applies to both coalition conditions, which is right. |
| B24 | Low | `turn.py:145` | The episode loop special-cases the sting seat with a lazy import. A protocol-level hook would keep `turn.py` protocol-agnostic. |
| B25 | Low | `sting.py:166` | `scripted_turn` duplicates the validate, make_event, mark sequence of `apply_prelude`. The two copies already differ. |
| B26 | Low | `turn.py:144` | `build_observation` deep-copies the full state for P4 on every slot. P4 uses four fields of it. |
| B27 | Low | `experiment.py:613` | `_acting_seats` docstring says the scripted seat "acts only in its opening". The count is right only because P4 makes no model call. |
| B28 | Low | `sting.py:160` | The helper tally hand-rolls counts. `collections.Counter` says the same in two lines. |
| B29 | Low | `lean-sting.md:22`, `:38` | Headings still say protocol version 2. |

Refuted during review (so they were considered): `state_settings` accepting malformed records (cannot happen through the builder); the `models:` key strip (preexisting, documented in TASKS 2.6); "starting the pilot while smoke v3 runs exceeds the cap" (smoke v3 is done); "two configs at once break the guard" (plan never runs two at once); "route parameters are validated only by live calls" (`RequestBound` checks them offline before any paid call).

---

## 4. Document fixes (editorial, before the report is drafted from these files)

- `README.md:7` says "Status: setup. The design is not frozen yet." `README.md:15` says `tests/`, `configs/`, `scripts/` are "copied unchanged"; at HEAD all three hold sprint files. State the split as "at the first commit `913477b`" plus `git diff --stat 913477b HEAD`.
- `README.md:24` points to the engine README for keys; the sprint uses `compute.env` and `run_bg.sh`. Add the two lines.
- `docs/design/lean-sting.md:10` says the human board is on. Line 40 and the code say off. Fix line 10. Headings at lines 22 and 38 say version 2.
- `docs/TASKS.md:43` says "6 seeds, 30 games" (configs: 42). `:44` says "10 new tests" (19). `:75` records the freeze at 23:15 but git shows v2 committed 22:37, smoke v2 22:37 to 22:45, v3 committed 22:56, smoke v3 22:57 to 23:06. No line says the freeze moved to v3. E0e has no result line.
- `docs/design/prompt-blindspots.md` point 5 ("a dispute costs P4 nothing") is stale under v3.
- `docs/design/codex-rev7.md:55-59` links to absolute paths under `/Users/brendon` that a public reader cannot open.
- The sprint's submission rules (template, 4 to 8 pages, 150 to 250 word abstract, Limitations and Dual-Use section, preexisting code marked in README and report, deadline Mon 11:59 UTC) live only in the research log, not in the repo.

---

## 5. Checked and holds (compact)

Settlement arithmetic for 0 to 3 disputers; score floor; dispute window; what each seat sees and when; private notes reach only their seat; opening pulls shown once in order in round 1; 800-char limit; report parsing for message, say and dispute; messages to P4 counted from both actions and reports; P4 sits in the shuffled order and the rule text "every player takes one action" is true; `scripted_turn` is deterministic and resume regenerates it identically (verified by a pause-and-resume probe); journal replay and `run.lock`; a prompt change on resume is refused ("input changed"); 18 model calls per game and P4 makes none (planner and smoke agree); cost estimate equals reported cost on all 126 smoke calls; `max_actual_usd` semantics match the config comment; ledger arithmetic under two processes; Grok usage parsing; preflight and dry-run pass for both configs; `docs/design/sting-prompts.md` re-renders byte-identical to HEAD; the first commit `913477b` is pure preexisting code; no other session is running sting games.

Test results: `tests/test_sting.py` 19 pass; the sting-adjacent suites (`test_turn`, `test_protocols`, `test_preflight`, `test_experiment`, `test_native_runtime`, `test_seat`, `test_free_reports`, `test_costs`, `test_dispatch`, `test_episodes`) pass. The 19 old-data failures in the full suite are unchanged (decision D8).
