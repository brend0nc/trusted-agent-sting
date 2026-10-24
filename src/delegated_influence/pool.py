"""Which episodes count: arm ownership, instrument selection, and the paper pool.

One episode joins a config arm only through its embedded experiment stamp or, for marker-less
legacy logs, the run-directory prefix rule. A complete confirmatory arm feeds the ``paper`` pool;
everything else is ``in_progress``, ``held_out``, ``historical``, or ``exploratory``. The site
(site/derive.py) and the report CLI (report.py) both call this module, so they agree on which
episodes count.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from delegated_influence import experiment


def load_arms(configs):
    """[(arm config dict, {planned signature set})] from every configs/*.yaml that has arms,
    built by the real task constructors (experiment.build_tasks), so planned ids/metadata are
    exact — never a formula. The planner expands focal arms into one task shard per model, named
    '<arm>-<model>'; shards are aggregated back onto their config arm here so coverage stays
    one row per arm. A config that fails to build (mid-edit, unknown task) is skipped loudly."""
    arms = []
    for path in sorted(Path(configs).glob("*.yaml")):
        try:
            cfg = experiment.load_config(path)
            if not isinstance(cfg, dict) or not cfg.get("arms"):
                continue
            stage = cfg.get("stage")  # smoke/instrument configs set stage: exploratory
            planned = {arm["name"]: set() for arm in cfg["arms"]}
            legacy_planned = {arm["name"]: set() for arm in cfg["arms"]}
            for name, t in experiment.build_tasks(cfg, config_name=path.stem):
                owner = (t.dataset[0].metadata.get("experiment_arm")
                         or experiment.task_arm_name(name, cfg["arms"]))
                planned[owner] |= {experiment.episode_identity(s.id, s.metadata) for s in t.dataset}
                legacy_planned[owner] |= {
                    experiment.legacy_episode_identity(s.id, s.metadata) for s in t.dataset
                }
        except Exception as ex:
            print(f"skip coverage for {path.name}: {ex}", file=sys.stderr)
            continue
        for arm_cfg in cfg["arms"]:
            arms.append((dict(arm_cfg, config=path.stem, stage=experiment.resolve_arm_stage(stage, arm_cfg),
                              _legacy_planned=legacy_planned[arm_cfg["name"]]),
                         planned[arm_cfg["name"]]))
    if not arms:
        print(f"skip coverage: no buildable arms under {configs}", file=sys.stderr)
    return arms
def arm_conditions(arm):
    """A human-readable one-liner for an arm's condition cell."""
    spr = arm.get("self_pull_reward") or 0
    parts = []
    if arm.get("topologies"):
        parts.append("+".join(arm["topologies"]) + " topology")
    parts += [f"mixed economy (self-pull {spr})" if spr else "pure economy",
              "messages off" if arm.get("messages_off") else "messages on"]
    if arm["task"] == "focal_resist":
        inocs = arm.get("inoculations", [False, True])
        parts.append("targets " + "+".join("inoculated" if i else "naive" for i in inocs))
    if arm["task"] == "arena":
        parts.append("seeds " + ",".join(str(s) for s in arm.get("seeds", [41, 42, 43])))
    return "; ".join(parts)
def match_arms(episodes, arms):
    """Tag each episode with the config arm it realizes (exact sample-id + metadata match),
    and return coverage rows. done = distinct planned episodes finished (re-runs counted once).

    New logs use their embedded experiment_config/experiment_arm provenance; moving a file cannot
    change its owner. Marker-less legacy logs fall back to the run-directory prefix rule
    (logs/01_smoke_v3 -> config 01_smoke). An episode with conflicting or unknown provenance is
    left unclaimed with a warning, so it cannot leak into a paper arm."""
    coverage = []
    warned = set()
    for arm, planned in arms:
        current_done = set()
        # cohort -> planned condition -> exact recovered sampling-instrument ids. Prompt and
        # observation define the cohort; sampling may differ across models by design, but two
        # variants for the same planned condition make that cell ambiguous and unusable.
        legacy_cells = defaultdict(lambda: defaultdict(set))
        legacy_exact_done = defaultdict(set)
        legacy_model_variants = defaultdict(lambda: defaultdict(set))
        owned_episodes = []
        legacy_planned = arm.get("_legacy_planned") or set()
        for ep in episodes:
            if ep["metadata"].get("sampling_instrument_mismatch"):
                key = (ep["run"], ep["sample_id"], "sampling_instrument_mismatch")
                if key not in warned:
                    warned.add(key)
                    print(
                        f"unclaimed: {ep['run']!r}/{ep['sample_id']!r} used sampling settings "
                        "that conflict with its experiment stamp; left exploratory",
                        file=sys.stderr,
                    )
                ep["instrument_validation_error"] = "sampling_metadata_event_mismatch"
                continue
            try:
                sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
            except ValueError as exc:
                key = (ep["run"], ep["sample_id"], str(exc))
                if key not in warned:
                    warned.add(key)
                    print(f"unclaimed: {exc}; left exploratory", file=sys.stderr)
                ep["instrument_validation_error"] = "invalid_experiment_stamp"
                continue
            if sig in planned:
                instrument_match = "current"
                candidates = [a for a, p in arms if sig in p]
                matched_identity = sig
            else:
                legacy_sig = experiment.legacy_episode_identity(ep["sample_id"], ep["metadata"])
                if legacy_sig not in legacy_planned:
                    continue
                instrument_match = experiment.legacy_cohort_identity(ep["metadata"])
                sampling_instrument_match = experiment.legacy_instrument_identity(ep["metadata"])
                candidates = [a for a, _p in arms
                              if legacy_sig in (a.get("_legacy_planned") or set())]
                matched_identity = legacy_sig
            if candidates:
                stamped_config = ep["metadata"].get("experiment_config")
                stamped_arm = ep["metadata"].get("experiment_arm")
                if stamped_config is not None or stamped_arm is not None:
                    owners = [a for a in candidates
                              if (a.get("config"), a["name"])
                              == (stamped_config, stamped_arm)]
                    provenance = f"embedded provenance {stamped_config!r}/{stamped_arm!r}"
                else:
                    owners = [a for a in candidates if ep["run"].startswith(a["config"])]
                    provenance = f"legacy run dir {ep['run']!r}"
                if not owners:  # orphan run dir: leave unclaimed, never count as coverage
                    key = (ep["run"], stamped_config, stamped_arm,
                           tuple(sorted((a["config"], a["name"]) for a in candidates)))
                    if key not in warned:
                        warned.add(key)
                        print(f"unclaimed: {provenance} conflicts with matching planned arms "
                              f"{sorted((a['config'], a['name']) for a in candidates)}; "
                              "left exploratory", file=sys.stderr)
                    continue
                owner = (owners[0] if stamped_config is not None or stamped_arm is not None
                         else max(owners, key=lambda a: len(a["config"])))
                if ((owner.get("config"), owner["name"])
                        != (arm.get("config"), arm["name"])):
                    continue  # this arm matched, but the episode's run dir belongs elsewhere
                stamped_stage = ep["metadata"].get("experiment_stage")
                if stamped_stage is not None and stamped_stage != owner.get("stage", "confirmatory"):
                    key = (ep["run"], stamped_config, stamped_arm, stamped_stage,
                           owner.get("stage", "confirmatory"))
                    if key not in warned:
                        warned.add(key)
                        print(
                            f"unclaimed: embedded stage {stamped_stage!r} conflicts with "
                            f"configured stage {owner.get('stage', 'confirmatory')!r} for "
                            f"{owner.get('config')!r}/{owner['name']!r}; left exploratory",
                            file=sys.stderr,
                        )
                    continue
                ep["arm"] = arm["name"]
                ep["arm_config"] = arm.get("config")
                ep["arm_stage"] = arm.get("stage", "confirmatory")
                ep["instrument_match"] = instrument_match
                ep["matched_condition_identity"] = matched_identity
                owned_episodes.append(ep)
                if instrument_match == "current":
                    current_done.add(matched_identity)
                else:
                    ep["sampling_instrument_match"] = sampling_instrument_match
                    legacy_cells[instrument_match][matched_identity].add(
                        sampling_instrument_match
                    )
                    legacy_exact_done[sampling_instrument_match].add(matched_identity)
                    for model, variant in experiment.legacy_sampling_profile(ep["metadata"]).items():
                        legacy_model_variants[instrument_match][model].add(variant)
        conflicted_models = {
            cohort: {model for model, variants in by_model.items() if len(variants) > 1}
            for cohort, by_model in legacy_model_variants.items()
        }
        legacy_done = {
            cohort: {
                condition for condition, variants in cells.items()
                if (len(variants) == 1 and not any(
                    model in conflicted_models.get(cohort, set())
                    for ep in owned_episodes
                    if (ep.get("instrument_match") == cohort
                        and ep.get("matched_condition_identity") == condition)
                    for model in experiment.legacy_sampling_profile(ep["metadata"])
                ))
            }
            for cohort, cells in legacy_cells.items()
        }
        ambiguous_legacy_cells = {
            cohort: sorted(condition for condition, variants in cells.items() if len(variants) > 1)
            for cohort, cells in legacy_cells.items()
            if any(len(variants) > 1 for variants in cells.values())
        }
        for ep in owned_episodes:
            if ep.get("instrument_match") != "current":
                variants = legacy_cells[ep["instrument_match"]][ep["matched_condition_identity"]]
                ep_models = set(experiment.legacy_sampling_profile(ep["metadata"]))
                ep["instrument_cell_ambiguous"] = (
                    len(variants) > 1
                    or bool(ep_models & conflicted_models.get(ep["instrument_match"], set()))
                )
        complete_legacy = sorted(
            instrument_id for instrument_id, done_ids in legacy_done.items()
            if len(done_ids) >= len(planned)
        )
        ambiguous = False
        if len(current_done) >= len(planned):
            selected = "current"
            done = current_done
        elif len(complete_legacy) == 1:
            selected = complete_legacy[0]
            done = legacy_done[selected]
        elif len(complete_legacy) > 1:
            # Two complete historical instruments are both valid data, but neither may be chosen
            # silently. Keep their coverage visible and exclude both from the paper pool.
            selected = None
            done = max((legacy_done[key] for key in complete_legacy), key=len)
            ambiguous = True
        elif current_done:
            selected = "current"
            done = current_done
        elif len(legacy_done) == 1:
            selected = next(iter(legacy_done))
            done = legacy_done[selected]
        elif legacy_done:
            selected = None
            done = max(legacy_done.values(), key=len)
            ambiguous = True
        else:
            selected = "current"
            done = set()
        # Check the whole selected cohort before choosing a duplicate condition. A newer
        # runtime must not disappear merely because an earlier copy wins the date tie-break.
        runtime_variants = {
            experiment.runtime_compatibility_key(ep["metadata"].get("runtime_instrument")) or 'null'
            for ep in owned_episodes
            if ep.get("instrument_match") == selected and not ep.get("instrument_cell_ambiguous")
        }
        runtime_ambiguous = len(runtime_variants) > 1
        ambiguous = ambiguous or runtime_ambiguous
        selected_eps = select_analysis_episodes(owned_episodes, selected)
        served_summary = summarize_served_instruments(selected_eps)
        # A provider may route the same requested model through several hosts; that stays an audit
        # covariate. A changed served model/version is a changed instrument and blocks paper use.
        realized_model_ambiguous = served_summary["served_model_drift"]
        ambiguous = ambiguous or realized_model_ambiguous
        coverage.append({"arm": arm["name"], "task": arm["task"], "config": arm.get("config"),
                         "stage": arm.get("stage", "confirmatory"),
                         "planned_episodes": len(planned),
                         "done_episodes": len(done),
                         "current_done_episodes": len(current_done),
                         "legacy_done_episodes": max(
                             (len(value) for value in legacy_done.values()), default=0
                         ),
                         "legacy_instruments": {
                             key: len(value) for key, value in sorted(legacy_exact_done.items())
                         },
                         "legacy_cohorts": {
                             key: len(value) for key, value in sorted(legacy_done.items())
                         },
                         "ambiguous_legacy_cells": ambiguous_legacy_cells,
                         "conflicted_sampling_models": {
                             key: sorted(value) for key, value in sorted(conflicted_models.items())
                             if value
                         },
                         "selected_instrument": selected,
                         "instrument_ambiguous": ambiguous,
                         "runtime_ambiguous": runtime_ambiguous,
                         "runtime_instruments": [json.loads(value) for value in sorted(runtime_variants)],
                         "served_instruments": served_summary["models"],
                         "provider_drift": served_summary["provider_drift"],
                         "served_model_drift": served_summary["served_model_drift"],
                         "analysis_episodes": len(selected_eps),
                         "duplicate_episodes": sum(
                             "duplicate_of" in ep for ep in owned_episodes
                         ),
                         "conditions": arm_conditions(arm)})
    for ep in episodes:
        ep.setdefault("arm", None)
        ep.setdefault("arm_config", None)
    tag_pools(episodes, coverage)
    return coverage
def select_analysis_episodes(episodes, selected_instrument):
    """Choose one deterministic episode per planned condition; retain reruns only for audit."""
    eligible = [
        ep for ep in episodes
        if (selected_instrument is not None
            and ep.get("instrument_match") == selected_instrument
            and not ep.get("instrument_cell_ambiguous"))
    ]
    by_condition = defaultdict(list)
    for ep in eligible:
        by_condition[ep.get("matched_condition_identity")].append(ep)
    selected = []
    for copies in by_condition.values():
        copies.sort(key=lambda ep: (
            ep.get("started") or "", ep.get("eval_file") or "", ep.get("id") or "",
        ))
        winner = copies[0]
        winner["analysis_selected"] = True
        selected.append(winner)
        for duplicate in copies[1:]:
            duplicate["analysis_selected"] = False
            duplicate["duplicate_of"] = winner.get("id")
    for ep in episodes:
        if ep not in eligible:
            ep["analysis_selected"] = False
    return selected
def summarize_served_instruments(episodes):
    """Aggregate realized provider/model counts for one selected arm instrument."""
    models_out = {}
    for ep in episodes:
        for model, values in ((ep.get("metadata") or {}).get("served_instruments") or {}).items():
            row = models_out.setdefault(
                model,
                {"calls": 0, "failed_calls": 0, "pending_calls": 0,
                 "unknown_provider_calls": 0,
                 "requested_models": {}, "served_models": {}, "failed_served_models": {},
                 "pending_served_models": {}, "providers": {}},
            )
            for scalar in ("calls", "failed_calls", "pending_calls",
                           "unknown_provider_calls"):
                row[scalar] += int(values.get(scalar) or 0)
            for field in ("requested_models", "served_models", "failed_served_models",
                          "pending_served_models", "providers"):
                for key, count in (values.get(field) or {}).items():
                    row[field][key] = row[field].get(key, 0) + int(count or 0)
    return {
        "models": models_out,
        "provider_drift": any(len(row["providers"]) > 1 for row in models_out.values()),
        "served_model_drift": any(len(row["served_models"]) > 1 for row in models_out.values()),
    }
def tag_pools(episodes, coverage):
    """Sort each episode into a `pool`, so figures never mix test data with paper results:
      - "paper": from a confirmatory config arm that is COMPLETE (done == planned).
      - "in_progress": confirmatory arm, not yet complete (shown in coverage, kept out of figures).
      - "held_out": causal-control arms (stage held-out, exp-13) — complete or not, they stay out
        of the paper pool and every headline figure until the pooling decision lands (D21).
      - "historical": a non-selected legacy instrument retained for audit, never mixed into paper.
      - "exploratory": smoke/instrument arms (stage exploratory) or calibration/smoke runs (no arm).
    Only the paper pool feeds the leaderboard, question figures, and metrics preview."""
    complete = {(c["config"], c["arm"]): c.get("selected_instrument", "current")
                for c in coverage
                if c["stage"] == "confirmatory"
                and not c.get("instrument_ambiguous")
                and ("selected_instrument" not in c or c["selected_instrument"] is not None)
                and c["planned_episodes"] and c["done_episodes"] >= c["planned_episodes"]}
    selected = {(c["config"], c["arm"]): c.get("selected_instrument", "current")
                for c in coverage}
    confirmatory = {(c["config"], c["arm"]) for c in coverage if c["stage"] == "confirmatory"}
    held_out = {(c["config"], c["arm"]) for c in coverage if c["stage"] == "held-out"}
    configs_by_arm = defaultdict(set)
    for row in coverage:
        configs_by_arm[row["arm"]].add(row["config"])
    for ep in episodes:
        arm = ep.get("arm")
        config = ep.get("arm_config")
        if config is None and arm and len(configs_by_arm[arm]) == 1:
            config = next(iter(configs_by_arm[arm]))  # legacy in-memory/site records
        key = (config, arm) if arm else None
        instrument_match = ep.get("instrument_match", "current")
        if ep.get("analysis_selected") is False and arm:
            ep["pool"] = "historical"
        elif ep.get("instrument_cell_ambiguous"):
            ep["pool"] = "historical"
        elif key in complete and instrument_match == complete[key]:
            ep["pool"] = "paper"
        elif key in confirmatory:
            ep["pool"] = ("in_progress" if selected.get(key) == instrument_match
                          else "historical")
        elif key in held_out:
            ep["pool"] = ("held_out" if selected.get(key) == instrument_match
                          else "historical")
        else:
            ep["pool"] = "exploratory"
def pool_summary(episodes):
    """Per-pool episode counts + the run dirs in each, for the site's paper-vs-test split."""
    out = {}
    for pool in ("paper", "in_progress", "held_out", "historical", "exploratory"):
        eps = [ep for ep in episodes if ep["pool"] == pool]
        runs = sorted({ep["run"] for ep in eps})
        out[pool] = {"n_episodes": len(eps), "runs": runs}
    return out
