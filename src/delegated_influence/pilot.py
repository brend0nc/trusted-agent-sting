"""Plan full-length cost pilots and summarize saved provider evidence. Never dispatches calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sqlite3

import yaml

from delegated_influence import costs, episodes, experiment
from delegated_influence.journal import canonical, unpack, valid_cost
from delegated_influence.settings import GameSettings

PILOT_SEED_OFFSET = 100000

def _condition(metadata):
    """Seed/seat permutations are replicates; retain model roles and all game instruments."""
    seats = metadata["seat_map"]
    roles = sorted((model, (metadata.get("personas") or {}).get(seat, "neutral"),
                    seat == metadata.get("focal"), seat in metadata.get("principal_seats", []))
                   for seat, model in seats.items())
    value = {"settings": GameSettings.from_metadata(metadata).as_dict(), "model_roles": roles,
             "sampling_instruments": metadata.get("sampling_instruments"),
             "model_configs": metadata.get("model_configs") or {},
             "observation_schema": metadata.get("observation_schema"),
             "gameplay_protocol_version": metadata.get("gameplay_protocol_version")}
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:24]


def _single_arms(arm):
    """Split constructor dimensions while retaining the original task and first seed."""
    params = inspect.signature(experiment.TASK_BUILDERS[arm["task"]]).parameters
    base = deepcopy(arm)
    if "reps" in params:
        base["reps"] = 1
    if "seeds" in params:
        base["seeds"] = [base.get("seeds", params["seeds"].default)[0]]
    dimensions = {}
    for key in ("topologies", "inoculations", "candidates"):
        if key in params:
            dimensions[key] = base.get(key, params[key].default)
    if arm["task"] in experiment.MODEL_ARGUMENTS:
        dimensions["models"] = experiment.expand_models(arm.get("models", "all"))
    for choices in itertools.product(*dimensions.values()):
        yield {**base, **{key: [value] for key, value in zip(dimensions, choices)}}


def _planning_cost(configs):
    """A shared conservative planning assumption, separate from dispatch price ceilings."""
    configured = [{**experiment.DEFAULT_COST, **(cfg.get("cost") or {})} for cfg in configs]
    priced = [bool(cost.get("model_prices")) for cost in configured]
    if any(priced) and not all(priced):
        raise ValueError("pilot sources must use the same pricing basis: all per-model or all blended")
    merged = {}
    for key in (*experiment.DEFAULT_COST, "avg_input_tokens_per_call", "avg_output_tokens_per_call"):
        values = [cost[key] for cost in configured if key in cost]
        if values:
            merged[key] = max(costs.number(value, f"cost.{key}") for value in values)
    if all(priced):
        merged["model_prices"] = {}
        for cost in configured:
            for model, rates in costs.prices(cost).items():
                target = merged["model_prices"].setdefault(model, {})
                for key, value in rates.items():
                    target[key] = max(target.get(key, 0), value)
    return merged


def prepare(config_paths, output_dir, *, allowance_usd=25.0, spending_manifest=None, ledger=None):
    """Write pilot.yaml and manifest.json. This function never executes a provider."""
    costs.number(allowance_usd, "allowance_usd", positive=True)
    paths = [Path(path).resolve() for path in config_paths]
    if not paths:
        raise ValueError("at least one target config is required")
    out = Path(output_dir).resolve()
    config_path, manifest_path = out / "pilot.yaml", out / "manifest.json"
    if config_path.exists() or manifest_path.exists():
        raise FileExistsError("pilot artifacts already exist; use a new output directory")
    configs = [experiment.load_config(path) for path in paths]
    if (spending_manifest is None) != (ledger is None):
        raise ValueError("provide both spending_manifest and ledger, or neither")
    guard = spending_manifest is not None
    guard_path = Path(spending_manifest).resolve() if guard else None
    guard_bytes = guard_path.read_bytes() if guard else None
    guard_document = json.loads(guard_bytes) if guard else None
    planning_cost = _planning_cost(configs)
    conditions = {}
    for path, cfg in zip(paths, configs):
        for _, task in experiment.build_tasks(cfg, config_name=path.stem):
            for sample in task.dataset:
                md = sample.metadata
                key = _condition(md)
                row = conditions.setdefault(key, {
                    "id": key, "settings": GameSettings.from_metadata(md).as_dict(),
                    "model_counts": dict(Counter(md["seat_map"].values())),
                    "sampling_instruments": md.get("sampling_instruments"),
                    "target_episodes": 0, "sources": [], "selected": False})
                row["target_episodes"] += 1
                source = {"config": str(path), "arm": md["experiment_arm"]}
                if source not in row["sources"]:
                    row["sources"].append(source)
        for arm in cfg["arms"]:
            for candidate in _single_arms(arm):
                built = experiment.build_tasks({"stage": "exploratory", "arms": [candidate]})
                samples = [sample for _, task in built for sample in task.dataset]
                if len(samples) != 1:
                    raise ValueError(f"cannot isolate one pilot sample for {arm['name']}")
                key = _condition(samples[0].metadata)
                conditions[key].setdefault("pilot_arm", candidate)
    selected, total = [], 0.0
    for row in conditions.values():
        arm = row.pop("pilot_arm")
        arm["name"] = f"pilot_{len(selected) + 1:03d}"
        cfg = {"stage": "exploratory", "pilot_seed_offset": PILOT_SEED_OFFSET,
               "arms": [arm], "cost": planning_cost}
        built = experiment.build_tasks(cfg, config_name="pilot")
        price = experiment.estimate_cost(cfg, built)["total_usd"]
        row["planning_cost_usd"] = price
        row["selected"] = total + price <= allowance_usd + 1e-12
        row["selection_reason"] = "whole episode fits planning allowance" if row["selected"] else "planning allowance exhausted"
        if row["selected"]:
            total += price
            selected.append(arm)
            row["pilot_arm"] = arm["name"]
            row["pilot_sample_id"] = str(built[0][1].dataset[0].id)
            if guard:
                from delegated_influence.dispatch import RequestBound
                md = built[0][1].dataset[0].metadata
                row["spending_instruments"] = {}
                for model in row["model_counts"]:
                    bound = RequestBound(guard_document, model, (md.get("model_configs") or {}).get(model))
                    row["spending_instruments"][model] = {
                        "route": bound.row, "maximum_usd": str(bound.maximum_usd),
                        "provider_routing": bound.routing}
    cfg = {"stage": "exploratory", "pilot_seed_offset": PILOT_SEED_OFFSET,
           "log_dir": str(out / "run"), "max_tasks": 1,
           "native": {"max_concurrent_episodes": 1, "allow_estimated_cost": False},
           "budget": {"max_estimated_usd": allowance_usd, "max_actual_usd": allowance_usd},
           "cost": planning_cost, "arms": selected}
    if guard:
        cfg["spending"] = {"ledger": str(Path(ledger).resolve()), "cap_usd": allowance_usd,
                           "manifest": str(guard_path)}
    manifest = {"schema_version": 1, "created": datetime.now(timezone.utc).isoformat(),
                "status": "planned_unrun", "pilot_config": str(config_path), "log_dir": cfg["log_dir"],
                "pilot_seed_offset": PILOT_SEED_OFFSET,
                "allowance_usd": allowance_usd, "planning_cost_usd": total,
                "guard_configured": guard, "selected_episodes": len(selected),
                "launch_note": "Dispatch must validate routes, prices, credentials and the shared cap."
                    if guard else "Missing spending manifest or shared ledger; this plan cannot launch.",
                "conditions": list(conditions.values()),
                "spending_manifest": {"path": str(guard_path),
                                      "sha256": hashlib.sha256(guard_bytes).hexdigest()} if guard else None,
                "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in paths],
                "planning_note": "Input order sets priority. Estimates include parse correction allowance. "
                    "They are not provider billing or a hard cap. Full rounds, actions and sampling settings stay fixed."}
    out.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def _compact_event(event):
    event = {key: value for key, value in event.items() if key not in ("raw_responses", "usage")}
    event["calls"] = [{key: ({k: v for k, v in value.items()
                             if k not in ("provider_request_body", "provider_request_snapshot")}
                            if key == "usage" else value)
                       for key, value in call.items()
                       if key not in ("input", "response", "provider_request_body")}
                      for call in event.get("calls", [])]
    return event


def _native_records(paths):
    """Read completed exports and journal snapshots without opening a writable Journal."""
    native_paths = [path for path, _ in episodes.expand_native_paths(paths)]
    records = {}
    for rec in episodes.load_episodes(native_paths):
        key = (rec["sample_uuid"], rec.get("started", ""))
        records[key] = {**rec, "status": "complete", "extra_calls": []}
    journals = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            journals.update(path.glob("**/run.sqlite3"))
        elif path.is_file() and path.suffix in (".sqlite3", ".sqlite", ".db"):
            journals.add(path)
    for path in sorted(journals):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            for row in db.execute("SELECT id,metadata,status,started,sample_id FROM episodes"):
                md = unpack(row["metadata"])
                events = [_compact_event(unpack(event[0])) for event in db.execute(
                    "SELECT payload FROM events WHERE episode=? ORDER BY sequence", (row["id"],))]
                attached = {call["call_id"] for event in events for call in event.get("calls", [])}
                extra = []
                for call in db.execute("SELECT id,seat,status,result,error FROM calls WHERE episode=?", (row["id"],)):
                    if call["id"] not in attached:
                        extra.append({"call_id": call["id"], "agent": call["seat"],
                                      "status": call["status"], "error": call["error"],
                                      "usage": (unpack(call["result"]) or {}).get("usage", {})})
                records[(row["id"], row["started"])] = {**md, "metadata": md, "events": events,
                    "status": row["status"], "extra_calls": extra, "sample_id": row["sample_id"],
                    "source_eval_file": str(path.resolve()), "sample_uuid": row["id"]}
    return list(records.values())


def _provider_cost(usage):
    raw = (usage.get("provider_usage") or {}).get("cost")
    request_id = usage.get("provider_request_id")
    requests = usage.get("http_requests")
    if (valid_cost(raw) and valid_cost(usage.get("total_cost"))
            and abs(raw - usage["total_cost"]) < 1e-12
            and isinstance(request_id, str) and request_id.strip()
            and isinstance(requests, int) and not isinstance(requests, bool) and requests >= 1
            and usage.get("provider_request_source") == "http_request_body"
            and usage.get("cost_source") == "openrouter_response_usage_cost"):
        return raw
    return None


def _stats(values):
    known = [value for value in values if valid_cost(value)]
    return {"known": len(known), "unknown": len(values) - len(known),
            "sum": sum(known) if known else None,
            "mean": sum(known) / len(known) if known else None,
            "max": max(known) if known else None}


MEASUREMENTS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cache_read_tokens",
                "cache_write_tokens", "latency_seconds", "input_characters", "observation_characters")


def calibrate(manifest, paths=None):
    """Summarize compact native calls. Missing evidence blocks full-target extrapolation."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported pilot manifest")
    paths = list(paths) if paths is not None else [manifest["log_dir"]]
    conditions = {row["id"]: row for row in manifest["conditions"]}
    groups, all_calls, complete, episode_rows = defaultdict(list), [], defaultdict(list), []
    ignored, action_count, forfeits, retries = 0, 0, 0, 0
    for rec in _native_records(paths):
        md = rec["metadata"]
        cid = _condition(md)
        condition = conditions.get(cid)
        if (condition is None or not condition["selected"] or md.get("experiment_stage") != "exploratory"
                or md.get("experiment_config") != Path(manifest["pilot_config"]).stem
                or md.get("experiment_arm") != condition.get("pilot_arm")
                or str(rec["sample_id"]) != condition.get("pilot_sample_id")):
            ignored += 1
            continue
        per_episode, slots = [], Counter()
        for index, event in enumerate(rec["events"]):
            if event.get("forced"):
                continue
            seat, rnd = event["agent"], event["round"]
            kind = "intent" if event.get("type") == "intent" else "action"
            if kind == "action":
                action_count += 1
                forfeits += bool(event.get("forfeit"))
                retries += event.get("parse_retries", 0)
            calls = event.get("calls", [])
            if calls:
                slots[(seat, rnd, 0 if kind == "intent" else event.get("action_number"))] += 1
            for call in calls:
                usage = call.get("usage") or {}
                measured = {**usage, "provider_cost_usd": _provider_cost(usage),
                            "call_id": call["call_id"], "condition": cid, "round": rnd,
                            "model": md["seat_map"][seat], "event_index": index,
                            "episode_id": rec["sample_uuid"], "source": rec["source_eval_file"],
                            "error": call.get("error") or usage.get("error"),
                            "status": call.get("status", "complete")}
                context = [usage.get(key) for key in ("prompt_tokens", "cache_read_tokens", "cache_write_tokens")]
                measured["context_tokens"] = sum(context) if all(valid_cost(value) for value in context) else None
                groups[(cid, measured["model"], rnd)].append(measured)
                per_episode.append(measured)
        for call in rec["extra_calls"]:
            usage = call.get("usage") or {}
            per_episode.append({**usage, "provider_cost_usd": _provider_cost(usage),
                                "call_id": call["call_id"], "condition": cid, "round": None,
                                "model": md["seat_map"][call["agent"]], "event_index": None,
                                "episode_id": rec["sample_uuid"], "source": rec["source_eval_file"],
                                "status": call["status"], "error": call.get("error")})
        expected = Counter({(seat, rnd, action): 1 for seat in md["seat_map"]
                            for rnd in range(1, md["rounds"] + 1)
                            for action in range(0 if md.get("declared_intents") else 1, md["actions_per_round"] + 1)})
        known = [call["provider_cost_usd"] for call in per_episode if call["provider_cost_usd"] is not None]
        expected_guard = condition.get("spending_instruments")
        guard_matches = bool(expected_guard) and canonical(md.get("spending_instrument")) == canonical(expected_guard)
        providers_match = bool(expected_guard) and all(
            call.get("served_provider") == expected_guard[call["model"]]["route"]["provider_name"]
            for call in per_episode)
        covered = (rec["status"] == "complete" and slots == expected and not rec["extra_calls"]
                   and len(known) == len(per_episode) and bool(per_episode)
                   and all(call.get("context_tokens") is not None for call in per_episode)
                   and guard_matches and providers_match)
        if covered:
            complete[cid].append(sum(known))
        episode_rows.append({"episode_id": rec["sample_uuid"], "condition": cid,
                             "source": rec["source_eval_file"], "status": rec["status"], "covered": covered,
                             "guard_matches": guard_matches, "served_providers_match": providers_match,
                             "slots_complete": slots == expected,
                             "provider_cost_usd": sum(known) if len(known) == len(per_episode) and known else None})
        all_calls.extend(per_episode)
    rows, covered_cells = [], 0
    for cid, condition in conditions.items():
        for model in condition["model_counts"]:
            for rnd in range(1, condition["settings"]["rounds"] + 1):
                calls = groups[(cid, model, rnd)]
                row = {"condition": cid, "model": model, "round": rnd, "calls": len(calls),
                       **{key: _stats([call.get(key) for call in calls])
                          for key in (*MEASUREMENTS, "context_tokens", "provider_cost_usd")}}
                row["covered"] = bool(complete[cid])
                covered_cells += row["covered"]
                rows.append(row)
    known = [call["provider_cost_usd"] for call in all_calls if call["provider_cost_usd"] is not None]
    full = (bool(conditions) and all(complete[cid] for cid in conditions)
            and all(row["covered"] for row in episode_rows))
    # Every target condition must have a complete billed game. No pooling across arms or models.
    point = sum(sum(complete[cid]) / len(complete[cid]) * condition["target_episodes"]
                for cid, condition in conditions.items()) if full else None
    return {"schema_version": 1, "status": "covered" if full else "incomplete" if all_calls else "unrun",
            "evidence_note": "Provider billing evidence requires raw usage.cost, request ID and captured HTTP dispatch. "
                "Saved records are not an independent provider invoice. Offline fixtures are not live validation.",
            "measured": {"calls": len(all_calls), "known_provider_cost_usd": sum(known) if known else None,
                         "provider_cost_usd": sum(known) if known and len(known) == len(all_calls) else None,
                         "unknown_cost_calls": len(all_calls) - len(known), "actions": action_count,
                         "forfeits": forfeits, "forfeit_rate": forfeits / action_count if action_count else None,
                         "parse_retries": retries, "unassigned_calls": sum(call["round"] is None for call in all_calls),
                         "uncertain_calls": sum(call.get("status") in ("pending", "uncertain") for call in all_calls),
                         "http_retries": sum(max(call.get("http_requests", 1) - 1, 0) for call in all_calls),
                         **{key: _stats([call.get(key) for call in all_calls]) for key in MEASUREMENTS}},
            "coverage": {"conditions": len(conditions), "complete_conditions": sum(bool(complete[cid]) for cid in conditions),
                         "model_round_cells": len(rows), "covered_model_round_cells": covered_cells,
                         "ignored_episodes": ignored},
            "target_point_estimate_usd": point,
            "estimate_note": "Tiny sample point estimate only, conditional on the same guarded routes and settings. "
                "No confidence interval; prices, retries and behavior can change."
                if full else "No target estimate: every condition and accepted game needs complete slots, provider cost, context and matching guarded routes.",
            "episodes": episode_rows, "model_rounds": rows,
            "calls": [{key: value for key, value in call.items()
                       if key not in ("provider_request_body", "provider_request_snapshot")} for call in all_calls]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare", help="write a planned, unrun pilot; no provider calls")
    plan.add_argument("configs", nargs="+")
    plan.add_argument("--out", required=True)
    plan.add_argument("--allowance-usd", type=float, default=25)
    plan.add_argument("--spending-manifest")
    plan.add_argument("--ledger")
    summary = commands.add_parser("summarize", help="read saved native exports and journals")
    summary.add_argument("manifest")
    summary.add_argument("paths", nargs="*")
    summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.configs, args.out, allowance_usd=args.allowance_usd,
                         spending_manifest=args.spending_manifest, ledger=args.ledger)
        print(json.dumps({key: result[key] for key in ("status", "pilot_config", "guard_configured", "selected_episodes", "planning_cost_usd")}, indent=2))
    else:
        result = calibrate(args.manifest, args.paths or None)
        Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("status", "coverage", "target_point_estimate_usd")}, indent=2))


if __name__ == "__main__":
    main()
