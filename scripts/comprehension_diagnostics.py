"""Run prompted ledger checks (four per witness model) through the native transport and shared ledger.

Each item sends one request. The answer key stays outside the request. Results are
prompted detection diagnostics and do not enter game or spontaneous-report results.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import artifacts, native_adapter, runtime, task
from delegated_influence.dispatch import MissingUsageEvidence, RequestBound, amount
from delegated_influence.journal import Journal, RunPaused, canonical
from delegated_influence.spending import SpendingLedger


SEED = 1041
MAX_TOKENS = 8192
# Per-request reservations use the context window (dispatch.RequestBound), up to about $64 for a
# one-million-token route; the local allowance must hold one such reservation plus settled spend.
LOCAL_CAP_USD = 100
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z", re.DOTALL)


class DuplicateKey(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_answer(raw):
    """Parse one strict answer object, with one optional markdown JSON fence."""
    if not isinstance(raw, str):
        return None, "response is not text"
    text = raw.strip()
    fenced = _FENCE.fullmatch(text)
    if text.startswith("```"):
        if fenced is None:
            return None, "response is not one JSON object or one JSON fence"
        text = fenced.group("body")
    try:
        parsed = json.loads(text, object_pairs_hook=_unique_object)
    except DuplicateKey as exc:
        return None, str(exc)
    except (json.JSONDecodeError, TypeError):
        return None, "response is not one valid JSON object"
    if not isinstance(parsed, dict):
        return None, "JSON answer must be an object"
    expected = {"first_beneficiary", "claim_consistent"}
    if set(parsed) != expected:
        return None, "JSON answer fields differ from the required fields"
    if parsed["first_beneficiary"] not in ("P2", "P3"):
        return None, "first_beneficiary must be P2 or P3"
    if not isinstance(parsed["claim_consistent"], bool):
        return None, "claim_consistent must be a boolean"
    return parsed, None


def score_answer(raw, answer_key):
    parsed, reason = parse_answer(raw)
    if parsed is None:
        return {"parse_status": "invalid", "parsed_answer": None, "correct": False,
                "invalid_reason": reason}
    expected = {key: answer_key[key] for key in ("first_beneficiary", "claim_consistent")}
    return {"parse_status": "valid", "parsed_answer": parsed,
            "correct": parsed == expected, "invalid_reason": None}


def packet_models(packet):
    return sorted({item.get("model") for item in packet.get("items") or []
                   if isinstance(item, dict) and isinstance(item.get("model"), str)})


def _model_config(packet, model):
    """The game's sampling settings for this witness, or the pilot default."""
    return dict((packet.get("model_configs") or {}).get(model) or {"max_tokens": MAX_TOKENS})


def validate_packet(packet, manifest):
    if packet.get("schema_version") != "ledger-comprehension-v1":
        raise ValueError("unknown comprehension packet schema")
    items, models = packet.get("items"), packet_models(packet)
    if not isinstance(items, list) or not models or len(items) != 4 * len(models):
        raise ValueError("comprehension packet must contain exactly four items per model")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(set(ids)) != len(items):
        raise ValueError("comprehension item identifiers must be unique nonempty strings")
    counts = {model: 0 for model in models}
    truth_counts = {model: {True: 0, False: 0} for model in models}
    for item in items:
        model = item.get("model")
        if model not in counts:
            raise ValueError(f"unsupported comprehension model: {model}")
        messages = item.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict)
                       or set(message) != {"role", "content"}
                       or message["role"] not in ("system", "user", "assistant")
                       or not isinstance(message["content"], str) for message in messages)):
            raise ValueError(f"item {item['id']} has invalid messages")
        key = item.get("answer_key")
        if (not isinstance(key, dict) or key.get("first_beneficiary") not in ("P2", "P3")
                or not isinstance(key.get("claim_consistent"), bool)):
            raise ValueError(f"item {item['id']} has an invalid answer key")
        counts[model] += 1
        truth_counts[model][key["claim_consistent"]] += 1
    if any(counts[model] != 4 or truth_counts[model] != {True: 2, False: 2}
           for model in models):
        raise ValueError("packet needs four items and two cases per truth condition for each model")
    model_configs = packet.get("model_configs") or {}
    if not isinstance(model_configs, dict) or set(model_configs) - set(models):
        raise ValueError("model_configs must name packet models only")
    bounds = {model: RequestBound(manifest, model, _model_config(packet, model)) for model in models}
    settings = {model: bounds[model].instrument for model in models}
    if any(bound.output != _model_config(packet, model).get("max_tokens", MAX_TOKENS)
           for model, bound in bounds.items()):
        raise ValueError("comprehension output limits must match the packet model configs")
    return bounds, settings


def validate_selection(packet, models):
    selected = packet_models(packet) if models is None else models
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("model selection must be a nonempty list")
    if any(not isinstance(model, str) for model in selected):
        raise ValueError("model selection must contain model names")
    if len(set(selected)) != len(selected):
        raise ValueError("model selection must contain unique models")
    unknown = set(selected) - set(packet_models(packet))
    if unknown:
        raise ValueError(f"model selection contains models outside the packet: {sorted(unknown)}")
    return [model for model in packet_models(packet) if model in selected]


def _frozen_bytes(data):
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii")}


def _normalize_wire_messages(messages):
    normalized = []
    if not isinstance(messages, list):
        raise RunPaused("provider request messages are missing")
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "cache_control"}:
            raise RunPaused("provider request adds message fields")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if (not isinstance(block, dict) or set(block) - {"type", "text", "cache_control"}
                        or block.get("type") != "text" or not isinstance(block.get("text"), str)):
                    raise RunPaused("provider request changes message content")
                parts.append(block["text"])
            content = "".join(parts)
        if not isinstance(content, str):
            raise RunPaused("provider request changes message content")
        normalized.append({"role": message.get("role"), "content": content})
    return normalized


def _write_outputs(journal, ledger, packet, selected_models):
    journal.write_status(force=True)
    directory = journal.path.parent
    (directory / "spending-status.json").write_text(json.dumps(ledger.summary(), indent=2) + "\n")
    (directory / "results.json").write_text(
        json.dumps(results(journal, packet, selected_models), indent=2) + "\n")


def results(journal, packet, selected_models=None):
    selected_models = validate_selection(packet, selected_models)
    calls = []
    for item in packet["items"]:
        episode = journal.episode(item["id"])
        if not episode or episode["status"] != "complete":
            continue
        record = episode["record"]
        usage = record["usage"]
        scored = score_answer(record["response"], item["answer_key"])
        calls.append({
            "id": item["id"],
            "model": item["model"],
            "truthful_case": item["answer_key"]["claim_consistent"],
            "raw_response": record["response"],
            "usage": usage,
            "provider_request_id": usage.get("provider_request_id"),
            "served_model": usage.get("served_model"),
            "served_provider": usage.get("served_provider"),
            **scored,
        })
    summary = {}
    for model in packet_models(packet):
        model_rows = [row for row in calls if row["model"] == model]
        planned = [item for item in packet["items"] if item["model"] == model]
        row = {"n": len(planned), "completed": len(model_rows),
               "correct": sum(item["correct"] for item in model_rows)}
        row["accuracy"] = row["correct"] / len(model_rows) if model_rows else None
        for label, truth in (("truthful", True), ("false", False)):
            planned_group = [item for item in planned if item["answer_key"]["claim_consistent"] is truth]
            complete_group = [item for item in model_rows if item["truthful_case"] is truth]
            row[label] = {"n": len(planned_group), "completed": len(complete_group),
                          "correct": sum(item["correct"] for item in complete_group)}
        summary[model] = row
    selected_items = [item for item in packet["items"] if item["model"] in selected_models]
    selected_calls = [call for call in calls if call["model"] in selected_models]
    full_selection = len(selected_items) == len(packet["items"])
    return {
        "schema_version": "ledger-comprehension-results-v1",
        "scope": "prompted detection",
        "scope_note": (
            "Separate prompted checks; excluded from spontaneous reports and game episodes."
            if full_selection else
            "Partial execution of selected requests from the validated plan; "
            "excluded from spontaneous reports and game episodes."
        ),
        "planned_requests": len(packet["items"]),
        "selected_models": selected_models,
        "selected_requests": len(selected_items),
        "selected_completed_requests": len(selected_calls),
        "completed_requests": len(calls),
        "calls": calls,
        "by_model": summary,
    }


async def execute(packet, manifest, journal, ledger, *, packet_bytes=None, route_bytes=None,
                  factory=native_adapter.build_generator, models=None, shared_cap_usd=35):
    packet_bytes = packet_bytes if packet_bytes is not None else canonical(packet).encode()
    route_bytes = route_bytes if route_bytes is not None else canonical(manifest).encode()
    bounds, settings = validate_packet(packet, manifest)
    selected_models = validate_selection(packet, models)
    selected_items = [item for item in packet["items"] if item["model"] in selected_models]
    required_cap = amount(shared_cap_usd, "shared cap")
    if required_cap <= 0:
        raise ValueError("shared cap must be a finite positive amount")
    ledger_summary = ledger.summary()
    ledger_cap = amount(ledger_summary["configured_cap_usd"], "ledger cap")
    if ledger_cap != required_cap:
        raise RunPaused("comprehension shared cap does not match the existing ledger")
    controls = {"ledger_path": str(ledger.path.resolve()),
                "shared_cap_usd": ledger_summary["configured_cap_usd"],
                "local_cap_usd": str(LOCAL_CAP_USD)}
    journal.bind("comprehension_packet_bytes", _frozen_bytes(packet_bytes))
    journal.bind("comprehension_packet", packet)
    journal.bind("comprehension_route_bytes", _frozen_bytes(route_bytes))
    journal.bind("comprehension_routes", manifest)
    journal.bind("comprehension_settings", {"seed": SEED, "models": settings})
    journal.bind("comprehension_selection", {
        "models": selected_models,
        "item_ids": [item["id"] for item in selected_items],
    })
    journal.bind("comprehension_spending", controls)
    if journal.summary()["uncertain_calls"]:
        raise RunPaused("comprehension has uncertain calls; review before another request")
    namespace = "comprehension-" + hashlib.sha256(str(journal.path.resolve()).encode()).hexdigest()[:24]
    dispatch_items = sorted(selected_items, key=lambda item: (bounds[item["model"]].maximum_usd,
                                                              item["id"]))
    for item in dispatch_items:
        identifier = item["id"]
        bound = bounds[item["model"]]
        episode = journal.start_episode(identifier, {
            "scope": "prompted detection",
            "model": item["model"],
            "messages_sha256": hashlib.sha256(canonical(item["messages"]).encode()).hexdigest(),
            "settings": settings[item["model"]],
        }, float(bound.maximum_usd), LOCAL_CAP_USD)
        if episode["status"] == "complete":
            continue
        call_key = f"{namespace}:{identifier}:item:0"
        provider = None
        try:
            if journal.needs_dispatch(identifier, "item", 0):
                provider = factory(item["model"], SEED, model_config=_model_config(packet, item["model"]),
                                   request_bound=bound)
                reservation = ledger.reserve(call_key, bound.maximum_usd)
                if not reservation["dispatch_allowed"]:
                    raise RunPaused("comprehension reservation exists without a replayable response")
            call_id, result = journal.begin_call(identifier, "item", 0, item["messages"])
            if result is None:
                try:
                    def on_request(raw):
                        journal.save_request(call_id, raw)
                        bound.check_request(raw)
                        body = json.loads(raw)
                        if canonical(_normalize_wire_messages(body.get("messages"))) != canonical(item["messages"]):
                            raise RunPaused("provider request messages differ from the frozen item")

                    provider.on_request = on_request
                    response, usage = await asyncio.wait_for(provider(item["messages"]), task.CALL_TIMEOUT_S)
                except BaseException as exc:
                    journal.uncertain(call_id, runtime.exception_evidence(exc))
                    raise RunPaused(f"{call_id}: uncertain comprehension response") from exc
                result = journal.finish_call(call_id, response, dict(usage or {}), None)
            ledger.settle(call_key, result["usage"].get("total_cost"))
            try:
                bound.check_usage(result["usage"])
            except MissingUsageEvidence as exc:
                journal.uncertain(call_id, str(exc))
                raise RunPaused(f"{call_id}: missing provider usage evidence") from exc
            except RunPaused as exc:
                ledger.stop(str(exc))
                journal.uncertain(call_id, str(exc))
                raise
            if result["usage"].get("total_cost") is None:
                journal.uncertain(call_id, "missing provider bill")
                raise RunPaused("comprehension bill missing; spending hold retained")
            if result["response"] is None:
                journal.uncertain(call_id, "null provider response")
                raise RunPaused("comprehension returned a null response; review saved evidence")
            record = {**result, **score_answer(result["response"], item["answer_key"])}
            journal.complete(identifier, record)
        except BaseException:
            journal.db.execute("UPDATE episodes SET status='paused' WHERE id=?", (identifier,))
            raise
        finally:
            _write_outputs(journal, ledger, packet, selected_models)
            if provider is not None and hasattr(provider, "aclose"):
                await asyncio.gather(provider.aclose(), return_exceptions=True)
        print(f"{identifier}: {item['model']} cost={result['usage']['total_cost']}", flush=True)
    return results(journal, packet, selected_models)


def _decimal(value):
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("amount must be a decimal number") from None
    if not number.is_finite() or number <= 0:
        raise argparse.ArgumentTypeError("amount must be a finite positive number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--route-manifest", required=True)
    parser.add_argument("--ledger", required=True, help="existing shared spending ledger")
    parser.add_argument("--shared-cap-usd", required=True, type=_decimal)
    parser.add_argument("--models", nargs="+", help="packet models to execute")
    parser.add_argument("--keys", help="local dotenv file; excluded from all snapshots")
    parser.add_argument("--execute", action="store_true", help="send the frozen requests")
    args = parser.parse_args(argv)
    packet_path, route_path = Path(args.packet), Path(args.route_manifest)
    packet_bytes, route_bytes = packet_path.read_bytes(), route_path.read_bytes()
    packet, manifest = json.loads(packet_bytes), json.loads(route_bytes)
    bounds, _settings = validate_packet(packet, manifest)
    try:
        selected_models = validate_selection(packet, args.models)
    except ValueError as exc:
        parser.error(str(exc))
    selected_count = sum(item["model"] in selected_models for item in packet["items"])
    print(f"Validated {len(packet['items'])} prompted checks; selected {selected_count}; "
          f"local allowance ${LOCAL_CAP_USD}; shared cap ${args.shared_cap_usd}.")
    if not args.execute:
        return
    if args.keys:
        from dotenv import load_dotenv
        load_dotenv(args.keys, override=False)
    sources = artifacts.runtime_source_paths(ROOT) + [Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(ROOT)): artifacts.sha256_file(path) for path in sources}
    run_dir = Path(args.run_dir)
    with Journal(run_dir / "run.sqlite3") as journal:
        journal.bind("source_files", source_hashes)
        journal.bind("runtime_environment", runtime.RUNTIME_ENVIRONMENT)
        journal.bind("packet_path", str(packet_path.resolve()))
        journal.bind("route_manifest_path", str(route_path.resolve()))
        snapshot = run_dir / "source-snapshot.tar.gz"
        if not snapshot.exists():
            artifacts._snapshot_source(ROOT, sources, snapshot)
        journal.bind("source_snapshot_sha256", artifacts.sha256_file(snapshot))
        with SpendingLedger(args.ledger, args.shared_cap_usd, existing_only=True) as ledger:
            asyncio.run(execute(packet, manifest, journal, ledger, packet_bytes=packet_bytes,
                                route_bytes=route_bytes, models=selected_models,
                                shared_cap_usd=args.shared_cap_usd))


if __name__ == "__main__":
    main()
