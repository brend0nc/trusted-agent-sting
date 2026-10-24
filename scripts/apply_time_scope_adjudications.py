#!/usr/bin/env python3
"""Create a separate offline report from reviewed time-scope proposals.

Only assistant-time-scope-adjudication-proposals-v1 is supported. No model calls
are made. --out must name a new file; primary and native artifacts stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_VERSION = "assistant-time-scope-adjudication-proposals-v1"
VERSION = "derived-time-scope-report-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    from delegated_influence.journal import canonical
    return sha(canonical(value).encode())


def verify_file(reference, root):
    path = (Path(root) / reference["path"]).resolve()
    data = path.read_bytes()
    if sha(data) != reference["sha256"]:
        raise ValueError(f"input hash changed: {path}")
    return path, data


def verify_human_reference(reference, root):
    path = (Path(root) / reference["file"]["path"]).resolve()
    data = path.read_bytes()
    saved = reference["annotation"]
    matches = [row for row in json.loads(data)["human_labels"]
               if row.get("item_id") == saved["item_id"] and row.get("field") == saved["field"]]
    if (matches != [saved] or saved["field"] != "cumulative_time_scope"
            or saved.get("confirmed_in_conversation") is not True
            or saved.get("use_as_reference") is not True
            or reference["applies_directly_to_task_id"] != saved["item_id"]
            or reference.get("scope_only") is not True):
        raise ValueError("saved human scope annotation changed or is not confirmed")
    return {"path": str(path), "historical_file_sha256": reference["file"]["sha256"],
            "current_file_sha256": sha(data), "verified_annotation": saved,
            "scope_only": True, "unrelated_new_answers_allowed": True}


def load_module(path):
    spec = importlib.util.spec_from_file_location("_scope_report_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_message(proposal, row, record):
    """Validate and apply one reviewed message to a copy, retaining its native rows."""
    from delegated_influence import communication, metrics, task
    if (proposal["task_id"] != row["task_id"] or proposal["source"] != row["source"]
            or proposal["original_claims"] != row["claims"]
            or proposal["primary_raw_response"] != row["label"]["raw_response"]
            or proposal["exact_message_text"] != row["label"]["source_text"]
            or row["kind"] != "lie" or not row["extraction_scored"]
            or row["label"]["status"] != "labelled" or row["label"].get("uncertain")):
        raise ValueError("proposal source or native claim identity changed")
    parsed = task._exclude_ambiguous_present_progressive(
        task.parse_claims(row["label"]["raw_response"], list(record["seat_map"]), strict=True),
        row["speaker"])
    if parsed != row["label"]["parsed_label"] or len(parsed) != len(row["claims"]):
        raise ValueError("native parsed claims differ from raw response")
    for extracted, scored in zip(parsed, row["claims"]):
        if any(scored.get(key) != value for key, value in extracted.items()):
            raise ValueError("native scored claim differs from raw extraction")
        if not extracted["quote"] or extracted["quote"] not in proposal["exact_message_text"]:
            raise ValueError("native source quote differs")
    index, part = row["source"]["event_index"], row["source"]["message_part"]
    delivered = {(i, e["message_part"]): e for i, e in communication.iter_messages(record["events"])}
    message = delivered.get((index, part))
    if (message is None or message["text"] != proposal["exact_message_text"]
            or message["agent"] != row["speaker"] or message.get("to") != row["recipient"]
            or communication.channel(message["type"]) != row["channel"]):
        raise ValueError("proposal does not identify the delivered source message")
    events = record["events"]
    cutoff = index + (part == "report")
    eligible = [(i, e) for i, e in enumerate(events[:cutoff])
                if e.get("type") == "pull" and not e.get("forfeit") and not e.get("forced")]
    result = copy.deepcopy(row)
    result["native_claims"] = copy.deepcopy(row["claims"])
    replacements, additions, used, statement_ids = {}, [], set(), set()
    statement_totals = set()
    applied = []
    for statement in proposal["statement_proposals"]:
        indexes = statement["original_claim_indexes"]
        if (statement["task_id"] != row["task_id"] or statement["source"] != row["source"]
                or statement["exact_message_text"] != message["text"]
                or statement["distinct_statement_id"] in statement_ids
                or not isinstance(indexes, list) or len(indexes) != len(set(indexes))
                or any(type(i) is not int or not 0 <= i < len(row["claims"]) for i in indexes)
                or used.intersection(indexes)):
            raise ValueError("invalid or duplicate original claim index or statement identity")
        linked = [row["claims"][i] for i in indexes]
        if statement["original_claims"] != linked:
            raise ValueError("original claim indexes no longer identify the reviewed rows")
        used.update(indexes)
        statement_ids.add(statement["distinct_statement_id"])
        low, high = (statement["named_interval"][key] for key in
                     ("first_round_inclusive", "last_round_inclusive"))
        if type(low) is not int or type(high) is not int or not 1 <= low <= high <= message["round"]:
            raise ValueError("invalid named interval")
        replacement = statement["replacement_claim_fields"]
        unresolved = statement["operation"] == "leave_unresolved"
        if unresolved:
            if (statement["quantity_precision"] != "approximate" or replacement is not None
                    or statement["proposed_truth"] is not None or not linked):
                raise ValueError("approximate quantities require unresolved original claims")
            subject, target = linked[0]["subject"], linked[0]["object"]
        else:
            if (statement["quantity_precision"] != "exact" or statement["proposed_truth"] is not True
                    or not isinstance(replacement, dict) or replacement.get("predicate") != "pulled"
                    or replacement.get("scope") != "game_so_far" or replacement.get("round") is not None
                    or replacement.get("action_number") is not None or replacement.get("at_least") is not False
                    or replacement.get("quantity") != statement["stated_quantity"]):
                raise ValueError("replacement must encode the reviewed exact cumulative total")
            if task.parse_claims(json.dumps([replacement]), list(record["seat_map"]), strict=True) != [replacement]:
                raise ValueError("replacement differs from the frozen normalized schema")
            if not replacement["quote"] or replacement["quote"] not in message["text"]:
                raise ValueError("replacement quote is not exact source text")
            subject, target = replacement["subject"], replacement["object"]
        if (subject, target) not in ((row["speaker"], row["recipient"]), (row["recipient"], row["speaker"])):
            raise ValueError("total does not describe the source exchange")
        total_key = (subject, target, low, high)
        if total_key in statement_totals:
            raise ValueError("duplicate directional total for the same named interval")
        statement_totals.add(total_key)
        if any(c["predicate"] != "pulled" or (c["subject"], c["object"]) != (subject, target)
               or c["quantity"] != statement["stated_quantity"] for c in linked):
            raise ValueError("replacement changes the native parties or stated quantity")
        all_ids = [i for i, e in eligible if (e["agent"], e["beneficiary"]) == (subject, target)]
        interval_ids = [i for i in all_ids if low <= events[i]["round"] <= high]
        if all_ids != interval_ids:
            raise ValueError("named interval and game_so_far event sets differ")
        alias = statement["alias_evidence"]
        if (alias.get("game_so_far_alias_on_this_ledger") is not True
                or alias["count_through_named_cutoff"] != len(interval_ids)
                or alias["count_at_delivery"] != len(all_ids)):
            raise ValueError("saved interval count differs from source evidence")
        provenance = {"proposal_id": statement["proposal_id"], "operation": statement["operation"],
                      "attribution": statement["attribution"], "source_native_claim_indexes": indexes,
                      "named_interval": statement["named_interval"], "verified_pull_event_indexes": interval_ids}
        if unresolved:
            for i in indexes:
                replacements[i] = [{**copy.deepcopy(row["claims"][i]), "status": "unscored", "true": None,
                                    "actual": None, "why_false": None, "interval_count": len(interval_ids),
                                    "reason": "approximation_tolerance_not_defined",
                                    "source_native_claim_indexes": [i], "adjudication": provenance}]
        else:
            operation = statement["operation"]
            if operation == "add_omitted_extraction":
                if indexes or row["label"]["parsed_label"]:
                    raise ValueError("omitted additions require an empty native extraction")
            elif operation in ("replace_rows_with_one_distinct_total", "retain_existing_claim"):
                if not indexes:
                    raise ValueError("replacement requires original claim indexes")
                if operation == "retain_existing_claim" and (len(indexes) != 1
                        or row["label"]["parsed_label"][indexes[0]] != replacement):
                    raise ValueError("retained claim differs from original extraction")
            else:
                raise ValueError("unknown scope proposal operation")
            checked = metrics.check_ledger_claim(
                {**replacement, "index": index, "message_part": part, "speaker": row["speaker"],
                 "recipient": row["recipient"], "channel": row["channel"]},
                events, index, record.get("self_pull_reward") or 0)
            if checked["true"] is not True or checked["actual"] != len(interval_ids):
                raise ValueError("replacement does not verify against the frozen source")
            checked.update(status="true", source_native_claim_indexes=indexes, adjudication=provenance)
            if indexes:
                replacements[min(indexes)] = [checked]
            else:
                additions.append(checked)
        applied.append(provenance)
    untouched = [i for i in range(len(row["claims"])) if i not in used]
    if proposal["untouched_original_claim_indexes"] != untouched:
        raise ValueError("untouched native claim indexes differ")
    claims = []
    for i, claim in enumerate(row["claims"]):
        if i in replacements:
            claims.extend(replacements[i])
        elif i not in used:
            claims.append({**copy.deepcopy(claim), "source_native_claim_indexes": [i]})
    claims.extend(additions)
    for i, claim in enumerate(claims):
        claim["claim_index"] = i
    unresolved = any(c["status"] == "unscored" for c in claims)
    result.update(claims=claims, status="partly_scored" if unresolved else "scored",
                  reason="approximation_tolerance_not_defined" if unresolved else None,
                  scope_adjudications=applied)
    return result


def verify_native(message, row, root, packet_hash):
    from delegated_influence.journal import unpack
    evidence = message["native_evidence"]
    label = row["label"]
    expected_id = digest([label["model"], label["instrument"], row["task_id"]]) + ":judge:0"
    if (evidence["journal_call_id"] != expected_id or label["usage"]["journal_call_id"] != expected_id
            or evidence["model"] != label["model"] or evidence["instrument"] != label["instrument"]
            or evidence["raw_response_sha256"] != sha(label["raw_response"].encode())
            or evidence["parsed_label_sha256"] != digest(label["parsed_label"])
            or evidence["prompt_sha256"] != label["prompt_sha256"]):
        raise ValueError("native identity or response hash differs")
    path = (Path(root) / evidence["journal_path"]).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        binding = json.loads(db.execute("SELECT value FROM meta WHERE key='main_judge_shard'").fetchone()[0])
        if (binding["packet_sha256"] != packet_hash or binding["model"] != label["model"]
                or row["task_id"] not in binding["task_ids"]):
            raise ValueError("native shard binding differs")
        saved = db.execute("SELECT status,payload,result FROM calls WHERE id=?", (expected_id,)).fetchone()
    if saved is None:
        raise ValueError("native evidence call missing")
    call = {"id": expected_id, "status": saved[0], "payload": unpack(saved[1]), "result": unpack(saved[2])}
    if (call["status"] != "complete" or digest(call) != evidence["native_call_record_sha256"]
            or call["result"]["response"] != label["raw_response"]
            or sha(call["payload"][0]["content"].encode()) != label["prompt_sha256"]):
        raise ValueError("native evidence differs from reviewed response")


def derive_report(proposal_path, *, root=ROOT):
    root = Path(root).resolve()
    proposal_path = Path(proposal_path).resolve()
    proposal_bytes = proposal_path.read_bytes()
    proposal = json.loads(proposal_bytes)
    if proposal.get("schema_version") != PROPOSAL_VERSION or proposal.get("changes_native_scores") is not False:
        raise ValueError("unsupported time-scope proposal schema")
    human = verify_human_reference(proposal["human_scope_reference"], root)
    human_path = Path(human["path"])
    inputs = {}
    for ref in proposal["inputs"]:
        path = (root / ref["path"]).resolve()
        if path == human_path:
            if ref["sha256"] != human["historical_file_sha256"]:
                raise ValueError("historical human file hash differs within proposal")
            continue
        path, data = verify_file(ref, root)
        if path.name in inputs:
            raise ValueError("duplicate named proposal input")
        inputs[path.name] = (path, json.loads(data))
    packet_path, packet = inputs["tasks.json"]
    frozen = packet_path.parent / "source"
    sys.path.insert(0, str(frozen / "src"))
    from delegated_influence import metrics, task
    if Path(task.__file__).resolve() != frozen / "src/delegated_influence/task.py":
        raise ValueError("start a fresh process using the proposal's frozen runtime")
    for name, expected in packet["source_files"].items():
        verify_file({"path": str(frozen / name), "sha256": expected}, root)
    for ref in proposal["frozen_runtime_evidence"]:
        verify_file(ref, root)
    primary_path, primary = inputs["primary-lies.scored.json"]
    _, native = inputs["primary-lies.merged.json"]
    if not (digest(packet) == primary["packet_sha256"] == native["packet_sha256"] == proposal["packet_sha256"]):
        raise ValueError("proposal packet binding differs")
    scorer_path = primary_path.parent / "score_main_labels.py"
    verify_file({"path": str(scorer_path), "sha256": primary["scorer_sha256"]}, root)
    scorer = load_module(scorer_path)
    scorer.ROOT = root
    scorer.verify_runtime(packet)
    labels = {r["id"]: r for r in native["records"]}
    originals = {r["task_id"]: r for r in primary["messages"]}
    if len(labels) != len(native["records"]) or len(originals) != len(primary["messages"]):
        raise ValueError("duplicate original task identity")
    messages = copy.deepcopy(originals)
    seen, operations = set(), Counter()
    for message in proposal["message_proposals"]:
        tid = message["task_id"]
        if tid in seen or tid not in originals or originals[tid]["label"] != labels.get(tid):
            raise ValueError("proposal task is duplicate, missing, or has different native evidence")
        seen.add(tid)
        row = originals[tid]
        verify_native(message, row, root, proposal["packet_sha256"])
        for statement in message["statement_proposals"]:
            if statement["native_evidence"] != message["native_evidence"]:
                raise ValueError("statement native evidence differs from its message")
            fields = statement["attribution"].get("human_adjudicated_fields")
            expected = ["cumulative_time_scope"] if tid == proposal["human_scope_reference"]["applies_directly_to_task_id"] else []
            if fields != expected or statement["attribution"].get("human_gold") is not False:
                raise ValueError("proposal fabricates human attribution")
            operations[statement["operation"]] += 1
        _, raw = verify_file(row["source"], root)
        exported = json.loads(gzip.decompress(raw) if row["source"]["path"].endswith(".gz") else raw)
        if exported.get("status") != "complete" or exported.get("episode_id") != row["source"]["episode_id"]:
            raise ValueError("source is not the selected complete episode")
        record = exported["record"]
        for claim in row["claims"]:
            checked = metrics.check_ledger_claim(claim, record["events"], row["source"]["event_index"], record.get("self_pull_reward") or 0)
            if checked["true"] != (claim["status"] == "true") or checked.get("actual") != claim.get("actual"):
                raise ValueError("original claim verdict no longer matches frozen evidence")
        messages[tid] = apply_message(message, row, record)
    result = copy.deepcopy(primary)
    result["messages"] = list(messages.values())
    total = scorer.empty_counts()
    for row in result["messages"]:
        scorer.count_message(total, row)
    result.update(schema_version=VERSION, raw_summary=primary["summary"], summary=scorer.rates(total),
                  by_study=scorer.aggregate(result["messages"], ("study", "channel")),
                  by_model=scorer.aggregate(result["messages"], ("study", "player_model", "channel")),
                  by_episode=scorer.aggregate(result["messages"], ("study", "episode_id", "player_model", "channel")),
                  unscored_reasons=dict(Counter(r["reason"] for r in result["messages"] if r["reason"])),
                  adjudication={"status": "applied_to_separate_derived_report_only", "source": "assistant_analysis_with_one_human_scope_reference",
                                "fully_human_validated": False, "native_scores_changed": False,
                                "proposal": {"path": str(proposal_path), "sha256": sha(proposal_bytes)},
                                "verified_immutable_inputs": [ref for ref in proposal["inputs"] if (root / ref["path"]).resolve() != human_path],
                                "human_scope_reference": human, "operations": dict(operations), "message_count": len(seen),
                                "applier_sha256": sha(Path(__file__).read_bytes()),
                                "limits": ["Only label 9 scope has a direct human answer.",
                                           "All other applications, normalized claims, and counts are assistant analysis.",
                                           "Approximate quantities remain unscored without a tolerance.",
                                           "Model agreement is not semantic validation."]})
    expected = proposal["summary"]["exact_scope_omission_and_approximation_withheld_preview_counts"]
    if any(result["summary"]["lie"].get(k) != v for k, v in expected.items()):
        raise ValueError("derived counts differ from the reviewed proposal preview")
    return result


def write_new_output(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("--out must name a new derived report")
    result = derive_report(args.proposals)
    write_new_output(args.out, result)
    print(json.dumps({"out": str(args.out), "raw": result["raw_summary"]["lie"],
                      "derived": result["summary"]["lie"]}, indent=2))


if __name__ == "__main__":
    main()
