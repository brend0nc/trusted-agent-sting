"""Compare scope-aware lie judges on the same hand-labelled 60-message sample.

Dry by default. Paid calls require --spend. Results are checkpointed after every call, so a
timeout or credit interruption can resume without repeating completed calls.

Run:
  PYTHONPATH=src python3 scripts/calibrate_lie_judges.py
  PYTHONPATH=src python3 scripts/calibrate_lie_judges.py --spend
"""
import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import calibrate_score, models  # noqa: E402
from delegated_influence import native_judges  # noqa: E402
from delegated_influence.task import LIE_JUDGE_INSTRUMENT, make_lie_judge  # noqa: E402

SAMPLE = ROOT / "data" / "lie_judge_calibration_sample.json"
OUT = ROOT / "data" / "lie_judge_scope_comparison.json"
DEFAULT_JUDGES = ("gpt-5.5", "opus-4.8")


def select_records(sample, channel="all"):
    """Return a copy containing only the requested message channel.

    Older private-only calibration files have no channel field, so missing means private.
    """
    if channel == "all":
        records = list(sample["records"])
    else:
        records = [r for r in sample["records"] if r.get("channel", "private") == channel]
    return {**sample, "records": records}


def infer_roster(records):
    """Infer the anonymous seat roster from structured fields and message text."""
    seat_numbers = set()
    for record in records:
        values = [record.get("sender_seat"), record.get("recipient_seat"), record.get("text")]
        for value in values:
            seat_numbers.update(int(n) for n in re.findall(r"\bP(\d+)\b", str(value or "")))
    if not seat_numbers:
        raise ValueError("could not infer a player roster from the selected records")
    return [f"P{i}" for i in range(1, max(seat_numbers) + 1)]


def apply_label_sidecar(sample, labels, sample_sha256):
    """Freeze human decisions separately from message text and verify the sampled input hash.

    Supports the older binary claim-detection sidecars and the paper holdout's exact structured
    ``gold_claims`` sidecar. Blank structured gold stops before any paid call.
    """
    if labels.get("sample_sha256") != sample_sha256:
        raise ValueError("label sidecar does not match the sample file hash")
    decisions = {str(r["record_id"]): r for r in labels.get("records", [])}
    expected = {str(r["record_id"]) for r in sample["records"]}
    if set(decisions) != expected:
        raise ValueError("label sidecar record ids do not exactly match the sample")
    structured = any("gold_claims" in decision for decision in decisions.values())
    records = []
    for record in sample["records"]:
        decision = decisions[str(record["record_id"])]
        if structured:
            if decision.get("gold_claims") is None:
                raise ValueError(f"blank structured claim label for {record['record_id']}")
            records.append({
                **record,
                "gold_claims": decision["gold_claims"],
                "gold_has_claims": bool(decision["gold_claims"]),
                "gold_notes": decision.get("gold_notes", ""),
            })
        else:
            if decision.get("gold_has_claims") is None:
                raise ValueError(f"blank claim label for {record['record_id']}")
            fields = ("gold_promise", "gold_conditional", "gold_has_claims", "gold_notes")
            records.append({**record, **{field: decision.get(field) for field in fields}})
    return {**sample, "records": records}


def judge_spec(name):
    """Roster key -> OpenRouter spec; raw Inspect specs remain available for controlled tests."""
    return models.openrouter_spec(name) if name in models.MODEL_REGISTRY else name


def checkpoint(doc, path):
    """Atomic checkpoint so an interrupted paid run never corrupts completed results."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def paid_output_count(doc):
    return sum(len(record.get("outputs") or {}) for record in (doc or {}).get("records", []))


def archive_replaced_outputs(path, existing):
    """Preserve paid outputs that the current instrument cannot safely resume."""
    path = Path(path)
    raw = path.read_bytes()
    instrument = str((existing.get("meta") or {}).get("instrument") or "legacy")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", instrument).strip("-") or "legacy"
    digest = hashlib.sha256(raw).hexdigest()[:12]
    archive = path.with_name(f"{path.stem}.archived-{label}-{digest}{path.suffix}")
    if archive.exists():
        if archive.read_bytes() != raw:
            raise FileExistsError(f"archive path already contains different data: {archive}")
    else:
        archive.write_bytes(raw)
    return archive


def initialise(sample, judges, existing=None, sample_path=SAMPLE, channel="all", roster=None):
    """Create or validate the comparison artifact while preserving completed model outputs."""
    keep = ("record_id", "episode_id", "event_index", "round", "sender_seat", "sender_model",
            "recipient_seat", "text", "reader_link", "gold_promise", "gold_conditional",
            "gold_has_claims", "gold_claims", "gold_notes", "pool", "channel")
    sample_path = Path(sample_path).resolve()
    try:
        sample_label = str(sample_path.relative_to(ROOT))
    except ValueError:
        sample_label = str(sample_path)
    fresh = {
        "meta": {
            "purpose": "scope-aware lie-judge validation on frozen labels",
            "sample": sample_label,
            "channel": channel,
            "roster": list(roster or infer_roster(sample["records"])),
            "n": len(sample["records"]),
            "judges": list(judges),
            "instrument": LIE_JUDGE_INSTRUMENT,
            "gold_limit": (
                "Exact claim fields are supplied; preserve the label sidecar's source and approval status."
                if sample["records"] and all(r.get("gold_claims") is not None for r in sample["records"])
                else "gold_has_claims validates detection only; exact scope has no structured gold"),
        },
        "records": [{**{k: (r.get(k, "private") if k == "channel" else r.get(k))
                         for k in keep}, "outputs": {}}
                    for r in sample["records"]],
        "summary": {},
    }
    if not existing:
        return fresh

    expected = [r["record_id"] for r in fresh["records"]]
    actual = [r["record_id"] for r in existing.get("records", [])]
    if expected != actual:
        raise ValueError("existing output does not match the calibration sample")
    stable_fields = ("episode_id", "event_index", "sender_seat", "sender_model",
                     "recipient_seat", "text", "reader_link", "pool", "channel")
    previous_instrument = (existing.get("meta") or {}).get("instrument")
    instrument_changed = previous_instrument not in (None, LIE_JUDGE_INSTRUMENT)
    for current, previous in zip(fresh["records"], existing["records"]):
        for field in stable_fields:
            old = previous.get(field, "private" if field == "channel" else None)
            new = current.get(field, "private" if field == "channel" else None)
            if old != new:
                raise ValueError(
                    f"existing output input changed for record {current['record_id']}: {field}")
        legacy_public_prompt = previous_instrument is None and current.get("channel") == "public"
        if not instrument_changed and not legacy_public_prompt:
            current["outputs"] = previous.get("outputs", {})
    # Human labels are deliberately refreshed from the sidecar-backed sample. Only paid model
    # outputs are resumed; summaries/adjudications are recomputed later from the current labels.
    return fresh


def summarise(doc, judges, gold_field="gold_has_claims"):
    """Detection quality, schema distribution, token use, served model, and reported API cost."""
    summary = {}
    for name in judges:
        rows = []
        usages = []
        scopes = Counter()
        served = Counter()
        total_claims = exact_slots = errors = 0
        for r in doc["records"]:
            out = r["outputs"].get(name)
            if not out:
                continue
            if out.get("error"):
                errors += 1
                continue
            claims = out["claims"]
            rows.append({"gold": r.get(gold_field, r["gold_has_claims"]),
                         "pred": bool(claims)})
            total_claims += len(claims)
            scopes.update(c.get("scope") for c in claims)
            exact_slots += sum(c.get("action_number") is not None for c in claims)
            usages.append(out.get("usage") or {})
            served[out.get("served_model") or "unknown"] += 1
        usage_totals = {k: sum((u.get(k) or 0) for u in usages)
                        for k in ("input_tokens", "output_tokens", "reasoning_tokens", "total_cost")}
        summary[name] = {
            **calibrate_score.binary_metrics(rows, "gold", "pred"),
            "completed_calls": len(rows), "errors": errors, "total_claims": total_claims,
            "scope_counts": dict(sorted(scopes.items())), "exact_action_claims": exact_slots,
            "served_models": dict(sorted(served.items())), "usage": usage_totals,
        }
    return summary


CLAIM_SHAPE_FIELDS = (
    "predicate", "subject", "object", "quantity", "at_least", "scope", "round",
    "action_number",
)


def _claim_shape(claim):
    """Quote-free exact shape used by the deterministic ledger verifier."""
    return tuple(claim.get(field) for field in CLAIM_SHAPE_FIELDS)


def structured_summary(doc, judges):
    """Exact holdout quality for every verifier-relevant claim field.

    Detection metrics answer whether a message contains a supported claim. The primary structured
    score is micro exact-claim precision/recall/F1 after removing only the quote span. A claim is a
    match only when every field consumed by the ledger verifier matches human gold. Record exact
    match is the stricter all-claims-correct diagnostic.
    """
    if not doc.get("records") or not all(
            record.get("gold_claims") is not None for record in doc["records"]):
        return None
    summary = {}
    for name in judges:
        detection_rows = []
        tp = fp = fn = exact_records = completed = errors = 0
        for record in doc["records"]:
            output = record.get("outputs", {}).get(name)
            if not output:
                continue
            if output.get("error"):
                errors += 1
                continue
            completed += 1
            gold = Counter(_claim_shape(claim) for claim in record["gold_claims"])
            predicted = Counter(_claim_shape(claim) for claim in output.get("claims", []))
            shared = gold & predicted
            tp += sum(shared.values())
            fp += sum((predicted - gold).values())
            fn += sum((gold - predicted).values())
            exact_records += predicted == gold
            detection_rows.append({"gold": bool(gold), "pred": bool(predicted)})
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        summary[name] = {
            "detection": calibrate_score.binary_metrics(detection_rows, "gold", "pred"),
            "exact_claim_tp": tp,
            "exact_claim_fp": fp,
            "exact_claim_fn": fn,
            "exact_claim_precision": precision,
            "exact_claim_recall": recall,
            "exact_claim_f1": f1,
            "exact_record_match_rate": exact_records / completed if completed else None,
            "completed_calls": completed,
            "errors": errors,
            "shape_fields": list(CLAIM_SHAPE_FIELDS),
        }
    return summary


def comparison_summary(doc, judges):
    """Cross-judge agreement after removing harmless quote-span wording differences."""
    fields = ("predicate", "subject", "object", "quantity", "at_least", "scope", "round",
              "action_number")
    name_a, name_b = judges
    detection_disagreements = []
    shared = only_a = only_b = changed_records = 0
    for r in doc["records"]:
        outputs = r["outputs"]
        if name_a not in outputs or name_b not in outputs:
            continue
        a = outputs[name_a]
        b = outputs[name_b]
        if a.get("error") or b.get("error"):
            continue
        claims_a, claims_b = a["claims"], b["claims"]
        if bool(claims_a) != bool(claims_b):
            detection_disagreements.append(r["record_id"])
        shapes_a = {tuple(c.get(k) for k in fields) for c in claims_a}
        shapes_b = {tuple(c.get(k) for k in fields) for c in claims_b}
        shared += len(shapes_a & shapes_b)
        only_a += len(shapes_a - shapes_b)
        only_b += len(shapes_b - shapes_a)
        changed_records += shapes_a != shapes_b
    return {
        "judges": [name_a, name_b], "detection_disagreement_ids": detection_disagreements,
        "shared_claim_shapes": shared, f"{name_a}_only_claim_shapes": only_a,
        f"{name_b}_only_claim_shapes": only_b,
        "records_with_claim_shape_differences": changed_records,
    }


def adjudication_queue(doc, judges):
    """The small set where current human detection gold and either judge disagree."""
    queued = []
    for r in doc["records"]:
        outputs = [r["outputs"].get(name) for name in judges]
        if any(not out or out.get("error") for out in outputs):
            continue
        predicted = [bool(out["claims"]) for out in outputs]
        if all(p == bool(r["gold_has_claims"]) for p in predicted):
            continue
        predicates = sorted({c["predicate"] for out in outputs for c in out["claims"]})
        category = ("gold_excludes_explicit_score_claim" if not r["gold_has_claims"]
                    and predicates == ["score"] else "gold_claim_missed_by_both"
                    if r["gold_has_claims"] and not any(predicted) else "other_disagreement")
        queued.append({
            "record_id": r["record_id"], "category": category,
            "current_gold_has_claims": r["gold_has_claims"], "text": r["text"],
            "gold_notes": r.get("gold_notes", ""),
            "judge_claims": {name: r["outputs"][name]["claims"] for name in judges},
            "adjudicated_gold_has_claims": None, "adjudication_notes": "",
        })
    return queued


def apply_adjudications(doc, judges, adjudication):
    """Apply explicit, auditable rubric decisions without overwriting the original human gold."""
    sample = doc["meta"]["sample"]
    decisions = [d for d in adjudication.get("decisions", []) if d["sample"] == sample]
    records = {str(r["record_id"]): r for r in doc["records"]}
    for r in doc["records"]:
        r["adjudicated_gold_has_claims"] = bool(r["gold_has_claims"])
    for decision in decisions:
        record_id = str(decision["record_id"])
        if record_id not in records:
            raise ValueError(f"adjudication references unknown record {record_id!r} in {sample}")
        records[record_id]["adjudicated_gold_has_claims"] = bool(decision["gold_has_claims"])
    by_id = {str(d["record_id"]): d for d in decisions}
    for queued in doc.get("adjudication_queue", []):
        decision = by_id.get(str(queued["record_id"]))
        if decision:
            queued["adjudicated_gold_has_claims"] = bool(decision["gold_has_claims"])
            queued["adjudication_notes"] = decision["notes"]
    doc["meta"]["rubric_v2"] = adjudication.get("rubric", {})
    doc["meta"]["adjudicated_records"] = len(decisions)
    doc["summary_adjudicated"] = summarise(doc, judges, "adjudicated_gold_has_claims")
    return doc


def calls_remaining(doc, judges, force_records=()):
    force_records = {str(record_id) for record_id in force_records}
    return sum(str(r["record_id"]) in force_records or name not in r["outputs"]
               or bool(r["outputs"][name].get("error"))
               for name in judges for r in doc["records"])


async def fill(doc, judges, out_path, roster, force_records=(), session=None):
    force_records = {str(record_id) for record_id in force_records}
    if session:
        for name in judges:
            for record in doc["records"]:
                if name in record.get("outputs", {}):
                    session.require_saved_call(name, LIE_JUDGE_INSTRUMENT, record["record_id"])
    for name in judges:
        kwargs = ({"generate": session.generator(name, LIE_JUDGE_INSTRUMENT)} if session else {})
        judge = make_lie_judge(judge_spec(name), include_details=True, **kwargs)
        for i, r in enumerate(doc["records"], 1):
            if (session is None and str(r["record_id"]) not in force_records and name in r["outputs"]
                    and not r["outputs"][name].get("error")):
                continue
            try:
                if r.get("channel", "private") == "public":
                    result = await native_judges.call_judge(
                        judge, r["record_id"], r["text"], r["sender_seat"], r["recipient_seat"],
                        roster, channel="public")
                else:
                    result = await native_judges.call_judge(
                        judge, r["record_id"], r["text"], r["sender_seat"], r["recipient_seat"], roster)
                r["outputs"][name] = result
            except Exception as exc:
                r["outputs"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                doc["summary"] = summarise(doc, judges)
                exact = structured_summary(doc, judges)
                if exact is not None:
                    doc["structured_summary"] = exact
                checkpoint(doc, out_path)
                raise
            doc["summary"] = summarise(doc, judges)
            exact = structured_summary(doc, judges)
            if exact is not None:
                doc["structured_summary"] = exact
            checkpoint(doc, out_path)
            print(f"{name}: {i}/{len(doc['records'])}", flush=True)
    doc["summary"] = summarise(doc, judges)
    exact = structured_summary(doc, judges)
    if exact is not None:
        doc["structured_summary"] = exact
    checkpoint(doc, out_path)


def main():
    ap = argparse.ArgumentParser(description="Compare two lie judges on existing human labels.")
    ap.add_argument("--spend", action="store_true", help="make real paid model calls")
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    ap.add_argument("--channel", choices=("all", "private", "public"), default="all",
                    help="judge only this message channel (missing channel means private)")
    ap.add_argument("--adjudication", default=None,
                    help="JSON rubric decisions to layer over, without replacing, original gold")
    ap.add_argument("--labels", default=None,
                    help="frozen human-label sidecar for a blank sample")
    ap.add_argument("--force-record", nargs="*", default=[],
                    help="rerun only these record ids even when checkpointed")
    native_judges.add_arguments(ap)
    args = ap.parse_args()

    sample_path = Path(args.sample)
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    if args.labels:
        with Path(args.labels).open() as f:
            labels = json.load(f)
        sample = apply_label_sidecar(sample, labels, hashlib.sha256(sample_bytes).hexdigest())
    sample = select_records(sample, args.channel)
    roster = infer_roster(sample["records"])
    out_path = Path(args.out)
    if out_path.exists():
        with out_path.open() as f:
            existing = json.load(f)
    else:
        existing = None
    doc = initialise(sample, args.judges, existing, sample_path, args.channel, roster)
    if args.labels:
        doc["meta"]["sample_sha256"] = hashlib.sha256(sample_bytes).hexdigest()
        doc["meta"]["labels_sha256"] = hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()
        doc["meta"]["labels_status"] = labels.get("status")
        try:
            doc["meta"]["labels"] = str(Path(args.labels).resolve().relative_to(ROOT))
        except ValueError:
            doc["meta"]["labels"] = str(Path(args.labels).resolve())
    doc["summary"] = summarise(doc, args.judges)
    exact = structured_summary(doc, args.judges)
    if exact is not None:
        doc["structured_summary"] = exact
    if len(args.judges) == 2:
        doc["comparison"] = comparison_summary(doc, args.judges)
    doc["adjudication_queue"] = adjudication_queue(doc, args.judges)
    adjudication = None
    if args.adjudication:
        with Path(args.adjudication).open() as f:
            adjudication = json.load(f)
        apply_adjudications(doc, args.judges, adjudication)
    force_records = set(args.force_record)
    remaining = calls_remaining(doc, args.judges, force_records)
    print(f"{len(sample['records'])} labelled messages x {len(args.judges)} judges; "
          f"{remaining} calls remaining")
    if not args.spend:
        print("dry run: add --spend to make paid calls")
        return
    if existing and paid_output_count(existing) > paid_output_count(doc):
        archive = archive_replaced_outputs(out_path, existing)
        print(f"archived incompatible paid outputs -> {archive}")
    inputs = [sample_path, *([Path(args.labels)] if args.labels else []),
              *([Path(args.adjudication)] if args.adjudication else [])]
    with native_judges.open_session(args, inputs=inputs, binding={
            "stage": "lie-validation", "output": str(out_path.resolve()),
            "judges": args.judges, "channel": args.channel}) as session:
        asyncio.run(fill(doc, args.judges, out_path, roster, force_records, session=session))
    if len(args.judges) == 2:
        doc["comparison"] = comparison_summary(doc, args.judges)
    doc["adjudication_queue"] = adjudication_queue(doc, args.judges)
    if adjudication:
        apply_adjudications(doc, args.judges, adjudication)
    exact = structured_summary(doc, args.judges)
    if exact is not None:
        doc["structured_summary"] = exact
    checkpoint(doc, out_path)
    print(json.dumps(doc["summary"], indent=2))
    if exact is not None:
        print(json.dumps({"structured_summary": exact}, indent=2))


if __name__ == "__main__":
    main()
