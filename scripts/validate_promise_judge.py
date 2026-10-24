"""Validate the current breach-test promise judge on the frozen experiment-11 holdout.

This runner never edits the sample or human-label sidecar. A dry run validates hashes and reports
the remaining call count. Paid calls require ``--spend`` and checkpoint after every judgment.

Run without spend:
  PYTHONPATH=src python3 scripts/validate_promise_judge.py

Run paid:
  PYTHONPATH=src python3 scripts/validate_promise_judge.py --spend
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import calibrate_score, models  # noqa: E402
from delegated_influence import native_judges  # noqa: E402
from delegated_influence.task import PROMISE_JUDGE_INSTRUMENT, make_promise_judge  # noqa: E402


SAMPLE = ROOT / "data" / "exp11_holdout_sample.json"
LABELS = ROOT / "data" / "exp11_holdout_labels.json"
OUT = ROOT / "data" / "promise_judge_breach_holdout_opus.json"
JUDGE = "opus-4.8"


def _load(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_records(sample_path=SAMPLE, labels_path=LABELS):
    """Join hash-bound human labels without exposing author model to the judge."""
    sample_path, labels_path = Path(sample_path), Path(labels_path)
    sample, labels = _load(sample_path), _load(labels_path)
    digest = _sha256(sample_path)
    if labels.get("sample_sha256") != digest:
        raise ValueError("promise label sidecar does not match the frozen sample")
    decisions = {str(record["record_id"]): record for record in labels.get("records", [])}
    expected = {str(record["record_id"]) for record in sample.get("records", [])}
    if set(decisions) != expected:
        raise ValueError("promise label sidecar record ids do not exactly match the sample")
    records = []
    for source in sample["records"]:
        gold = decisions[str(source["record_id"])]
        if gold.get("gold_promise") is None or gold.get("gold_conditional") is None:
            raise ValueError(f"blank promise label for {source['record_id']}")
        records.append({
            "record_id": source["record_id"],
            "text": source["text"],
            "gold_promise": bool(gold["gold_promise"]),
            "gold_conditional": bool(gold["gold_conditional"]),
            "gold_notes": gold.get("gold_notes", ""),
            "output": None,
        })
    return records, digest, _sha256(labels_path)


def _judge_spec(name):
    return models.openrouter_spec(name) if name in models.MODEL_REGISTRY else name


def initialise(records, sample_hash, labels_hash, judge, existing=None):
    meta = {
        "purpose": "Current breach-test promise-judge validation on frozen human labels",
        "sample": "data/exp11_holdout_sample.json",
        "sample_sha256": sample_hash,
        "labels": "data/exp11_holdout_labels.json",
        "labels_sha256": labels_hash,
        "instrument": PROMISE_JUDGE_INSTRUMENT,
        "judge": judge,
        "gold_limit": (
            "Human gold validates promise detection and conditionality only; exact commitment, "
            "beneficiary scope, and deadline parsing remain unvalidated."
        ),
    }
    doc = {"meta": meta, "records": records, "summary": {}}
    if not existing:
        return doc
    if existing.get("meta") != meta:
        raise ValueError("existing promise validation output belongs to different inputs")
    old = {str(record["record_id"]): record for record in existing.get("records", [])}
    if set(old) != {str(record["record_id"]) for record in records}:
        raise ValueError("existing promise validation output has different records")
    for record in doc["records"]:
        prior = old[str(record["record_id"])]
        if prior.get("text") != record["text"]:
            raise ValueError(f"existing promise validation text changed: {record['record_id']}")
        record["output"] = prior.get("output")
    return doc


def summarise(doc):
    completed = [record for record in doc["records"] if record.get("output") is not None]
    detection = [
        {"gold": record["gold_promise"], "pred": bool(record["output"].get("promise"))}
        for record in completed
    ]
    positive_gold = [record for record in completed if record["gold_promise"]]
    conditional = [
        {"gold": record["gold_conditional"],
         "pred": bool(record["output"].get("conditional"))
         if record["output"].get("promise") else False}
        for record in positive_gold
    ]
    predicted_promises = [record["output"] for record in completed
                          if record["output"].get("promise")]
    return {
        "completed_calls": len(completed),
        "remaining_calls": len(doc["records"]) - len(completed),
        "detection": calibrate_score.binary_metrics(detection, "gold", "pred"),
        "conditional_on_gold_promises": calibrate_score.binary_metrics(
            conditional, "gold", "pred"),
        "predicted_promises": len(predicted_promises),
        "beneficiary_scope_present": sum(
            output.get("beneficiary_scope") in ("recipient", "triggering_agent")
            for output in predicted_promises
        ),
        "deadline_hint_present": sum(output.get("deadline_hint") is not None
                                     for output in predicted_promises),
        "gold_limit": doc["meta"]["gold_limit"],
    }


def checkpoint(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


async def fill(doc, out_path, session=None):
    if session:
        for record in doc["records"]:
            if record.get("output") is not None:
                session.require_saved_call(doc["meta"]["judge"], PROMISE_JUDGE_INSTRUMENT, record["record_id"])
    kwargs = ({"generate": session.generator(doc["meta"]["judge"], PROMISE_JUDGE_INSTRUMENT),
               "include_details": True} if session else {})
    judge = make_promise_judge(doc["meta"]["judge"], **kwargs)
    for index, record in enumerate(doc["records"], 1):
        if session is None and record.get("output") is not None:
            continue
        record["output"] = await native_judges.call_judge(judge, record["record_id"], record["text"])
        doc["summary"] = summarise(doc)
        checkpoint(doc, out_path)
        print(f"promise judge: {index}/{len(doc['records'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Validate the current promise judge on frozen breach-test labels.")
    parser.add_argument("--spend", action="store_true", help="make real paid judge calls")
    parser.add_argument("--sample", default=str(SAMPLE))
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--judge", default=JUDGE)
    native_judges.add_arguments(parser)
    args = parser.parse_args()

    records, sample_hash, labels_hash = load_frozen_records(args.sample, args.labels)
    judge = _judge_spec(args.judge)
    out_path = Path(args.out)
    existing = _load(out_path) if out_path.exists() else None
    doc = initialise(records, sample_hash, labels_hash, judge, existing)
    doc["summary"] = summarise(doc)
    print(f"{len(records)} frozen messages; {doc['summary']['remaining_calls']} calls remaining")
    if not args.spend:
        print("dry run: add --spend to make paid calls")
        return
    with native_judges.open_session(args, inputs=[args.sample, args.labels], binding={
            "stage": "promise-validation", "output": str(out_path.resolve()), "judge": judge}) as session:
        asyncio.run(fill(doc, out_path, session=session))
    doc["summary"] = summarise(doc)
    checkpoint(doc, out_path)
    print(json.dumps(doc["summary"], indent=2))


if __name__ == "__main__":
    main()
