"""The current promise holdout runner preserves frozen human evidence."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "validate_promise_judge", ROOT / "scripts" / "validate_promise_judge.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def test_shipped_promise_labels_are_hash_bound_and_complete():
    records, sample_hash, labels_hash = validator.load_frozen_records()

    assert len(records) == 20
    assert sample_hash
    assert labels_hash
    assert all(record["gold_promise"] is not None for record in records)
    assert all(record["gold_conditional"] is not None for record in records)


def test_mismatched_sidecar_stops_before_judging(tmp_path):
    sample = tmp_path / "sample.json"
    labels = tmp_path / "labels.json"
    sample.write_text(json.dumps({"records": [{"record_id": "x", "text": "hello"}]}))
    labels.write_text(json.dumps({
        "sample_sha256": "wrong",
        "records": [{"record_id": "x", "gold_promise": None, "gold_conditional": None}],
    }))

    with pytest.raises(ValueError, match="does not match"):
        validator.load_frozen_records(sample, labels)


def test_summary_includes_detection_and_states_structured_gold_limit():
    records = [
        {"record_id": "a", "text": "x", "gold_promise": True,
         "gold_conditional": True, "gold_notes": "",
         "output": {"promise": True, "conditional": True,
                    "beneficiary_scope": "recipient", "deadline_hint": "next round"}},
        {"record_id": "b", "text": "y", "gold_promise": False,
         "gold_conditional": False, "gold_notes": "",
         "output": {"promise": False, "conditional": False,
                    "beneficiary_scope": None, "deadline_hint": None}},
    ]
    doc = validator.initialise(records, "sample", "labels", "judge")
    result = validator.summarise(doc)

    assert result["detection"]["accuracy"] == 1
    assert result["conditional_on_gold_promises"]["accuracy"] == 1
    assert result["beneficiary_scope_present"] == 1
    assert "remain unvalidated" in result["gold_limit"]
