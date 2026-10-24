"""Targeted checks must keep pending work and previous paid calls separate."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "artifacts/labeling/2026-09-10/execution/run_targeted_audit.py"


def module():
    spec = importlib.util.spec_from_file_location("targeted_audit", PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def row(identifier, claims=(), *, extracted=True, reason=None, kind="lie"):
    return {"task_id": identifier, "kind": kind, "extraction_scored": extracted,
            "reason": reason, "claims": [{"status": status} for status in claims]}


def test_selection_keeps_pending_and_negative_messages_out():
    selection = module().select_candidates({"messages": [
        row("false", ["true", "false"]), row("unknown", ["unscored"]),
        row("parse", extracted=False, reason="parse_error"),
        row("pending", extracted=False, reason="not_run"),
        row("missing", extracted=False, reason="missing_label"),
        row("true", ["true"]), row("empty"), row("promise", kind="promise"),
    ]})
    assert selection == [
        {"task_id": "false", "reasons": ["false_claim"]},
        {"task_id": "parse", "reasons": ["extraction_failure"]},
        {"task_id": "unknown", "reasons": ["unscored_claim"]},
    ]


def test_frozen_manifest_cannot_change_after_first_write(tmp_path):
    audit = module()
    path = tmp_path / "selection.json"
    audit.write_fixed(path, {"task_ids": ["a"]})
    audit.write_fixed(path, {"task_ids": ["a"]})
    with pytest.raises(ValueError, match="changed"):
        audit.write_fixed(path, {"task_ids": ["a", "b"]})


def test_same_verdict_does_not_hide_structured_claim_disagreement():
    audit = module()
    primary = {"parsed_label": [{"scope": "specific_round", "round": 4}], "status": "labelled", "uncertain": False}
    second = {"parsed_label": [{"scope": "game_so_far"}], "status": "labelled", "uncertain": False}
    assert audit.comparison(primary, second) == "structured_disagreement"
    assert audit.comparison(primary, {**second, "status": "unscored"}) == "unresolved"


def test_previous_bound_ids_block_new_paid_journal_even_if_no_result_exists(tmp_path):
    audit = module()
    previous = tmp_path / "previous"
    previous.mkdir()
    audit.write_fixed(previous / "run.json", {"packet_sha256": "packet", "model": "gpt-5.5", "task_ids": ["a"]})
    with pytest.raises(ValueError, match="already bound"):
        audit.check_prior_bindings(tmp_path, "packet", {"a"}, allowed=set())
    audit.check_prior_bindings(tmp_path, "packet", {"a"}, allowed={previous.resolve()})
