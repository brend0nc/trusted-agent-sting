"""The 13b pipeline stops at human and validation boundaries."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "run_13b_deception_validation", ROOT / "scripts" / "run_13b_deception_validation.py"
)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_status_recognises_existing_run_and_frozen_structured_gold():
    result = pipeline.status()

    assert result["existing_13b_logs"] is True
    assert result["lie_structured_gold"] == {"complete": 50, "total": 50}
    assert result["next_safe_stage"] == "add OPENROUTER_API_KEY"


def test_paid_stage_requires_explicit_spend():
    args = type("Args", (), {"spend": False})()

    with pytest.raises(SystemExit, match="add --spend"):
        pipeline._require_spend(args)


def test_bulk_stage_needs_validation_and_explicit_acceptance(monkeypatch):
    args = type("Args", (), {"spend": True, "validation_approved": False})()
    monkeypatch.setenv("OPENROUTER_API_KEY", "present-for-offline-test")
    monkeypatch.setattr(pipeline, "_lie_validation_complete", lambda: True)
    monkeypatch.setattr(pipeline, "_promise_validation_complete", lambda: True)

    with pytest.raises(SystemExit, match="validation-approved"):
        pipeline.run_ledgers(args)


def test_gate_includes_held_out_13b_records(monkeypatch, tmp_path):
    for name in ("LIE_LEDGER", "PROMISE_LEDGER", "GATE", "MANIFEST"):
        monkeypatch.setattr(pipeline, name, tmp_path / f"{name}.json")
    pipeline.LIE_LEDGER.write_text("{}")
    pipeline.PROMISE_LEDGER.write_text("{}")
    def run(arguments, capture=False):
        assert arguments[arguments.index("--pool") + 1] == "all"
        return '{"validated_message_gate":{"event_gate_passed":false}}'
    monkeypatch.setattr(pipeline, "_run", run)
    pipeline.run_gate()
