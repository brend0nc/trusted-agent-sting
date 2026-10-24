"""The static site must make experiment-instrument audit status visible."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))

import render  # noqa: E402


def test_completed_experiments_are_not_labeled_in_progress():
    groups = render.experiment_groups([
        {"id": "done", "status": "done"},
        {"id": "partial", "status": "partial"},
    ])
    assert [(g["label"], [e["id"] for e in g["tiles"]]) for g in groups] == [
        ("Completed", ["done"]),
        ("In progress", ["partial"]),
    ]


def test_flatten_stats_recurses_into_list_rows():
    rows = render.flatten_stats({"per_model": [
        {"model": "alpha", "value": 0.2, "ci": [0.1, 0.3]},
    ]})
    assert {r["key"]: r["value"] for r in rows} == {
        "per_model.0.model": "alpha",
        "per_model.0.value": "0.2",
        "per_model.0.ci": "[0.1, 0.3]",
    }


def test_coverage_rows_expose_historical_instrument_and_provider_audit():
    rows = render.coverage_rows({"coverage": [{
        "arm": "a", "task": "lineup", "config": "03_five_principals",
        "stage": "confirmatory", "conditions": "pure", "done_episodes": 30,
        "planned_episodes": 30, "selected_instrument": "legacy-cohort-v1:abc",
        "instrument_ambiguous": False, "served_instruments": {"m": {"calls": 1}},
        "provider_drift": True, "served_model_drift": False,
        "analysis_episodes": 30, "duplicate_episodes": 2,
    }]})
    assert rows[0]["instrument"] == "historical (recovered)"
    assert rows[0]["provenance"] == "provider mix (audit)"
    assert rows[0]["analysis"] == "30"
    assert rows[0]["duplicates"] == "2"


def test_coverage_rows_make_served_model_drift_a_visible_block():
    row = render.coverage_rows({"coverage": [{
        "done_episodes": 2, "planned_episodes": 2, "selected_instrument": "current",
        "instrument_ambiguous": True, "served_model_drift": True,
    }]})[0]
    assert row["instrument"] == "blocked: served-model drift"
    assert row["provenance"] == "model-version mix (blocked)"
