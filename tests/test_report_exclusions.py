"""One invalid scientific comparison must not remove valid report blocks."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from delegated_influence import report
from test_analysis_instruments import attach_instruments, comparison_cases
from test_report import population_stat, rec, SEATS


CASES = comparison_cases() + [(report.same_model_favoritism_summary, [
    population_stat(41, "five_by_five", .4), population_stat(42, "five_by_five", .5)])]


def separated_blocks(rows):
    good = attach_instruments(deepcopy(rows))
    bad = attach_instruments(deepcopy(rows))
    for row in good:
        row["observation_schema"] = "good-schema"
    for row in bad:
        row["observation_schema"] = "bad-schema"
    bad[-1]["sampling_instrument_mismatch"] = True
    return good, bad


@pytest.mark.parametrize("analysis,rows", CASES)
def test_invalid_block_is_reported_and_valid_block_survives(analysis, rows):
    good, bad = separated_blocks(rows)
    result = analysis(bad + good, trials=20)
    valid = [row for row in result if row.get("status") != "excluded"]
    excluded = [row for row in result if row.get("status") == "excluded"]
    assert valid == analysis(good, trials=20)
    assert len(excluded) == 1
    row = excluded[0]
    assert "sampling mismatch" in row["reason"]
    assert row["n_episodes"] == len(bad)
    assert row["n_seeds"] == len({e["seed"] for e in bad})
    assert row["block"]["observation_schema"] == "bad-schema"
    assert sum(cell["n_episodes"] for cell in row["conditions"]) == len(bad)
    assert json.loads(json.dumps(result))  # exclusion metadata is also an export contract


@pytest.mark.parametrize("analysis,rows", CASES)
def test_only_invalid_block_is_explicit_and_empty_input_stays_empty(analysis, rows):
    _, bad = separated_blocks(rows)
    assert analysis([], trials=20) == []
    result = analysis(bad, trials=20)
    assert len(result) == 1
    assert result[0]["status"] == "excluded"
    assert "arm_means" not in result[0]


@pytest.mark.parametrize("analysis,rows", CASES)
def test_duplicate_block_is_excluded_with_every_duplicate_counted(analysis, rows):
    good, bad = separated_blocks(rows)
    bad[-1].pop("sampling_instrument_mismatch")
    bad.append(deepcopy(bad[0]))
    result = analysis(bad + good, trials=20)
    excluded = [row for row in result if row.get("status") == "excluded"]
    assert len(excluded) == 1
    assert "duplicate" in excluded[0]["reason"]
    assert excluded[0]["n_episodes"] == len(bad)
    assert len(result) == 2


@pytest.mark.parametrize("analysis,rows", CASES)
def test_duplicate_partial_block_is_not_silently_dropped(analysis, rows):
    _, bad = separated_blocks(rows)
    duplicate = deepcopy(bad[0])
    duplicate.pop("sampling_instrument_mismatch", None)
    result = analysis([duplicate, deepcopy(duplicate)], trials=20)
    assert len(result) == 1 and result[0]["status"] == "excluded"
    assert result[0]["n_episodes"] == 2
    assert "duplicate" in result[0]["reason"]


@pytest.mark.parametrize("analysis,rows", CASES)
def test_runtime_environment_is_provenance_not_a_scientific_treatment(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    for i, row in enumerate(rows):
        row["runtime_instrument"] = {"runner": "native", "adapter_version": 1, "sdk_retries": 0,
                                     "python": f"3.13.{i}", "packages": {"httpx": str(i)}}
    result = analysis(rows, trials=20)
    assert len(result) == 1 and result[0].get("status") != "excluded"


def test_paper_claims_keep_exclusions_and_do_not_treat_them_as_multiple_valid_blocks():
    analysis, rows = CASES[0]
    good, bad = separated_blocks(rows)
    for row in good + bad:
        row["experiment_config"] = "20_arena_25"
    result = report.paper_claim_summary(good + bad, trials=20)
    assert result["claim_2"]["estimate"] == analysis(good, trials=20)[0]["communication_main_effect"]
    assert len(result["excluded_comparison_blocks"]) == 1
    only_bad = report.paper_claim_summary(bad, trials=20)
    assert only_bad["claim_2"]["status"] == "unavailable"
    assert len(only_bad["excluded_comparison_blocks"]) == 1
    assert only_bad["claim_3_family_size"] == 4


def test_ambiguous_paper_aggregate_is_unavailable_and_retains_each_valid_block():
    analysis, rows = CASES[0]
    first, second = separated_blocks(rows)
    second[-1].pop("sampling_instrument_mismatch")
    for row in first + second:
        row["experiment_config"] = "20_arena_25"
    result = report.paper_claim_summary(first + second, trials=20)
    assert result["claim_2"]["status"] == "unavailable"
    assert result["claim_2"]["estimate"] is None
    assert len(result["comparison_results"]["factorial"]) == 2
    assert result["ambiguous_comparisons"][0]["n_eligible_blocks"] == 2
    assert "multiple" in result["ambiguous_comparisons"][0]["reason"]


@pytest.mark.parametrize("analysis,rows", CASES)
def test_unexpected_validation_errors_are_not_reported_as_data_exclusions(analysis, rows):
    with patch.object(report, "_check_comparison_instruments", side_effect=ValueError("unexpected bug")):
        with pytest.raises(ValueError, match="unexpected bug"):
            analysis(rows, trials=20)


def test_report_renders_excluded_rows_before_accessing_numeric_results():
    measured = report.measure_episode(rec([], {who: 0 for who in SEATS}, SEATS))
    patches = []
    for analysis, rows in CASES:
        good, bad = separated_blocks(rows)
        patches.append(patch.object(report, analysis.__name__, return_value=analysis(good + bad, trials=20)))
    from contextlib import ExitStack
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        rendered = report.format_report([measured])
    assert rendered.count("EXCLUDED") >= 4
    assert "sampling mismatch" in rendered
    assert "bad-schema" in rendered
    assert "primary communication effect" in rendered


def test_paper_cli_exports_valid_results_and_exclusions(monkeypatch, capsys):
    records = []
    for analysis, rows in CASES:
        if analysis not in (report.action_allocation_factorial, report.intent_reactivity_analysis):
            continue
        good, bad = separated_blocks(rows)
        config = ("20_arena_25" if analysis == report.action_allocation_factorial
                  else "23_deception_intent_validation")
        for row in good + bad:
            row["experiment_config"] = config
        records.extend(good + bad)
    monkeypatch.setattr(report.episodes, "load_episodes", lambda paths: records)
    monkeypatch.setattr(report.pool, "load_arms", lambda path: [])
    monkeypatch.setattr(report.pool, "match_arms", lambda records, arms: None)
    monkeypatch.setattr(report, "measure_episode", lambda row: row)
    monkeypatch.setattr("sys.argv", ["report", "unused", "--paper-claims", "--pool", "all"])
    report.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["paper_claims"]["claim_2"]["estimate"] is not None
    assert len(payload["paper_claims"]["excluded_comparison_blocks"]) == 1
    assert len(payload["intent_reactivity"]) == 2
    assert sum(row.get("status") == "excluded" for row in payload["intent_reactivity"]) == 1
