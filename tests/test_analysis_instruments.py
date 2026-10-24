"""Registered comparisons must not mistake a changed model instrument for a treatment."""
from copy import deepcopy
import re

import pytest

from delegated_influence import report
from test_report import allocation_stat, cost_stat, intent_reactivity_stat, population_stat, rec


def attach_instruments(rows):
    for row in rows:
        models = {seat["model"] for seat in row["seats"]}
        row["sampling_instruments"] = {
            model: {"model_spec": model, "max_tokens": 1024,
                    "reasoning_effort": "none", "temperature": None,
                    "provider_routing": None}
            for model in models
        }
        row["served_instruments"] = {
            model: {"served_models": {model + "-v1": 10}, "providers": {"host-a": 10}}
            for model in models
        }
    return rows


def comparison_cases():
    return [
        (report.action_allocation_factorial,
         [allocation_stat(41, talk, punish, 0.4 if talk else 0.2)
          for talk in (False, True) for punish in (False, True)]),
        (report.cooperation_cost_analysis,
         [cost_stat(41, reward, 1 - reward / 2) for reward in (0, .25, .5, .75, 1)]),
        (report.population_composition_analysis,
         [population_stat(41, "diverse", .2), population_stat(41, "five_by_five", .4)]),
        (report.intent_reactivity_analysis,
         [intent_reactivity_stat(41, False, .4, .2),
          intent_reactivity_stat(41, True, .6, .3)]),
    ]


def assert_excluded(analysis, rows, reason="incompatible.*instrument"):
    result = analysis(rows, trials=20)
    assert len(result) == 1
    assert result[0]["status"] == "excluded"
    assert re.search(reason, result[0]["reason"])
    assert result[0]["n_episodes"] == len(rows)
    # Only the report boundary changed. The underlying scientific guard remains strict.
    with pytest.raises(ValueError, match=reason):
        report._check_comparison_instruments(rows, "test")


@pytest.mark.parametrize("analysis,rows", comparison_cases())
@pytest.mark.parametrize("field,value", [
    ("reasoning_effort", "high"),
    ("max_tokens", 2048),
    ("provider_routing", {"order": ["other-host"], "allow_fallbacks": False}),
])
def test_registered_comparisons_reject_changed_sampling(analysis, rows, field, value):
    rows = attach_instruments(deepcopy(rows))
    common = set.intersection(*(set(row["sampling_instruments"]) for row in rows))
    model = sorted(common)[0]
    rows[-1]["sampling_instruments"][model][field] = value
    assert_excluded(analysis, rows)


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_reject_served_version_change(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    common = set.intersection(*(set(row["sampling_instruments"]) for row in rows))
    model = sorted(common)[0]
    rows[-1]["served_instruments"][model]["served_models"] = {model + "-v2": 10}
    assert_excluded(analysis, rows)


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_keep_host_counts_as_audit_data(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    for instrument in rows[-1]["served_instruments"].values():
        instrument["providers"] = {"host-b": 6, "host-c": 3}
        instrument["served_models"] = {key: 9 for key in instrument["served_models"]}
    assert len(analysis(rows, trials=20)) == 1


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_reject_partial_instrument_evidence(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    rows[-1].pop("sampling_instruments")
    assert_excluded(analysis, rows)


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_keep_consistently_unknown_legacy_rows(analysis, rows):
    assert len(analysis(deepcopy(rows), trials=20)) == 1


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_reject_recorded_sampling_mismatch(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    rows[-1]["sampling_instrument_mismatch"] = True
    assert_excluded(analysis, rows)


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_human_board_conditions_never_mix(analysis, rows):
    rows = deepcopy(rows)
    rows[-1]["human_board"] = True
    assert analysis(rows, trials=20) == []


def test_measured_rows_keep_recorded_instruments_and_mismatch_status():
    metadata = {
        "model_configs": {"m": {"reasoning_effort": "none"}},
        "sampling_instruments": {"m": {"reasoning_effort": "high"}},
        "stamped_sampling_instruments": {"m": {"reasoning_effort": "none"}},
        "served_instruments": {"m": {"served_models": {"m-v1": 2}}},
        "sampling_instrument_mismatch": True,
        "runtime_instrument": {"runner": "native", "adapter_version": 1, "sdk_retries": 0},
        "runtime_environment": {"python": "3.13.8", "packages": {"httpx": "0.28.1"}},
    }
    record = rec([], {"P1": 0, "P2": 0}, {"P1": "m", "P2": "m"},
                 metadata=metadata, human_board=True, seeded_favours=1)
    measured = report.measure_episode(record)
    for key, value in metadata.items():
        assert measured.get(key) == value
    assert measured.get("human_board") is True
    assert measured.get("seeded_favours") == 1


@pytest.mark.parametrize("analysis,rows", comparison_cases())
@pytest.mark.parametrize("replacement", [
    None,
    {"runner": "native", "adapter_version": 2, "sdk_retries": 0},
    {"runner": "native", "adapter_version": 1, "sdk_retries": 2},
])
def test_registered_comparisons_reject_changed_runtime(analysis, rows, replacement):
    rows = deepcopy(rows)
    for row in rows:
        row["runtime_instrument"] = {"runner": "native", "adapter_version": 1, "sdk_retries": 0}
    rows[-1]["runtime_instrument"] = replacement
    assert_excluded(analysis, rows, "incompatible.*runtime instrument")


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_accept_consistent_native_runtime(analysis, rows):
    rows = deepcopy(rows)
    for row in rows:
        row["runtime_instrument"] = {"runner": "native", "adapter_version": 1, "sdk_retries": 0}
    assert len(analysis(rows, trials=20)) == 1


@pytest.mark.parametrize("analysis,rows", comparison_cases())
def test_registered_comparisons_keep_price_changes_out_of_pinned_guard_identity(analysis, rows):
    rows = attach_instruments(deepcopy(rows))
    for index, row in enumerate(rows):
        row['spending_instrument'] = {model: {'provider_routing': {
            'only': ['host-a'], 'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': 2 + index, 'completion': 10 + index, 'request': 0}}}
            for model in row['sampling_instruments']}
    before = deepcopy(rows)
    result = analysis(rows, trials=20)
    assert len(result) == 1
    assert result[0].get('status') != 'excluded'
    assert rows == before  # All original price evidence survives the comparison.


@pytest.mark.parametrize('field,replacement', [
    ('only', ['host-b']), ('allow_fallbacks', True), ('require_parameters', False),
    ('data_collection', 'deny'),
])
def test_price_projection_preserves_nonfinancial_guard_changes(field, replacement):
    analysis, source = comparison_cases()[2]
    rows = attach_instruments(deepcopy(source))
    for row in rows:
        row['spending_instrument'] = {model: {'provider_routing': {
            'only': ['host-a'], 'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': 2, 'completion': 10, 'request': 0}}}
            for model in row['sampling_instruments']}
    common = set(rows[0]['sampling_instruments']) & set(rows[-1]['sampling_instruments'])
    rows[-1]['spending_instrument'][sorted(common)[0]]['provider_routing'][field] = replacement
    assert_excluded(analysis, rows)


def test_factorial_rejects_version_drift_between_paired_seeds():
    rows = comparison_cases()[0][1]
    later = deepcopy(rows)
    for row in later:
        row["seed"] = 42
    rows = attach_instruments(rows + later)
    for row in rows[4:]:
        for instrument in row["sampling_instruments"].values():
            instrument["reasoning_effort"] = "high"
    assert_excluded(report.action_allocation_factorial, rows)
