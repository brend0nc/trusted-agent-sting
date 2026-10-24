"""Offline fixtures test pilot accounting. They are not live provider validation."""
import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest
import yaml


def config(tmp_path, *, arms=None, name="target"):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({
        "cost": {"avg_tokens_per_turn": 1000, "usd_per_mtok": 2},
        "budget": {"max_estimated_usd": 100},
        "arms": arms or [{"name": "talk", "task": "lineup", "seats": ["gpt-5.5", "glm-5.2"],
                          "reps": 3, "rounds": 2, "actions_per_round": 2,
                          "model_configs": {"gpt-5.5": {"max_tokens": 777}}}],
    }))
    return path


def routes(tmp_path):
    from delegated_influence.task import sampling_instrument
    data = {"schema_version": 1, "verified_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model in ("gpt-5.5", "glm-5.2"):
        data["models"][model] = {
            "model": sampling_instrument(model)["model_spec"], "provider_tag": "fixture",
            "provider_name": "Fixture", "context_length": 100000, "max_completion_tokens": 100000,
            "price_ceiling": {key: "0.000001" for key in
                              ("prompt", "completion", "input_cache_read", "input_cache_write", "request")},
            "supported_parameters": ["max_tokens", "reasoning", "temperature"],
            "reasoning": {}, "evidence": "https://example.invalid/offline-fixture"}
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(data))
    return path


def prepare(tmp_path, **kwargs):
    from delegated_influence.pilot import prepare
    kwargs.setdefault("spending_manifest", routes(tmp_path))
    kwargs.setdefault("ledger", tmp_path / "cap.sqlite3")
    return prepare([config(tmp_path)], tmp_path / "pilot", **kwargs)


def episode(manifest, tmp_path, *, cost=.01):
    from delegated_influence.experiment import build_tasks, load_config
    cfg = load_config(manifest["pilot_config"])
    sample = build_tasks(cfg, config_name="pilot")[0][1].dataset[0]
    md = deepcopy(sample.metadata)
    md["spending_instrument"] = manifest["conditions"][0].get("spending_instruments", {})
    events = []
    for rnd in range(1, md["rounds"] + 1):
        for action in range(1, md["actions_per_round"] + 1):
            for seat in md["seat_map"]:
                cid = f"ep:{len(events)}"
                usage = {"total_cost": cost, "provider_usage": {"cost": cost},
                         "cost_source": "openrouter_response_usage_cost",
                         "provider_request_id": f"gen-{cid}", "http_requests": 1,
                         "served_provider": "Fixture",
                         "provider_request_source": "http_request_body",
                         "prompt_tokens": rnd * 100, "completion_tokens": 10,
                         "cache_read_tokens": 20, "cache_write_tokens": 0,
                         "reasoning_tokens": 5, "latency_seconds": rnd,
                         "input_characters": rnd * 400, "observation_characters": rnd * 80}
                events.append({"round": rnd, "action_number": action, "agent": seat,
                               "type": "pull", "beneficiary": seat, "forfeit": False,
                               "calls": [{"call_id": cid, "usage": usage}]})
    saved = {"schema_version": "native-episode-v1", "status": "complete", "episode_id": "ep",
             "sample_id": sample.id, "metadata": md,
             "record": {**md, "events": events, "scores": {s: 0 for s in md["seat_map"]}}}
    path = tmp_path / "fixture.episode.json"
    path.write_text(json.dumps(saved))
    return path, saved


def test_prepare_retains_full_context_settings_and_never_marks_run(tmp_path):
    from delegated_influence.pilot import prepare
    source = config(tmp_path)
    original = source.read_bytes()
    manifest = prepare([source], tmp_path / "pilot")
    cfg = yaml.safe_load(open(manifest["pilot_config"]))
    arm = cfg["arms"][0]
    assert arm["reps"] == 1
    assert (arm["rounds"], arm["actions_per_round"]) == (2, 2)
    assert arm["model_configs"]["gpt-5.5"]["max_tokens"] == 777
    assert cfg["stage"] == "exploratory"
    assert manifest["status"] == "planned_unrun"
    assert manifest["guard_configured"] is False
    assert manifest["conditions"][0]["target_episodes"] == 3
    assert source.read_bytes() == original
    from delegated_influence.experiment import build_tasks
    pilot_sample = build_tasks(cfg, config_name="pilot")[0][1].dataset[0]
    assert pilot_sample.metadata["seed"] == 100041
    assert "pilot" in str(pilot_sample.id)


def test_prepare_deduplicates_equivalent_conditions_and_respects_whole_episode_allowance(tmp_path):
    from delegated_influence.pilot import prepare
    one = config(tmp_path)
    two = config(tmp_path, name="duplicate")
    manifest = prepare([one, two], tmp_path / "pilot", allowance_usd=.031)
    assert len(manifest["conditions"]) == 1
    assert manifest["conditions"][0]["target_episodes"] == 6
    assert not manifest["conditions"][0]["selected"]
    assert manifest["guard_configured"] is False
    assert yaml.safe_load(open(manifest["pilot_config"]))["arms"] == []


def test_prepare_requires_shared_guard_and_does_not_overwrite_artifacts(tmp_path):
    manifest = prepare(tmp_path, spending_manifest=tmp_path / "routes.json", ledger=tmp_path / "cap.sqlite3")
    cfg = yaml.safe_load(open(manifest["pilot_config"]))
    assert cfg["spending"]["cap_usd"] == 25
    assert cfg["spending"]["ledger"] == str(tmp_path / "cap.sqlite3")
    assert manifest["guard_configured"] is True
    assert "launch_ready" not in manifest  # preparation cannot establish access or remaining funds
    with pytest.raises(FileExistsError):
        prepare(tmp_path)


def test_calibration_requires_all_calls_and_late_rounds_then_scales_complete_games(tmp_path):
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    report = calibrate(manifest, [path])
    assert report["status"] == "covered"
    assert report["measured"]["provider_cost_usd"] == pytest.approx(.08)
    assert report["target_point_estimate_usd"] == pytest.approx(.24)
    assert report["coverage"]["covered_model_round_cells"] == 4
    late = next(row for row in report["model_rounds"] if row["round"] == 2)
    assert late["prompt_tokens"]["mean"] == 200
    assert late["context_tokens"]["max"] == 220
    assert late["completion_tokens"]["sum"] == 20  # reasoning is already included
    saved["record"]["events"] = saved["record"]["events"][:4]
    path.write_text(json.dumps(saved))
    report = calibrate(manifest, [path])
    assert report["status"] == "incomplete"
    assert report["target_point_estimate_usd"] is None


@pytest.mark.parametrize("missing", ["provider_usage", "provider_request_id", "http_requests"])
def test_cost_flags_or_offline_cost_alone_cannot_prove_live_billing(tmp_path, missing):
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    saved["record"]["events"][0]["calls"][0]["usage"].pop(missing)
    path.write_text(json.dumps(saved))
    report = calibrate(manifest, [path])
    assert report["measured"]["provider_cost_usd"] is None
    assert report["measured"]["unknown_cost_calls"] == 1
    assert report["measured"]["known_provider_cost_usd"] == pytest.approx(.07)
    assert report["target_point_estimate_usd"] is None


def test_unrun_has_unknown_cost_and_no_estimate(tmp_path):
    from delegated_influence.pilot import calibrate
    report = calibrate(prepare(tmp_path), [])
    assert report["status"] == "unrun"
    assert report["measured"]["provider_cost_usd"] is None
    assert report["target_point_estimate_usd"] is None


def test_journal_and_export_count_once_and_keep_prompt_bodies_on_disk(tmp_path):
    from delegated_influence.journal import Journal
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    with Journal(tmp_path / "run.sqlite3") as journal:
        md = saved["metadata"]
        journal.start_episode("ep", md, 1, 25, sample_id=saved["sample_id"])
        sequence = {}
        for index, event in enumerate(saved["record"]["events"]):
            seat = event["agent"]
            seq = sequence.get(seat, 0)
            sequence[seat] = seq + 1
            cid, _ = journal.begin_call("ep", seat, seq, [{"role": "user", "content": "SECRET PROMPT"}])
            usage = event["calls"][0]["usage"]
            usage["provider_request_body"] = "SECRET BODY"
            journal.finish_call(cid, "SECRET OUTPUT", usage, None)
            event["calls"] = [journal.call(cid)]
            journal.save_event("ep", index, event)
        journal.complete("ep", saved["record"])
        journal.export_episode("ep", path)
        report = calibrate(manifest, [tmp_path, path])
        assert report["measured"]["calls"] == 8
        assert report["target_point_estimate_usd"] == pytest.approx(.24)
        assert "SECRET" not in json.dumps(report)


def test_partial_journal_retains_uncertain_call_without_mutating_status(tmp_path):
    from delegated_influence.journal import Journal
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    path.unlink()
    with Journal(tmp_path / "run.sqlite3") as journal:
        journal.start_episode("ep", saved["metadata"], 1, 25, sample_id=saved["sample_id"])
        journal.begin_call("ep", "P1", 0, [])
        report = calibrate(manifest, [tmp_path])
        assert report["measured"]["calls"] == 1
        assert report["measured"]["unknown_cost_calls"] == 1
        assert report["measured"]["unassigned_calls"] == 1
        assert report["measured"]["uncertain_calls"] == 1
        assert report["target_point_estimate_usd"] is None
        assert journal.db.execute("SELECT status FROM calls").fetchone()[0] == "pending"


def test_incomplete_additional_game_cannot_be_silently_dropped_from_estimate(tmp_path):
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    saved["episode_id"] = "incomplete"
    saved["record"]["events"] = saved["record"]["events"][:4]
    (tmp_path / "partial.episode.json").write_text(json.dumps(saved))
    report = calibrate(manifest, [tmp_path])
    assert report["measured"]["calls"] == 12
    assert report["target_point_estimate_usd"] is None


def test_sampling_or_stage_mismatch_cannot_supply_coverage(tmp_path):
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    saved["metadata"]["sampling_instruments"]["gpt-5.5"]["max_tokens"] = 1
    path.write_text(json.dumps(saved))
    report = calibrate(manifest, [path])
    assert report["coverage"]["ignored_episodes"] == 1
    assert report["measured"]["calls"] == 0
    assert report["target_point_estimate_usd"] is None


def test_changed_guard_route_keeps_measured_cost_but_blocks_extrapolation(tmp_path):
    from delegated_influence.pilot import calibrate
    manifest = prepare(tmp_path)
    path, saved = episode(manifest, tmp_path)
    saved["metadata"]["spending_instrument"] = {"gpt-5.5": {"provider_routing": {"only": ["other"]}}}
    path.write_text(json.dumps(saved))
    report = calibrate(manifest, [path])
    assert report["measured"]["provider_cost_usd"] == pytest.approx(.08)
    assert report["target_point_estimate_usd"] is None
    assert manifest["spending_manifest"]["sha256"]
