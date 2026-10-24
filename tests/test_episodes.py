"""Canonical saved-episode ingestion, ids, and replay."""
from types import SimpleNamespace

from delegated_influence import episodes, replay


def _sample():
    return SimpleNamespace(
        id="sample|1", uuid="uuid-1",
        metadata={
            "seat_map": {"P1": "gpt-5.5", "P2": "glm-5.2"},
            "seed": 41,
            "gameplay_protocol_version": 2,
            "observation_schema": "bounded-test",
            "experiment_config": "21_arena_5x5",
            "experiment_arm": "five_by_five",
        },
        events=[],
        store={
            "Episode:seat_map": {"P1": "gpt-5.5", "P2": "glm-5.2"},
            "Episode:scores": {"P1": 0, "P2": 1},
            "Episode:events": [{
                "agent": "P1", "type": "pull", "beneficiary": "P2", "forfeit": False,
            }],
        },
    )


def test_loader_attaches_one_canonical_source_context(monkeypatch, tmp_path):
    source = tmp_path / "run-a" / "one.eval"
    source.parent.mkdir()
    source.write_text("stub")
    log = SimpleNamespace(
        samples=[_sample()],
        eval=SimpleNamespace(
            revision=SimpleNamespace(commit="abc123"), created="2026-07-17T00:00:00Z"
        ),
    )
    monkeypatch.setattr(episodes, "read_eval_log", lambda *a, **kw: log)
    rows = episodes.load_episode_rows([tmp_path])
    assert len(rows) == 1
    row = rows[0]
    assert row["run"] == "run-a"
    assert row["eval_file"] == "run-a/one.eval"
    assert row["source_eval_file"] == str(source.resolve())
    assert row["sample_id"] == "sample|1"
    assert row["sample_uuid"] == "uuid-1"
    assert row["record"]["scores"] == {"P1": 0, "P2": 1}
    assert row["record"]["seed"] == 41
    assert row["record"]["gameplay_protocol_version"] == 2
    assert row["record"]["observation_schema"] == "bounded-test"
    assert row["record"]["experiment_config"] == "21_arena_5x5"
    assert row["record"]["experiment_arm"] == "five_by_five"


def test_nested_retry_folder_keeps_first_directory_below_collection_as_run(monkeypatch, tmp_path):
    source = tmp_path / "08_resist" / "retry" / "one.eval"
    source.parent.mkdir(parents=True)
    source.write_text("stub")
    log = SimpleNamespace(samples=[_sample()], eval=SimpleNamespace(revision=None, created="now"))
    monkeypatch.setattr(episodes, "read_eval_log", lambda *a, **kw: log)
    row = episodes.load_episode_rows([tmp_path])[0]
    assert row["run"] == "08_resist"
    assert row["eval_file"] == "08_resist/retry/one.eval"


def test_nested_retry_folder_keeps_run_when_run_directory_is_input(monkeypatch, tmp_path):
    run_root = tmp_path / "08_resist"
    source = run_root / "retry" / "one.eval"
    source.parent.mkdir(parents=True)
    source.write_text("stub")
    log = SimpleNamespace(samples=[_sample()], eval=SimpleNamespace(revision=None, created="now"))
    monkeypatch.setattr(episodes, "read_eval_log", lambda *a, **kw: log)
    row = episodes.load_episode_rows([run_root])[0]
    assert row["run"] == "08_resist"
    assert row["eval_file"] == "retry/one.eval"


def test_glob_input_keeps_the_run_directory_as_owner(monkeypatch, tmp_path):
    run_root = tmp_path / "08_resist"
    source = run_root / "one.eval"
    run_root.mkdir()
    source.write_text("stub")
    log = SimpleNamespace(samples=[_sample()], eval=SimpleNamespace(revision=None, created="now"))
    monkeypatch.setattr(episodes, "read_eval_log", lambda *a, **kw: log)
    monkeypatch.chdir(tmp_path)

    row = episodes.load_episode_rows(["08_resist/*.eval"])[0]

    assert row["run"] == "08_resist"
    assert row["eval_file"] == "one.eval"


def test_loader_marks_stamped_sampling_settings_that_conflict_with_model_events(
        monkeypatch, tmp_path):
    source = tmp_path / "04_run" / "one.eval"
    source.parent.mkdir()
    source.write_text("stub")
    sample = _sample()
    sample.metadata["sampling_instruments"] = {"gpt-5.5": {
        "instrument_version": 1,
        "model_spec": "openrouter/openai/gpt-5.5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "reasoning_effort": None,
        "provider_routing": None,
    }}
    sample.events = [
        SimpleNamespace(event="span_begin", type="turn", span_id="turn", name="P1 · gpt-5.5"),
        SimpleNamespace(
            event="model", span_id="turn", model="openrouter/openai/gpt-5.5",
            config=SimpleNamespace(max_tokens=512, temperature=0.7,
                                   reasoning_effort=None, extra_body=None),
        ),
    ]
    log = SimpleNamespace(samples=[sample], eval=SimpleNamespace(revision=None, created="now"))
    monkeypatch.setattr(episodes, "read_eval_log", lambda *a, **kw: log)

    metadata = episodes.load_episode_rows([tmp_path], resolve_instruments=True)[0]["metadata"]

    assert metadata["sampling_instrument_mismatch"] is True
    assert metadata["stamped_sampling_instruments"]["gpt-5.5"]["max_tokens"] == 1024
    assert metadata["sampling_instruments"]["gpt-5.5"]["status"] == \
        "metadata_event_mismatch"


def test_reader_ids_disambiguate_repeated_samples_by_eval_time():
    rows = [
        {"run": "run", "sample_id": "same|id", "eval_file": "t1_task_x.eval"},
        {"run": "run", "sample_id": "same|id", "eval_file": "t2_task_x.eval"},
    ]
    episodes.assign_reader_ids(rows)
    assert rows[0]["id"] == "run--t1--same_id"
    assert rows[1]["id"] == "run--t2--same_id"


def test_replay_is_the_single_score_and_counter_state_machine():
    seat_map = {"P1": "a", "P2": "b", "P3": "c"}
    events = [
        {"agent": "P1", "type": "pull", "beneficiary": "P2", "forfeit": False},
        {"agent": "P2", "type": "pull", "beneficiary": "P2", "forfeit": False},
        {"agent": "P3", "type": "message", "forfeit": False},
        {"agent": "P1", "type": "say", "forfeit": False},
        {"agent": "P2", "type": "punish", "target": "P1", "forfeit": False},
        {"agent": "P1", "type": "pull", "beneficiary": "P3", "forfeit": True},
        {"agent": "P1", "type": "intent", "forfeit": False},
    ]
    steps, final = replay.events(seat_map, events, self_pull_reward=0.5)
    assert steps[0]["before"]["scores"] == {"P1": 0, "P2": 0, "P3": 0}
    assert final == {
        "scores": {"P1": 0, "P2": 0.5, "P3": 0},
        "public_actions": 3,
        "private_messages": 1,
        "public_messages": 1,
    }


def test_legacy_inspect_catalog_cost_is_not_provider_billing():
    from types import SimpleNamespace
    sample = SimpleNamespace(uuid='old',metadata={},store={
        'Episode:events':[{'usage':[{'prompt_tokens':10,'completion_tokens':5,'total_cost':.12}]}],
        'Episode:seat_map':{'P1':'m'},'Episode:scores':{'P1':0}})
    record = episodes.read_episode(sample)
    usage = record['events'][0]['usage'][0]
    assert usage['total_cost'] is None
    assert usage['estimated_cost_usd'] == .12
    assert sample.store['Episode:events'][0]['usage'][0]['total_cost'] == .12
