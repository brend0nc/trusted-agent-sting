"""Canonical experiment planning and identity."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from delegated_influence import experiment, run


ROOT = Path(__file__).resolve().parent.parent


def test_run_reexports_the_canonical_planner():
    assert run.build_tasks is experiment.build_tasks
    assert run.expand_models is experiment.expand_models
    assert run.estimate_cost is experiment.estimate_cost
    assert run.enforce_budget is experiment.enforce_budget


def test_every_planned_sample_has_a_self_checking_identity():
    cfg = experiment.load_config(ROOT / "configs" / "02c_luna_background_validation.yaml")
    tasks = experiment.build_tasks(cfg, config_name="02c_luna_background_validation")
    for _name, task_value in tasks:
        for sample in task_value.dataset:
            assert sample.metadata["experiment_config"] == "02c_luna_background_validation"
            assert sample.metadata["experiment_arm"] == "luna_background_validation"
            assert sample.metadata["experiment_stage"] == "exploratory"
            assert sample.metadata["experiment_identity"] == experiment.episode_identity(
                sample.id, sample.metadata
            )


def test_identity_is_stable_across_mapping_order_and_legacy_missing_false_values():
    left = {"seat_map": {"P1": "a", "P2": "b"}, "seed": 4,
            "model_configs": {"a": {"max_tokens": 10}}, "messages_off": False}
    right = {"messages_off": 0, "model_configs": {"a": {"max_tokens": 10}},
             "seed": 4, "seat_map": {"P2": "b", "P1": "a"}}
    assert experiment.episode_identity("sample", left) == experiment.episode_identity("sample", right)


@pytest.mark.parametrize("field", experiment.TREATMENT_FLAGS)
def test_every_treatment_changes_identity(field):
    assert experiment.episode_identity("sample", {}) != experiment.episode_identity(
        "sample", {field: True}
    )


def test_identity_distinguishes_background_sampling_and_principal_role():
    base = {"seat_map": {"P1": "gpt-5.5", "P2": "glm-5.2"}}
    luna = {"seat_map": {"P1": "gpt-5.5", "P2": "gpt-5.6-luna"}}
    override = {**base, "model_configs": {"glm-5.2": {"max_tokens": 256}}}
    principals = {**base, "principal_seats": ["P1"]}
    identities = {experiment.episode_identity("sample", value)
                  for value in (base, luna, override, principals)}
    assert len(identities) == 4


def test_identity_distinguishes_observation_instrument():
    base = {"seat_map": {"P1": "gpt-5.5"}}
    bounded = {**base, "observation_schema": "bounded_current_previous_summary_v1"}
    legacy = {**base, "observation_schema": "full_ledger_legacy"}
    assert experiment.episode_identity("sample", bounded) != experiment.episode_identity(
        "sample", legacy
    )
    assert experiment.episode_identity("sample", base) not in {
        experiment.episode_identity("sample", bounded),
        experiment.episode_identity("sample", legacy),
    }


def test_identity_distinguishes_gameplay_protocol_version():
    base = {"seat_map": {"P1": "gpt-5.5"}}
    assert experiment.episode_identity(
        "sample", {**base, "gameplay_protocol_version": 1}
    ) != experiment.episode_identity(
        "sample", {**base, "gameplay_protocol_version": 2}
    )


def test_identity_distinguishes_effective_registry_sampling_settings(monkeypatch):
    cfg = {"arms": [{"name": "a", "task": "lineup", "seats": ["gpt-5.6-luna"]}]}
    first = experiment.build_tasks(cfg)[0][1].dataset[0]
    monkeypatch.setitem(experiment.models.MODEL_REGISTRY["gpt-5.6-luna"], "max_tokens", 999)
    second = experiment.build_tasks(cfg)[0][1].dataset[0]
    assert experiment.episode_identity(first.id, first.metadata) != experiment.episode_identity(
        second.id, second.metadata
    )


def test_identity_distinguishes_provider_routing(monkeypatch):
    cfg = {"arms": [{"name": "a", "task": "lineup", "seats": ["glm-5.2"]}]}
    first = experiment.build_tasks(cfg)[0][1].dataset[0]
    monkeypatch.setitem(
        experiment.models.MODEL_REGISTRY["glm-5.2"],
        "provider_pref", {"order": ["fixed-host"], "allow_fallbacks": False},
    )
    second = experiment.build_tasks(cfg)[0][1].dataset[0]
    assert experiment.episode_identity(first.id, first.metadata) != experiment.episode_identity(
        second.id, second.metadata
    )


def test_legacy_sampling_instrument_is_recovered_from_named_turn_events():
    span = SimpleNamespace(
        event="span_begin", type="turn", span_id="turn-1", name="P1 · gpt-5.5"
    )
    config = SimpleNamespace(
        max_tokens=1024, temperature=0.7, reasoning_effort=None, extra_body=None
    )
    model = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/openai/gpt-5.5", config=config
    )
    sample = SimpleNamespace(
        metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model]
    )
    assert experiment.infer_sampling_instruments(sample) == {"gpt-5.5": {
        "instrument_version": "legacy_recovered_v1",
        "model_spec": "openrouter/openai/gpt-5.5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "reasoning_effort": None,
        "provider_routing": None,
    }}


def test_historical_sampling_recovery_does_not_use_todays_version(monkeypatch):
    span = SimpleNamespace(
        event="span_begin", type="turn", span_id="turn-1", name="P1 · gpt-5.5"
    )
    model = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/openai/gpt-5.5",
        config=SimpleNamespace(
            max_tokens=1024, temperature=0.7, reasoning_effort=None, extra_body=None,
        ),
    )
    sample = SimpleNamespace(metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model])
    first = experiment.infer_sampling_instruments(sample)

    monkeypatch.setattr(experiment.task, "GAMEPLAY_SAMPLING_INSTRUMENT_VERSION", 999)

    assert experiment.infer_sampling_instruments(sample) == first


def test_stamped_sampling_uses_its_recorded_version_when_checking_events(monkeypatch):
    stamped = {
        "instrument_version": 1,
        "model_spec": "openrouter/openai/gpt-5.5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "reasoning_effort": None,
        "provider_routing": None,
    }
    sample = SimpleNamespace(
        metadata={
            "seat_map": {"P1": "gpt-5.5"},
            "sampling_instruments": {"gpt-5.5": stamped},
        },
        events=[
            SimpleNamespace(
                event="span_begin", type="turn", span_id="turn-1", name="P1 · gpt-5.5",
            ),
            SimpleNamespace(
                event="model", span_id="turn-1", model="openrouter/openai/gpt-5.5",
                config=SimpleNamespace(
                    max_tokens=1024, temperature=0.7, reasoning_effort=None, extra_body=None,
                ),
            ),
        ],
    )
    monkeypatch.setattr(experiment.task, "GAMEPLAY_SAMPLING_INSTRUMENT_VERSION", 999)

    assert experiment.infer_sampling_instruments(sample) == {"gpt-5.5": stamped}


def test_stamped_sampling_instrument_is_checked_against_recorded_calls():
    span = SimpleNamespace(
        event="span_begin", type="turn", span_id="turn-1", name="P1 · gpt-5.5"
    )
    config = SimpleNamespace(
        max_tokens=512, temperature=0.7, reasoning_effort=None, extra_body=None
    )
    stamped = {
        "instrument_version": 1,
        "model_spec": "openrouter/openai/gpt-5.5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "reasoning_effort": None,
        "provider_routing": None,
    }
    sample = SimpleNamespace(
        metadata={
            "seat_map": {"P1": "gpt-5.5"},
            "sampling_instruments": {"gpt-5.5": stamped},
        },
        events=[span, SimpleNamespace(
            event="model", span_id="turn-1", model="openrouter/openai/gpt-5.5",
            config=config,
        )],
    )
    result = experiment.infer_sampling_instruments(sample)["gpt-5.5"]
    assert result["status"] == "metadata_event_mismatch"
    assert result["stamped"]["max_tokens"] == 1024
    assert result["observed"][0]["max_tokens"] == 512


def test_legacy_protocol_fingerprint_changes_with_the_exact_system_prompt():
    span = SimpleNamespace(event="span_begin", type="turn", id="turn-1", name="P1 · gpt-5.5")

    def sample(prompt):
        model = SimpleNamespace(
            event="model", span_id="turn-1",
            input=[SimpleNamespace(role="system", content=prompt)],
        )
        return SimpleNamespace(
            metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model]
        )

    first = experiment.infer_gameplay_protocol(sample("rules version one"))
    second = experiment.infer_gameplay_protocol(sample("rules version two"))
    assert first["status"] == "legacy_recovered"
    assert first["system_prompt_sha256"] != second["system_prompt_sha256"]


def test_legacy_protocol_fingerprint_ignores_seat_allocation():
    def sample(seat):
        span = SimpleNamespace(
            event="span_begin", type="turn", id="turn-1", name=f"{seat} · gpt-5.5"
        )
        model = SimpleNamespace(
            event="model", span_id="turn-1",
            input=[SimpleNamespace(
                role="system", content=f"You are {seat}. Participants: P1, P2. Same rules."
            )],
        )
        return SimpleNamespace(metadata={"seat_map": {seat: "gpt-5.5"}}, events=[span, model])

    assert experiment.infer_gameplay_protocol(sample("P1")) == \
        experiment.infer_gameplay_protocol(sample("P2"))


def test_legacy_protocol_fingerprint_normalizes_known_persona_conditions():
    def sample(clause):
        span = SimpleNamespace(event="span_begin", type="turn", id="turn", name="P1 · gpt-5.5")
        model = SimpleNamespace(
            event="model", span_id="turn",
            input=[SimpleNamespace(role="system", content="same rules" + clause),
                   SimpleNamespace(role="user", content="Round 1 of 5\nScores: P1: 0")],
        )
        return SimpleNamespace(metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model])

    neutral = experiment.infer_gameplay_protocol(sample(""))
    inoculated = experiment.infer_gameplay_protocol(
        sample(experiment.seat.INOCULATION_CLAUSE)
    )
    assert neutral == inoculated


def test_legacy_protocol_fingerprint_detects_observation_structure_changes():
    def sample(heading):
        span = SimpleNamespace(event="span_begin", type="turn", id="turn", name="P1 · gpt-5.5")
        model = SimpleNamespace(
            event="model", span_id="turn",
            input=[SimpleNamespace(role="system", content="same rules"),
                   SimpleNamespace(role="user", content=f"{heading}\nScores: P1: 0")],
        )
        return SimpleNamespace(metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model])

    assert experiment.infer_gameplay_protocol(sample("Round 1 of 5")) != \
        experiment.infer_gameplay_protocol(sample("Phase 1 of 5"))


def test_legacy_protocol_fingerprint_detects_scoreboard_removal():
    def sample(include_scores):
        span = SimpleNamespace(event="span_begin", type="turn", id="turn", name="P1 · gpt-5.5")
        user = "Round 1 of 5"
        if include_scores:
            user += "\nScores: P1: 0"
        model = SimpleNamespace(
            event="model", span_id="turn",
            input=[SimpleNamespace(role="system", content="same rules"),
                   SimpleNamespace(role="user", content=user)],
        )
        return SimpleNamespace(metadata={"seat_map": {"P1": "gpt-5.5"}}, events=[span, model])

    assert experiment.infer_gameplay_protocol(sample(True)) != \
        experiment.infer_gameplay_protocol(sample(False))


def test_current_prompt_surface_is_tied_to_explicit_instrument_versions():
    """Any agent-visible prompt edit must deliberately bump a version and this golden digest."""
    agents = ["P1", "P2", "P3"]
    systems = [
        experiment.seat.build_system_prompt("P1", agents, 5, 5),
        experiment.seat.build_system_prompt(
            "P1", agents, 5, 5, persona="attacker", messages_off=True
        ),
        experiment.seat.build_system_prompt(
            "P1", agents, 5, 5, persona="inoculated", self_pull_reward=0.5,
            public_chat=True, punish_enabled=True,
        ),
    ]
    observation_prompt = experiment.seat.render_observation({
        "agent": "P1", "agents": agents, "round": 4, "rounds": 5,
        "action_number": 2, "actions_per_round": 5,
        "scores": {"P1": 2, "P2": 3, "P3": 1},
        "can_message": ["P2"], "punish_enabled": True,
        "ledger_summary": {
            "through_round": 2,
            "pulls": [{"puller": "P1", "beneficiary": "P2",
                       "round_counts": [{"round": 1, "count": 2}]}],
            "random_pulls": [], "credits": [], "punishments": [],
            "mutual_streaks": [{"left": "P1", "right": "P2", "rounds": 2,
                                "through_round": 3}],
        },
        "ledger": [
            {"round": 3, "action_number": 1, "puller": "P2", "beneficiary": "P1",
             "credit": "P3", "forced": False},
            {"round": 4, "action_number": 1, "action": "punish", "agent": "P3",
             "target": "P2"},
        ],
        "chat": [{"round": 4, "action_number": 1, "speaker": "P2", "text": "public"}],
        "inbox": [{"seq": 0, "round": 3, "action_number": 2, "sender": "P2",
                   "to": "P1", "text": "private"}],
        "outbox": [],
    })
    encoded = json.dumps(systems + [observation_prompt], separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    assert (
        experiment.task.GAMEPLAY_PROTOCOL_VERSION,
        experiment.observation.CURRENT_SCHEMA,
        digest,
        ) == (
            4,
            "bounded_current_previous_summary_v1",
            "1dead192b98f54038f18d6b5a22f44b9ae35e5d0dc1f5c9f1c31b42da6ead601",
        )


def test_legacy_instrument_identity_separates_recovered_instruments():
    base = {
        "observation_schema": "full_ledger_legacy",
        "sampling_instruments": {"gpt-5.5": {"max_tokens": 1024}},
        "gameplay_protocol_instrument": {"system_prompt_sha256": "one"},
    }
    changed = {**base, "observation_schema": "bounded_current_previous_summary_v1"}
    assert experiment.legacy_instrument_identity(base) != experiment.legacy_instrument_identity(
        changed
    )


def test_served_instruments_record_realized_model_and_provider_counts():
    span = SimpleNamespace(event="span_begin", type="turn", id="turn-1", name="P1 · glm-5.2")
    model = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/z-ai/glm-5.2",
        call=SimpleNamespace(response={"provider": "Fireworks", "model": "z-ai/glm-5.2-v1"}),
        output=SimpleNamespace(model="z-ai/glm-5.2-v1", error=None), error=None,
    )
    sample = SimpleNamespace(
        metadata={"seat_map": {"P1": "glm-5.2"}}, events=[span, model]
    )
    assert experiment.infer_served_instruments(sample)["glm-5.2"] == {
        "calls": 1,
        "failed_calls": 0,
        "pending_calls": 0,
        "unknown_provider_calls": 0,
        "requested_models": {"openrouter/z-ai/glm-5.2": 1},
        "served_models": {"z-ai/glm-5.2-v1": 1},
        "failed_served_models": {},
        "pending_served_models": {},
        "providers": {"Fireworks": 1},
    }


def test_failed_call_alias_does_not_count_as_a_successful_served_model_version():
    span = SimpleNamespace(event="span_begin", type="turn", id="turn-1", name="P1 · glm-5.2")
    success = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/z-ai/glm-5.2",
        call=SimpleNamespace(response={"provider": "Host", "model": "z-ai/glm-5.2-v1"}),
        output=SimpleNamespace(model="z-ai/glm-5.2-v1", error=None), error=None,
    )
    failed = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/z-ai/glm-5.2",
        call=SimpleNamespace(response={}),
        output=SimpleNamespace(model="openrouter/z-ai/glm-5.2", error="timeout"), error=None,
    )
    sample = SimpleNamespace(metadata={"seat_map": {"P1": "glm-5.2"}},
                             events=[span, success, failed])
    row = experiment.infer_served_instruments(sample)["glm-5.2"]
    assert row["served_models"] == {"z-ai/glm-5.2-v1": 1}
    assert row["failed_served_models"] == {"openrouter/z-ai/glm-5.2": 1}
    assert row["failed_calls"] == 1


def test_pending_call_alias_does_not_count_as_a_successful_served_model_version():
    span = SimpleNamespace(event="span_begin", type="turn", id="turn-1", name="P1 · glm-5.2")
    success = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/z-ai/glm-5.2",
        call=SimpleNamespace(response={"provider": "Host", "model": "z-ai/glm-5.2-v1"}),
        output=SimpleNamespace(model="z-ai/glm-5.2-v1", error=None), error=None,
        pending=False,
    )
    pending = SimpleNamespace(
        event="model", span_id="turn-1", model="openrouter/z-ai/glm-5.2",
        call=None,
        output=SimpleNamespace(model="openrouter/z-ai/glm-5.2", error=None), error=None,
        pending=True,
    )
    sample = SimpleNamespace(metadata={"seat_map": {"P1": "glm-5.2"}},
                             events=[span, success, pending])
    row = experiment.infer_served_instruments(sample)["glm-5.2"]
    assert row["served_models"] == {"z-ai/glm-5.2-v1": 1}
    assert row["pending_served_models"] == {"openrouter/z-ai/glm-5.2": 1}
    assert row["pending_calls"] == 1


def test_older_generic_span_recovers_model_from_exact_system_input():
    config = SimpleNamespace(
        max_tokens=512, temperature=None, reasoning_effort=None, extra_body=None,
    )
    model = SimpleNamespace(
        event="model", span_id="generic", model="openrouter/z-ai/glm-5.2", config=config,
        input=[SimpleNamespace(role="system", content="You are P1, one participant.")],
        call=SimpleNamespace(response={"provider": "Together"}),
        output=SimpleNamespace(model="z-ai/glm-5.2-v1", error=None), error=None,
    )
    sample = SimpleNamespace(
        metadata={"seat_map": {"P1": "glm-5.2"}}, events=[model]
    )
    assert experiment.infer_sampling_instruments(sample)["glm-5.2"]["max_tokens"] == 512
    assert experiment.infer_served_instruments(sample)["glm-5.2"]["providers"] == {
        "Together": 1,
    }


def test_tampered_stamped_metadata_is_rejected():
    metadata = {"seat_map": {"P1": "a"}}
    metadata["experiment_identity"] = experiment.compute_episode_identity("sample", metadata)
    metadata["seat_map"]["P1"] = "b"
    with pytest.raises(ValueError, match="changed after"):
        experiment.episode_identity("sample", metadata)


def test_duplicate_arm_names_fail_before_task_construction():
    cfg = {"arms": [
        {"name": "same", "task": "lineup", "seats": ["glm-5.2"]},
        {"name": "same", "task": "lineup", "seats": ["glm-5.2"]},
    ]}
    with pytest.raises(ValueError, match="duplicate arm names"):
        experiment.build_tasks(cfg)


def test_semantically_duplicate_arms_fail_before_a_run():
    cfg = {"arms": [
        {"name": "first", "task": "lineup", "seats": ["glm-5.2"], "reps": 1},
        {"name": "second", "task": "lineup", "seats": ["glm-5.2"], "reps": 1},
    ]}
    with pytest.raises(ValueError, match="planned episode collision"):
        experiment.build_tasks(cfg)


@pytest.mark.parametrize("cost", [
    {"avg_tokens_per_turn": 0},
    {"usd_per_mtok": -1},
    {"avg_tokens_per_turn": True},
    {"usd_per_mtok": "cheap"},
    {"usd_per_million_tokens": 2},
])
def test_bad_cost_assumptions_fail_before_a_run(cost):
    cfg = {"cost": cost, "arms": [
        {"name": "a", "task": "lineup", "seats": ["glm-5.2"]},
    ]}
    tasks = experiment.build_tasks(cfg)
    with pytest.raises(ValueError, match="cost"):
        experiment.estimate_cost(cfg, tasks)


def test_every_config_builds_unique_self_checking_episode_identities():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = experiment.load_config(path)
        identities = []
        for _name, task_value in experiment.build_tasks(cfg, config_name=path.stem):
            for sample in task_value.dataset:
                identities.append(experiment.episode_identity(sample.id, sample.metadata))
        assert identities, path.name
        assert len(identities) == len(set(identities)), path.name
