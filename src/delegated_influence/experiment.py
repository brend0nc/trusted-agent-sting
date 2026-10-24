"""Canonical experiment planning and semantic episode identity.

This module is the single place that interprets ``configs/*.yaml``.  Live runs, preflight, and
the results site all consume the same expanded plan.  Every newly planned sample is stamped with
a versioned identity derived from the condition that can change play or interpretation; existing
logs without the stamp remain readable because their identity is reconstructed from metadata.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from delegated_influence import costs, models, observation, seat, task
from delegated_influence import protocols, sting


EXPERIMENT_IDENTITY_VERSION = 1
EXPERIMENT_IDENTITY_PREFIX = f"experiment-v{EXPERIMENT_IDENTITY_VERSION}:"
LEGACY_INSTRUMENT_VERSION = 1
LEGACY_INSTRUMENT_PREFIX = f"legacy-instrument-v{LEGACY_INSTRUMENT_VERSION}:"
LEGACY_COHORT_PREFIX = f"legacy-cohort-v{LEGACY_INSTRUMENT_VERSION}:"

# Every condition flag rendered by report.label_condition must live here.  The identity and report
# both import this one list, so a new treatment cannot be added to only one research path.
TREATMENT_FLAGS = (
    "messages_off",
    "public_chat",
    "human_board",
    "declared_intents",
    "seeded_favours",
    "self_pull_reward",
    "punish_enabled",
)

TASK_BUILDERS = {
    "focal_attack": task.focal_attack,
    "focal_resist": task.focal_resist,
    "arena": task.arena,
    "arena_25": task.arena_25,
    "arena_5x5": task.arena_5x5,
    "lineup": task.lineup,
    "forfeit_smoke": task.forfeit_smoke,
    "controlled_whistleblowing": protocols.controlled_whistleblowing,
    "matched_reasoning": protocols.matched_reasoning,
    "trusted_sting": sting.trusted_sting,
}
MODEL_ARGUMENTS = {
    "focal_attack": "model_keys",
    "focal_resist": "targets",
}
STAGES = {"confirmatory", "exploratory", "held-out"}
# Compatibility alias for existing planners. costs owns the shared defaults.
DEFAULT_COST = costs.DEFAULT_COST
COST_FIELDS = frozenset(DEFAULT_COST)
UNKNOWN_OBSERVATION_SCHEMA = observation.UNKNOWN_SCHEMA


def runtime_compatibility_key(instrument: dict[str, Any] | None) -> str | None:
    """Scientific execution identity, separate from machine/package provenance.

    Retain all semantic fields, including future fields. Only known environment
    fields are ignored. Old records with embedded versions remain readable.
    Checkpoint resume uses stricter source/environment checks in run.py.
    """
    if not instrument:
        return None
    semantic = {key: value for key, value in instrument.items() if key not in {'python', 'packages'}}
    return json.dumps(semantic, sort_keys=True, separators=(',', ':')) if semantic else None


def comparison_request_guard(routing: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compare pinned routing controls without their financial admission ceiling.

    Price filters on unpinned routes can select a different provider, so retain
    those filters. Never change the saved evidence or the live request guard.
    """
    if routing is None:
        return None
    pinned = (isinstance(routing.get('only'), list) and len(routing['only']) == 1
              and routing.get('allow_fallbacks') is False
              and routing.get('require_parameters') is True)
    return {key: value for key, value in routing.items() if key != 'max_price' or not pinned}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate one experiment YAML document."""
    path = Path(path)
    return validate_config(yaml.safe_load(path.read_text()), source=str(path))


def validate_config(cfg: Any, *, source: str = "config") -> dict[str, Any]:
    """Validate the plan-level shape before any Inspect task is constructed."""
    if not isinstance(cfg, dict):
        raise ValueError(f"{source}: config must be a mapping")
    arms = cfg.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{source}: config requires a non-empty arms list")
    stage = cfg.get("stage")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"{source}: unknown stage {stage!r} (have {sorted(STAGES)})")
    if 'pilot_seed_offset' in cfg:
        offset = cfg['pilot_seed_offset']
        if stage != 'exploratory' or isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0:
            raise ValueError('pilot_seed_offset requires an exploratory config and a positive integer')
    names = []
    for index, arm in enumerate(arms):
        label = f"{source}: arm {index}"
        if not isinstance(arm, dict):
            raise ValueError(f"{label} must be a mapping")
        name = arm.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} requires a non-empty name")
        task_name = arm.get("task")
        if task_name not in TASK_BUILDERS:
            raise ValueError(
                f"arm {name!r}: unknown task {task_name!r} (have {list(TASK_BUILDERS)})"
            )
        names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate arm names: {duplicates}")
    return cfg


def resolve_arm_stage(config_stage: str | None, arm: dict[str, Any]) -> str:
    """Resolve the paper pool stage using the locked exp-13 held-out rule."""
    if config_stage:
        return config_stage
    if arm.get("declared_intents") or arm.get("seeded_favours"):
        return "held-out"
    return "confirmatory"


def expand_models(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    """Expand ``all`` and validate every requested roster key."""
    keys = list(models.ROSTER) if spec == "all" else list(spec)
    bad = [key for key in keys if key not in models.MODEL_REGISTRY]
    if bad:
        raise ValueError(f"unknown model keys: {bad}")
    return keys


def _condition_payload(sample_id: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Canonical identity payload for one episode condition.

    Missing and explicit false/zero values deliberately collapse for treatment flags, matching
    historical logs.  Missing observation or sampling markers never inherit today's defaults:
    they become explicitly unknown, preventing an old run from silently pooling with a new one.
    """
    return {
        "identity_version": EXPERIMENT_IDENTITY_VERSION,
        "sample_id": str(sample_id),
        "seat_map": metadata.get("seat_map") or {},
        "topology": metadata.get("topology", "complete"),
        "seed": metadata.get("seed"),
        "focal": metadata.get("focal"),
        "personas": metadata.get("personas") or {},
        "principal_seats": sorted(metadata.get("principal_seats") or []),
        "brokerage": bool(metadata.get("brokerage")),
        "rounds": metadata.get("rounds"),
        "actions_per_round": metadata.get("actions_per_round"),
        "gameplay_protocol_version": metadata.get(
            "gameplay_protocol_version", "legacy_unversioned"
        ),
        "observation_schema": (
            metadata.get("observation_schema") or UNKNOWN_OBSERVATION_SCHEMA
        ),
        # Retain explicit overrides as historical evidence even when an old fixture/log has no
        # reconstructed effective instrument. New plans additionally carry the resolved settings.
        "model_configs": metadata.get("model_configs") or {},
        "sampling_instruments": metadata.get("sampling_instruments") or {
            model: {"status": "unknown"}
            for model in sorted(set((metadata.get("seat_map") or {}).values()))
        },
        "treatments": {name: metadata.get(name) or 0 for name in TREATMENT_FLAGS
                       if name != "human_board" or metadata.get(name)},
        **({'protocol': metadata['protocol']} if metadata.get('protocol') else {}),
    }


LEGACY_RECOVERED_SAMPLING_INSTRUMENT_VERSION = "legacy_recovered_v1"


def _event_sampling_instrument(
    event: Any, instrument_version: Any = LEGACY_RECOVERED_SAMPLING_INSTRUMENT_VERSION,
) -> dict[str, Any]:
    """Canonical planned sampling fields retained by an Inspect ModelEvent."""
    config = getattr(event, "config", None)
    extra_body = getattr(config, "extra_body", None) or {}
    return {
        "instrument_version": instrument_version,
        "model_spec": getattr(event, "model", None),
        "max_tokens": getattr(config, "max_tokens", None),
        "temperature": getattr(config, "temperature", None),
        "reasoning_effort": getattr(config, "reasoning_effort", None),
        "provider_routing": extra_body.get("provider"),
    }


def _turn_span_models(events: list[Any]) -> dict[Any, str]:
    """Map Inspect turn-span ids to the roster key named by our generator wrapper."""
    spans = {}
    for event in events:
        if (getattr(event, "event", None) == "span_begin"
                and getattr(event, "type", None) == "turn"):
            name = str(getattr(event, "name", "") or "")
            if " · " in name:
                span_id = getattr(event, "span_id", None) or getattr(event, "id", None)
                spans[span_id] = name.split(" · ", 1)[1]
    return spans


def _message_value(message: Any, field: str) -> Any:
    return message.get(field) if isinstance(message, dict) else getattr(message, field, None)


def _canonical_text(value: Any) -> str:
    """Stable text for prompt content, including Inspect's structured content blocks."""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_system_prompt(text: str) -> str:
    """Remove seat allocation from a prompt while retaining its experimental wording."""
    for clause in (*seat._PERSONA_CLAUSE.values(), seat.ATTACKER_CLAUSE_NOMSG):
        if clause:
            text = text.replace(clause, "")
    return re.sub(r"\bP\d+\b", "<SEAT>", text)


def _model_event_key(
    event: Any, span_models: dict[Any, str], seat_map: dict[str, str],
) -> str | None:
    """Recover the roster key from modern named spans or older exact model inputs."""
    key = span_models.get(getattr(event, "span_id", None))
    if key in set(seat_map.values()):
        return key
    for message in getattr(event, "input", None) or []:
        if str(_message_value(message, "role")) != "system":
            continue
        text = _canonical_text(_message_value(message, "content"))
        match = re.search(r"\bYou are (P\d+)\b", text)
        if match and match.group(1) in seat_map:
            return seat_map[match.group(1)]
    requested = getattr(event, "model", None)
    candidates = [model for model in sorted(set(seat_map.values()))
                  if requested in {model, models.openrouter_spec(model)}]
    return candidates[0] if len(candidates) == 1 else None


def _observation_structure(event: Any) -> list[str]:
    """Stable user-prompt frame, excluding behavior-dependent optional blocks.

    These lines are always part of the information instrument (apart from the deadline, whose
    presence is determined by the round).  Event/message bodies are deliberately excluded: two
    episodes should not become different instruments merely because different actions occurred.
    """
    structural = (
        "Round ", "Deadline:", "Scores:",
        "Reply with exactly one JSON action object.",
    )
    for message in getattr(event, "input", None) or []:
        if str(_message_value(message, "role")) != "user":
            continue
        text = _canonical_text(_message_value(message, "content"))
        fixed = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or not line.startswith(structural):
                continue
            line = re.sub(r"\bP\d+\b", "<SEAT>", line)
            line = re.sub(r"\b\d+(?:\.\d+)?\b", "<N>", line)
            fixed.append(line)
        return fixed
    return []


def infer_sampling_instruments(sample: Any) -> dict[str, dict[str, Any]]:
    """Recover effective per-model settings from legacy ModelEvents.

    New samples carry this mapping directly.  Older logs are reconstructed from the named turn
    spans and exact GenerateConfig stored with each call.  Missing or inconsistent evidence is
    marked unknown/ambiguous rather than guessed from today's mutable model registry.
    """
    metadata = getattr(sample, "metadata", None) or {}
    stored = metadata.get("sampling_instruments") or {}

    model_keys = sorted(set((metadata.get("seat_map") or {}).values()) | set(stored))
    seat_map = metadata.get("seat_map") or {}
    events = getattr(sample, "events", None) or []
    span_models = _turn_span_models(events)

    observed: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in model_keys}
    for event in events:
        if getattr(event, "event", None) != "model":
            continue
        key = _model_event_key(event, span_models, seat_map)
        if key not in observed:
            continue
        # A missing GenerateConfig contains no evidence about the effective settings. Keep the
        # stamp, if present, but do not pretend an all-null reconstruction verified it.
        if getattr(event, "config", None) is None:
            continue
        stamped = stored.get(key) or {}
        # GenerateConfig records the behavioral settings, but not the version label. A modern
        # metadata stamp supplies that label; otherwise use one permanent legacy identity rather
        # than today's mutable version constant.
        instrument = _event_sampling_instrument(
            event,
            stamped.get("instrument_version", LEGACY_RECOVERED_SAMPLING_INSTRUMENT_VERSION),
        )
        encoded = json.dumps(instrument, sort_keys=True, separators=(",", ":"))
        observed[key][encoded] = instrument

    inferred = {}
    for key in model_keys:
        values = list(observed[key].values())
        stamped = stored.get(key)
        if stamped is not None and not values:
            inferred[key] = stamped
        elif stamped is not None and len(values) == 1 and values[0] == stamped:
            inferred[key] = stamped
        elif stamped is not None and values:
            # Never let a self-reported current stamp overrule what the paid calls actually used.
            # The loader records this mismatch and the paper-pool matcher excludes the episode.
            inferred[key] = {
                "status": "metadata_event_mismatch",
                "stamped": stamped,
                "observed": values,
            }
        elif len(values) == 1:
            inferred[key] = values[0]
        elif not values:
            inferred[key] = {"status": "unknown"}
        else:
            inferred[key] = {"status": "ambiguous", "observed": values}
    return inferred


def infer_gameplay_protocol(sample: Any) -> dict[str, Any]:
    """Recover a stable gameplay-prompt fingerprint for marker-less historical logs.

    The current protocol carries an explicit version. Historical logs instead retain the exact
    system input on every ModelEvent; hashing the unique gameplay prompts prevents two old prompt
    revisions from being silently treated as one instrument.
    """
    metadata = getattr(sample, "metadata", None) or {}
    version = metadata.get("gameplay_protocol_version")
    if version is not None:
        return {"status": "versioned", "version": version}

    events = getattr(sample, "events", None) or []
    turn_spans = _turn_span_models(events)
    prompts = set()
    observation_lines = set()
    seat_map = metadata.get("seat_map") or {}
    for event in events:
        if (getattr(event, "event", None) != "model"
                or _model_event_key(event, turn_spans, seat_map) is None):
            continue
        for message in getattr(event, "input", None) or []:
            if str(_message_value(message, "role")) == "system":
                prompts.add(_normalize_system_prompt(
                    _canonical_text(_message_value(message, "content"))
                ))
        observation_lines.update(_observation_structure(event))
    if not prompts:
        return {"status": "unknown"}
    encoded = json.dumps(sorted(prompts), ensure_ascii=False, separators=(",", ":"))
    protocol = {
        "status": "legacy_recovered",
        "system_prompt_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "unique_system_prompts": len(prompts),
    }
    if observation_lines:
        encoded_observation = json.dumps(
            sorted(observation_lines), ensure_ascii=False, separators=(",", ":")
        )
        protocol["observation_structure_sha256"] = hashlib.sha256(
            encoded_observation.encode("utf-8")
        ).hexdigest()
    return protocol


def infer_served_instruments(sample: Any) -> dict[str, dict[str, Any]]:
    """Recover realized model version and OpenRouter host counts from ModelEvents.

    Served hosts are provenance, not planned identity: OpenRouter may use multiple hosts inside
    one episode. Keeping counts separate makes routing drift auditable without fragmenting the
    experiment into one-cell-per-host-mixture.
    """
    metadata = getattr(sample, "metadata", None) or {}
    model_keys = sorted(set((metadata.get("seat_map") or {}).values()))
    seat_map = metadata.get("seat_map") or {}
    events = getattr(sample, "events", None) or []
    span_models = _turn_span_models(events)
    out = {
        key: {"calls": 0, "failed_calls": 0, "pending_calls": 0,
              "unknown_provider_calls": 0,
              "requested_models": {}, "served_models": {}, "failed_served_models": {},
              "pending_served_models": {}, "providers": {}}
        for key in model_keys
    }

    def increment(bucket: dict[str, int], value: Any) -> None:
        if value:
            text = str(value)
            bucket[text] = bucket.get(text, 0) + 1

    for event in events:
        if getattr(event, "event", None) != "model":
            continue
        key = _model_event_key(event, span_models, seat_map)
        if key not in out:
            continue
        row = out[key]
        row["calls"] += 1
        increment(row["requested_models"], getattr(event, "model", None))
        call = getattr(event, "call", None)
        response = getattr(call, "response", None) if call is not None else None
        response = response if isinstance(response, dict) else {}
        output = getattr(event, "output", None)
        served_model = getattr(output, "model", None) or response.get("model")
        provider = response.get("provider")
        increment(row["providers"], provider)
        output_error = getattr(output, "error", None) if output is not None else None
        failed = bool(getattr(event, "error", None) or output_error)
        pending = bool(getattr(event, "pending", False))
        if pending:
            # Inspect writes an alias-only placeholder while a call is in flight. An interrupted
            # run can leave that event in the log without an error, response, or served version.
            # Preserve it for audit, but do not report the requested alias as realized drift.
            row["pending_calls"] += 1
            increment(row["pending_served_models"], served_model)
        elif failed:
            row["failed_calls"] += 1
            # Inspect often echoes the requested alias into output.model on a failed call. Keep it
            # for audit, but never mistake it for a second successfully served model version.
            increment(row["failed_served_models"], served_model)
        else:
            increment(row["served_models"], served_model)
        if not provider:
            row["unknown_provider_calls"] += 1
    return out


def legacy_instrument_identity(metadata: dict[str, Any]) -> str:
    """Identity of a recovered historical instrument, independent of episode condition."""
    payload = {
        "instrument_version": LEGACY_INSTRUMENT_VERSION,
        "observation_schema": metadata.get("observation_schema") or UNKNOWN_OBSERVATION_SCHEMA,
        "sampling_instruments": metadata.get("sampling_instruments") or {"status": "unknown"},
        "gameplay_protocol": metadata.get("gameplay_protocol_instrument") or {
            "status": "unknown"
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return LEGACY_INSTRUMENT_PREFIX + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def legacy_sampling_profile(metadata: dict[str, Any]) -> dict[str, str]:
    """Per-model recovered sampling variant ids for arm-wide consistency checks."""
    profile = {}
    for model, instrument in sorted((metadata.get("sampling_instruments") or {}).items()):
        encoded = json.dumps(
            instrument, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        profile[model] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return profile


def legacy_cohort_identity(metadata: dict[str, Any]) -> str:
    """Historical prompt/observation cohort; sampling is checked separately per plan cell.

    Arms intentionally contain different models with different sampling settings. A cohort can
    therefore span those planned differences, while matching refuses to combine two recovered
    sampling variants for the *same* planned condition.
    """
    payload = {
        "instrument_version": LEGACY_INSTRUMENT_VERSION,
        "observation_schema": metadata.get("observation_schema") or UNKNOWN_OBSERVATION_SCHEMA,
        "gameplay_protocol": metadata.get("gameplay_protocol_instrument") or {
            "status": "unknown"
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return LEGACY_COHORT_PREFIX + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_episode_identity(sample_id: Any, metadata: dict[str, Any]) -> str:
    """Compute the versioned semantic identity for a planned or recorded episode."""
    payload = json.dumps(
        _condition_payload(sample_id, metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return EXPERIMENT_IDENTITY_PREFIX + digest


def legacy_episode_identity(sample_id: Any, metadata: dict[str, Any]) -> str:
    """Instrument-unversioned compatibility key for marker-less historical logs only.

    This key may locate an old run's experiment arm, but it must never be treated as the current
    experiment identity or pooled with a current run. The site records which instrument won.
    """
    payload = _condition_payload(sample_id, metadata)
    for field in (
        "observation_schema", "sampling_instruments", "gameplay_protocol_version",
    ):
        payload.pop(field, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "legacy-condition-v1:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def episode_identity(sample_id: Any, metadata: dict[str, Any]) -> str:
    """Return canonical identity and reject a stale/tampered stamp on new logs."""
    computed = compute_episode_identity(sample_id, metadata)
    stored = metadata.get("experiment_identity")
    if stored is not None and stored != computed:
        raise ValueError(
            f"episode {sample_id!r} metadata changed after its experiment identity was stamped"
        )
    return computed


def _stamp_task(task_value: Any, *, config_name: str | None, arm_name: str,
                task_name: str, shard_name: str, stage: str) -> None:
    """Attach self-checking provenance to every planned sample without changing game inputs."""
    for sample in task_value.dataset:
        metadata = sample.metadata
        if config_name is not None:
            metadata["experiment_config"] = config_name
        metadata["experiment_arm"] = arm_name
        metadata["experiment_task"] = task_name
        metadata["experiment_shard"] = shard_name
        metadata["experiment_stage"] = stage
        metadata["experiment_identity"] = compute_episode_identity(sample.id, metadata)


def build_tasks(cfg: dict[str, Any], *, config_name: str | None = None) -> list[tuple[str, Any]]:
    """Expand one validated config into named Inspect tasks without making model calls."""
    validate_config(cfg)
    task.set_direct_models((cfg.get("native") or {}).get("direct_models"))
    out = []
    for arm in cfg["arms"]:
        name, task_name = arm["name"], arm["task"]
        stage = resolve_arm_stage(cfg.get("stage"), arm)
        kwargs = {key: value for key, value in arm.items()
                  if key not in ("name", "task", "models")}
        if task_name in MODEL_ARGUMENTS:
            # One task per model keeps .eval files committable and resume granularity fine.
            for model in expand_models(arm.get("models", "all")):
                shard = f"{name}-{model}"
                built = TASK_BUILDERS[task_name](
                    **{MODEL_ARGUMENTS[task_name]: [model], "name": shard, **kwargs}
                )
                _stamp_task(built, config_name=config_name, arm_name=name,
                            task_name=task_name, shard_name=shard, stage=stage)
                out.append((shard, built))
        else:
            built = TASK_BUILDERS[task_name](name=name, **kwargs)
            _stamp_task(built, config_name=config_name, arm_name=name,
                        task_name=task_name, shard_name=name, stage=stage)
            out.append((name, built))
    seen: dict[str, tuple[str, str]] = {}
    for shard_name, built in out:
        for sample in built.dataset:
            if cfg.get('pilot_seed_offset'):
                offset = cfg['pilot_seed_offset']
                if sample.metadata.get('protocol'):
                    raise ValueError('controlled protocol pilots must select fresh seeds in their builder')
                sample.metadata['seed'] += offset
                sample.id = f'{sample.id}|pilot-offset-{offset}'
                sample.metadata['experiment_identity'] = compute_episode_identity(sample.id, sample.metadata)
            identity = episode_identity(sample.id, sample.metadata)
            source = (sample.metadata["experiment_arm"], shard_name)
            if identity in seen:
                previous = seen[identity]
                raise ValueError(
                    "planned episode collision: semantic identity appears in both "
                    f"{previous[0]!r}/{previous[1]!r} and {source[0]!r}/{source[1]!r}"
                )
            seen[identity] = source
    return out


def task_arm_name(task_name: str, arms: list[dict[str, Any]]) -> str:
    """Find a shard's owning arm, preferring stamped exact names over prefix ambiguity."""
    matches = [arm["name"] for arm in arms
               if task_name == arm["name"] or task_name.startswith(arm["name"] + "-")]
    if not matches:
        raise ValueError(f"task {task_name!r} does not belong to a configured arm")
    return max(matches, key=len)


def _turns_per_episode(task_value: Any) -> int:
    metadata = task_value.dataset[0].metadata
    return metadata["rounds"] * metadata["actions_per_round"] * len(metadata["seat_map"])


def _calls_per_episode(task_value: Any) -> int:
    """Worst-case paid calls: every gameplay turn may consume its correction retry."""
    metadata = task_value.dataset[0].metadata
    intents = (metadata["rounds"] * len(metadata["seat_map"])
               if metadata.get("declared_intents") else 0)
    return (_turns_per_episode(task_value) * task.turn.MAX_TURN_ATTEMPTS) + intents


def _base_calls_per_episode(task_value: Any) -> int:
    """Normal calls before reserving for the one parse-correction retry."""
    metadata = task_value.dataset[0].metadata
    intents = (metadata["rounds"] * len(metadata["seat_map"])
               if metadata.get("declared_intents") else 0)
    return _turns_per_episode(task_value) + intents


def estimate_cost(cfg: dict[str, Any], tasks: list[tuple[str, Any]]) -> dict[str, Any]:
    """Estimate normal calls plus one parse-correction allowance for every gameplay turn.

    This is a pre-launch planning estimate, not a runtime spending cap. Inspect task retries reuse
    completed samples but may repeat a failed sample, and provider billing can differ from the flat
    token-price assumption; exact realized cost is retained in the run export.
    """
    configured = cfg.get("cost") or {}
    if not isinstance(configured, dict):
        raise ValueError("cost must be a mapping")
    unknown = set(configured) - COST_FIELDS - costs.EXTRA_COST_FIELDS
    if unknown:
        raise ValueError(f"unknown cost fields: {sorted(unknown)}")
    cost = {**DEFAULT_COST, **configured}
    for field in DEFAULT_COST:
        costs.number(cost[field], f"cost.{field}", positive=True)
    costs.prices(cost)
    rows, total, base_total, scheduled_calls, base_calls = [], 0.0, 0.0, 0, 0
    for name, task_value in tasks:
        episodes = len(task_value.dataset)
        turns = _turns_per_episode(task_value)
        calls_per_episode = _calls_per_episode(task_value)
        base_calls_per_episode = _base_calls_per_episode(task_value)
        calls = episodes * calls_per_episode
        ordinary_calls = episodes * base_calls_per_episode
        call_price = sum(costs.planning_call_cost(cost, sample.metadata["seat_map"])
                         for sample in task_value.dataset) / max(episodes, 1)
        usd = calls * call_price
        base_usd = ordinary_calls * call_price
        total += usd
        base_total += base_usd
        scheduled_calls += calls
        base_calls += ordinary_calls
        rows.append({
            "arm": name,
            "episodes": episodes,
            "turns_per_episode": turns,
            "calls_per_episode": calls_per_episode,
            "base_calls_per_episode": base_calls_per_episode,
            "base_calls": ordinary_calls,
            "scheduled_calls": calls,
            "base_usd": base_usd,
            "usd": usd,
        })
    return {"arms": rows, "base_calls": base_calls, "scheduled_calls": scheduled_calls,
            "pricing_basis": "per_model" if cost.get("model_prices") else "legacy_blended",
            "base_usd": base_total, "total_usd": total,
            "scope": "normal gameplay plus one parse-correction allowance per turn; "
                     "not a runtime cap and excludes repeated failed samples"}


def enforce_budget(cfg: dict[str, Any], estimate: dict[str, Any]) -> None:
    """Require a deliberate ceiling before a paid run; never impose a mid-game call cap."""
    budget = cfg.get("budget") or {}
    if not isinstance(budget, dict):
        raise ValueError("budget must be a mapping")
    unknown = set(budget) - {"max_estimated_usd", "max_actual_usd"}
    if unknown:
        raise ValueError(f"unknown budget fields: {sorted(unknown)}")
    if "max_actual_usd" in budget:
        costs.number(budget["max_actual_usd"], "budget.max_actual_usd", positive=True)
    ceiling = budget.get("max_estimated_usd")
    if ceiling is None:
        raise ValueError(
            "a live run requires budget.max_estimated_usd; use --dry-run to inspect the estimate, "
            "then set a deliberate ceiling in the config"
        )
    costs.number(ceiling, "budget.max_estimated_usd", positive=True)
    if estimate["total_usd"] > ceiling:
        raise ValueError(
            f"estimated cost ${estimate['total_usd']:,.2f} exceeds "
            f"budget.max_estimated_usd ${ceiling:,.2f}; review the plan or raise the ceiling"
        )
