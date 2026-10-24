"""Immutable run manifests and analysis-ready exports for arena `.eval` logs.

The raw Inspect log remains the source of truth.  Before a live launch we freeze the exact YAML
and the runtime source files that can affect play.  After the run we write a compressed JSONL
export which joins each canonical game event to its exact model input, output, timing, provider,
token usage, and OpenRouter-reported cost.

No environment variables or `.env` files are read or copied here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from inspect_ai.log import read_eval_log

from delegated_influence import episodes as episode_io
from delegated_influence import replay


MANIFEST_SCHEMA_VERSION = 1
EXPORT_SCHEMA_VERSION = 1
RUNTIME_ROOTS = ("src/delegated_influence",)
RUNTIME_FILES = ("pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class RunArtifacts:
    """Paths and identifiers for one launch; every launch gets a fresh directory."""

    launch_id: str
    directory: Path
    manifest_path: Path
    config_path: Path
    source_snapshot_path: Path

    @property
    def eval_metadata(self) -> dict[str, Any]:
        return {
            "run_manifest_id": self.launch_id,
            "run_manifest_schema": MANIFEST_SCHEMA_VERSION,
        }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Write JSON atomically; exclusive=True refuses to replace a frozen artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(tmp, path)
        path.chmod(0o444)
    finally:
        if tmp.exists():
            tmp.unlink()


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def runtime_source_paths(root: Path) -> list[Path]:
    """Whitelisted source snapshot; deliberately excludes `.env`, data, logs, and user files."""
    paths = []
    for rel in RUNTIME_FILES:
        path = root / rel
        if path.is_file():
            paths.append(path)
    for rel in RUNTIME_ROOTS:
        base = root / rel
        if base.is_dir():
            paths.extend(p for p in base.rglob("*.py") if p.is_file())
    return sorted(set(paths), key=lambda p: _relative_label(p, root))


def _source_tree_hash(source_files: dict[str, str]) -> str:
    canonical = "".join(f"{path}\0{digest}\n" for path, digest in sorted(source_files.items()))
    return _sha256_bytes(canonical.encode("utf-8"))


def _git_value(root: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def git_state(root: Path) -> dict[str, Any]:
    status = _git_value(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "branch": _git_value(root, "branch", "--show-current"),
        "dirty": bool(status),
        # Filenames make a dirty launch auditable without copying unrelated workspace contents.
        "changed_paths": sorted(line[3:] for line in status.splitlines()) if status else [],
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _snapshot_source(root: Path, source_paths: list[Path], destination: Path) -> None:
    with tarfile.open(destination, "x:gz") as archive:
        for path in source_paths:
            archive.add(path, arcname=_relative_label(path, root), recursive=False)
    destination.chmod(0o444)


def create_run_artifacts(
    config_file: str | Path,
    resolved_config: dict[str, Any],
    estimate: dict[str, Any],
    tasks: Iterable[tuple[str, Any]],
    log_dir: str | Path,
    *,
    retry_attempts: int,
    max_tasks: int,
    runner: str = "inspect",
    source_root: str | Path | None = None,
    now: datetime | None = None,
) -> RunArtifacts:
    """Freeze one live launch before any API call is made."""
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    config_file = Path(config_file).resolve()
    config_bytes = config_file.read_bytes()
    config_hash = _sha256_bytes(config_bytes)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    launch_id = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{config_hash[:12]}"
    directory = Path(log_dir) / "run-artifacts" / launch_id
    directory.mkdir(parents=True, exist_ok=False)

    frozen_config = directory / "run-config.yaml"
    with frozen_config.open("xb") as f:
        f.write(config_bytes)
    frozen_config.chmod(0o444)

    sources = runtime_source_paths(root)
    source_hashes = {_relative_label(path, root): sha256_file(path) for path in sources}
    snapshot = directory / "source-snapshot.tar.gz"
    _snapshot_source(root, sources, snapshot)

    task_rows = []
    for name, task in tasks:
        task_rows.append({
            "arm": name,
            "task_name": getattr(task, "name", None),
            "episodes": len(getattr(task, "dataset", []) or []),
        })

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "launch_id": launch_id,
        "created_at": timestamp.isoformat(),
        "input_config": {
            "path": _relative_label(config_file, root),
            "sha256": config_hash,
            "snapshot": frozen_config.name,
            "resolved": resolved_config,
        },
        "launch": {
            "log_dir": str(Path(log_dir)),
            "retry_attempts": retry_attempts,
            "max_tasks": max_tasks,
            "log_model_api": runner == "inspect",
            "runner": runner,
            "estimate": estimate,
            "tasks": task_rows,
        },
        "source": {
            "git": git_state(root),
            "tree_sha256": _source_tree_hash(source_hashes),
            "files": source_hashes,
            "snapshot": snapshot.name,
            "snapshot_sha256": sha256_file(snapshot),
            "scope": [*RUNTIME_FILES, *RUNTIME_ROOTS],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "inspect_ai": _package_version("inspect_ai"),
                "openai": _package_version("openai"),
                "httpx": _package_version("httpx"),
                "delegated-influence": _package_version("delegated-influence"),
            },
        },
        "privacy": {
            "environment_copied": False,
            "dotenv_copied": False,
        },
    }
    manifest_path = directory / "manifest.json"
    _atomic_json(manifest_path, manifest, exclusive=True)
    return RunArtifacts(launch_id, directory, manifest_path, frozen_config, snapshot)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _resolve_attachments(value: Any, attachments: dict[str, Any]) -> Any:
    """Replace Inspect's attachment:// references with the exact stored prompt text."""
    if isinstance(value, str) and value.startswith("attachment://"):
        return attachments.get(value.removeprefix("attachment://"), value)
    if isinstance(value, dict):
        return {k: _resolve_attachments(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_attachments(v, attachments) for v in value]
    return value


def _raw_usage(model_event: Any) -> dict[str, Any]:
    call = getattr(model_event, "call", None)
    response = getattr(call, "response", None) if call is not None else None
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def normalize_model_call(model_event: Any, attachments: dict[str, Any]) -> dict[str, Any]:
    """One Inspect ModelEvent -> stable, provider-neutral research fields."""
    call = getattr(model_event, "call", None)
    response = getattr(call, "response", None) if call is not None else None
    response = response if isinstance(response, dict) else {}
    output = getattr(model_event, "output", None)
    output_usage = _jsonable(getattr(output, "usage", None)) or {}
    raw_usage = _raw_usage(model_event)
    completion_details = raw_usage.get("completion_tokens_details") or {}
    prompt_details = raw_usage.get("prompt_tokens_details") or {}

    def usage_value(key, fallback):
        value = output_usage.get(key)
        return fallback if value is None else value
    input_messages = [
        _resolve_attachments(_jsonable(message), attachments)
        for message in (getattr(model_event, "input", None) or [])
    ]
    return {
        "timestamp": _jsonable(getattr(model_event, "timestamp", None)),
        "completed_at": _jsonable(getattr(model_event, "completed", None)),
        "latency_seconds": (
            getattr(call, "time", None) if call is not None
            else getattr(model_event, "working_time", None)
        ),
        "requested_model": getattr(model_event, "model", None),
        "served_model": getattr(output, "model", None) or response.get("model"),
        "provider": response.get("provider"),
        "service_tier": response.get("service_tier"),
        "system_fingerprint": response.get("system_fingerprint"),
        "response_id": response.get("id"),
        "config": _jsonable(getattr(model_event, "config", None)) or {},
        "input": input_messages,
        "output": {
            "completion": getattr(output, "completion", None),
            "error": _jsonable(getattr(output, "error", None)),
            "metadata": _jsonable(getattr(output, "metadata", None)),
        },
        "event_error": _jsonable(getattr(model_event, "error", None)),
        "retries": getattr(model_event, "retries", None),
        "usage": {
            "input_tokens": usage_value("input_tokens", raw_usage.get("prompt_tokens", 0)),
            "output_tokens": usage_value("output_tokens", raw_usage.get("completion_tokens", 0)),
            "total_tokens": usage_value("total_tokens", raw_usage.get("total_tokens", 0)),
            "reasoning_tokens": usage_value(
                "reasoning_tokens", completion_details.get("reasoning_tokens", 0)),
            "cache_read_tokens": usage_value(
                "input_tokens_cache_read", prompt_details.get("cached_tokens", 0)),
            "cache_write_tokens": usage_value(
                "input_tokens_cache_write", prompt_details.get("cache_write_tokens", 0)),
            "reported_total_cost": output_usage.get("total_cost"),
            "openrouter_cost": raw_usage.get("cost"),
            "openrouter_cost_details": _jsonable(raw_usage.get("cost_details")),
        },
    }


def _turn_model_events(sample: Any) -> tuple[dict[str, list[Any]], int]:
    spans = {}
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) == "span_begin" and getattr(event, "type", None) == "turn":
            spans[getattr(event, "id", None)] = getattr(event, "name", "")
    by_agent: dict[str, list[Any]] = defaultdict(list)
    unassigned = 0
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        name = spans.get(getattr(event, "span_id", None))
        if not name:
            continue  # a judge/scorer call, not an arena turn
        agent = name.split(" · ", 1)[0]
        if not agent:
            unassigned += 1
            continue
        by_agent[agent].append(event)
    return by_agent, unassigned


def _usage_matches(event_usage: dict[str, Any], call: dict[str, Any]) -> bool:
    actual = call["usage"]
    return (
        int(event_usage.get("prompt_tokens") or 0) == int(actual.get("input_tokens") or 0)
        and int(event_usage.get("completion_tokens") or 0) == int(actual.get("output_tokens") or 0)
    )


def _call_matches_usage(event_usage: dict[str, Any], call: dict[str, Any]) -> bool:
    """A conservative join: never consume a later successful call for a missing error call."""
    if not _usage_matches(event_usage, call):
        return False
    if event_usage.get("error"):
        return bool(call["output"].get("error") or call.get("event_error"))
    return not bool(call["output"].get("error") or call.get("event_error"))


def sample_export_records(sample: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join one stored episode to its low-level calls and reconstruct score state per event."""
    store = getattr(sample, "store", None) or {}
    events = store.get("Episode:events")
    if events is None:
        raise ValueError(f"sample {getattr(sample, 'id', None)!r} has no stored Episode:events")
    seat_map = store.get("Episode:seat_map") or {}
    metadata = getattr(sample, "metadata", None) or {}
    attachments = getattr(sample, "attachments", None) or {}
    calls_by_agent, unassigned = _turn_model_events(sample)
    positions = defaultdict(int)
    skipped_by_agent = defaultdict(int)
    replay_steps, replay_final = replay.events(
        seat_map, events, metadata.get("self_pull_reward") or 0
    )
    stored_scores = store.get("Episode:scores") or {}
    if replay_final["scores"] != stored_scores:
        raise ValueError(
            f"sample {getattr(sample, 'id', None)!r} replay scores do not match stored scores"
        )
    records = []

    for index, raw_event in enumerate(events):
        event = _jsonable(raw_event)
        agent = event.get("agent")
        event_usages = event.get("usage") or []
        expected = len(event_usages)
        calls = []
        skipped = 0
        for event_usage in event_usages:
            position = positions[agent]
            available = calls_by_agent.get(agent, [])
            if position >= len(available):
                continue
            matched_at = None
            candidate = None
            for candidate_index in range(position, len(available)):
                normalized = normalize_model_call(available[candidate_index], attachments)
                if _call_matches_usage(event_usage, normalized):
                    matched_at, candidate = candidate_index, normalized
                    break
            if matched_at is not None:
                skipped += matched_at - position
                skipped_by_agent[agent] += matched_at - position
                calls.append(candidate)
                positions[agent] = matched_at + 1
        usage_match = len(calls) == expected and skipped == 0
        if expected == 0:
            join_status = "not_applicable"
        elif len(calls) != expected:
            join_status = "missing_model_calls"
        elif skipped:
            join_status = "usage_mismatch"
        else:
            join_status = "matched"

        state_before = replay_steps[index]["before"]
        state_after = replay_steps[index]["after"]
        records.append({
            "record_type": "event",
            "schema_version": EXPORT_SCHEMA_VERSION,
            "sample_id": str(getattr(sample, "id", "")),
            "event_index": index,
            "event": event,
            "state_before": state_before,
            "state_after": state_after,
            "join": {
                "status": join_status,
                "expected_model_calls": expected,
                "matched_model_calls": len(calls),
                "usage_matches": usage_match,
                "skipped_model_calls": skipped,
            },
            "model_calls": calls,
        })

    extras = {
        agent: len(calls_by_agent.get(agent, [])) - positions[agent]
        for agent in calls_by_agent
        if len(calls_by_agent.get(agent, [])) != positions[agent]
    }
    episode = {
        "record_type": "episode",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "sample_id": str(getattr(sample, "id", "")),
        "sample_uuid": str(getattr(sample, "uuid", "") or ""),
        "metadata": _jsonable(metadata),
        "seat_map": _jsonable(seat_map),
        "final_scores": _jsonable(store.get("Episode:scores") or {}),
        "started_at": _jsonable(getattr(sample, "started_at", None)),
        "completed_at": _jsonable(getattr(sample, "completed_at", None)),
        "total_time_seconds": getattr(sample, "total_time", None),
        "working_time_seconds": getattr(sample, "working_time", None),
        "event_count": len(events),
        "unassigned_turn_model_calls": unassigned,
        "unmatched_turn_model_calls": extras,
        "skipped_turn_model_calls": dict(skipped_by_agent),
    }
    return episode, records


def export_eval_log(eval_log: Any, destination_dir: str | Path, source_file: str | Path | None = None) -> dict[str, Any]:
    """Write one EvalLog to compressed JSONL, preserving the raw `.eval` as source of truth."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    location = source_file or getattr(eval_log, "location", None)
    source = Path(str(location)) if location else None
    stem = source.stem if source else str(getattr(getattr(eval_log, "eval", None), "eval_id", "eval"))
    output = destination_dir / f"{stem}.events.jsonl.gz"
    tmp = output.with_name(output.name + f".{os.getpid()}.tmp")
    samples_written = events_written = 0
    usage_summary = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "openrouter_cost_usd": 0.0,
        "calls_with_exact_cost": 0,
        "latency_seconds": 0.0,
        "calls_with_latency": 0,
        "model_errors": 0,
        "parse_retries": 0,
    }
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            header = {
                "record_type": "eval",
                "schema_version": EXPORT_SCHEMA_VERSION,
                "source_eval_file": str(source) if source else None,
                "source_eval_sha256": sha256_file(source) if source and source.is_file() else None,
                "eval": _jsonable(getattr(eval_log, "eval", None)),
            }
            f.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
            for sample in getattr(eval_log, "samples", None) or []:
                if "Episode:events" not in (getattr(sample, "store", None) or {}):
                    continue
                episode, records = sample_export_records(sample)
                f.write(json.dumps(episode, sort_keys=True, ensure_ascii=False) + "\n")
                samples_written += 1
                for record in records:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    events_written += 1
                    usage_summary["parse_retries"] += int(
                        record["event"].get("parse_retries") or 0
                    )
                    for call in record["model_calls"]:
                        usage = call["usage"]
                        usage_summary["model_calls"] += 1
                        for target, source_key in (
                            ("input_tokens", "input_tokens"),
                            ("output_tokens", "output_tokens"),
                            ("reasoning_tokens", "reasoning_tokens"),
                            ("cache_read_tokens", "cache_read_tokens"),
                            ("cache_write_tokens", "cache_write_tokens"),
                        ):
                            usage_summary[target] += int(usage.get(source_key) or 0)
                        cost = usage.get("openrouter_cost")
                        if cost is not None:
                            usage_summary["openrouter_cost_usd"] += float(cost)
                            usage_summary["calls_with_exact_cost"] += 1
                        latency = call.get("latency_seconds")
                        if latency is not None:
                            usage_summary["latency_seconds"] += float(latency)
                            usage_summary["calls_with_latency"] += 1
                        if call["output"].get("error") or call.get("event_error"):
                            usage_summary["model_errors"] += 1
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "samples": samples_written,
        "events": events_written,
        "usage": usage_summary,
    }


def finalize_run(artifacts: RunArtifacts, eval_logs: Iterable[Any], success: bool) -> dict[str, Any]:
    """Rehydrate eval-set headers, export their samples, then seal the completion record.

    Inspect's ``eval_set`` deliberately returns header-only EvalLog objects. Never treat those
    headers as full logs: doing so produces a valid-looking, empty export. A successful eval must
    export every logged sample or finalisation fails before ``completion.json`` is written.
    """
    export_dir = artifacts.directory / "analysis-ready"
    exports = []
    for header in eval_logs:
        location = getattr(header, "location", None)
        log = header
        if getattr(log, "samples", None) is None:
            if not location:
                raise RuntimeError("eval_set returned a header without a readable log location")
            log = read_eval_log(str(location))
        if getattr(log, "samples", None) is None:
            raise RuntimeError(f"full eval log has no samples: {location}")
        result = export_eval_log(log, export_dir, location)
        if success and result["samples"] != len(log.samples):
            raise RuntimeError(
                f"analysis export incomplete for {location}: "
                f"exported {result['samples']} of {len(log.samples)} samples"
            )
        exports.append(result)
    completion = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "launch_id": artifacts.launch_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "success": bool(success),
        "exports": exports,
    }
    _atomic_json(artifacts.directory / "completion.json", completion, exclusive=True)
    error_path = artifacts.directory / "finalization-error.json"
    if error_path.exists():
        error_path.unlink()
    return completion


def record_finalization_error(
        artifacts: RunArtifacts, error_type: str, *, eval_success: bool) -> None:
    """Record a retryable export failure without mislabelling the paid evaluation itself."""
    value = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "launch_id": artifacts.launch_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "eval_success": bool(eval_success),
        "finalization_success": False,
        "retryable": True,
        "error_type": error_type,
    }
    _atomic_json(artifacts.directory / "finalization-error.json", value)


def record_interrupted_run(artifacts: RunArtifacts, error_type: str) -> None:
    """Record a failed launch without copying exception text that could contain provider secrets."""
    value = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "launch_id": artifacts.launch_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "error_type": error_type,
        "exports": [],
    }
    _atomic_json(artifacts.directory / "completion.json", value, exclusive=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export arena .eval logs to analysis-ready JSONL.")
    ap.add_argument("paths", nargs="+", help=".eval files or directories")
    ap.add_argument("--out-dir", help="destination; default is analysis-ready/ beside each log")
    args = ap.parse_args()
    files = [path for path, _root in episode_io.expand_eval_paths(args.paths)]
    if not files:
        raise SystemExit("no .eval files found")
    for path in files:
        destination = Path(args.out_dir) if args.out_dir else path.parent / "analysis-ready"
        result = export_eval_log(read_eval_log(str(path)), destination, path)
        print(f"{path.name}: {result['events']} events -> {result['path']}")


if __name__ == "__main__":
    main()
