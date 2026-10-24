"""Canonical ingestion and deterministic replay for saved game episodes.

All readers use this module to expand paths, read Inspect logs, recover legacy instrument
metadata, assign transcript ids, and replay score/counter state. The raw ``.eval`` file remains
the source of truth; callers receive plain dictionaries suited to the analysis pipeline.
"""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from inspect_ai.log import read_eval_log

from delegated_influence import experiment, observation, replay


def _glob_input_root(pattern: str) -> Path:
    """The literal directory prefix before the first glob segment."""
    path = Path(pattern)
    parts = path.parts
    literal_count = next(
        (i for i, part in enumerate(parts) if any(char in part for char in "*?[")),
        len(parts),
    )
    if literal_count == 0:
        return Path(path.anchor or ".")
    return Path(*parts[:literal_count])


def expand_eval_paths(paths: Iterable[str | Path]) -> list[tuple[Path, Path]]:
    """Expand files, directories, and globs into unique ``(file, input_root)`` pairs."""
    found: dict[Path, Path] = {}
    for value in paths:
        text = str(value)
        path = Path(text)
        if path.is_dir():
            candidates = sorted(path.glob("**/*.eval"))
            root = path
        elif path.is_file() and path.suffix == ".eval":
            candidates = [path]
            root = path.parent
        else:
            candidates = [Path(item) for item in sorted(glob.glob(text, recursive=True))
                          if item.endswith(".eval")]
            root = _glob_input_root(text)
        for candidate in candidates:
            resolved = candidate.resolve()
            found.setdefault(resolved, root.resolve())
    return sorted(found.items(), key=lambda pair: str(pair[0]))


def run_name(file: Path, root: Path) -> tuple[Path, str]:
    """(path relative to its input root, run label) for one ``.eval`` file.

    A caller may pass either a collection (logs/) or one run (logs/08_resist). The run owner is
    the same in both cases, including when the only file is inside retry/. Run roots are
    identified by a direct eval/manifest or the repository's numbered run convention."""
    try:
        relative = file.relative_to(root)
    except ValueError:
        relative = Path(file.name)
    explicit_run_root = (
        any(root.glob("*.eval"))
        or (root / "run.sqlite3").exists()
        or (root / "run_manifest.json").exists()
        or (root / "completion.json").exists()
        or bool(re.match(r"^\d{2}[a-z]?(?:_|-)", root.name))
    )
    run = (
        root.name if explicit_run_root
        else relative.parts[0] if len(relative.parts) > 1
        else file.parent.name
    )
    return relative, run



def _legacy_usage(usage):
    """Inspect's ModelUsage.total_cost can come from its local price catalog."""
    if usage.get("cost_source") == "openrouter_response_usage_cost":
        return usage
    if usage.get("total_cost") is None:
        return usage
    return {**usage, "total_cost": None,
            "estimated_cost_usd": usage.get("estimated_cost_usd", usage["total_cost"]),
            "cost_source": "unverified_legacy_or_inspect_estimate"}

def read_episode(sample: Any) -> dict[str, Any] | None:
    """Read one persisted Episode store; return ``None`` for stubs/aborted samples."""
    store = getattr(sample, "store", None) or {}
    if "Episode:events" not in store:
        return None
    metadata = getattr(sample, "metadata", None) or {}
    return {
        "checkpoint_episode_id": str(getattr(sample, "uuid", "") or "") or None,
        "seat_map": store["Episode:seat_map"],
        "scores": store["Episode:scores"],
        "events": [{**event, **({"usage": [_legacy_usage(usage) for usage in event["usage"]]}
                                   if "usage" in event else {})}
                   for event in store["Episode:events"]],
        "focal": metadata.get("focal"),
        "seed": metadata.get("seed"),
        "topology": metadata.get("topology", "complete"),
        "messages_off": bool(metadata.get("messages_off")),
        "public_chat": bool(metadata.get("public_chat")),
        "human_board": bool(metadata.get("human_board")),
        "self_pull_reward": metadata.get("self_pull_reward") or 0,
        "personas": metadata.get("personas") or {},
        "principal_seats": metadata.get("principal_seats") or [],
        "rounds": metadata.get("rounds") or max(
            (event.get("round", 0) for event in store["Episode:events"]), default=0),
        "actions_per_round": metadata.get("actions_per_round", 5),
        "declared_intents": bool(metadata.get("declared_intents")),
        "seeded_favours": metadata.get("seeded_favours") or 0,
        "punish_enabled": bool(metadata.get("punish_enabled")),
        "observation_schema": metadata.get("observation_schema"),
        "gameplay_protocol_version": metadata.get("gameplay_protocol_version"),
        "experiment_config": metadata.get("experiment_config"),
        "experiment_arm": metadata.get("experiment_arm"),
        "experiment_stage": metadata.get("experiment_stage"),
        "experiment_identity": metadata.get("experiment_identity"),
    }


def load_episode_rows(
    paths: Iterable[str | Path], *, resolve_instruments: bool = False,
    skip_unreadable: bool = False, include_call_evidence: bool = False,
) -> list[dict[str, Any]]:
    """Load canonical source/context/metadata/record rows from raw Inspect logs."""
    paths = list(paths)
    rows = []
    for file, root in expand_eval_paths(paths):
        try:
            log = read_eval_log(str(file), resolve_attachments=resolve_instruments)
        except Exception as exc:
            if not skip_unreadable:
                raise
            print(f"skip (unreadable): {file}: {exc}", file=sys.stderr)
            continue
        before = len(rows)
        relative, run = run_name(file, root)
        for sample in getattr(log, "samples", None) or []:
            record = read_episode(sample)
            if record is None:
                continue
            metadata = dict(getattr(sample, "metadata", None) or {})
            if resolve_instruments:
                metadata.setdefault(
                    "observation_schema", observation.infer_schema(sample)
                )
                stamped_sampling = metadata.get("sampling_instruments")
                effective_sampling = experiment.infer_sampling_instruments(sample)
                if stamped_sampling and effective_sampling != stamped_sampling:
                    metadata["stamped_sampling_instruments"] = stamped_sampling
                    metadata["sampling_instrument_mismatch"] = True
                metadata["sampling_instruments"] = effective_sampling
                metadata.setdefault(
                    "gameplay_protocol_instrument", experiment.infer_gameplay_protocol(sample)
                )
                metadata.setdefault(
                    "served_instruments", experiment.infer_served_instruments(sample)
                )
                record["observation_schema"] = metadata["observation_schema"]
            rows.append({
                "run": run,
                "eval_file": str(relative),
                "source_eval_file": str(file),
                "commit": getattr(getattr(log.eval, "revision", None), "commit", None),
                "started": str(getattr(log.eval, "created", "")),
                "sample_id": str(getattr(sample, "id", "")),
                "sample_uuid": str(getattr(sample, "uuid", "") or "") or None,
                "metadata": metadata,
                "record": record,
            })
        if len(rows) == before and skip_unreadable:
            print(f"skip (no episodes): {file}", file=sys.stderr)
    for file, root in expand_native_paths(paths):
        try:
            opener = gzip.open if file.suffix == ".gz" else open
            with opener(file, "rt") as stream:
                saved = json.load(stream)
            if saved.get("schema_version") != "native-episode-v1":
                raise ValueError("unsupported native episode schema")
            if saved.get("status") != "complete":
                continue
            # Native records carry an explicit protocol version, without Inspect
            # ModelEvents. Recover the same instrument even for older exports.
            metadata = saved['metadata']
            if not metadata.get('gameplay_protocol_instrument'):
                version = metadata.get('gameplay_protocol_version')
                metadata['gameplay_protocol_instrument'] = (
                    {'status':'versioned', 'version':version} if version is not None
                    else {'status':'unknown'})
            if not saved['record'].get('gameplay_protocol_instrument'):
                saved['record']['gameplay_protocol_instrument'] = metadata['gameplay_protocol_instrument']
            if not include_call_evidence:
                saved["record"] = _compact_native_record(saved["record"], file)
            relative, run = run_name(file, root)
            rows.append({"run": run, "eval_file": str(relative), "source_eval_file": str(file),
                         "commit": saved.get("commit"), "started": saved.get("started", ""),
                         "sample_id": saved["sample_id"], "sample_uuid": saved["episode_id"],
                         "metadata": saved["metadata"], "record": saved["record"]})
        except Exception as exc:
            if not skip_unreadable:
                raise
            print(f"skip (unreadable): {file}: {exc}", file=sys.stderr)
    return rows



def _compact_native_record(record, source):
    """Keep exact histories on disk during bulk measurement; hydrate one transcript on demand."""
    def usage_summary(usage):
        return {key: value for key, value in usage.items()
                if key not in ("provider_request_body", "provider_request_snapshot")}
    events = []
    for event in record["events"]:
        compact = {**event, "usage": [usage_summary(usage) for usage in event.get("usage", [])]}
        if "calls" in event:
            compact["calls"] = [{key: usage_summary(value) if key == "usage" else value
                                 for key, value in call.items()
                                 if key not in ("input", "response", "provider_request_body")}
                                for call in event["calls"]]
        events.append(compact)
    return {**record, "events": events, "call_evidence_source": str(source)}

def expand_native_paths(paths):
    """Find complete native exports; the journal and incomplete games stay out of analysis."""
    found = {}
    for value in paths:
        path = Path(value)
        if path.is_dir():
            candidates = sorted(path.glob("**/*.episode.json")) + sorted(path.glob("**/*.episode.json.gz"))
            root = path
        else:
            candidates = [Path(item) for item in sorted(glob.glob(str(value), recursive=True))
                          if item.endswith((".episode.json", ".episode.json.gz"))]
            root = path.parent if path.is_file() else _glob_input_root(str(value))
        for candidate in candidates:
            found.setdefault(candidate.resolve(), root.resolve())
    return sorted(found.items(), key=lambda pair: str(pair[0]))


_CONTEXT_FIELDS = ("run", "eval_file", "source_eval_file", "commit", "started", "sample_id",
                   "sample_uuid", "metadata")


def load_episodes(paths: Iterable[str | Path], *, skip_unreadable: bool = False,
                  include_call_evidence: bool = False) -> list[dict[str, Any]]:
    """Every readable episode as one flat dict: the game record, its source context
    (run, eval_file, source_eval_file, commit, started, sample_id, sample_uuid, metadata) and a
    stable reader ``id``. Instrument metadata is always resolved. This is the one episode shape
    every analysis, ledger builder, and the site read. Native call bodies stay on disk by default;
    include_call_evidence=True restores them for an individual transcript export."""
    rows = load_episode_rows(paths, resolve_instruments=True, skip_unreadable=skip_unreadable,
                             include_call_evidence=include_call_evidence)
    return assign_reader_ids([
        {**row["record"], **{field: row[field] for field in _CONTEXT_FIELDS}} for row in rows
    ])


def _sanitize(value: Any) -> str:
    return str(value).replace("|", "_").replace("/", "_")


def assign_reader_ids(rows: list[dict[str, Any]], *, field: str = "id") -> list[dict[str, Any]]:
    """Assign stable transcript-reader ids, disambiguating repeated sample ids by file time."""
    counts = {}
    for row in rows:
        key = (row["run"], row["sample_id"])
        counts[key] = counts.get(key, 0) + 1
    for row in rows:
        run, sample_id = row["run"], row["sample_id"]
        if counts[(run, sample_id)] > 1:
            timestamp = Path(row["eval_file"]).name.split("_task_")[0]
            row[field] = f"{_sanitize(run)}--{_sanitize(timestamp)}--{_sanitize(sample_id)}"
        else:
            row[field] = f"{_sanitize(run)}--{_sanitize(sample_id)}"
    return rows
