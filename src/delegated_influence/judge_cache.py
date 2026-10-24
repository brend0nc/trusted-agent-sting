"""Atomic per-message checkpoints for paid post-hoc judge passes.

Only successful structured judgments are cached. A timeout, provider error, or strict-schema
failure remains retryable, while every earlier paid result survives process interruption.
"""
from __future__ import annotations

import hashlib
import json
import os
import fcntl
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


def _input_hash(judge_input: str) -> str:
    return hashlib.sha256(str(judge_input).encode("utf-8")).hexdigest()


def checkpoint_identity(record: dict[str, Any]) -> str:
    """Stable episode identity, with a content hash fallback for in-memory/test records."""
    if record.get("_judgment_checkpoint_identity"):
        return str(record["_judgment_checkpoint_identity"])
    if record.get("checkpoint_episode_id"):
        identity = str(record["checkpoint_episode_id"])
    elif record.get("sample_uuid"):
        identity = str(record["sample_uuid"])
    else:
        canonical = json.dumps(
            {"sample_id": record.get("sample_id"), "seat_map": record.get("seat_map"),
             "events": record.get("events")},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
        )
        identity = "content:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    record["_judgment_checkpoint_identity"] = identity
    return identity


def judgment_key(record: dict[str, Any], event_index: int, channel: str) -> str:
    return f"{checkpoint_identity(record)}|{channel}|{event_index}"


class JudgmentCheckpoint:
    """Validated, atomic JSON cache tied to one exact judge instrument and parameter set."""

    def __init__(
        self, path: str | Path, instrument: str, parameters: dict[str, Any],
        compatible_parameters: list[dict[str, Any]] | None = None,
    ):
        self.path = Path(path)
        self.expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "instrument": instrument,
            "parameters": parameters,
        }
        self.compatible = [
            {**self.expected, "parameters": value}
            for value in (compatible_parameters or [])
        ]
        if self.path.exists():
            doc = json.loads(self.path.read_text())
            self.doc = self._validate(doc)
        else:
            self.doc = {**self.expected, "records": {}}

    @property
    def lock_path(self):
        return self.path.with_name(self.path.name + ".lock")

    def _validate(self, doc):
        actual = {key: doc.get(key) for key in self.expected}
        if actual != self.expected and actual not in self.compatible:
            raise ValueError(
                f"judge checkpoint {self.path} belongs to a different instrument or run")
        if not isinstance(doc.get("records"), dict):
            raise ValueError(f"judge checkpoint {self.path} has invalid records")
        return doc

    def _latest(self):
        return (self._validate(json.loads(self.path.read_text())) if self.path.exists()
                else {**self.expected, "records": {}})

    def get(self, key: str, judge_input: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            self.doc = self._latest()
        entry = self.doc["records"].get(key)
        if entry is None:
            return None
        if entry.get("input_sha256") != _input_hash(judge_input):
            raise ValueError(f"judge checkpoint input changed for {key}")
        return entry.get("judgment")

    def put(self, key: str, judge_input: str, judgment: Any) -> None:
        entry = {"input_sha256": _input_hash(judge_input), "judgment": judgment}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            latest = self._latest()
            previous = latest["records"].get(key)
            if previous is not None and previous != entry:
                raise ValueError(f"judge checkpoint conflicts for {key}")
            latest["records"][key] = entry
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(latest, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n")
                os.replace(tmp, self.path)
                self.doc = latest
            finally:
                if tmp.exists():
                    tmp.unlink()
