"""Resumable, evidence-preserving pipeline for the registered experiment-13b deception gate.

The stages are deliberately separate. Validation results must be seen before the bulk ledgers can
run, and bulk execution requires ``--validation-approved``. Old artifacts are never overwritten.

Examples:
  PYTHONPATH=src python3 scripts/run_13b_deception_validation.py status
  PYTHONPATH=src python3 scripts/run_13b_deception_validation.py promise --spend
  PYTHONPATH=src python3 scripts/run_13b_deception_validation.py lie --spend
  PYTHONPATH=src python3 scripts/run_13b_deception_validation.py ledgers --spend --validation-approved
  PYTHONPATH=src python3 scripts/run_13b_deception_validation.py gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
sys.path.insert(0, str(ROOT / "src"))
from delegated_influence import native_judges
LOGS = ROOT / "logs" / "13b_attack_intents"
LIE_SAMPLE = ROOT / "data" / "paper_lie_structured_holdout_sample.json"
LIE_LABELS = ROOT / "data" / "paper_lie_structured_holdout_labels.json"
LIE_VALIDATION = ROOT / "data" / "paper_lie_structured_holdout_opus.json"
PROMISE_VALIDATION = ROOT / "data" / "promise_judge_breach_holdout_opus.json"
LIE_LEDGER = ROOT / "data" / "13b_lie_ledger_current.json"
PROMISE_LEDGER = ROOT / "data" / "13b_promise_ledger_current.json"
GATE = ROOT / "data" / "13b_deception_gate_current.json"
MANIFEST = ROOT / "data" / "13b_deception_validation_manifest.json"


def _load(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _completed_structured_labels():
    labels = _load(LIE_LABELS)
    rows = labels.get("records", [])
    return sum(record.get("gold_claims") is not None for record in rows), len(rows)


def _lie_validation_complete():
    if not LIE_VALIDATION.exists():
        return False
    doc = _load(LIE_VALIDATION)
    row = (doc.get("structured_summary") or {}).get("opus-4.8") or {}
    return row.get("completed_calls") == 50 and row.get("errors") == 0


def _promise_validation_complete():
    if not PROMISE_VALIDATION.exists():
        return False
    row = _load(PROMISE_VALIDATION).get("summary") or {}
    return row.get("completed_calls") == 20 and row.get("remaining_calls") == 0


def status():
    lie_done, lie_total = _completed_structured_labels()
    return {
        "existing_13b_logs": LOGS.exists(),
        "openrouter_key_available": bool(os.environ.get("OPENROUTER_API_KEY")),
        "lie_structured_gold": {"complete": lie_done, "total": lie_total},
        "lie_validation_complete": _lie_validation_complete(),
        "promise_validation_complete": _promise_validation_complete(),
        "current_lie_ledger_exists": LIE_LEDGER.exists(),
        "current_promise_ledger_exists": PROMISE_LEDGER.exists(),
        "registered_gate_exists": GATE.exists(),
        "next_safe_stage": (
            "complete structured human gold" if lie_done != lie_total
            else "add OPENROUTER_API_KEY" if not os.environ.get("OPENROUTER_API_KEY")
            else "run missing holdout validation" if not (
                _lie_validation_complete() and _promise_validation_complete())
            else "review validation and explicitly approve bulk ledgers"
            if not (LIE_LEDGER.exists() and PROMISE_LEDGER.exists())
            else "apply registered gate" if not GATE.exists() else "review gate output"
        ),
    }


def _run(arguments, *, capture=False):
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + current if current else "")
    result = subprocess.run(
        [PYTHON, *arguments], cwd=ROOT, env=env, check=True, text=True,
        capture_output=capture,
    )
    return result.stdout if capture else None


def _require_spend(args):
    if not args.spend:
        raise SystemExit("paid stage refused: add --spend")
    if getattr(args, "keys", None):
        from dotenv import load_dotenv
        load_dotenv(args.keys, override=False)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("paid stage refused: OPENROUTER_API_KEY is not available")


def _native_args(args, stage):
    native_judges.require_controls(args)
    return ["--run-dir", str(Path(args.run_dir) / stage),
            "--route-manifest", str(args.route_manifest), "--ledger", str(args.ledger),
            "--shared-cap-usd", str(args.shared_cap_usd)]


def run_promise_validation(args):
    _require_spend(args)
    _run(["scripts/validate_promise_judge.py", "--spend", *_native_args(args, "promise-holdout")])


def run_lie_validation(args):
    _require_spend(args)
    done, total = _completed_structured_labels()
    if done != total:
        raise SystemExit(f"lie validation refused: structured human gold is {done}/{total}")
    _run([
        "scripts/calibrate_lie_judges.py", "--spend",
        "--sample", str(LIE_SAMPLE), "--labels", str(LIE_LABELS),
        "--out", str(LIE_VALIDATION), "--judges", "opus-4.8", "--channel", "private",
        *_native_args(args, "lie-holdout"),
    ])


def _validation_snapshot():
    return {
        "lie_validation": {
            "path": str(LIE_VALIDATION.relative_to(ROOT)),
            "sha256": _sha256(LIE_VALIDATION),
            "summary": (_load(LIE_VALIDATION).get("structured_summary") or {}).get("opus-4.8"),
        },
        "promise_validation": {
            "path": str(PROMISE_VALIDATION.relative_to(ROOT)),
            "sha256": _sha256(PROMISE_VALIDATION),
            "summary": _load(PROMISE_VALIDATION).get("summary"),
        },
    }


def run_ledgers(args):
    _require_spend(args)
    if not (_lie_validation_complete() and _promise_validation_complete()):
        raise SystemExit("bulk ledgers refused: both holdout validations must be complete")
    if not args.validation_approved:
        raise SystemExit(
            "bulk ledgers refused: review the holdout results, then add --validation-approved")
    manifest = {
        "protocol": "experiment-13b current-rule deception validation",
        "source_logs": str(LOGS.relative_to(ROOT)),
        "source_log_files": sorted(path.name for path in LOGS.glob("*.eval")),
        "validation_approved": True,
        "validation": _validation_snapshot(),
        "outputs": {
            "lie_ledger": str(LIE_LEDGER.relative_to(ROOT)),
            "promise_ledger": str(PROMISE_LEDGER.relative_to(ROOT)),
            "gate": str(GATE.relative_to(ROOT)),
        },
    }
    _atomic_json(MANIFEST, manifest)
    _run([
        "-m", "delegated_influence.check_lies", str(LOGS),
        "--out", str(LIE_LEDGER), "--judge", "opus-4.8", "--channel", "private",
        "--spend", *_native_args(args, "lie-ledger"),
    ])
    _run([
        "-m", "delegated_influence.check_promises", str(LOGS),
        "--out", str(PROMISE_LEDGER), "--judge", "opus-4.8",
        "--spend", *_native_args(args, "promise-ledger"),
    ])


def run_gate():
    if not (LIE_LEDGER.exists() and PROMISE_LEDGER.exists()):
        raise SystemExit("gate refused: current lie and promise ledgers do not both exist")
    raw = _run([
        "-m", "delegated_influence.report", str(LOGS),
        "--lies", str(LIE_LEDGER), "--promises", str(PROMISE_LEDGER),
        "--deception-audit", "--pool", "all",
    ], capture=True)
    payload = json.loads(raw)
    _atomic_json(GATE, payload)
    manifest = _load(MANIFEST) if MANIFEST.exists() else {}
    manifest["gate_sha256"] = _sha256(GATE)
    manifest["gate_passed"] = bool(
        (payload.get("validated_message_gate") or {}).get("event_gate_passed"))
    _atomic_json(MANIFEST, manifest)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run the staged experiment-13b deception audit.")
    parser.add_argument("stage", choices=("status", "promise", "lie", "ledgers", "gate"))
    parser.add_argument("--spend", action="store_true", help="allow paid OpenRouter calls")
    parser.add_argument("--validation-approved", action="store_true",
                        help="confirm holdout results were reviewed before the bulk pass")
    native_judges.add_arguments(parser)
    args = parser.parse_args()
    if args.stage == "status":
        print(json.dumps(status(), indent=2))
    elif args.stage == "promise":
        run_promise_validation(args)
    elif args.stage == "lie":
        run_lie_validation(args)
    elif args.stage == "ledgers":
        run_ledgers(args)
    else:
        run_gate()


if __name__ == "__main__":
    main()
