"""Guarded native calls for the registered post-hoc lie and promise judges.

The journal saves raw responses before parsing or settling the shared bill.
Unknown responses retain their spending holds. Resuming never buys a known response again.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from delegated_influence import artifacts, episodes, models, native_adapter, runtime, task
from delegated_influence.dispatch import MissingUsageEvidence, RequestBound, amount
from delegated_influence.journal import Journal, RunPaused, canonical


ROOT = Path(__file__).resolve().parents[2]


def model_key(value):
    if value in models.MODEL_REGISTRY:
        return value
    for key in models.MODEL_REGISTRY:
        if models.openrouter_spec(key) == value:
            return key
    raise ValueError("guarded judges require a registered OpenRouter model with a saved route")


async def call_judge(judge, request_key, *args, **kwargs):
    """Keep one paid request identity per source record, including identical message text."""
    if getattr(judge, "native_judge", False):
        kwargs["request_key"] = str(request_key)
    return await judge(*args, **kwargs)


def source_inputs(paths):
    return sorted({path for path, _ in [*episodes.expand_eval_paths(paths),
                                       *episodes.expand_native_paths(paths)]})


class NativeJudgeSession:
    def __init__(self, journal, ledger, manifest, *, factory=None):
        self.journal, self.ledger, self.manifest = journal, ledger, manifest
        self.factory = factory or native_adapter.build_generator
        self.namespace = "judge-" + hashlib.sha256(str(journal.path.resolve()).encode()).hexdigest()[:24]
        journal.bind("judge_routes", manifest)
        journal.bind("judge_spending", {"ledger": str(ledger.path.resolve())})

    def require_saved_call(self, model, instrument, request_key):
        identifier = hashlib.sha256(canonical([model_key(model), instrument, str(request_key)]).encode()).hexdigest()
        row = self.journal.db.execute("SELECT id,status FROM calls WHERE id=?",
                                      (f"{identifier}:judge:0",)).fetchone()
        if row is None or row["status"] != "complete" or self.journal.call(row["id"])["response"] is None:
            raise RunPaused("unbound or uncertain holdout output; retain its original native journal before resume")

    def generator(self, model, instrument):
        key = model_key(model)
        settings = {"max_tokens": task.JUDGE_MAX_TOKENS}
        sampling = task.sampling_instrument(key, settings)
        # Do not inherit a gameplay temperature or reasoning policy into the old judge prompt.
        if sampling["temperature"] is not None or sampling["reasoning_effort"] is not None:
            raise ValueError("native judge sampling must preserve temperature=None and reasoning_effort=None")
        row = self.manifest.get("models", {}).get(key)
        bound = RequestBound.from_recorded(row, sampling)
        self.journal.bind(f"judge_instrument:{key}:{instrument}", sampling)

        async def generate(prompt, *, request_key):
            messages = [{"role": "user", "content": prompt}]
            identifier = hashlib.sha256(canonical([key, instrument, request_key]).encode()).hexdigest()
            metadata = {"purpose": "post-hoc judge", "model": key, "instrument": instrument,
                        "request_key": request_key, "sampling": sampling,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            if self.journal.summary()["uncertain_calls"]:
                raise RunPaused("judge has uncertain calls; review saved evidence before another request")
            budget = float(self.ledger.summary()["configured_cap_usd"])
            episode = self.journal.start_episode(identifier, metadata, 0, budget)
            call_key = f"{self.namespace}:{identifier}:judge:0"
            provider = None
            try:
                if self.journal.needs_dispatch(identifier, "judge", 0):
                    # Fresh route evidence is required only for a new provider request.
                    RequestBound(self.manifest, key, settings)
                    provider = self.factory(key, 0, model_config=settings, request_bound=bound)
                    reservation = self.ledger.reserve(call_key, bound.maximum_usd)
                    if not reservation["dispatch_allowed"]:
                        raise RunPaused("judge spending hold exists without a replayable response")
                call_id, result = self.journal.begin_call(identifier, "judge", 0, messages)
                if result is None:
                    try:
                        def on_request(body):
                            self.journal.save_request(call_id, body)
                            bound.check_request(body)
                            sent = json.loads(body).get("messages")
                            if sent != messages:
                                # The adapter can express text as a single text block.
                                normalized = [{**message, "content": "".join(
                                    block["text"] for block in message["content"])}
                                    if isinstance(message.get("content"), list) else message
                                    for message in (sent or [])]
                                if normalized != messages:
                                    raise RunPaused("judge wire messages differ from the frozen prompt")
                        provider.on_request = on_request
                        response, usage = await asyncio.wait_for(provider(messages), task.CALL_TIMEOUT_S)
                    except BaseException as exc:
                        self.journal.uncertain(call_id, runtime.exception_evidence(exc))
                        raise RunPaused(f"{call_id}: uncertain judge response") from exc
                    result = self.journal.finish_call(call_id, response, dict(usage or {}), None)
                # Raw response and raw usage now survive interruption or a parser failure.
                self.ledger.settle(call_key, result["usage"].get("total_cost"))
                if result["usage"].get("total_cost") is None or result["response"] is None:
                    self.journal.uncertain(call_id, "missing provider bill or response")
                    raise RunPaused("judge bill or response missing; review saved evidence")
                try:
                    bound.check_usage(result["usage"])
                except MissingUsageEvidence as exc:
                    self.journal.uncertain(call_id, str(exc))
                    raise
                except RunPaused as exc:
                    self.ledger.stop(str(exc))
                    self.journal.uncertain(call_id, str(exc))
                    raise
                self.journal.complete(identifier, result)
                input_buckets = [result["usage"].get(field) for field in
                                 ("prompt_tokens", "cache_read_tokens", "cache_write_tokens")]
                usage = {**result["usage"], "journal_call_id": call_id,
                         "input_tokens": sum(input_buckets) if all(value is not None for value in input_buckets) else None,
                         "output_tokens": result["usage"].get("completion_tokens")}
                return result["response"], usage
            except BaseException:
                self.journal.db.execute("UPDATE episodes SET status='paused' WHERE id=?", (identifier,))
                raise
            finally:
                self.journal.write_status(force=True)
                if provider is not None:
                    await provider.aclose()
        return generate


def add_arguments(parser):
    parser.add_argument("--run-dir", help="durable native judge journal directory")
    parser.add_argument("--route-manifest", help="fresh saved OpenRouter routes")
    parser.add_argument("--ledger", help="existing shared spending ledger")
    parser.add_argument("--shared-cap-usd", type=lambda value: amount(value, "shared cap"))
    parser.add_argument("--keys", help="private dotenv file; never copied into artifacts")


def require_controls(args):
    for name in ("run_dir", "route_manifest", "ledger", "shared_cap_usd"):
        if getattr(args, name, None) is None:
            raise SystemExit(f"paid judge refused: add --{name.replace('_', '-')}")
    if args.shared_cap_usd <= 0:
        raise SystemExit("paid judge refused: shared cap must be positive")


@contextmanager
def open_session(args, *, inputs, binding):
    require_controls(args)
    if getattr(args, "keys", None):
        from dotenv import load_dotenv
        load_dotenv(args.keys, override=False)
    route_path = Path(args.route_manifest)
    manifest = json.loads(route_path.read_bytes())
    source_paths = artifacts.runtime_source_paths(ROOT) + [ROOT / "scripts" / name for name in (
        "run_13b_deception_validation.py", "calibrate_lie_judges.py", "validate_promise_judge.py")]
    source_paths = sorted(set(source_paths))
    hashes = {str(path.relative_to(ROOT)): artifacts.sha256_file(path) for path in source_paths}
    input_hashes = {str(Path(path).resolve()): artifacts.sha256_file(Path(path)) for path in inputs}
    with Journal(Path(args.run_dir) / "run.sqlite3") as journal:
        journal.bind("source_files", hashes)
        journal.bind("runtime_environment", runtime.RUNTIME_ENVIRONMENT)
        journal.bind("judge_inputs", input_hashes)
        journal.bind("judge_output", binding)
        snapshot = Path(args.run_dir) / "source-snapshot.tar.gz"
        if not snapshot.exists():
            artifacts._snapshot_source(ROOT, source_paths, snapshot)
        journal.bind("source_snapshot_sha256", artifacts.sha256_file(snapshot))
        previous = journal.db.execute("SELECT value FROM meta WHERE key='spending'").fetchone()
        if previous is None and journal.db.execute("SELECT 1 FROM calls LIMIT 1").fetchone():
            raise RunPaused("existing judge calls have no bound spending policy")
        policy = (json.loads(previous["value"]) if previous else
                  {"ledger": str(Path(args.ledger).resolve()), "cap_usd": str(args.shared_cap_usd)})
        options = {"ledger": str(Path(args.ledger).resolve()), "cap_usd": args.shared_cap_usd}
        # Pass a policy even on first launch: this task must use an existing shared ledger.
        ledger, audits = runtime.open_audited_spending_ledger(policy, options)
        try:
            journal.bind("spending", policy)
            if audits:
                runtime.record_spending_cap_increase(journal, policy, str(args.shared_cap_usd), audits)
            yield NativeJudgeSession(journal, ledger, manifest)
        finally:
            (journal.path.parent / "spending-status.json").write_text(json.dumps(ledger.summary(), indent=2) + "\n")
            ledger.close()
