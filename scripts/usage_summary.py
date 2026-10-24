#!/usr/bin/env python3
"""Offline usage summary over Inspect logs and native episode exports. No API calls.

Prints tokens per turn / episode / arm, split by seat count and model. Input includes cached
tokens once; reasoning is a subset of completion tokens. Reports provider billing separately
from the optional blended estimate, plus retries, forfeits, and context growth by round.
Served-provider host mix comes from raw Inspect responses or native usage records.

Usage:
  .venv/bin/python scripts/usage_summary.py logs/03_five_principals logs/04_attack_complete
  .venv/bin/python scripts/usage_summary.py logs            # every experiment under logs/
  .venv/bin/python scripts/usage_summary.py logs/04* --usd-per-mtok 2.0
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Runnable as a plain script (no PYTHONPATH=src needed): put src/ on the path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inspect_ai.log import read_eval_log  # noqa: E402

from delegated_influence import costs, episodes, models  # noqa: E402

# spec (as it appears on a model event, e.g. 'openrouter/z-ai/glm-5.2') -> roster key ('glm-5.2').
_SPEC_TO_KEY = {}
for _key in models.MODEL_REGISTRY:
    # MODEL_REGISTRY lists paper instruments before deprecated auxiliary aliases. The first
    # name is therefore the canonical reporting bucket when two keys share an endpoint.
    _SPEC_TO_KEY.setdefault(models.openrouter_spec(_key), _key)


def model_key(spec):
    """A model event's spec -> the roster key when known, else the spec without 'openrouter/'."""
    spec = spec or ""
    return _SPEC_TO_KEY.get(spec, spec.removeprefix(models.OPENROUTER))


def event_tokens(event):
    """(total input, completion) tokens for one event, retries summed; None for call-less events.

    Forced control pulls (and any event with an empty `usage` list) made no model call, so they are
    not turns and return None. Forfeited turns DID call the model and are counted (they cost money).
    Within the returned tuple, a missing token bucket makes that input or output total None.
    """
    usage = event.get("usage") or []
    if not usage:
        return None
    prompt = _complete_sum([_input_tokens(u) for u in usage])
    completion = _complete_sum([_usage_count(u, "completion_tokens", "output_tokens") for u in usage])
    return prompt, completion


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage_count(usage, *keys):
    # Canonical keys come first and win conflicts. Only absent/null values use an alias;
    # an invalid canonical count stays unknown instead of falling back to the alias.
    return _count(next((usage[key] for key in keys if usage.get(key) is not None), None))


def _complete_sum(values):
    return sum(values) if all(value is not None for value in values) else None


def _input_tokens(usage):
    # Inspect and the native adapter store uncached input separately. Reasoning is already output.
    return _complete_sum([_usage_count(usage, *keys) for keys in (
        ("prompt_tokens", "input_tokens"), ("cache_read_tokens", "input_tokens_cache_read"),
        ("cache_write_tokens", "input_tokens_cache_write"))])


TOKEN_BUCKETS = (
    ("uncached_input_tokens", "unknown_uncached_input_calls", ("prompt_tokens", "input_tokens")),
    ("completion", "unknown_completion_calls", ("completion_tokens", "output_tokens")),
    ("cache_read_tokens", "unknown_cache_read_calls", ("cache_read_tokens", "input_tokens_cache_read")),
    ("cache_write_tokens", "unknown_cache_write_calls", ("cache_write_tokens", "input_tokens_cache_write")),
    ("reasoning_tokens", "unknown_reasoning_calls", ("reasoning_tokens",)),
)


def _known_cost(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _totals():
    return {"turns": 0, "calls": 0, "prompt": 0, "unknown_input_calls": 0,
            **{name: 0 for field, unknown, _ in TOKEN_BUCKETS for name in (field, unknown)},
            "retries": 0, "forfeits": 0, "reported_cost_usd": 0.0,
            "cost_calls": 0, "unknown_cost_calls": 0,
            "stop_reasons": Counter(), "unknown_stop_reason_calls": 0}


def _round_totals():
    return {"calls": 0, "input_tokens": 0, "input_known_calls": 0,
            "observation_characters": 0, "observation_known_calls": 0}


def new_aggregates():
    """The three empty accumulators: (seats, model) totals, per-arm totals, (model, arm) host mix."""
    by_model = defaultdict(lambda: {**_totals(), "eps": set()})
    by_arm = defaultdict(lambda: {**_totals(), "seats": set(), "episodes": 0,
                                  "ep_tokens": [], "rounds": defaultdict(_round_totals)})
    host_mix = defaultdict(Counter)
    return by_model, by_arm, host_mix


def _accumulate_events(arm, sample_id, seat_map, events, by_model, by_arm):
    seats = len(seat_map)
    a = by_arm[arm]
    a["seats"].add(seats)
    a["episodes"] += 1
    ep_tokens = 0
    for e in events:
        usage = e.get("usage") or []
        m = by_model[(seats, seat_map.get(e.get("agent")))]
        for bucket in (a, m):
            bucket["forfeits"] += bool(e.get("forfeit"))
            bucket["retries"] += max(len(usage) - 1, _count(e.get("parse_retries")) or 0)
            bucket["calls"] += len(usage)
            for u in usage:
                reason = u.get("stop_reason")
                if isinstance(reason, str) and reason:
                    bucket["stop_reasons"][reason] += 1
                else:
                    bucket["unknown_stop_reason_calls"] += 1
                values = [(field, unknown, _usage_count(u, *keys)) for field, unknown, keys in TOKEN_BUCKETS]
                values.append(("prompt", "unknown_input_calls", _input_tokens(u)))
                for field, unknown, value in values:
                    bucket[unknown if value is None else field] += 1 if value is None else value
        if e.get("round") is not None:
            r = a["rounds"][e["round"]]
            for u in usage:
                r["calls"] += 1
                input_tokens = _input_tokens(u)
                if input_tokens is not None:
                    r["input_tokens"] += input_tokens
                    r["input_known_calls"] += 1
                characters = _count(u.get("observation_characters"))
                if characters is not None:
                    r["observation_characters"] += characters
                    r["observation_known_calls"] += 1
        toks = event_tokens(e)
        if toks is None:
            continue
        prompt, completion = toks
        for bucket in (a, m):
            bucket["turns"] += 1
        m["eps"].add(sample_id)
        if prompt is None or completion is None:
            ep_tokens = None
        elif ep_tokens is not None:
            ep_tokens += prompt + completion
    a["ep_tokens"].append(ep_tokens)


def _billing(bucket, cost):
    bucket["cost_calls"] += 1
    if _known_cost(cost):
        bucket["reported_cost_usd"] += cost
    else:
        bucket["unknown_cost_calls"] += 1


def accumulate_sample(arm, sample, by_model, by_arm, host_mix):
    """Fold a historical EvalSample into the aggregates, retaining raw billing evidence."""
    store = sample.store or {}
    if "Episode:events" not in store:
        return False
    seat_map = store["Episode:seat_map"]
    _accumulate_events(arm, sample.id, seat_map, store["Episode:events"], by_model, by_arm)
    # served-provider host mix, from Inspect's own model events (not our Episode events).
    billing_calls = Counter()
    for me in (sample.events or []):
        if getattr(me, "event", None) != "model":
            continue
        call = getattr(me, "call", None)
        resp = getattr(call, "response", None) if call is not None else None
        mdl = model_key(getattr(me, "model", ""))
        raw_usage = (resp.get("usage") or {}) if isinstance(resp, dict) else {}
        # ModelUsage.total_cost can be a catalog estimate; only raw OpenRouter bills count here.
        cost = (raw_usage.get("cost")
                if str(getattr(me, "model", "")).startswith(models.OPENROUTER) else None)
        _billing(by_arm[arm], cost)
        _billing(by_model[(len(seat_map), mdl)], cost)
        billing_calls[mdl] += 1
        host = resp.get("provider") if isinstance(resp, dict) else None
        if host:
            host_mix[(mdl, arm)][host] += 1
    expected_calls = Counter()
    for event in store["Episode:events"]:
        expected_calls[seat_map.get(event.get("agent"))] += len(event.get("usage") or [])
    missing_calls = max(sum(expected_calls.values()) - sum(billing_calls.values()), 0)
    for mdl, expected in expected_calls.items():
        missing = min(max(expected - billing_calls[mdl], 0), missing_calls)
        for _ in range(missing):
            _billing(by_arm[arm], None)
            _billing(by_model[(len(seat_map), mdl)], None)
        missing_calls -= missing
    return True


def accumulate_record(arm, record, by_model, by_arm, host_mix):
    """Accumulate one native record returned by the canonical episode loader."""
    seat_map = record["seat_map"]
    _accumulate_events(arm, record.get("id"), seat_map, record["events"], by_model, by_arm)
    for event in record["events"]:
        mdl = seat_map.get(event.get("agent"))
        for usage in event.get("usage") or []:
            cost = (usage.get("total_cost")
                    if usage.get("cost_source") == "openrouter_response_usage_cost" else None)
            _billing(by_arm[arm], cost)
            _billing(by_model[(len(seat_map), mdl)], cost)
            if usage.get("served_provider"):
                host_mix[(mdl, arm)][usage["served_provider"]] += 1


def collect(files):
    """Read the logs once and accumulate the token + host-mix aggregates.

    ``files`` are (file, input_root) pairs from episodes.expand_eval_paths; the arm label is the
    run name every other reader uses."""
    by_model, by_arm, host_mix = new_aggregates()
    for f, root in files:
        _relative, arm = episodes.run_name(f, root)
        log = read_eval_log(str(f))
        for s in (log.samples or []):
            accumulate_sample(arm, s, by_model, by_arm, host_mix)
    return by_model, by_arm, host_mix


def collect_paths(paths):
    """Read historical logs and native exports once each; retain existing legacy run labels."""
    paths = list(paths)
    return _collect_expanded(episodes.expand_eval_paths(paths), episodes.expand_native_paths(paths))


def _collect_expanded(files, native_files):
    """Collect already expanded files so the CLI does not scan its input directories twice."""
    by_model, by_arm, host_mix = collect(files)
    for file, root in native_files:
        # One native file per load bounds memory when saved prompts make a run large.
        for record in episodes.load_episodes([file]):
            metadata = record.get("metadata") or {}
            arm = metadata.get("experiment_arm") or record.get("experiment_arm")
            config = metadata.get("experiment_config") or record.get("experiment_config")
            if arm and config:
                arm = f"{config}/{arm}"
            elif not arm:
                _relative, arm = episodes.run_name(file, root)
            accumulate_record(arm, record, by_model, by_arm, host_mix)
    return by_model, by_arm, host_mix


def _seats_label(seat_set):
    return ",".join(str(n) for n in sorted(seat_set))


def _format_number(value, places=0):
    return "unknown" if value is None else f"{value:,.{places}f}"


def _complete_tokens(bucket):
    inputs = bucket["prompt"] if bucket["calls"] and not bucket["unknown_input_calls"] else None
    outputs = bucket["completion"] if bucket["calls"] and not bucket["unknown_completion_calls"] else None
    return inputs, outputs, _complete_sum([inputs, outputs])


def _per(value, count):
    return value / count if value is not None and count else None


def print_pricing_table(by_model):
    """Tokens PER TURN, split by seat count and model — the basis for pricing a bigger arm."""
    print("== tokens per turn, by model x seat-count (pooled over arms) "
          "— the pricing basis ==")
    print(f"{'model':<20} {'seats':>5} {'turns':>7} {'input/turn':>12} "
          f"{'compl/turn':>11} {'total/turn':>11} {'total_tok':>12}")
    rows = sorted(by_model.items(), key=lambda kv: (str(kv[0][1]), kv[0][0]))
    combined = _totals()
    for (seats, mdl), v in rows:
        t = v["turns"]
        prompt, completion, total = _complete_tokens(v)
        print(f"{str(mdl):<20} {seats:>5} {t:>7} {_format_number(_per(prompt, t)):>12} "
              f"{_format_number(_per(completion, t)):>11} {_format_number(_per(total, t)):>11} {_format_number(total):>12}")
        for key in ("turns", "calls", "prompt", "completion", "unknown_input_calls", "unknown_completion_calls"):
            combined[key] += v[key]
    if combined["turns"]:
        t = combined["turns"]
        prompt, completion, total = _complete_tokens(combined)
        print(f"{'ALL':<20} {'':>5} {t:>7} {_format_number(_per(prompt, t)):>12} "
              f"{_format_number(_per(completion, t)):>11} {_format_number(_per(total, t)):>11} {_format_number(total):>12}")
    print()


def print_arm_table(by_arm, usd_per_mtok):
    """Tokens PER EPISODE and PER ARM, with an estimated cost at a flat blended price."""
    print(f"== tokens per episode & per arm  (est cost at ${usd_per_mtok:g}/Mtok, blended) ==")
    print(f"{'arm':<26} {'seats':>5} {'eps':>4} {'turns':>7} {'tok/turn':>9} "
          f"{'tok/episode':>12} {'total_tok':>13} {'est_usd':>9}")
    arm_totals = []
    for arm, v in sorted(by_arm.items()):
        _prompt, _completion, total = _complete_tokens(v)
        tok_turn = _per(total, v["turns"])
        tok_ep = _per(_complete_sum(v["ep_tokens"]), len(v["ep_tokens"]))
        usd = total * usd_per_mtok / 1e6 if total is not None else None
        print(f"{arm:<26} {_seats_label(v['seats']):>5} {v['episodes']:>4} {v['turns']:>7} "
              f"{_format_number(tok_turn):>9} {_format_number(tok_ep):>12} {_format_number(total):>13} {_format_number(usd, 2):>9}")
        arm_totals.append(total)
    grand_tok = _complete_sum(arm_totals)
    grand_usd = grand_tok * usd_per_mtok / 1e6 if grand_tok is not None else None
    print(f"{'TOTAL':<26} {'':>5} {'':>4} {'':>7} {'':>9} {'':>12} "
          f"{_format_number(grand_tok):>13} {_format_number(grand_usd, 2):>9}")
    print()


def print_host_mix(host_mix):
    """Served-provider host mix per model per arm — the cross-host routing drift check."""
    if not host_mix:
        print("== served-provider host mix ==  (no provider host recorded in these logs)\n")
        return
    print("== served-provider host mix, by model x arm  "
          "(open-weight models drift across hosts) ==")
    for (mdl, arm) in sorted(host_mix):
        counter = host_mix[(mdl, arm)]
        total = sum(counter.values())
        parts = ", ".join(f"{host} {n} ({n / total:.0%})"
                          for host, n in counter.most_common())
        print(f"  {mdl:<18} {arm:<26} {total:>5} calls: {parts}")
    print()


def print_stop_reason_table(by_arm):
    """Show one normalized stop reason per recorded model call."""
    print("== generation stop reasons, by arm ==")
    for arm, values in sorted(by_arm.items()):
        parts = [f"{reason} {count}" for reason, count in values["stop_reasons"].most_common()]
        if values["unknown_stop_reason_calls"]:
            parts.append(f"unknown {values['unknown_stop_reason_calls']}")
        details = ", ".join(parts) if parts else "none recorded"
        print(f"  {arm:<26} {values['calls']:>5} calls: {details}")
    print()


def print_cost_table(by_arm):
    """Reported bills are a subtotal; missing bills never imply free calls."""
    print("== provider billing and attempts (USD; reported subtotal excludes unknown bills) ==")
    print(f"{'arm':<26} {'calls':>7} {'reported subtotal':>18} {'unknown bills':>14} "
          f"{'retries':>7} {'forfeits':>8} {'uncached':>12} {'cache read':>12} {'cache write':>12} "
          f"{'reasoning':>12}")
    for arm, v in sorted(by_arm.items()):
        def subtotal(field, unknown):
            if not v["calls"] or v[unknown] == v["calls"]:
                return "unknown"
            return f"{v[field]:,}" + ("+?" if v[unknown] else "")
        print(f"{arm:<26} {v['calls']:>7,} {v['reported_cost_usd']:>18.6f} "
              f"{v['unknown_cost_calls']:>14,} {v['retries']:>7,} {v['forfeits']:>8,} "
              f"{subtotal('uncached_input_tokens', 'unknown_uncached_input_calls'):>12} "
              f"{subtotal('cache_read_tokens', 'unknown_cache_read_calls'):>12} "
              f"{subtotal('cache_write_tokens', 'unknown_cache_write_calls'):>12} "
              f"{subtotal('reasoning_tokens', 'unknown_reasoning_calls'):>12}")
    print("Token columns show recorded subtotals; +? marks additional calls with unknown tokens. "
          "Cache input is counted once; reasoning is already in completion.\n")


def print_context_table(by_arm):
    """Show context growth without turning absent measurements into zero."""
    print("== context by round (means use known calls only; coverage is known/stored calls) ==")
    print(f"{'arm':<26} {'round':>6} {'mean input/call':>16} {'input coverage':>16} "
          f"{'mean observation chars':>24} {'obs coverage':>16}")
    for arm, v in sorted(by_arm.items()):
        if not v["rounds"]:
            print(f"{arm:<26} {'unknown':>6} {'unknown':>16} {'unknown':>16} "
                  f"{'unknown':>24} {'unknown':>16}")
        for round_, r in sorted(v["rounds"].items()):
            input_mean = (f"{r['input_tokens'] / r['input_known_calls']:,.0f}"
                          if r["input_known_calls"] else "unknown")
            obs_mean = (f"{r['observation_characters'] / r['observation_known_calls']:,.0f}"
                        if r["observation_known_calls"] else "unknown")
            input_coverage = f"{r['input_known_calls']}/{r['calls']}"
            obs_coverage = f"{r['observation_known_calls']}/{r['calls']}"
            print(f"{arm:<26} {str(round_):>6} {input_mean:>16} {input_coverage:>16} "
                  f"{obs_mean:>24} {obs_coverage:>16}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Offline usage summary over .eval logs and native episode exports (no API calls).")
    ap.add_argument("paths", nargs="+", help=".eval / .episode.json[.gz] files, directories, or globs")
    ap.add_argument("--usd-per-mtok", type=float, default=costs.DEFAULT_COST['usd_per_mtok'],
                    help="blended $/Mtok for the separate estimate column (default %(default)g)")
    args = ap.parse_args()

    files = episodes.expand_eval_paths(args.paths)
    native_files = episodes.expand_native_paths(args.paths)
    if not files and not native_files:
        sys.exit(f"no .eval logs or native episode exports matched: {args.paths}")
    by_model, by_arm, host_mix = _collect_expanded(files, native_files)
    if not by_arm:
        sys.exit(f"no stored episodes found in {len(files) + len(native_files)} file(s)")

    print(f"# usage_summary over {len(files)} .eval file(s), {len(native_files)} native export(s), "
          f"{sum(a['episodes'] for a in by_arm.values())} episodes\n")
    print("Native exports contain completed episodes only. For running or failed native games, use:\n"
          "  python -m delegated_influence.run <config.yaml> --status\n")
    print("Input tokens include uncached input plus cache reads and writes once. "
          "Missing token buckets make totals and blended estimates unknown.\n")
    print_pricing_table(by_model)
    print_arm_table(by_arm, args.usd_per_mtok)
    print_cost_table(by_arm)
    print_stop_reason_table(by_arm)
    print_context_table(by_arm)
    print_host_mix(host_mix)


if __name__ == "__main__":
    main()
