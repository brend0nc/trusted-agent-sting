"""Preflight: prove every experiment config produces readable, analyzable output BEFORE money.

For each config it (1) builds the tasks (catching config errors), (2) plays one scripted offline
episode per distinct condition through the real engine, (3) runs the measurements and asserts each
config's promised numbers actually populate, and (4) lints every prompt variant against the game
rules. The class of bug this guards against: experiments whose data cannot be read, and prompts
that contradict the validator (both happened; see wiki/pages/analyses/2026-07-03-pre-spend-gap-review.md).

Usage:  uv run python -m delegated_influence.preflight              # all configs
        uv run python -m delegated_influence.preflight configs/03_five_principals.yaml
"""
import asyncio
import glob
import json
import re
import math
import random
import sys
from pathlib import Path

from delegated_influence import experiment, game, report, seat, turn


MINIMUM_OUTPUT_TOKEN_RESERVE = 256


def scripted_action(state, obs, agents):
    """A deterministic bot exercising every mechanic the condition allows: messages (with a
    promise), public statements, pulls, one VERIFIED credit chain, one fabricated credit, and a
    self-pull in the mixed economy."""
    who = obs["agent"]
    idx = agents.index(who)
    nxt = agents[(idx + 1) % len(agents)]
    nxt2 = agents[(idx + 2) % len(agents)]
    rnd, slot = obs["round"], obs["action_number"]
    if not state["messages_off"] and rnd == 1 and slot == 1:
        # ask nxt to pull for nxt2 -> enables a VERIFIED credit later, and is also a promise
        return {"action": "message", "to": nxt,
                "text": f"please pull for {nxt2}; I'll pull for you next round"}
    if state["public_chat"] and rnd == 1 and slot == 2:
        return {"action": "say", "text": f"{who} here: let's all cooperate"}
    if state.get("human_board") and rnd == 1 and slot == 3:
        return {"action": "say_human", "text": f"{who} here: this is my game message for human readers"}
    if state["self_pull_reward"] and rnd == 2 and slot == 1:
        return {"action": "pull", "beneficiary": who}  # the mixed economy's safe option
    if not state["messages_off"] and rnd == 2 and slot == 2:
        prev = agents[(idx - 1) % len(agents)]
        beneficiary = nxt if nxt2 in (who, prev) else nxt2
        credit = prev if prev not in (who, beneficiary) else None
        return {"action": "pull", "beneficiary": beneficiary, "credit": credit}  # prev asked us
    if rnd == 3 and slot == 1 and nxt2 != who:
        return {"action": "pull", "beneficiary": nxt, "credit": nxt2}  # fabricated: nxt2 never asked
    if state["punish_enabled"] and rnd == 3 and slot == 2 and state["scores"][who] >= 1:
        return {"action": "punish", "target": nxt}
    if state["declared_intents"] and rnd == 4 and slot == 1:
        prev = agents[(idx - 1) % len(agents)]
        return {"action": "pull", "beneficiary": prev}  # prev messaged us in round 1: a
        # toward-messager pull, so the intent GAP has both buckets to compare
    return {"action": "pull", "beneficiary": nxt}


def scripted_intent(who, agents, rnd):
    """The scripted pre-round plan: rounds 1 and 4 a pull-plan the turns depart from, round 2
    undecided (exercises the missing-plan guard), later rounds the plan the turns follow."""
    idx = agents.index(who)
    nxt = agents[(idx + 1) % len(agents)]
    nxt2 = agents[(idx + 2) % len(agents)]
    if rnd in (1, 4):
        return {"action": "pull", "beneficiary": nxt2}
    if rnd == 2:
        return None
    return {"action": "pull", "beneficiary": nxt}


def simulate_episode(metadata):
    """Run scripted generators through the same episode loop used by paid runs."""
    seat_map = metadata["seat_map"]
    agents = list(seat_map)
    state = game.state_from_metadata(metadata)
    prompt_chars = []

    def generator(who):
        async def generate(messages):
            prompt_chars.append(sum(len(m["content"]) for m in messages))
            # The bot reads the actual user prompt. It does not own a second turn schedule.
            header = messages[1]["content"].splitlines()[0]
            match = re.match(r"Round (\d+) of \d+, (?:your action (\d+)|before your first action)", header)
            if match is None:
                raise ValueError(f"unrecognized preflight prompt header: {header}")
            rnd, slot = int(match[1]), int(match[2] or 0)
            if slot == 0:
                action = scripted_intent(who, agents, rnd) or {"action": "undecided"}
            else:
                obs = {"agent": who, "round": rnd, "action_number": slot}
                action = scripted_action(state, obs, agents)
                if state.get("free_reports"):
                    # Exercise the same zero-cost report path as live normal turns.
                    action["report"] = {"action": "say_human" if slot % 2 and state.get("human_board") else "say",
                                        "text": "This is a free report in an offline test."}
            return json.dumps(action), {}
        return generate

    asyncio.run(turn.drive_episode(
        state, random.Random(metadata["seed"]), {who: generator(who) for who in agents},
        personas=metadata.get("personas"),
    ))
    if any(event["forfeit"] or any(usage.get("error") for usage in event["usage"])
           for event in state["events"]):
        raise ValueError("scripted preflight failed: a turn forfeited or a generator returned an error")
    average_chars = sum(prompt_chars) / len(prompt_chars) if prompt_chars else 0
    max_tokens_by_model = {
        model: int(instrument.get("max_tokens") or 0)
        for model, instrument in (metadata.get("sampling_instruments") or {}).items()
    }
    output_allowance = (sum(max_tokens_by_model.get(model, 0) for model in seat_map.values())
                        / len(seat_map)) if seat_map else 0
    prompt_profile = {
        "calls": len(prompt_chars),
        "average_input_chars": round(average_chars),
        "maximum_input_chars": max(prompt_chars, default=0),
        "approx_average_input_tokens": math.ceil(average_chars / 4),
        "configured_average_output_allowance": math.ceil(output_allowance),
        "minimum_tokens_per_call": (
            math.ceil(average_chars / 4) + MINIMUM_OUTPUT_TOKEN_RESERVE
        ),
        "conservative_tokens_per_call": math.ceil(average_chars / 4 + output_allowance),
    }
    return {"seat_map": seat_map, "scores": state["scores"], "events": state["events"],
            "focal": metadata.get("focal"), "topology": metadata.get("topology", "complete"),
            "messages_off": bool(metadata.get("messages_off")),
            "public_chat": bool(metadata.get("public_chat")),
            "human_board": bool(metadata.get("human_board")),
            "self_pull_reward": metadata.get("self_pull_reward") or 0,
            "personas": metadata.get("personas") or {},
            "actions_per_round": metadata["actions_per_round"],
            "declared_intents": bool(metadata.get("declared_intents")),
            "seeded_favours": metadata.get("seeded_favours") or 0,
            "punish_enabled": bool(metadata.get("punish_enabled")),
            "_prompt_profile": prompt_profile,
            "principal_seats": metadata.get("principal_seats") or []}


def _condition_key(md):
    return (md.get("topology"), bool(md.get("messages_off")), bool(md.get("public_chat")),
            bool(md.get("human_board")), md.get("self_pull_reward") or 0, tuple(sorted((md.get("personas") or {}).values())),
            bool(md.get("principal_seats")), bool(md.get("declared_intents")),
            int(md.get("seeded_favours") or 0), bool(md.get("punish_enabled")),
            (md.get("protocol") or {}).get("kind"), (md.get("protocol") or {}).get("condition"))


def lint_prompt(md):
    """The prompt an agent would see must match the rules the validator enforces."""
    problems = []
    settings = game.state_from_metadata(md)
    free_reports = bool(settings.get("free_reports"))
    personas = set((md.get("personas") or {}).values()) or {"neutral"}
    for persona in personas:
        p = seat.build_system_prompt("P1", list(md["seat_map"]), md["rounds"],
                                     md["actions_per_round"], persona,
                                     md.get("self_pull_reward") or 0,
                                     bool(md.get("messages_off")), bool(md.get("public_chat")),
                                     bool(md.get("punish_enabled")), bool(md.get("human_board")),
                                     free_reports, settings.get("stakes"))
        low = p.lower()
        can_be_asked = (not md.get("messages_off")) or md.get("public_chat") or md.get("human_board")
        if "any available action on every" not in low:
            problems.append(f"{persona}: missing you-may-act-every-turn rule")
        if ("credit" in low) != bool(can_be_asked):
            problems.append(f"{persona}: credit clause presence contradicts channels")
        if ('"say"' in low) != bool(md.get("public_chat")):
            problems.append(f"{persona}: say action presence contradicts public_chat")
        if ('"say_human"' in low) != bool(md.get("human_board")):
            problems.append(f"{persona}: say_human action presence contradicts human_board")
        if ('"punish"' in low) != bool(md.get("punish_enabled")):
            problems.append(f"{persona}: punish action presence contradicts punish_enabled")
        if "goodwill" in low:
            problems.append(f"{persona}: the payback nudge is back")
        if persona == "inoculated" and "ledger" in low:
            if "current and previous round" not in low or "summarizes older" not in low:
                problems.append("inoculated: public-ledger retention window is not stated")
    return problems


def check_config(path):
    """Preflight one config. Returns a list of problem strings (empty = pass)."""
    problems = []
    try:
        cfg = experiment.load_config(path)
        tasks = experiment.build_tasks(cfg, config_name=Path(path).stem)
    except Exception as e:
        return [f"build failed: {e}"]
    assumed_tokens = {**experiment.DEFAULT_COST, **(cfg.get("cost") or {})}[
        "avg_tokens_per_turn"
    ]
    for arm_name, t in tasks:
        seen = set()
        sims = []
        for sample in t.dataset:
            md = sample.metadata
            key = _condition_key(md)
            if key in seen:
                continue
            seen.add(key)
            problems += [f"{arm_name}: {p}" for p in lint_prompt(md)]
            simulated = simulate_episode(md)
            profile = simulated["_prompt_profile"]
            if profile["minimum_tokens_per_call"] > assumed_tokens:
                problems.append(
                    f"{arm_name}: cost.avg_tokens_per_turn {assumed_tokens:g} is below the "
                    f"offline minimum {profile['minimum_tokens_per_call']} "
                    "(representative input plus a small output reserve)"
                )
            sim = report.measure_episode(simulated)
            # the exp-13 controls must yield their promised numbers from simulated data
            if md.get("declared_intents") and not md.get("messages_off") and sim["intent_gap"] is None:
                problems.append(f"{arm_name}: declared intents on but the intent gap is unmeasurable")
            if md.get("seeded_favours") and sim["repay_forced"] is None:
                problems.append(f"{arm_name}: seeded favours on but forced repayment is unmeasurable")
            sims.append(sim)
        # every simulated model must be readable in the per-model table
        expected = {m for e in sims for m in
                    {r["model"] for r in e["seats"]}}
        led = {r["model"]: r for r in report.model_leaderboard(sims)}
        for m in sorted(expected):
            if m not in led or led[m]["score"] is None:
                problems.append(f"{arm_name}: model {m} unreadable in the per-model table")
        if any(r["principal"] for e in sims for r in e["seats"]) and not report.principal_vs_crowd(sims):
            problems.append(f"{arm_name}: principals tagged but principal-vs-crowd table empty")
        personas_present = {v for e in sims for v in e["personas"].values()}
        if {"neutral", "inoculated"} <= personas_present and not report.resistance(sims):
            problems.append(f"{arm_name}: resist conditions present but resistance table empty")
        try:
            report.format_report(sims)
        except Exception as e:
            problems.append(f"{arm_name}: report crashed on simulated data: {e}")
    return problems


def main():
    paths = sys.argv[1:] or sorted(glob.glob("configs/*.yaml"))
    failed = False
    for path in paths:
        problems = check_config(path)
        status = "OK" if not problems else "FAIL"
        print(f"{status:>4}  {path}")
        for p in problems:
            failed = True
            print(f"      - {p}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
