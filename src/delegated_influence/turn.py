"""The async turn loop + episode driver. Model calls are injected, so both are offline-testable.
"""
import asyncio

from delegated_influence import game, protocols, seat


class OutOfCredits(RuntimeError):
    """The provider account has no credits (OpenRouter 402). Raised instead of forfeiting so the
    run dies loudly and can resume after a top-up — a 2026-07-12 run forfeited 405 whole games."""


MAX_TURN_ATTEMPTS = 2


def check_credits(e):
    msg = str(e)
    if "Insufficient credits" in msg or "'code': 402" in msg or "Error code: 402" in msg:
        error = OutOfCredits(msg)
        if isinstance(e, BaseException):
            raise error from e
        raise error


def system_prompt(obs, persona="neutral"):
    """The seat's rules. The trusted-agent sting uses its own short rule text (sprint work)."""
    if obs.get("system_prompt") == "trusted_sting":
        from delegated_influence import sting
        return sting.system_prompt(obs)
    return seat.build_system_prompt(obs["agent"], obs["agents"], obs["rounds"], obs["actions_per_round"],
                                    persona, obs["self_pull_reward"], obs["messages_off"],
                                    obs.get("public_chat", False), obs.get("punish_enabled", False),
                                    obs.get("human_board", False), obs.get("free_reports", False),
                                    obs.get("stakes"), rules_note=obs.get("rules_note"))


async def take_turn(state, obs, generate, persona="neutral", max_attempts=MAX_TURN_ATTEMPTS,
                    call_timeout=None):
    """One turn's event: call the model, classify, reprompt once, else forfeit.

    `generate(messages)` returns `(content, usage)`; an exception or None content = `api_error`.
    `call_timeout` (seconds) bounds each model call so one hung request can't freeze the episode;
    a timeout is treated as an api_error like any other call failure.
    """
    who = obs["agent"]
    messages = [
        {"role": "system", "content": system_prompt(obs, persona)},
        {"role": "user", "content": seat.render_observation(obs)},
    ]
    raws, usages = [], []
    last_reason = "parse_failure"
    for attempt in range(max_attempts):
        try:
            if call_timeout:
                content, usage = await asyncio.wait_for(generate(messages), call_timeout)
            else:
                content, usage = await generate(messages)
        except Exception as e:
            check_credits(e)  # out-of-credits must kill the run, not forfeit through it
            # str(TimeoutError()) is "" — always record WHY, so timeout forfeits are
            # distinguishable from provider errors in the stored events.
            usages.append({"prompt_tokens": 0, "completion_tokens": 0,
                           "error": str(e) or type(e).__name__})
            return game.make_event(obs, None, forfeit_reason="api_error",
                                   raw_responses=raws, usage=usages, parse_retries=attempt)
        usages.append(usage)
        if content is None:
            if isinstance(usage, dict):
                check_credits(usage.get("error", ""))
            return game.make_event(obs, None, forfeit_reason="api_error",
                                   raw_responses=raws, usage=usages, parse_retries=attempt)
        raws.append(content)
        outcome = seat.classify_reply(content, state, who)
        if isinstance(outcome, seat.Parsed):
            return game.make_event(obs, outcome.action, raw_responses=raws, usage=usages, parse_retries=attempt)
        last_reason = outcome.reason
        if attempt < max_attempts - 1:  # a try remains: reprompt with the correction
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": outcome.correction},
            ]
    return game.make_event(obs, None, forfeit_reason=last_reason,
                           raw_responses=raws, usage=usages, parse_retries=max_attempts - 1)


async def elicit_intent(state, rng, generators, who, rnd, persona="neutral", call_timeout=None):
    """One pre-round intent call: ask the seat its planned action, record it (never forfeits).

    The event's `usage` records WHY a plan is missing: a timeout/provider failure carries an
    "error" key, so flakiness is distinguishable from a genuinely undecided seat (planned None
    with a clean call)."""
    obs = game.build_observation(state, who, rnd, 0, 0)
    messages = [
        {"role": "system", "content": system_prompt(obs, persona)},
        {"role": "user", "content": seat.build_intent_prompt(obs)},
    ]
    try:
        if call_timeout:
            content, usage = await asyncio.wait_for(generators[who](messages), call_timeout)
        else:
            content, usage = await generators[who](messages)
    except Exception as e:
        check_credits(e)  # out-of-credits must kill the run, not forfeit through it
        content = None  # str(TimeoutError()) is "" — fall back to the type name
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "error": str(e) or type(e).__name__}
    if content is None and isinstance(usage, dict):
        check_credits(usage.get("error", ""))
    planned, reason = seat.parse_intent(content, state, who)
    event = game.make_intent_event(who, rnd, planned, reason=reason, usage=usage)
    event["raw_responses"] = [content] if content is not None else []
    game.apply_event(state, event)
    return event


async def drive_episode(state, rng, generators, personas=None, call_timeout=None, on_event=None):
    """Run a full episode: each action slot, every seat acts once in freshly shuffled order.

    `generators` and `personas` map seat name -> generate callable / persona. Mutates `state`.
    With `declared_intents`, each round opens with a private intent call per seat; with
    `seeded_favours` > 0, the engine injects that many random control pulls at seeded slots.
    `call_timeout` (seconds) bounds every model call so a hung request can't freeze the run.
    `on_event(state, event)` is a synchronous checkpoint callback after every applied event.
    """
    personas = personas or {}
    protocols.apply_prelude(state, on_event)
    forced_slots = []
    if state.get("seeded_favours"):
        n_slots = state["rounds"] * state["actions_per_round"]
        k = min(state["seeded_favours"], n_slots)
        forced_slots = sorted(rng.sample(range(n_slots), k))  # 0-based (round, action) slot indices
    slot = 0
    for rnd in range(1, state["rounds"] + 1):
        if state.get("declared_intents"):
            for who in state["agents"]:  # deterministic order: intents are private, order is irrelevant
                event = await elicit_intent(state, rng, generators, who, rnd, personas.get(who, "neutral"),
                                            call_timeout=call_timeout)
                if on_event is not None:
                    on_event(state, event)
        for action_number in range(1, state["actions_per_round"] + 1):
            order = list(state["agents"])
            rng.shuffle(order)
            order = protocols.order_for_slot(state, order, rnd, action_number)
            for turn_index, who in enumerate(order):
                obs = game.build_observation(state, who, rnd, action_number, turn_index)
                event = await take_turn(state, obs, generators[who], persona=personas.get(who, "neutral"),
                                        call_timeout=call_timeout)
                game.apply_event(state, event)
                if on_event is not None:
                    on_event(state, event)
            if slot in forced_slots:  # after the slot's turns: inject the control favour
                game.inject_forced_pull(state, rng, rnd, action_number)
                if on_event is not None:
                    on_event(state, state["events"][-1])
            slot += 1
        for event in game.settle_statements(state, rnd):  # stakes protocols: after the round's last slot
            if on_event is not None:
                on_event(state, event)
    return state
