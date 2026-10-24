"""Write docs/design/sting-prompts.md: the exact prompts players see, rendered offline. Sprint work; no API calls.

    uv run python scripts/render_sting_prompts.py
"""
import asyncio
import json
import random
from pathlib import Path

from delegated_influence import game, sting, turn

OUT = Path(__file__).resolve().parents[1] / "docs" / "design" / "sting-prompts.md"


async def prompts_for(sample):
    calls = {}

    def make(seat):
        async def generate(messages):
            calls.setdefault(seat, []).append(messages)
            return json.dumps({"action": "pull", "beneficiary": "P4"}), {"prompt_tokens": 1, "completion_tokens": 1}
        return generate
    state = game.state_from_metadata(sample.metadata)
    await turn.drive_episode(state, random.Random(sample.metadata["seed"]),
                             {seat: make(seat) for seat in (*sting.ORDINARY, sting.TRUSTED)})
    return calls


def block(text):
    return ["```text", text, "```", ""]


def main():
    task = sting.trusted_sting(player_models=("grok-4.3",), seeds=(61, 62, 63, 64, 65, 66))
    first = {}
    for sample in task.dataset:
        first.setdefault(sample.metadata["protocol"]["condition"], sample)
    rendered = {condition: asyncio.run(prompts_for(sample)) for condition, sample in first.items()}
    p = first["direct"].metadata["protocol"]
    doc = ["# Sting prompts (exact text, rendered offline)", "",
           f"Rendered by `scripts/render_sting_prompts.py` from protocol version {sting.STING_VERSION}. "
           f"Seed 61: named = {p['named']}, actual = {p['actual']}, neutral = the third ordinary seat.", "",
           "## System prompt (every ordinary seat; shown for P2)", ""]
    doc += block(rendered["direct"]["P2"][0][0]["content"])
    ally = first["alliance"].metadata["protocol"]["ally"]
    doc += [f"## `alliance`: the allied seat ({ally}) gets this paragraph at the end of its system prompt", ""]
    doc += block(sting.ALLIANCE_NOTE)
    pair = first["coalition_direct"].metadata["protocol"]["coalition"]
    doc += [f"## `coalition_direct` and `coalition_none`: each coalition member ({pair[0]} and {pair[1]}) gets this paragraph, "
            "with the other member's name", ""]
    doc += block(sting.COALITION_NOTE.format(partner=pair[1]))
    doc += ["## P4's private messages (matched: only the middle sentence changes)", ""]
    for kind in ("direct", "hint", "benign"):
        doc += [f"`{kind}` ({len(sting.OFFER_TEXTS[kind])} characters):", ""] + block(sting.OFFER_TEXTS[kind])
    doc += ["`none`, `alliance` and `coalition_none`: no message. `coalition_direct`: the `direct` message.", ""]
    for condition in sting.CONDITIONS:
        doc += [f"## `{condition}`: P2's first turn (round 1, action 1)", ""]
        doc += block(rendered[condition]["P2"][0][1]["content"])
    doc += ["## `direct`: P2's first turn of round 2 (shows the earlier-record summary)", ""]
    doc += block(rendered["direct"]["P2"][3][1]["content"])
    OUT.write_text("\n".join(doc))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
