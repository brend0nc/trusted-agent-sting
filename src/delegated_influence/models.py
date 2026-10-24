"""The model roster: 27 models across 10 providers, plus the background filler and judges.

Within-family capability ladders keep scaling effects unconfounded with provider.
Ids verified live on OpenRouter 2026-06-23 (fable-5, sonnet-5: 2026-07-03).
"""

OPENROUTER = "openrouter/"

# key -> dict(id, provider, ladder=(family, rank) | None, tags, provider_pref?, temperature?)
# `ladder` rank is the within-family capability order used by the capability-scaling analysis.
# `provider_pref` (optional) is an OpenRouter routing object to pin an open-weight model to a
# fixed host (drift control) — see provider_pref() below and the glm-5.2 row for how to enable.
# `temperature: None` omits an unsupported request parameter without changing reasoning or
# output limits. Public endpoint metadata was checked on 2026-09-08; evidence is saved
# in artifacts/pilots/catalog/2026-09-08-temperature-policy.json. Live provider errors can
# override incorrect catalogue declarations, as recorded for Opus 4.8. Historical stamps stay intact.
ROSTER = {
    # Anthropic ladder: sonnet 4.6 -> sonnet 5 -> opus 4.6 -> 4.7 -> 4.8 -> fable 5
    "sonnet-4.6": {
        "id": "anthropic/claude-sonnet-4.6",
        "provider": "anthropic",
        "ladder": ("anthropic", 1),
        "tags": [],
    },
    "sonnet-5": {
        "id": "anthropic/claude-sonnet-5",
        "provider": "anthropic",
        "ladder": ("anthropic", 2),
        "tags": [],
        "temperature": None,
    },
    "opus-4.6": {
        "id": "anthropic/claude-opus-4.6",
        "provider": "anthropic",
        "ladder": ("anthropic", 3),
        "tags": [],
    },
    "opus-4.7": {
        "id": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "ladder": ("anthropic", 4),
        "tags": [],
        "temperature": None,
    },
    "opus-4.8": {
        "id": "anthropic/claude-opus-4.8",
        "provider": "anthropic",
        "ladder": ("anthropic", 5),
        "tags": ["frontier"],
        # Live Azure HTTP 400 on 2026-09-08; its catalogue advertised this unsupported field.
        "temperature": None,
    },
    "fable-5": {
        "id": "anthropic/claude-fable-5",
        "provider": "anthropic",
        "ladder": ("anthropic", 6),
        "tags": ["frontier"],
        "temperature": None,
    },
    # OpenAI ladder (nano -> mini -> full -> 5.5), plus o4-mini as a reasoning anchor
    "gpt-5.4-nano": {
        "id": "openai/gpt-5.4-nano",
        "provider": "openai",
        "ladder": ("openai", 1),
        "tags": ["weak-rung"],
        "temperature": None,
    },
    "gpt-5.4-mini": {
        "id": "openai/gpt-5.4-mini",
        "provider": "openai",
        "ladder": ("openai", 2),
        "tags": [],
        "temperature": None,
    },
    "gpt-5.4": {
        "id": "openai/gpt-5.4",
        "provider": "openai",
        "ladder": ("openai", 3),
        "tags": [],
        "temperature": None,
    },
    "gpt-5.5": {
        "id": "openai/gpt-5.5",
        "provider": "openai",
        "ladder": ("openai", 4),
        "tags": ["frontier"],
        "temperature": None,
    },
    "o4-mini": {
        "id": "openai/o4-mini",
        "provider": "openai",
        "ladder": None,
        "tags": ["reasoning"],
    },
    # Google ladder (flash-lite -> flash -> pro)
    "gemini-flash-lite": {
        "id": "google/gemini-3.1-flash-lite",
        "provider": "google",
        "ladder": ("google", 1),
        "tags": ["weak-rung"],
    },
    "gemini-flash": {
        "id": "google/gemini-3.5-flash",
        "provider": "google",
        "ladder": ("google", 2),
        "tags": [],
        # Fixed-prompt pilot, 2026-09-08: 18/26 valid at 1024; 26/26 at 4096.
        # Reasoning shares this allowance. Keep temperature and requested effort unchanged.
        # Evidence: artifacts/pilots/2026-09-08-remaining-pilots/diagnostics/output-summary.json.
        "max_tokens": 4096,
    },
    "gemini-pro": {
        "id": "google/gemini-3.1-pro-preview",
        "provider": "google",
        "ladder": ("google", 3),
        "tags": ["frontier"],
    },
    # DeepSeek (open-weight; r1 = susceptibility anchor + reasoning)
    "deepseek-r1": {
        "id": "deepseek/deepseek-r1",
        "provider": "deepseek",
        "ladder": None,
        "tags": ["open-weight", "reasoning", "susceptible-anchor"],
    },
    "deepseek-v3.2": {
        "id": "deepseek/deepseek-v3.2",
        "provider": "deepseek",
        "ladder": ("deepseek", 1),
        "tags": ["open-weight"],
    },
    "deepseek-v4-pro-reasoning": {
        "id": "deepseek/deepseek-v4-pro",
        "provider": "deepseek",
        "ladder": ("deepseek", 2),
        # "reasoning": hybrid reasoner — at 1024 tokens it burned the whole budget on hidden
        # reasoning and returned EMPTY completions (~13-26 forfeits/game in every 10-agent
        # preliminary + exp-11 game, verified 2026-07-14). Same failure and fix as glm-5.2.
        "tags": ["open-weight", "frontier", "reasoning"],
    },
    # Qwen (open-weight; 235b instruct/thinking = the clean same-weights reasoning pair)
    "qwen3-235b": {
        "id": "qwen/qwen3-235b-a22b-2507",
        "provider": "qwen",
        "ladder": None,
        "tags": ["open-weight", "reasoning-pair"],
    },
    "qwen3-235b-thinking": {
        "id": "qwen/qwen3-235b-a22b-thinking-2507",
        "provider": "qwen",
        "ladder": None,
        "tags": ["open-weight", "reasoning-pair", "reasoning"],
    },
    "qwen3.7-max": {
        "id": "qwen/qwen3.7-max",
        "provider": "qwen",
        "ladder": None,
        "tags": ["frontier"],
    },
    # Moonshot (open-weight; k2/k2-thinking = second-lab reasoning pair)
    "kimi-k2": {
        "id": "moonshotai/kimi-k2",
        "provider": "moonshot",
        "ladder": None,
        "tags": ["open-weight", "reasoning-pair"],
    },
    "kimi-k2-thinking": {
        "id": "moonshotai/kimi-k2-thinking",
        "provider": "moonshot",
        "ladder": None,
        "tags": ["open-weight", "reasoning-pair", "reasoning"],
    },
    # Single-provider anchors / diversity
    "grok-4.3": {
        "id": "x-ai/grok-4.3",
        "provider": "xai",
        "ladder": None,
        "tags": ["frontier", "resistance-anchor"],
    },
    "glm-5.2": {
        "id": "z-ai/glm-5.2",
        "provider": "zhipu",
        "ladder": None,
        # "reasoning": GLM-5 is a hybrid reasoner; the smoke run showed empty completions at 1024.
        # PROVIDER PINNING (drift control): glm-5.2 (the open-weight crowd) is served by ~25
        # OpenRouter hosts whose mix can vary across arms and bias cross-arm contrasts. To pin it
        # to a fixed host before the leaderboard, add an OpenRouter routing object here, e.g.:
        #   "provider_pref": {"order": ["z-ai"], "allow_fallbacks": False, "provider_pref": {"order": ["fireworks"], "allow_fallbacks": False}},
        # It is passed verbatim as the request's extra_body["provider"] (see provider_pref() below
        # and task.build_generate_config). Confirm the exact host slug on the model's OpenRouter
        # page; usage_summary.py prints the served-host mix so you can SEE the drift before pinning.
        # Left unset (the current state) = OpenRouter's normal routing — do not pin until run time.
        "tags": ["open-weight", "frontier", "reasoning"],
    },
    "llama-4-scout": {
        "id": "meta-llama/llama-4-scout",
        "provider": "meta",
        "ladder": ("meta", 2),
        "tags": ["open-weight"],
    },
    "llama-3.2-3b": {
        "id": "meta-llama/llama-3.2-3b-instruct",
        "provider": "meta",
        "ladder": ("meta", 1),
        "tags": ["open-weight", "weak-rung"],
    },
    "mistral-large": {
        "id": "mistralai/mistral-large-2512",
        "provider": "mistral",
        "ladder": None,
        "tags": ["frontier"],
    },
}

# Models used to build or validate the experimental environment, but not automatically added to
# focal sweeps or headline arenas. Luna earned the background role in the 02b bake-off while
# remaining outside the paper roster.
AUXILIARY_MODELS = {
    # September refresh candidates. Explicit pilot configs select these models;
    # the registered paper roster and its 25-seat arenas remain fixed.
    "gpt-5.6-sol": {
        "id": "openai/gpt-5.6-sol",
        "provider": "openai",
        "ladder": None,
        "tags": ["reasoning", "pilot-candidate"],
    },
    "gpt-6-astra": {
        "id": "openai/gpt-6-astra",
        "provider": "openai",
        "ladder": None,
        "tags": ["reasoning", "pilot-candidate"],
    },
    "fable-5.1": {
        "id": "anthropic/claude-fable-5.1",
        "provider": "anthropic",
        "ladder": None,
        "tags": ["reasoning", "pilot-candidate"],
    },
    "gemini-3.8-flash": {
        "id": "google/gemini-3.8-flash",
        "provider": "google",
        "ladder": None,
        "tags": ["reasoning", "pilot-candidate"],
    },
    "qwen3.8-max-0902": {
        "id": "qwen/qwen3.8-max-0902",
        "provider": "qwen",
        "ladder": None,
        "tags": ["reasoning", "pilot-candidate"],
    },
    "kimi-k3": {
        "id": "moonshotai/kimi-k3",
        "provider": "moonshot",
        "ladder": None,
        "tags": ["open-weight", "reasoning", "pilot-candidate"],
    },
    # Frozen legacy instrument used by preliminary + exp-11: temperature .7, 1024 output tokens.
    # It stays resolvable so old arms reproduce exactly, but is not in ROSTER and therefore cannot
    # silently pool with the corrected reasoning-aware instrument above.
    "deepseek-v4-pro": {
        "id": "deepseek/deepseek-v4-pro",
        "provider": "deepseek",
        "ladder": None,
        "tags": ["open-weight", "legacy-instrument"],
    },
    "gpt-5.6-luna": {
        "id": "openai/gpt-5.6-luna",
        "provider": "openai",
        "ladder": None,
        # The 02b bake-off found reasoning=none reliable and dramatically faster than GLM-5.2.
        # Keep the output tightly bounded because background seats dominate focal-run volume.
        "reasoning_effort": "none",
        "max_tokens": 256,
        "tags": ["reasoning", "background-candidate"],
    },
}

# Resolution and explicit config validation accept auxiliary models. ROSTER remains the paper
# roster, so `models: all` and the headline arenas do not silently acquire a new model.
MODEL_REGISTRY = {**ROSTER, **AUXILIARY_MODELS}

# The fixed neutral filler for focal episodes. When the focal IS the filler it self-plays —
# seats are anonymous, and one condition for every leaderboard row beats a swapped background.
BACKGROUND = "gpt-5.6-luna"

# Post-hoc transcript judges: capable primary + different-provider cross-check + a cheap option.
JUDGE_PRIMARY = "openai/gpt-5.5"
JUDGE_SECONDARY = "anthropic/claude-opus-4.8"
JUDGE_CHEAP = "z-ai/glm-5.2"
# Lie extraction has its own calibrated default. Keep this separate so changing it cannot
# silently change promise or attribution measurements.
LIE_JUDGE_PRIMARY = "anthropic/claude-opus-4.8"

# Same-weights thinking-vs-not pairs for the reasoning-mode control.
REASONING_PAIRS = [
    ("qwen3-235b", "qwen3-235b-thinking"),
    ("kimi-k2", "kimi-k2-thinking"),
]


def openrouter_spec(key):
    """Inspect model spec for a roster key, e.g. 'opus-4.8' -> 'openrouter/anthropic/claude-opus-4.8'."""
    return OPENROUTER + MODEL_REGISTRY[key]["id"]


DIRECT_PROVIDERS = ("openai", "anthropic")


def direct_spec(key):
    """Inspect model spec that calls the model's own provider instead of OpenRouter.

    Only OpenAI and Anthropic roster models have a direct route; everything else returns None
    and stays on OpenRouter. Anthropic's own ids spell the version with dashes where the
    OpenRouter slug uses a dot ('claude-opus-4.8' -> 'claude-opus-4-8'); OpenAI's match.
    """
    row = MODEL_REGISTRY.get(key)
    if not row or row.get("provider") not in DIRECT_PROVIDERS:
        return None
    provider, _, slug = row["id"].partition("/")
    return f"{provider}/{slug.replace('.', '-') if provider == 'anthropic' else slug}"


def resolve_judge_spec(key):
    """Judge model spec: a roster key resolves through OpenRouter; an explicit Inspect spec passes through."""
    if key is None:
        return None
    return openrouter_spec(key) if key in MODEL_REGISTRY else key


def provider_pref(key):
    """The optional OpenRouter provider-routing object for a roster key, or None (normal routing).

    An open-weight model (e.g. glm-5.2) is served by many OpenRouter hosts whose mix can drift
    across arms; a `provider_pref` entry on the ROSTER row pins routing so cross-arm contrasts stay
    comparable. The dict is an OpenRouter routing object
    (https://openrouter.ai/docs/features/provider-routing), e.g.
    {"order": ["z-ai"], "allow_fallbacks": False}; it is passed verbatim as the request's
    extra_body["provider"] by task.build_generate_config. Default None = do not pin.
    """
    return MODEL_REGISTRY.get(key, {}).get("provider_pref")


def ladders():
    """{family: [keys ordered by capability rank]} for the within-family scaling regressions."""
    fams = {}
    for k, v in ROSTER.items():
        if v["ladder"]:
            fam, rank = v["ladder"]
            fams.setdefault(fam, []).append((rank, k))
    return {fam: [k for _, k in sorted(rks)] for fam, rks in fams.items()}
