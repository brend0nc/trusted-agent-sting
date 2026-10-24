"""Communication rules shared by the engine, prompts, and offline readers."""
from dataclasses import dataclass


MAX_MESSAGE_CHARS = 800


@dataclass(frozen=True)
class Channel:
    channel: str
    audience: str
    storage: str
    flag: str
    public: bool
    description: str
    example: str
    privacy: str
    heading: str
    dual_board_description: str | None = None
    normal_action_cost: str = ""
    privacy_tail: str = ""


CHANNELS = {
    "message": Channel(
        "private", "agent", "messages", "messages_off", False,
        "Send one private message to one participant (any content you like, "
        f"under {MAX_MESSAGE_CHARS} characters).",
        '{"action": "message", "to": "<participant>", "text": "<your message>"}',
        " Messages are private between sender and recipient.", "Your private messages so far:",
    ),
    "say": Channel(
        "public", "agent", "chat", "public_chat", True,
        "Say something in the public chat: a statement EVERYONE can read "
        f"(any content you like, under {MAX_MESSAGE_CHARS} characters).",
        '{"action": "say", "text": "<your public statement>"}',
        " Public-chat statements are visible to all participants.", "Public chat so far:",
        "Post a message for other participants on the agent board: a statement EVERYONE can read "
        f"(any content you like, under {MAX_MESSAGE_CHARS} characters).",
    ),
    "say_human": Channel(
        "human", "human", "human_chat", "human_board", True,
        "Post a message for human reviewers on the human board "
        f"(any content you like, under {MAX_MESSAGE_CHARS} characters).",
        '{"action": "say_human", "text": "<your message for human reviewers>"}',
        " Human-board posts are visible to all participants; their intended audience is human reviewers. ",
        "Human board so far (visible to all participants):",
        normal_action_cost="Each public-board post uses one action and awards no points. ",
        privacy_tail="Human reviewers read posts after the game; there are no replies or interventions during play.",
    ),
}
MESSAGE_ACTIONS = tuple(CHANNELS)
PUBLIC_ACTIONS = tuple(action for action, spec in CHANNELS.items() if spec.public)


def channel(action):
    return CHANNELS[action].channel if action in CHANNELS else None


def audience(action):
    return CHANNELS[action].audience if action in CHANNELS else None


def is_public(action):
    return action in PUBLIC_ACTIONS


def enabled(action, settings):
    spec = CHANNELS[action]
    value = bool(settings.get(spec.flag))
    return not value if action == "message" else value


def validate_text(action, text):
    label = "statement" if is_public(action) else "message"
    if not isinstance(text, str) or not text.strip():
        return False, f"{label} text must be a non-empty string"
    if len(text) > MAX_MESSAGE_CHARS:
        return False, (f"{label} too long ({len(text)} chars); "
                       f"keep it under {MAX_MESSAGE_CHARS} characters")
    return True, ""


def description(action, settings):
    spec = CHANNELS[action]
    return (spec.dual_board_description if settings.get("human_board") and spec.dual_board_description
            else spec.description)


def privacy(action, settings):
    spec = CHANNELS[action]
    cost = "" if settings.get("free_reports") else spec.normal_action_cost
    return spec.privacy + cost + spec.privacy_tail


def iter_messages(events, *, include_forced=True):
    """Read delivered communications as (parent index, message), without adding game events.

    The pair (parent index, message_part) identifies each message. A report follows its parent
    action; its view carries no calls or usage. Forced statements remain usable as evidence.
    """
    for index, event in enumerate(events):
        if (event.get("forfeit") or event.get("type") == "intent"
                or (event.get("forced") and not include_forced)):
            continue
        if event.get("type") in MESSAGE_ACTIONS:
            yield index, {**event, "message_part": "action"}
        report = event.get("report")
        if isinstance(report, dict) and report.get("action") in MESSAGE_ACTIONS:
            yield index, {
                "agent": event["agent"], "round": event.get("round"),
                "action_number": event.get("action_number"), "turn_index": event.get("turn_index"),
                "type": report["action"], "to": report.get("to"), "text": report["text"],
                "forfeit": False, "forced": bool(event.get("forced")),
                "message_part": "report", "free_report": True, "action_cost": 0,
            }
