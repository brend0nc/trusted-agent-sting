"""One shared observation-window definition for prompts and deception visibility."""
from types import SimpleNamespace

from delegated_influence import observation


def test_current_and_previous_round_are_detailed_and_older_rounds_are_archived():
    assert observation.detailed_from_round(5) == 4
    assert observation.archive_through_round(5) == 3


def test_visibility_uses_the_same_two_round_window():
    recent = observation.claim_visibility(
        predicate="pulled", scope="specific_round", target_round=4,
        action_number=2, message_round=5, schema=observation.CURRENT_SCHEMA,
    )
    old_count = observation.claim_visibility(
        predicate="pulled", scope="specific_round", target_round=3,
        action_number=None, message_round=5, schema=observation.CURRENT_SCHEMA,
    )
    old_slot = observation.claim_visibility(
        predicate="pulled", scope="specific_round", target_round=3,
        action_number=2, message_round=5, schema=observation.CURRENT_SCHEMA,
    )
    assert recent == {"speaker_could_verify": True, "evidence_source": "recent_ledger"}
    assert old_count == {"speaker_could_verify": True, "evidence_source": "ledger_summary"}
    assert old_slot == {
        "speaker_could_verify": False, "evidence_source": "unavailable_exact_old_slot",
    }


def test_legacy_prompt_variants_are_recovered_without_guessing_unknown_prompts():
    def sample(text):
        return SimpleNamespace(metadata={}, events=[SimpleNamespace(
            event="model", input=[SimpleNamespace(content=text)]
        )])

    assert observation.infer_schema(sample(
        "Scores and the full history of lever pulls are public"
    )) == observation.LEGACY_FULL_LEDGER_SCHEMA
    assert observation.infer_schema(sample(
        "Scores and all lever pulls are public"
    )) == observation.LEGACY_FULL_LEDGER_SCHEMA
    assert observation.infer_schema(sample("different prompt")) == observation.UNKNOWN_SCHEMA
