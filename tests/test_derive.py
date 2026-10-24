"""Regression tests for site/derive.py arm matching (the paper-pool guard).

An episode joins a config arm only when that arm's config stem prefixes the episode's
run dir. A smoke rerun of a confirmatory signature must stay exploratory, and an orphan
run dir (no config prefixes it) must never be claimed or counted as coverage.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))

import derive
from delegated_influence import coalition, experiment, pool


SMOKE_ARM = {"name": "smoke_attack", "config": "01_smoke", "stage": "exploratory",
             "task": "focal_attack"}
PAPER_ARM = {"name": "attack_complete", "config": "04_attack_complete", "stage": "confirmatory",
             "task": "focal_attack"}


def test_prompt_export_follows_lie_prompt_helper_call():
    lie = next(section for section in derive._judge_sections() if section["id"] == "judge-lie")
    assert "first-person present-progressive" in lie["text"]
    assert "Answer with ONLY a JSON array" in lie["text"]


def test_system_prompt_export_keeps_credit_from_the_rendered_prompt():
    sections = {section["id"]: section for section in derive.build_prompt_sections()}
    assert {"system-neutral", "system-attacker", "system-inoculated", "system-credit"} <= sections.keys()
    credit = sections["system-credit"]["text"]
    assert credit.startswith("Credit: every pull records who brokered it.")
    assert credit in sections["system-neutral"]["text"]


def _episode(run, sample_id="focal-attack--opus-4.8_complete_r1", md=None):
    return {"run": run, "sample_id": sample_id, "metadata": md or {}, "id": f"{run}--{sample_id}"}


def _cov(coverage, arm):
    return next(c for c in coverage if c["arm"] == arm)


def test_smoke_rerun_of_confirmatory_signature_stays_exploratory():
    """A smoke rerun signature-identical to a paper cell belongs to the smoke arm
    (provenance: config stem prefixes the run dir), never the confirmatory arm."""
    ep = _episode("01_smoke_v3")
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arms = [(SMOKE_ARM, {sig}), (PAPER_ARM, {sig})]
    coverage = pool.match_arms([ep], arms)
    assert ep["arm"] == "smoke_attack"
    assert ep["pool"] == "exploratory"
    assert _cov(coverage, "smoke_attack")["done_episodes"] == 1
    assert _cov(coverage, "attack_complete")["done_episodes"] == 0


def test_sig_distinguishes_punish_enabled():
    """The 20_arena_25 2x2 (talk on/off x punish on/off) needs punish_enabled in the signature —
    otherwise the punish-on and punish-off arms collapse onto the same planned signature and the
    punish contrast gets silently pooled together."""
    md_on = {"punish_enabled": True}
    md_off = {"punish_enabled": False}
    assert experiment.episode_identity("s", md_on) != experiment.episode_identity("s", md_off)
    assert experiment.episode_identity("s", {"punish_enabled": True}) == experiment.episode_identity("s", md_on)


def test_sig_distinguishes_explicit_sampling_overrides():
    base = {"model_configs": {"gpt-5.6-luna": {
        "reasoning_effort": "none", "max_tokens": 256,
    }}}
    low = {"model_configs": {"gpt-5.6-luna": {
        "reasoning_effort": "low", "max_tokens": 512,
    }}}
    assert experiment.episode_identity("s", base) != experiment.episode_identity("s", low)
    assert experiment.episode_identity("s", {}) == experiment.episode_identity("s", {"model_configs": {}})


@pytest.mark.parametrize("flag", derive.report.TREATMENT_FLAGS)
def test_every_treatment_flag_distinguishes_sig_and_label(flag):
    """A flag added to TREATMENT_FLAGS but wired into only one of _sig / label_condition fails here:
    toggling each flag must change BOTH the planned/run signature and the condition tag."""
    assert experiment.episode_identity("s", {}) != experiment.episode_identity("s", {flag: True}), f"episode_identity ignores {flag}"
    base = {"topology": "complete", "self_pull_reward": 0, "messages_off": False}
    assert (derive.report.label_condition(base)
            != derive.report.label_condition({**base, flag: True})), f"label_condition ignores {flag}"


def test_orphan_run_dir_never_claimed_or_counted(capsys):
    """A run dir no config prefixes (orphan/renamed run) is left unclaimed with a stderr
    warning — it must not join the arm, its coverage, or the paper pool."""
    ep = _episode("oneoff_rerun")
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arms = [(PAPER_ARM, {sig})]
    coverage = pool.match_arms([ep], arms)
    assert ep["arm"] is None
    assert ep["pool"] == "exploratory"
    assert _cov(coverage, "attack_complete")["done_episodes"] == 0
    err = capsys.readouterr().err
    assert "oneoff_rerun" in err and "attack_complete" in err


def test_same_named_arms_from_different_configs_keep_config_locality():
    ep = _episode("04_second")
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    first = {"name": "shared", "config": "01_first", "stage": "exploratory", "task": "lineup"}
    second = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup"}
    coverage = pool.match_arms([ep], [(first, {sig}), (second, {sig})])
    assert ep["arm"] == "shared"
    assert ep["arm_config"] == "04_second"
    assert ep["pool"] == "paper"
    by_config = {row["config"]: row for row in coverage}
    assert by_config["01_first"]["done_episodes"] == 0
    assert by_config["04_second"]["done_episodes"] == 1


def test_embedded_provenance_survives_a_moved_run_directory():
    md = {"experiment_config": "04_second", "experiment_arm": "shared"}
    md["experiment_identity"] = derive.experiment.compute_episode_identity("same", md)
    ep = _episode("renamed_elsewhere", sample_id="same", md=md)
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup"}
    coverage = pool.match_arms([ep], [(arm, {sig})])
    assert ep["arm_config"] == "04_second"
    assert ep["arm"] == "shared"
    assert coverage[0]["done_episodes"] == 1


def test_conflicting_embedded_provenance_cannot_be_reassigned_by_directory(capsys):
    md = {"experiment_config": "wrong", "experiment_arm": "wrong"}
    md["experiment_identity"] = derive.experiment.compute_episode_identity("same", md)
    ep = _episode("04_second", sample_id="same", md=md)
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup"}
    coverage = pool.match_arms([ep], [(arm, {sig})])
    assert ep["arm"] is None
    assert coverage[0]["done_episodes"] == 0
    assert "embedded provenance" in capsys.readouterr().err


def test_sampling_stamp_event_mismatch_is_unclaimed_before_paper_matching(capsys):
    md = {"sampling_instrument_mismatch": True}
    ep = _episode("04_second", sample_id="same", md=md)
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory",
           "task": "lineup"}
    coverage = pool.match_arms([ep], [(arm, {sig})])
    assert coverage[0]["done_episodes"] == 0
    assert ep["arm"] is None and ep["pool"] == "exploratory"
    assert ep["instrument_validation_error"] == "sampling_metadata_event_mismatch"
    assert "conflict with its experiment stamp" in capsys.readouterr().err


def _current_and_legacy_plan():
    cfg = {"arms": [{"name": "shared", "task": "lineup", "seats": ["glm-5.2"]}]}
    sample = derive.experiment.build_tasks(cfg, config_name="04_second")[0][1].dataset[0]
    current_md = dict(sample.metadata)
    planned = {derive.experiment.episode_identity(sample.id, current_md)}
    legacy_planned = {derive.experiment.legacy_episode_identity(sample.id, current_md)}
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory",
           "task": "lineup", "_legacy_planned": legacy_planned}
    legacy_md = {k: v for k, v in current_md.items() if not k.startswith("experiment_")}
    legacy_md.pop("gameplay_protocol_version", None)
    legacy_md.pop("sampling_instruments", None)
    legacy_md["observation_schema"] = "full_ledger_legacy"
    return sample.id, current_md, legacy_md, arm, planned


def test_markerless_historical_run_keeps_coverage_without_pretending_to_be_current():
    sample_id, _current_md, legacy_md, arm, planned = _current_and_legacy_plan()
    ep = _episode("04_second", sample_id=sample_id, md=legacy_md)
    coverage = pool.match_arms([ep], [(arm, planned)])
    assert coverage[0]["done_episodes"] == 1
    assert coverage[0]["current_done_episodes"] == 0
    assert coverage[0]["legacy_done_episodes"] == 1
    assert coverage[0]["selected_instrument"].startswith("legacy-cohort-v1:")
    assert ep["pool"] == "paper"
    assert ep["instrument_match"] == coverage[0]["selected_instrument"]


def test_complete_current_instrument_supersedes_legacy_without_pooling_them():
    sample_id, current_md, legacy_md, arm, planned = _current_and_legacy_plan()
    current = _episode("moved-current", sample_id=sample_id, md=current_md)
    legacy = _episode("04_second", sample_id=sample_id, md=legacy_md)
    coverage = pool.match_arms([legacy, current], [(arm, planned)])
    assert coverage[0]["selected_instrument"] == "current"
    assert current["pool"] == "paper"
    assert legacy["pool"] == "historical"


def test_two_complete_legacy_instruments_are_not_silently_pooled_or_selected():
    sample_id, _current_md, legacy_md, arm, planned = _current_and_legacy_plan()
    full = _episode("04_second", sample_id=sample_id, md={
        **legacy_md,
        "observation_schema": "full_ledger_legacy",
        "gameplay_protocol_instrument": {"system_prompt_sha256": "old"},
    })
    bounded = _episode("04_second", sample_id=sample_id, md={
        **legacy_md,
        "observation_schema": "bounded_current_previous_summary_v1",
        "gameplay_protocol_instrument": {"system_prompt_sha256": "new"},
    })
    coverage = pool.match_arms([full, bounded], [(arm, planned)])
    assert coverage[0]["instrument_ambiguous"] is True
    assert coverage[0]["selected_instrument"] is None
    assert len(coverage[0]["legacy_instruments"]) == 2
    assert full["pool"] == bounded["pool"] == "historical"


def test_two_sampling_variants_for_one_legacy_cell_make_that_cell_unusable():
    sample_id, _current_md, legacy_md, arm, planned = _current_and_legacy_plan()
    base = {
        **legacy_md,
        "gameplay_protocol_instrument": {"system_prompt_sha256": "same"},
    }
    first = _episode("04_second", sample_id=sample_id, md={
        **base, "sampling_instruments": {"glm-5.2": {"max_tokens": 512}},
    })
    second = _episode("04_second", sample_id=sample_id, md={
        **base, "sampling_instruments": {"glm-5.2": {"max_tokens": 1024}},
    })
    coverage = pool.match_arms([first, second], [(arm, planned)])
    assert coverage[0]["done_episodes"] == 0
    assert coverage[0]["ambiguous_legacy_cells"]
    assert first["pool"] == second["pool"] == "historical"


def test_same_model_sampling_change_across_repetitions_blocks_the_arm():
    base = {"seat_map": {"P1": "glm-5.2"}, "observation_schema": "full_ledger_legacy",
            "gameplay_protocol_instrument": {"system_prompt_sha256": "same"}}
    first_md = {**base, "seed": 1,
                "sampling_instruments": {"glm-5.2": {"max_tokens": 512}}}
    second_md = {**base, "seed": 2,
                 "sampling_instruments": {"glm-5.2": {"max_tokens": 1024}}}
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup",
           "_legacy_planned": {
               derive.experiment.legacy_episode_identity("one", first_md),
               derive.experiment.legacy_episode_identity("two", second_md),
           }}
    first = _episode("04_second", sample_id="one", md=first_md)
    second = _episode("04_second", sample_id="two", md=second_md)
    coverage = pool.match_arms([first, second], [(arm, {"planned-one", "planned-two"})])
    assert coverage[0]["done_episodes"] == 0
    assert coverage[0]["conflicted_sampling_models"]
    assert first["pool"] == second["pool"] == "historical"


def test_reruns_count_once_in_coverage_and_once_in_analysis():
    md = {}
    sig = experiment.episode_identity("same", md)
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup"}
    first = _episode("04_second", sample_id="same", md=md)
    first.update({"started": "2026-01-01", "eval_file": "first.eval"})
    rerun = _episode("04_second", sample_id="same", md=md)
    rerun.update({"started": "2026-01-02", "eval_file": "rerun.eval"})
    coverage = pool.match_arms([first, rerun], [(arm, {sig})])
    assert coverage[0]["done_episodes"] == 1
    assert coverage[0]["analysis_episodes"] == 1
    assert coverage[0]["duplicate_episodes"] == 1
    assert [first["pool"], rerun["pool"]] == ["paper", "historical"]


def test_reruns_do_not_create_served_model_version_drift():
    sig = experiment.episode_identity("one", {})
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory",
           "task": "lineup"}
    episodes = []
    for started, served in (("2026-01-01", "glm-5.2-v1"),
                            ("2026-01-02", "glm-5.2-v2")):
        ep = _episode("04_second", sample_id="one", md={"served_instruments": {
            "glm-5.2": {
                "calls": 1, "failed_calls": 0, "unknown_provider_calls": 0,
                "requested_models": {"openrouter/z-ai/glm-5.2": 1},
                "served_models": {served: 1}, "providers": {"host": 1},
            }
        }})
        ep.update({"started": started, "eval_file": f"{started}.eval"})
        episodes.append(ep)
    coverage = pool.match_arms(episodes, [(arm, {sig})])
    assert coverage[0]["served_model_drift"] is False  # reruns are not pooled together
    assert [ep["pool"] for ep in episodes] == ["paper", "historical"]


def test_served_model_version_drift_across_distinct_repetitions_blocks_paper_pool():
    first_sig = experiment.episode_identity("one", {})
    second_sig = experiment.episode_identity("two", {})
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory",
           "task": "lineup"}
    episodes = []
    for sample_id, served in (("one", "glm-5.2-v1"), ("two", "glm-5.2-v2")):
        ep = _episode("04_second", sample_id=sample_id, md={"served_instruments": {
            "glm-5.2": {
                "calls": 1, "failed_calls": 0, "unknown_provider_calls": 0,
                "requested_models": {"openrouter/z-ai/glm-5.2": 1},
                "served_models": {served: 1}, "providers": {"host": 1},
            }
        }})
        ep.update({"started": "2026-01-01", "eval_file": f"{sample_id}.eval"})
        episodes.append(ep)
    coverage = pool.match_arms(episodes, [(arm, {first_sig, second_sig})])
    assert coverage[0]["done_episodes"] == 2
    assert coverage[0]["served_model_drift"] is True
    assert coverage[0]["instrument_ambiguous"] is True
    assert [ep["pool"] for ep in episodes] == ["in_progress", "in_progress"]


def test_failed_requested_alias_does_not_create_served_model_drift():
    summary = pool.summarize_served_instruments([{ "metadata": {"served_instruments": {
        "glm-5.2": {
            "calls": 2, "failed_calls": 1, "unknown_provider_calls": 1,
            "requested_models": {"openrouter/z-ai/glm-5.2": 2},
            "served_models": {"z-ai/glm-5.2-v1": 1},
            "failed_served_models": {"openrouter/z-ai/glm-5.2": 1},
            "providers": {"Host": 1},
        }
    }}}])
    assert summary["served_model_drift"] is False
    assert summary["models"]["glm-5.2"]["failed_served_models"] == {
        "openrouter/z-ai/glm-5.2": 1
    }


def test_stamped_older_instrument_remains_attributed_as_historical_coverage():
    sample_id, _current_md, legacy_md, arm, planned = _current_and_legacy_plan()
    legacy_md.update({"experiment_config": "04_second", "experiment_arm": "shared",
                      "experiment_stage": "confirmatory"})
    legacy_md["experiment_identity"] = derive.experiment.compute_episode_identity(
        sample_id, legacy_md
    )
    ep = _episode("moved", sample_id=sample_id, md=legacy_md)
    coverage = pool.match_arms([ep], [(arm, planned)])
    assert coverage[0]["done_episodes"] == 1
    assert ep["arm"] == "shared"
    assert ep["pool"] == "paper"


def test_stamped_stage_conflict_cannot_promote_an_old_episode(capsys):
    md = {"experiment_config": "04_second", "experiment_arm": "shared",
          "experiment_stage": "exploratory"}
    md["experiment_identity"] = derive.experiment.compute_episode_identity("same", md)
    ep = _episode("04_second", sample_id="same", md=md)
    sig = experiment.episode_identity(ep["sample_id"], ep["metadata"])
    arm = {"name": "shared", "config": "04_second", "stage": "confirmatory", "task": "lineup"}
    coverage = pool.match_arms([ep], [(arm, {sig})])
    assert ep["arm"] is None
    assert ep["pool"] == "exploratory"
    assert coverage[0]["done_episodes"] == 0
    assert "embedded stage" in capsys.readouterr().err


def test_selected_instrument_exposes_provider_mix_without_fragmenting_coverage():
    sample_id, current_md, _legacy_md, arm, planned = _current_and_legacy_plan()
    current_md["served_instruments"] = {"glm-5.2": {
        "calls": 2, "failed_calls": 0, "unknown_provider_calls": 0,
        "requested_models": {"openrouter/z-ai/glm-5.2": 2},
        "served_models": {"z-ai/glm-5.2-v1": 2},
        "providers": {"Fireworks": 1, "Together": 1},
    }}
    ep = _episode("moved-current", sample_id=sample_id, md=current_md)
    coverage = pool.match_arms([ep], [(arm, planned)])
    assert coverage[0]["done_episodes"] == 1
    assert coverage[0]["provider_drift"] is True
    assert coverage[0]["served_instruments"]["glm-5.2"]["calls"] == 2


# --- golden fixtures for the paper-facing stats functions ----------------------------------------
# Small synthetic episodes with hand-computed expected outputs (NOT snapshots of today's data),
# covering the derive-local statistics: the retracted site headline was computed by these.


def ev(agent, type=None, beneficiary=None, to=None, text=None, round=1,
       forfeit=False, forced=False, credit=None):
    """One event dict, defaulted for the fields derive/metrics read."""
    return {"round": round, "agent": agent, "type": type, "beneficiary": beneficiary,
            "to": to, "text": text, "forfeit": forfeit, "forced": forced,
            "credit": credit, "forfeit_reason": None}


def rec(seat_map, events, scores=None, focal=None, actions_per_round=5,
        self_pull_reward=0, messages_off=False, public_chat=False,
        topology="complete", personas=None):
    """A report.read_episode-shaped record with the fields derive reads."""
    return {"seat_map": seat_map, "events": events, "scores": scores or {},
            "focal": focal, "actions_per_round": actions_per_round,
            "self_pull_reward": self_pull_reward, "messages_off": messages_off,
            "public_chat": public_chat, "topology": topology, "personas": personas or {}}


def ep(rec_, arm=None, md=None, **extra):
    """A derive episode dict; **extra sets already-tagged fields (seat_extraction, points_received…)."""
    d = {**rec_, "arm": arm, "metadata": md or {}, "run": "run", "sample_id": "s", "id": "id"}
    d.update(extra)
    return d


# --- mutual_k3_pairs / tag_coalitions ---------------------------------------------------------------

def _coalition_events():
    """P1<->P2 mutual for 3 consecutive rounds (a coalition); P1<->P3 for only 2 (not); plus a
    self-pull and a forfeit that must both be ignored."""
    events = []
    for r in (1, 2, 3):
        events += [ev("P1", "pull", "P2", round=r), ev("P2", "pull", "P1", round=r)]
    for r in (1, 2):
        events += [ev("P1", "pull", "P3", round=r), ev("P3", "pull", "P1", round=r)]
    events += [ev("P1", "pull", "P1", round=3),            # self-pull: excluded
               ev("P2", None, round=3, forfeit=True)]       # forfeit: excluded
    return events


def mutual_k3_pairs(r):
    return coalition.mutual_support_k_sweep(r["events"], ks=(3,))[3]["pair_list"]


def test_mutual_k3_requires_k3_consecutive_mutual_rounds():
    r = rec({"P1": "alpha", "P2": "beta", "P3": "beta"}, _coalition_events())
    assert mutual_k3_pairs(r) == [("P1", "P2")]


def test_mutual_k3_two_consecutive_rounds_is_not_enough():
    # Truncate to 2 rounds: even P1<->P2 now only spans 2 consecutive rounds -> no coalition.
    events = [e for e in _coalition_events() if e["round"] <= 2]
    r = rec({"P1": "alpha", "P2": "beta", "P3": "beta"}, events)
    assert mutual_k3_pairs(r) == []


def test_tag_coalitions_sets_pairs_and_agent_count():
    e = ep(rec({"P1": "alpha", "P2": "beta", "P3": "beta"}, _coalition_events()), md={"rounds": 3})
    derive.tag_coalitions([e])
    assert e["coalition_pairs"] == [("P1", "P2")]
    assert e["coalition_agents"] == 2


def test_site_coalition_uses_canonical_metric_and_excludes_forced_mutual_favours():
    events = _coalition_events()
    # A forced third round must not turn a two-round chosen pair into a coalition.
    events = [event for event in events if not (
        event["round"] == 3 and {event["agent"], event.get("beneficiary")} == {"P1", "P2"}
    )]
    events += [
        ev("P1", "pull", "P2", round=3, forced=True),
        ev("P2", "pull", "P1", round=3, forced=True),
    ]
    record = rec({"P1": "alpha", "P2": "beta", "P3": "beta"}, events)
    assert mutual_k3_pairs(record) == []


# --- tag_prereg (extraction rate, points, top-1 share) -------------------------------------------

def test_tag_prereg_extraction_points_and_top1_share():
    sm = {"P1": "alpha", "P2": "beta", "P3": "beta"}
    events = [ev("P2", "pull", "P1", round=1),   # P1 receives from a rival
              ev("P3", "pull", "P1", round=1),   # P1 receives from a rival
              ev("P1", "pull", "P1", round=2),   # self-pull: counts in received, not from_others
              ev("P2", "pull", "P3", round=2)]   # P3 receives from a rival
    e = ep(rec(sm, events, scores={"P1": 2, "P2": 0, "P3": 1}, focal="P1", actions_per_round=2))
    derive.tag_prereg([e])
    # budget = rounds(max=2) x actions_per_round(2) = 4; denom = 4 x (3 seats - 1) = 8
    assert e["points_received"] == 2
    assert e["extraction_rate"] == 0.25            # 2 / 8
    assert e["seat_points"] == {"P1": 2, "P2": 0, "P3": 1}
    assert e["seat_extraction"] == {"P1": 0.25, "P2": 0.0, "P3": 0.125}
    assert e["top1_share"] == 0.75                 # P1 got 3 of 4 total pulls (self-pull included)


def test_tag_prereg_without_focal_nulls_the_focal_fields():
    e = ep(rec({"P1": "a", "P2": "b"}, [ev("P1", "pull", "P2")],
               scores={"P1": 0, "P2": 1}, focal=None, actions_per_round=1))
    derive.tag_prereg([e])
    assert e["points_received"] is None and e["extraction_rate"] is None
    assert e["seat_extraction"] == {"P1": 0.0, "P2": 1.0}   # budget 1, denom 1


# --- extraction_rows -----------------------------------------------------------------------------

def test_extraction_rows_focal_and_arena_seats():
    focal_ep = ep(rec({"P1": "alpha", "P2": "x", "P3": "x"}, [], focal="P1"),
                  seat_extraction={"P1": 0.4}, seat_points={"P1": 4})
    arena_ep = ep(rec({"P1": "gamma", "P2": "gamma"}, [], focal=None),
                  seat_extraction={"P1": 0.2, "P2": 0.4}, seat_points={"P1": 2, "P2": 4})
    rate_rows, point_rows = derive.extraction_rows([focal_ep, arena_ep])
    assert rate_rows[0] == {"model": "alpha", "n": 1, "rate": 0.4, "ci": [0.4, 0.4]}
    assert rate_rows[1]["model"] == "gamma" and rate_rows[1]["n"] == 2
    assert rate_rows[1]["rate"] == pytest.approx(0.3) and len(rate_rows[1]["ci"]) == 2
    assert point_rows[0] == {"model": "alpha", "n": 1, "points_mean": 4, "ci": [4, 4]}
    assert point_rows[1]["points_mean"] == pytest.approx(3.0)


def test_extraction_rows_messages_and_focal_only_filters():
    on = ep(rec({"P1": "a"}, [], focal="P1", messages_off=False),
            seat_extraction={"P1": 0.5}, seat_points={"P1": 5})
    off = ep(rec({"P1": "a"}, [], focal="P1", messages_off=True),
             seat_extraction={"P1": 0.1}, seat_points={"P1": 1})
    arena = ep(rec({"P1": "a", "P2": "a"}, [], focal=None, messages_off=False),
               seat_extraction={"P1": 0.5, "P2": 0.5}, seat_points={"P1": 5, "P2": 5})
    assert derive.extraction_rows([on, off, arena])[0][0]["n"] == 3        # msgs-on: focal + 2 arena
    assert derive.extraction_rows([on, off, arena], focal_only=True)[0][0]["n"] == 1  # drops arena
    off_rows = derive.extraction_rows([on, off, arena], messages_off=True)[0]
    assert off_rows[0]["n"] == 1 and off_rows[0]["rate"] == 0.1            # only the msgs-off seat


# --- lift_diff_rows ------------------------------------------------------------------------------

def test_lift_diff_rows_diff_ci_filtering_and_sort():
    eps = []
    for pts in (5, 5):   # alpha talk side (mean 5)
        eps.append(ep(rec({"P1": "alpha"}, [], focal="P1"), arm="attack_complete", points_received=pts))
    for pts in (2, 2):   # alpha no-talk side (mean 2) -> diff 3
        eps.append(ep(rec({"P1": "alpha"}, [], focal="P1"), arm="attack_nomsg", points_received=pts))
    for pts in (10, 10):  # beta talk side
        eps.append(ep(rec({"P1": "beta"}, [], focal="P1"), arm="attack_complete", points_received=pts))
    for pts in (1, 1):   # beta no-talk side -> diff 9
        eps.append(ep(rec({"P1": "beta"}, [], focal="P1"), arm="attack_nomsg", points_received=pts))
    eps.append(ep(rec({"P1": "alpha"}, [], focal="P1"), arm="other", points_received=99))    # wrong arm
    eps.append(ep(rec({"P1": "alpha"}, [], focal=None), arm="attack_complete", points_received=99))  # no focal
    rows = derive.lift_diff_rows(eps)
    # constant sides -> every bootstrap draw is identical -> a degenerate CI at the diff
    assert rows == [{"model": "alpha", "n_on": 2, "n_off": 2, "diff": 3.0, "ci": [3.0, 3.0]},
                    {"model": "beta", "n_on": 2, "n_off": 2, "diff": 9.0, "ci": [9.0, 9.0]}]


def test_lift_diff_rows_drops_models_missing_a_side():
    eps = [ep(rec({"P1": "alpha"}, [], focal="P1"), arm="attack_complete", points_received=5)]
    assert derive.lift_diff_rows(eps) == []   # no no-talk side -> no row


# --- timing_series -------------------------------------------------------------------------------

def test_timing_series_cumulative_pull_share():
    L = derive.TIMING_LENGTH
    events = []
    for k in range(L):
        events.append(ev("P1", "pull", "P2") if k == 0 else ev("P1", "message", to="P2"))
        events.append(ev("P2", "pull", "P1"))   # P2 pulls on every one of its actions
    out = derive.timing_series([ep(rec({"P1": "alpha", "P2": "alpha"}, events))])
    assert out["action_index"] == list(range(1, L + 1))
    row = out["per_model"][0]
    assert row["model"] == "alpha" and row["n_agent_episodes"] == 2
    # P1 series[k]=1/k (one pull, then non-pulls); P2 series[k]=1 (all pulls); model = their mean
    assert row["series"] == pytest.approx([(1.0 + 1.0 / k) / 2 for k in range(1, L + 1)])
    assert out["per_model_key"] == {"alpha": row}


def test_timing_series_skips_seats_of_wrong_budget():
    out = derive.timing_series([ep(rec({"P1": "alpha"}, [ev("P1", "pull", "P1")]))])
    assert out["per_model"] == []   # only 1 action, != TIMING_LENGTH


# --- surplus_rows --------------------------------------------------------------------------------

def test_surplus_rows_message_linked_pull_surplus():
    sm = {"P1": "alpha", "P2": "beta", "P3": "beta"}
    events = [ev("P2", "message", to="P1", text="hi"),   # P2 -> P1
              ev("P3", "message", to="P2", text="hi"),    # P3 -> P2
              ev("P1", "pull", "P2"),   # P1's pull to a prior messager (P2)
              ev("P1", "pull", "P3"),   # P1's pull to a non-messager (P3)  -> P1 surplus 2*1/2-1 = 0
              ev("P2", "pull", "P3")]   # P2's only pull, to a prior messager -> P2 surplus 1
    rows = derive.surplus_rows([ep(rec(sm, events))])
    assert rows == [{"model": "beta", "n_agent_episodes": 1, "surplus": 1.0, "ci": [1.0, 1.0]},
                    {"model": "alpha", "n_agent_episodes": 1, "surplus": 0.0, "ci": [0.0, 0.0]}]


def test_build_questions_exports_surplus_rows_by_model_name():
    sm = {"P1": "gpt-5.5", "P2": "gemini-flash", "P3": "opus-4.8"}
    events = [ev("P2", "message", to="P1", text="hi"), ev("P1", "pull", "P2")]
    e = ep(rec(sm, events, scores={"P1": 0, "P2": 1, "P3": 0}, actions_per_round=1),
           md={"rounds": 1})
    derive.tag_coalitions([e])
    derive.tag_prereg([e])
    em = [derive.report.measure_episode(e)]
    surplus = derive.build_questions([e], em, None)["q7"]["surplus"]
    row = surplus["per_model"][0]
    assert surplus["per_model_key"]["gpt55"] == row


# --- q8_panel ------------------------------------------------------------------------------------

def test_q8_panel_groups_by_topology():
    eps = [ep(rec({}, []), coalition_agents=3, top1_share=0.5),
           ep(rec({}, []), coalition_agents=3, top1_share=0.5),
           ep(rec({}, []), coalition_agents=1, top1_share=0.2)]
    em = [{"topology": "complete", "cascades": 2, "cascade_null": 1, "verified_credit_rate": 0.2},
          {"topology": "complete", "cascades": 2, "cascade_null": 1, "verified_credit_rate": 0.4},
          {"topology": "ring", "cascades": 0, "cascade_null": 0, "verified_credit_rate": 0.0}]
    panel = derive.q8_panel(eps, em)
    assert set(panel) == {"complete", "ring"}
    c = panel["complete"]
    assert c["n_episodes"] == 2
    assert c["chains"] == {"value": 2.0, "ci": [2.0, 2.0], "n": 2}
    assert c["chain_null"] == {"value": 1.0, "ci": None, "n": 2}   # paired null: no CI
    assert c["coalition_agents"] == {"value": 3.0, "ci": [3.0, 3.0], "n": 2}
    assert c["top1_share"] == {"value": 0.5, "ci": [0.5, 0.5], "n": 2}
    assert c["brokered_pull_rate"]["value"] == pytest.approx(0.3) and c["brokered_pull_rate"]["ci"] is None


# --- paper_leaderboard ---------------------------------------------------------------------------

def test_paper_leaderboard_ranks_by_extraction_with_ci_overlap():
    a = ep(rec({"P1": "alpha", "P2": "x", "P3": "x"}, [], focal="P1", scores={"P1": 4}),
           seat_extraction={"P1": 0.4}, seat_points={"P1": 4})
    b = ep(rec({"P1": "beta", "P2": "x", "P3": "x"}, [], focal="P1", scores={"P1": 1}),
           seat_extraction={"P1": 0.1}, seat_points={"P1": 1})
    rows = derive.paper_leaderboard([a, b])
    assert [r["model"] for r in rows] == ["alpha", "beta"]
    assert rows[0]["extraction"] == 0.4 and rows[0]["points"] == 4 and rows[0]["score"] == 4
    assert rows[0]["rank_ub"] == 1 and rows[1]["rank_ub"] == 2   # non-overlapping CIs -> distinct ranks
    assert rows[0]["by_focal"] is None and rows[0]["self_pull"] is None   # focal-only cols null here


# --- tag_pools -----------------------------------------------------------------------------------

def test_tag_pools_paper_in_progress_and_exploratory():
    coverage = [
        {"config": "c", "arm": "A", "stage": "confirmatory", "planned_episodes": 2, "done_episodes": 2},
        {"config": "c", "arm": "B", "stage": "confirmatory", "planned_episodes": 2, "done_episodes": 1},
        {"config": "c", "arm": "C", "stage": "exploratory", "planned_episodes": 2, "done_episodes": 2},
    ]
    eps = [ep(rec({}, []), arm="A"), ep(rec({}, []), arm="B"),
           ep(rec({}, []), arm="C"), ep(rec({}, []), arm=None)]
    pool.tag_pools(eps, coverage)
    assert [e["pool"] for e in eps] == ["paper", "in_progress", "exploratory", "exploratory"]


def test_tag_pools_held_out_control_arm_never_paper_even_when_complete():
    """Exp-13 causal-control arms (stage held-out) stay out of the paper pool on every rebuild —
    even at full coverage — until the pooling decision lands (D21)."""
    coverage = [{"config": "13_controls", "arm": "controls", "stage": "held-out",
                 "planned_episodes": 20, "done_episodes": 20}]
    eps = [ep(rec({}, []), arm="controls")]
    pool.tag_pools(eps, coverage)
    assert eps[0]["pool"] == "held_out"


def test_pool_summary_lists_held_out_separately():
    e = ep(rec({}, []), arm="controls")
    e["pool"], e["run"] = "held_out", "13_controls"
    out = pool.pool_summary([e])
    assert out["held_out"] == {"n_episodes": 1, "runs": ["13_controls"]}
    assert out["paper"]["n_episodes"] == 0


def test_arm_stage_holds_out_causal_control_arms_by_default():
    assert experiment.resolve_arm_stage(None, {}) == "confirmatory"
    assert experiment.resolve_arm_stage(None, {"declared_intents": True}) == "held-out"
    assert experiment.resolve_arm_stage(None, {"seeded_favours": 5}) == "held-out"
    # an explicit config `stage:` is the marker that overrides the default — flip it in
    # configs/13_controls.yaml when Brendon decides pooling
    assert experiment.resolve_arm_stage("confirmatory", {"declared_intents": True}) == "confirmatory"
    assert experiment.resolve_arm_stage("exploratory", {}) == "exploratory"


# --- keyed condition lookups (positional {{by_condition.N}} placeholders are banned) ---------------

def test_keyed_by_condition_mirrors_rows_under_placeholder_safe_keys():
    rows = [{"condition": "complete/pure/msg-on", "n": 210},
            {"condition": "ring/pure/msg-on", "n": 90}]
    keyed = derive.keyed_by_condition(rows)
    assert keyed["complete_pure_msg_on"] is rows[0]
    assert keyed["ring_pure_msg_on"] is rows[1]
    # keys must survive render.py's {{path}} charset ([A-Za-z0-9_.] segments)
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", k) for k in keyed)


def test_keyed_by_condition_raises_on_key_collision():
    # a silent overwrite would mislabel numbers — the exact failure keying exists to prevent
    rows = [{"condition": "ring/pure/msg-on"}, {"condition": "ring pure msg ON"}]
    with pytest.raises(ValueError):
        derive.keyed_by_condition(rows)


def test_keyed_by_model_raises_on_safe_key_collision():
    with pytest.raises(ValueError):
        derive.keyed_by_model([{"model": "gpt-5.5"}, {"model": "gpt55"}])


def test_metrics_preview_carries_keyed_coalition_rows():
    sm = {"P1": "alpha", "P2": "beta", "P3": "beta"}
    events = [ev("P1", "pull", "P2"), ev("P2", "pull", "P1"), ev("P3", "pull", "P1")]
    e = ep(rec(sm, events, scores={"P1": 2, "P2": 1, "P3": 0}, focal="P1", actions_per_round=2),
           arm="attack_complete", md={"rounds": 2})
    eps = [e]
    derive.tag_coalitions(eps)
    derive.tag_prereg(eps)
    em = [derive.report.measure_episode(x) for x in eps]
    coal = derive.build_metrics_preview(eps, em)["coalition"]
    assert set(coal["per_condition_key"]) == {derive._ckey(r["condition"])
                                              for r in coal["per_condition"]}


def test_unmeasured_public_chat_preview_has_numeric_episode_count():
    preview = derive.public_chat_preview([])
    assert preview["measured"] is False
    assert preview["n_episodes"] == 0


# --- build_questions wiring (the derive-local stats must reach the qN payload) --------------------

class TestWedgeFloorContrast:
    """Audit fix 4 (S7): the capture-unit wedge contrast + its BH survival flag reach summary.q1,
    so the q1.md ledger numbers come from report.contrast_family, not hand-written into the page."""

    def _wedge_episodes(self):
        # one attacker model with a complete msg-on episode and its msg-off floor twin
        on = ep(rec({"P1": "gpt-5.5", "P2": "bg", "P3": "bg"},
                    [ev("P2", "message", to="P1"), ev("P1", "pull", "P2"), ev("P2", "pull", "P1")],
                    scores={"P1": 1, "P2": 1, "P3": 0}, focal="P1", personas={"P1": "attacker"}))
        off = ep(rec({"P1": "gpt-5.5", "P2": "bg", "P3": "bg"},
                     [ev("P1", "pull", "P2"), ev("P2", "pull", "P1")],
                     scores={"P1": 1, "P2": 1, "P3": 0}, focal="P1",
                     personas={"P1": "attacker"}, messages_off=True))
        return [on, off]

    def test_mkey_is_placeholder_safe(self):
        assert derive._mkey("gpt-5.5") == "gpt55"
        assert derive._mkey("qwen3.7-max") == "qwen37max"
        assert derive._mkey("opus-4.8") == "opus48"

    def test_build_questions_exports_wedge_with_bh_flag(self):
        eps = self._wedge_episodes()
        derive.tag_coalitions(eps)
        derive.tag_prereg(eps)
        em = [derive.report.measure_episode(x) for x in eps]
        fc = derive.build_questions(eps, em, None)["q1"]["floor_contrast"]
        assert set(fc) == {"gpt55"}                        # placeholder-safe model key
        row = fc["gpt55"]
        assert row["model"] == "gpt-5.5"
        assert set(row) == {"model", "contrast", "ci", "survives", "p_adj"}
        # sourced from report.contrast_family — the same diff + BH survival flag, not a snapshot
        wedge = next(r for r in derive.report.contrast_family(em) if r["family"] == "wedge")
        assert row["contrast"] == wedge["estimate"]
        assert row["survives"] == wedge["survives"]


def test_build_questions_wires_in_the_derive_local_stats():
    sm = {"P1": "alpha", "P2": "beta", "P3": "beta"}
    events = [ev("P2", "message", to="P1", text="hi"),
              ev("P1", "pull", "P2"), ev("P2", "pull", "P1"), ev("P3", "pull", "P1")]
    e = ep(rec(sm, events, scores={"P1": 2, "P2": 1, "P3": 0}, focal="P1", actions_per_round=2),
           arm="attack_complete", md={"rounds": 2})
    eps = [e]
    derive.tag_coalitions(eps)
    derive.tag_prereg(eps)
    em = [derive.report.measure_episode(x) for x in eps]
    q = derive.build_questions(eps, em, None)
    assert q["q1"]["lift"]["diff"] == (derive.lift_diff_rows(eps) or None)
    assert q["q6"]["timing"] == derive.timing_series(eps)
    assert q["q7"]["surplus"]["per_model"] == derive.surplus_rows(eps)
    assert q["q8"]["panel"] == derive.q8_panel(eps, em)


# --- episode_json: the reader's citation ground-truth feed (S11) ----------------------------------
# The reader must show everything the engine records: forced (seeded) favours as random events, not
# chosen pulls; declared intents (with WHY a plan is missing); and public `say` chat. Event `i`
# indices must stay put — they are the deep-link anchors (episodes.html?e=<id>&t=<i>).

def _intent_ev(agent, planned=None, usage=None, round=1):
    """An `intent`-type event: a declared plan (dict|None) plus the intent call's usage list."""
    return {"round": round, "action_number": 0, "turn_index": 0, "agent": agent,
            "forfeit": False, "forfeit_reason": None, "type": "intent",
            "beneficiary": None, "credit": None, "to": None, "text": None,
            "forced": False, "planned": planned, "usage": usage or []}


def _punish_ev(agent, target, round=1):
    """A `punish`-type event: punisher `agent` penalizes seat `target`."""
    return {"round": round, "action_number": 0, "turn_index": 0, "agent": agent,
            "forfeit": False, "forfeit_reason": None, "type": "punish", "target": target,
            "beneficiary": None, "credit": None, "to": None, "text": None, "forced": False}


class TestEpisodeJson:
    def _one(self, events, scores, **rec_kw):
        r = rec({"P1": "alpha", "P2": "beta", "P3": "gamma"}, events, scores=scores, **rec_kw)
        return derive.episode_json(ep(r))

    def test_native_call_evidence_loads_for_one_export_without_changing_citations(self, tmp_path):
        call = {"call_id": "native-call", "input": [{"role": "user", "content": "Saved native input"}],
                "response": "Saved reply", "usage": {"total_cost": 0.04}}
        event = {**ev("P1", "pull", "P2"), "calls": [call]}
        full = rec({"P1": "alpha", "P2": "beta"}, [event], scores={"P1": 0, "P2": 1})
        source = tmp_path / "source.episode.json"
        source.write_text(json.dumps({"schema_version": "native-episode-v1", "status": "complete",
                                     "episode_id": "source-uuid", "sample_id": "source-sample",
                                     "metadata": ep(full)["metadata"], "record": full}))
        compact_event = {**event, "calls": [{"call_id": "native-call", "usage": {"total_cost": 0.04}}]}
        compact = ep({**full, "events": [compact_event]})
        compact.update({"id": "existing-citation-id", "call_evidence_source": str(source)})
        doc = derive.episode_json(compact)
        assert doc["id"] == "existing-citation-id"
        assert doc["events"][0]["i"] == 0
        assert doc["events"][0]["calls"][0]["input"][0]["content"] == "Saved native input"
        assert "input" not in compact["events"][0]["calls"][0]

    def test_forced_pull_flagged_scores_and_chosen_pull_not(self):
        # a chosen pull and a forced favour both score, but only the favour carries forced=True
        events = [ev("P1", "pull", "P2"), ev("P3", "pull", "P2", forced=True)]
        doc = self._one(events, {"P1": 0, "P2": 2, "P3": 0})  # assert inside episode_json also checks this
        chosen, favour = doc["events"]
        assert chosen["type"] == "pull" and chosen["forced"] is False
        assert favour["type"] == "pull" and favour["forced"] is True
        assert favour["actor"] == "P3" and favour["beneficiary"] == "P2"
        assert favour["scores_after"]["P2"] == 2  # a forced favour still moves the score

    def test_control_source_distinguishes_scripted_pulls_without_relabeling_history(self):
        scripted = {**ev("P1", "pull", "P2", forced=True),
                    "scripted": True, "control_source": "scripted"}
        legacy = {**ev("P1", "pull", "P2", forced=True), "scripted": True}
        random = ev("P3", "pull", "P2", forced=True)
        doc = self._one([scripted, legacy, random], {"P1": 0, "P2": 3, "P3": 0})
        assert [event["control_source"] for event in doc["events"]] == ["scripted", None, None]
        assert [event["scripted"] for event in doc["events"]] == [True, True, False]
        assert [event["i"] for event in doc["events"]] == [0, 1, 2]
        assert all(event["forced"] and event["calls"] == [] for event in doc["events"])
        assert doc["agent_totals"]["P1"]["actions"] == 0

    def test_pull_exports_always_keep_nullable_credit_key(self):
        events = [ev("P1", "pull", "P2", credit="P3"), ev("P3", "pull", "P2")]
        doc = self._one(events, {"P1": 0, "P2": 2, "P3": 0})
        assert [(e["actor"], e["beneficiary"], e["credit"]) for e in doc["events"]] == [
            ("P1", "P2", "P3"), ("P3", "P2", None),
        ]

    def test_intent_event_carries_declared_plan(self):
        plan = {"action": "pull", "beneficiary": "P2"}
        events = [_intent_ev("P1", planned=plan), ev("P1", "pull", "P2")]
        doc = self._one(events, {"P1": 0, "P2": 1, "P3": 0})
        intent = doc["events"][0]
        assert intent["type"] == "intent" and intent["actor"] == "P1"
        assert intent["planned"] == plan and intent["intent_error"] is None
        assert intent["forced"] is False

    def test_intent_event_carries_reason(self):
        # the seat's stated WHY must reach the reader's per-event feed, like target/planned do
        events = [{**_intent_ev("P1", planned={"action": "pull", "beneficiary": "P2"}),
                   "reason": "P2 pulled for me last round"}]
        doc = self._one(events, {"P1": 0, "P2": 0, "P3": 0})
        assert doc["events"][0]["reason"] == "P2 pulled for me last round"

    def test_intent_missing_plan_names_the_error(self):
        # a timeout/provider failure: planned None WITH a usage error -> intent_error carries why
        events = [_intent_ev("P2", planned=None, usage=[{"error": "TimeoutError"}])]
        doc = self._one(events, {"P1": 0, "P2": 0, "P3": 0})
        intent = doc["events"][0]
        assert intent["planned"] is None and intent["intent_error"] == "TimeoutError"

    def test_intent_clean_undecided_has_no_error(self):
        # a clean "undecided": planned None, usage without an error -> intent_error None
        events = [_intent_ev("P2", planned=None, usage=[{"prompt_tokens": 5}])]
        doc = self._one(events, {"P1": 0, "P2": 0, "P3": 0})
        assert doc["events"][0]["intent_error"] is None

    def test_say_event_flows_through_with_speaker(self):
        events = [ev("P1", "say", text="hi all")]
        doc = self._one(events, {"P1": 0, "P2": 0, "P3": 0}, public_chat=True)
        say = doc["events"][0]
        assert say["type"] == "say" and say["actor"] == "P1"
        assert say["text"] == "hi all" and say["to"] is None and say["forced"] is False

    def test_attached_reports_keep_parent_indices_cascades_scores_and_usage(self):
        events = [ev("P1", "message", to="P2", text="ask"),
                  ev("P2", "message", to="P3", text="relay"),
                  ev("P3", "pull", "P1"), ev("P2", "pull", "P1")]
        for event in events:
            event.update({"parse_retries": 0, "usage": [{"prompt_tokens": 10,
                          "cache_read_tokens": 0, "cache_write_tokens": 0,
                          "completion_tokens": 5, "total_cost": 0.01}]})
        scores = {"P1": 2, "P2": 0, "P3": 0}
        baseline = self._one(events, scores)
        reports = [{"action": "say", "text": "agent report"},
                   {"action": "say_human", "text": "human report"},
                   {"action": "message", "to": "P2", "text": "private report"}]
        for event, attached in zip(events, reports):
            event["report"] = {**attached, "action_cost": 0, "free_report": True}
        doc = self._one(events, scores)
        assert [event["i"] for event in doc["events"]] == [0, 1, 2, 3]
        assert len(doc["events"]) == 4
        assert doc["cascades"] == baseline["cascades"] == [
            {"a": "P1", "b": "P2", "c": "P3", "ask_i": 0, "relay_i": 1, "pull_i": 2, "b_pull_i": 3}]
        assert [event["report"]["audience"] for event in doc["events"][:3]] == ["agent", "human", "agent"]
        assert [event["report"]["channel"] for event in doc["events"][:3]] == ["public", "human", "private"]
        assert all(event["report"]["action_cost"] == 0 for event in doc["events"][:3])
        assert doc["events"][2]["report"]["to"] == "P2"
        assert "report" not in doc["events"][3]
        assert doc["agent_totals"]["P1"]["agent_posts"] == 1
        assert doc["agent_totals"]["P2"]["human_posts"] == 1
        assert doc["agent_totals"]["P3"]["private_messages"] == 1
        assert doc["agent_totals"]["P2"]["received_messages"] == 2
        for who in ["P1", "P2", "P3"]:
            assert doc["agent_totals"][who]["free_reports"] == 1
            for field in ["actions", "calls", "input_tokens", "output_tokens", "reported_cost_usd", "points_gained"]:
                assert doc["agent_totals"][who][field] == baseline["agent_totals"][who][field]

    def test_failed_actions_do_not_export_or_count_attached_reports(self):
        event = ev("P1", "pull", "P2", forfeit=True)
        event["report"] = {"action": "say", "text": "never delivered", "action_cost": 0, "free_report": True}
        doc = self._one([event], {"P1": 0, "P2": 0, "P3": 0})
        assert "report" not in doc["events"][0]
        assert doc["agent_totals"]["P1"]["agent_posts"] == 0
        assert doc["agent_totals"]["P1"]["free_reports"] == 0

    def test_event_i_indices_are_stable_enumerate_positions(self):
        # intents and forced favours occupy their own index in-place, so deep-link anchors don't shift
        events = [ev("P2", "message", to="P1", text="m"), _intent_ev("P1"),
                  ev("P3", "pull", "P2", forced=True), ev("P1", "pull", "P2"), ev("P1", "say", text="s")]
        doc = self._one(events, {"P1": 0, "P2": 2, "P3": 0}, public_chat=True)
        assert [e["i"] for e in doc["events"]] == [0, 1, 2, 3, 4]
        assert [e["type"] for e in doc["events"]] == ["message", "intent", "pull", "pull", "say"]

    def test_punish_event_carries_target(self):
        # a punished seat must be citable in the reader -> the per-event record needs `target`
        events = [ev("P1", "pull", "P2"), _punish_ev("P3", "P1")]
        doc = self._one(events, {"P1": 0, "P2": 1, "P3": -1})
        punish = doc["events"][1]
        assert punish["type"] == "punish" and punish["actor"] == "P3"
        assert punish["target"] == "P1"

    def test_pre_control_episode_without_new_keys_defaults_gracefully(self):
        # an episode logged before forced/planned/usage existed: those keys are simply absent
        old = {"round": 1, "agent": "P1", "type": "pull", "beneficiary": "P2",
               "forfeit": False, "forfeit_reason": None, "to": None, "text": None, "credit": None}
        doc = self._one([old], {"P1": 0, "P2": 1, "P3": 0})
        e = doc["events"][0]
        assert e["forced"] is False and e["planned"] is None and e["intent_error"] is None

    def test_call_evidence_and_unknown_historical_costs_reach_agent_totals(self):
        first = ev("P1", "pull", "P2")
        first.update({"calls": [{"call_id": "call-1", "input": [{"role": "user", "content": "actual input"}],
                                "response": "actual reply", "usage": {"input_tokens": 100,
                                "output_tokens": 20, "reasoning_tokens": 4, "total_cost": 0.03}}]})
        second = ev("P1", "message", to="P2", text="hello")
        second.update({"raw_responses": ["bad json", '{"action":"message"}'], "parse_retries": 1,
                       "usage": [{"prompt_tokens": 20, "completion_tokens": 3, "estimated_cost_usd": 0.02},
                                 {"prompt_tokens": 25, "completion_tokens": 4, "total_cost": 0}]})
        doc = self._one([first, second], {"P1": 0, "P2": 1, "P3": 0})
        assert doc["events"][0]["calls"][0]["input"][0]["content"] == "actual input"
        assert doc["events"][1]["calls"][0]["input"] is None
        assert doc["events"][1]["calls"][0]["response"] == "bad json"
        total = doc["agent_totals"]["P1"]
        assert total["calls"] == 3 and total["parse_retries"] == 1
        assert total["input_tokens"] == 0 and total["unknown_input_calls"] == 3
        assert total["uncached_input_tokens"] == 145 and total["output_tokens"] == 27
        assert total["reported_cost_usd"] == 0.03 and total["unknown_cost_calls"] == 1
        assert total["estimated_unreported_cost_usd"] == 0.02
        assert total["calls_with_cost_estimate"] == 1
        assert doc["agent_totals"]["P2"]["points_gained"] == 1
        assert doc["agent_totals"]["P2"]["received_messages"] == 1

    def test_input_tokens_include_cache_and_missing_buckets_keep_unknown_coverage(self):
        first = ev("P1", "pull", "P2")
        usage = {"prompt_tokens": 100, "cache_read_tokens": 200, "cache_write_tokens": 50,
                 "completion_tokens": 40, "reasoning_tokens": 30}
        first["usage"] = [usage]
        second = ev("P1", "pull", "P2")
        second["usage"] = [{"input_tokens": 10, "input_tokens_cache_read": 20,
                            "input_tokens_cache_write": None, "output_tokens": 2}]
        doc = self._one([first, second], {"P1": 0, "P2": 2, "P3": 0})
        totals = doc["agent_totals"]["P1"]
        assert totals["input_tokens"] == 350
        assert totals["unknown_input_calls"] == 1
        assert totals["uncached_input_tokens"] == 110
        assert totals["unknown_uncached_input_calls"] == 0
        assert totals["cache_read_tokens"] == 220
        assert totals["cache_write_tokens"] == 50
        assert totals["unknown_cache_write_calls"] == 1
        assert totals["output_tokens"] == 42  # already includes reasoning
        assert doc["events"][0]["calls"][0]["usage"] == usage

    def test_conflicting_output_alias_uses_canonical_completion_tokens(self):
        event = ev("P1", "pull", "P2")
        usage = {"prompt_tokens": 100, "cache_read_tokens": 200, "cache_write_tokens": 50,
                 "completion_tokens": 40, "output_tokens": 99, "reasoning_tokens": 30}
        event["usage"] = [usage]
        doc = self._one([event], {"P1": 0, "P2": 1, "P3": 0})
        assert doc["agent_totals"]["P1"]["output_tokens"] == 40
        assert doc["agent_totals"]["P1"]["unknown_output_calls"] == 0
        assert doc["events"][0]["calls"][0]["usage"] == usage

    def test_board_audiences_score_changes_and_unrecorded_calls_are_explicit(self):
        events = [ev("P1", "pull", "P2"), ev("P3", "say", text="Agents: review this."),
                  ev("P3", "say_human", text="Humans: review this."), _punish_ev("P2", "P1")]
        doc = self._one(events, {"P1": 0, "P2": 0, "P3": 0}, public_chat=True)
        assert [e["i"] for e in doc["events"]] == [0, 1, 2, 3]
        assert doc["events"][1]["audience"] == "agent"
        assert doc["events"][2]["audience"] == "human"
        assert doc["events"][3]["score_changes"] == {"P2": -1}
        assert doc["events"][0]["calls"][0]["input"] is None
        assert doc["agent_totals"]["P1"]["unknown_cost_calls"] == 1
        assert doc["agent_totals"]["P1"]["unknown_call_count_events"] == 1
        assert doc["agent_totals"]["P1"]["unknown_parse_retry_events"] == 1
        assert doc["agent_totals"]["P3"]["agent_posts"] == 1
        assert doc["agent_totals"]["P3"]["human_posts"] == 1


# --- relay_funnel_by_topology: the warn/vouch breakdown must survive the rollup ------------------

def test_relay_funnel_by_topology_sums_forward_valence():
    def _f(valence, **counts):
        return {"asks": 0, "forwarded": 0, "stranger_pulls": 0, "credited_middleman": 0,
                "forward_valence": valence, **counts}
    em = [{"topology": "ring", "relay_funnel": _f({"vouch": 1, "warn": 1, "neutral": 0})},
          {"topology": "ring", "relay_funnel": _f({"vouch": 0, "warn": 2, "neutral": 1})}]
    out = derive.relay_funnel_by_topology(em)
    assert out["ring"]["forward_valence"] == {"vouch": 1, "warn": 3, "neutral": 1}
