"""Guards for the paper-pool structured lie holdout and its blinding boundary."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load(relative):
    return json.loads((ROOT / relative).read_text())


def test_shipped_sample_is_blinded_hash_bound_and_gold_frozen():
    sample_path = ROOT / "data" / "paper_lie_structured_holdout_sample.json"
    sample = json.loads(sample_path.read_text())
    sources = load("data/paper_lie_structured_holdout_sources.json")
    labels = load("data/paper_lie_structured_holdout_labels.json")
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()

    assert sources["sample_sha256"] == digest
    assert labels["sample_sha256"] == digest
    assert labels["status"] == "approved_by_delegation_not_human_labeled"
    assert all(isinstance(record["gold_claims"], list) for record in labels["records"])
    assert {record["record_id"] for record in sample["records"]} == {
        record["record_id"] for record in sources["records"]
    } == {record["record_id"] for record in labels["records"]}

    hidden = {"episode_id", "event_index", "sender_model", "arm", "reader_link", "screen_hit"}
    assert all(not hidden.intersection(record) for record in sample["records"])


def test_shipped_sources_are_paper_only_and_episode_disjoint_from_development():
    summary = load("site/data/summary.json")
    sources = load("data/paper_lie_structured_holdout_sources.json")["records"]
    calibration = load("data/lie_judge_calibration_sample.json")["records"]
    paper_runs = set(summary["pools"]["paper"]["runs"])
    development_episodes = {record["episode_id"] for record in calibration}

    assert {record["arm"] for record in sources} <= paper_runs
    assert all(record["analysis_selected"] for record in sources)
    assert not ({record["episode_id"] for record in sources} & development_episodes)
    assert len({record["episode_id"] for record in sources}) == len(sources)
    assert Counter(record["sender_model"] for record in sources) == {
        "gemini-flash": 10,
        "glm-5.2": 10,
        "gpt-5.5": 10,
        "opus-4.8": 10,
        "qwen3.7-max": 10,
    }

    for source in sources:
        episode = load(f"site/dist/data/episodes/{source['episode_id']}.json")
        event = next(event for event in episode["events"] if event["i"] == source["event_index"])
        assert source["arm"] == episode["run"]
        assert source["sender_model"] == episode["seat_map"][event["actor"]]
        assert source["text"] == event["text"]
        assert source["sender_seat"] == event["actor"]
        assert source["recipient_seat"] == event["to"]
