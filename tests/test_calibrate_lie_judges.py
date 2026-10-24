"""Offline checks for the resumable two-judge lie calibration artifact."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "calibrate_lie_judges", ROOT / "scripts" / "calibrate_lie_judges.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def output(claims=None, error=None):
    return {"claims": claims or [], "raw": "[]", "served_model": "m", "usage": {}} \
        if error is None else {"error": error}


class TestComparisonArtifact(unittest.TestCase):
    JUDGES = ["a", "b"]

    def doc(self):
        score = {"predicate": "score", "subject": "P1", "object": None, "quantity": 2,
                 "at_least": False, "scope": "game_so_far", "round": None,
                 "action_number": None, "quote": "P1 at 2"}
        return {"records": [
            {"record_id": 0, "gold_has_claims": False, "text": "P1 at 2", "gold_notes": "",
             "outputs": {"a": output([score]), "b": output([score])}},
            {"record_id": 1, "gold_has_claims": True, "text": "active exchange",
             "gold_notes": "qualitative", "outputs": {"a": output(), "b": output()}},
        ]}

    def test_completed_checkpoint_has_no_remaining_calls_unless_forced(self):
        doc = self.doc()
        self.assertEqual(cal.calls_remaining(doc, self.JUDGES), 0)
        self.assertEqual(cal.calls_remaining(doc, self.JUDGES, {1}), 2)
        doc["records"][0]["outputs"]["a"] = output(error="timeout")
        self.assertEqual(cal.calls_remaining(doc, self.JUDGES), 1)

    def test_queue_separates_score_gold_conflicts_from_missed_gold(self):
        queue = cal.adjudication_queue(self.doc(), self.JUDGES)
        self.assertEqual([q["category"] for q in queue],
                         ["gold_excludes_explicit_score_claim", "gold_claim_missed_by_both"])

    def test_cross_judge_detection_agrees(self):
        comparison = cal.comparison_summary(self.doc(), self.JUDGES)
        self.assertEqual(comparison["detection_disagreement_ids"], [])
        self.assertEqual(comparison["shared_claim_shapes"], 1)

    def test_channel_filter_treats_old_unmarked_records_as_private(self):
        sample = {"records": [
            {"record_id": "old"},
            {"record_id": "private", "channel": "private"},
            {"record_id": "public", "channel": "public"},
        ]}
        self.assertEqual([r["record_id"] for r in cal.select_records(sample, "private")["records"]],
                         ["old", "private"])
        self.assertEqual([r["record_id"] for r in cal.select_records(sample, "public")["records"]],
                         ["public"])

    def test_roster_is_inferred_from_fields_and_message_text(self):
        records = [{"sender_seat": "P10", "recipient_seat": "P2",
                    "text": "P7 pulled for P4"}]
        self.assertEqual(cal.infer_roster(records), [f"P{i}" for i in range(1, 11)])

    def test_adjudication_preserves_original_gold_and_reports_revised_metrics(self):
        doc = self.doc()
        doc["meta"] = {"sample": "sample.json"}
        doc["adjudication_queue"] = cal.adjudication_queue(doc, self.JUDGES)
        decisions = {"rubric": {"exact_scores": "include"}, "decisions": [
            {"sample": "sample.json", "record_id": 0, "gold_has_claims": True,
             "notes": "Exact score counts."},
            {"sample": "sample.json", "record_id": 1, "gold_has_claims": False,
             "notes": "Vague relationship language is not deterministic."},
        ]}
        cal.apply_adjudications(doc, self.JUDGES, decisions)
        self.assertFalse(doc["records"][0]["gold_has_claims"])
        self.assertTrue(doc["records"][0]["adjudicated_gold_has_claims"])
        self.assertEqual(doc["summary_adjudicated"]["a"]["accuracy"], 1.0)
        self.assertEqual(doc["adjudication_queue"][0]["adjudicated_gold_has_claims"], True)

    def test_shipped_promise_holdout_sample_is_private_and_blank_for_lies(self):
        holdout = json.loads((ROOT / "data" / "exp11_holdout_sample.json").read_text())
        self.assertEqual(len(holdout["records"]), 20)
        self.assertEqual({r["channel"] for r in holdout["records"]}, {"private"})
        self.assertTrue(all(r["gold_has_claims"] is None for r in holdout["records"]))

    def test_label_sidecar_merges_human_gold_without_mutating_the_sample(self):
        sample = {"records": [{"record_id": "x", "text": "hello",
                               "gold_has_claims": None}]}
        labels = {"sample_sha256": "abc", "records": [{
            "record_id": "x", "gold_promise": False, "gold_conditional": False,
            "gold_has_claims": True, "gold_notes": "score",
        }]}
        merged = cal.apply_label_sidecar(sample, labels, "abc")
        self.assertTrue(merged["records"][0]["gold_has_claims"])
        self.assertIsNone(sample["records"][0]["gold_has_claims"])
        with self.assertRaisesRegex(ValueError, "hash"):
            cal.apply_label_sidecar(sample, labels, "wrong")

    def test_structured_sidecar_refuses_blank_gold_before_judge_calls(self):
        sample = {"records": [{"record_id": "x", "text": "P2 pulled for me"}]}
        labels = {"sample_sha256": "hash", "records": [
            {"record_id": "x", "gold_claims": None, "gold_notes": ""},
        ]}

        with self.assertRaisesRegex(ValueError, "blank structured"):
            cal.apply_label_sidecar(sample, labels, "hash")

    def test_structured_summary_requires_every_ledger_field_to_match(self):
        gold = {"predicate": "pulled", "subject": "P2", "object": "P1",
                "quantity": 1, "at_least": False, "scope": "current_round",
                "round": None, "action_number": None, "quote": "gold wording"}
        exact = {**gold, "quote": "different exact substring"}
        wrong_scope = {**gold, "scope": "game_so_far"}
        doc = {"records": [
            {"gold_claims": [gold], "outputs": {"judge": output([exact])}},
            {"gold_claims": [gold], "outputs": {"judge": output([wrong_scope])}},
        ]}

        result = cal.structured_summary(doc, ["judge"])["judge"]

        self.assertEqual(result["exact_claim_tp"], 1)
        self.assertEqual(result["exact_claim_fp"], 1)
        self.assertEqual(result["exact_claim_fn"], 1)
        self.assertEqual(result["exact_record_match_rate"], 0.5)

    def test_resume_refreshes_human_gold_but_preserves_paid_outputs(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1, "sender_seat": "P1",
            "recipient_seat": "P2", "text": "P3 pulled for you", "channel": "private",
            "gold_has_claims": True, "gold_notes": "updated label",
        }]}
        existing = cal.initialise(sample, self.JUDGES, roster=["P1", "P2", "P3"])
        existing["records"][0]["gold_has_claims"] = False
        existing["records"][0]["gold_notes"] = "stale label"
        existing["records"][0]["outputs"]["a"] = output()

        resumed = cal.initialise(
            sample, self.JUDGES, existing=existing, roster=["P1", "P2", "P3"])

        self.assertTrue(resumed["records"][0]["gold_has_claims"])
        self.assertEqual(resumed["records"][0]["gold_notes"], "updated label")
        self.assertIn("a", resumed["records"][0]["outputs"])

    def test_resume_treats_missing_channel_as_private_on_both_sides(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1,
            "sender_seat": "P1", "recipient_seat": "P2", "text": "hello",
            "gold_has_claims": False,
        }]}
        existing = cal.initialise(sample, self.JUDGES, roster=["P1", "P2"])
        existing["records"][0].pop("channel")  # older private-only artifact
        existing["records"][0]["outputs"]["a"] = output()

        resumed = cal.initialise(sample, self.JUDGES, existing=existing,
                                 roster=["P1", "P2"])

        self.assertEqual(resumed["records"][0]["channel"], "private")
        self.assertIn("a", resumed["records"][0]["outputs"])

    def test_resume_refuses_changed_message_text(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1, "sender_seat": "P1",
            "recipient_seat": "P2", "text": "original", "channel": "private",
            "gold_has_claims": False,
        }]}
        existing = cal.initialise(sample, self.JUDGES, roster=["P1", "P2"])
        changed = {"records": [{**sample["records"][0], "text": "changed"}]}
        with self.assertRaisesRegex(ValueError, "input changed"):
            cal.initialise(changed, self.JUDGES, existing=existing, roster=["P1", "P2"])

    def test_resume_drops_stale_adjudicated_summary(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1, "sender_seat": "P1",
            "recipient_seat": "P2", "text": "hello", "channel": "private",
            "gold_has_claims": False,
        }]}
        existing = cal.initialise(sample, self.JUDGES, roster=["P1", "P2"])
        existing["summary_adjudicated"] = {"a": {"accuracy": 0.1}}
        existing["meta"]["rubric_v2"] = {"old": True}
        resumed = cal.initialise(sample, self.JUDGES, existing=existing,
                                 roster=["P1", "P2"])
        self.assertNotIn("summary_adjudicated", resumed)
        self.assertNotIn("rubric_v2", resumed["meta"])

    def test_resume_invalidates_outputs_from_the_legacy_public_prompt(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1, "sender_seat": "P1",
            "recipient_seat": None, "text": "public claim", "channel": "public",
            "gold_has_claims": False,
        }]}
        existing = cal.initialise(sample, self.JUDGES, roster=["P1", "P2"])
        existing["records"][0]["outputs"]["a"] = output()
        existing["meta"].pop("instrument")  # artifacts written before channel-aware prompting

        resumed = cal.initialise(sample, self.JUDGES, existing=existing,
                                 roster=["P1", "P2"])

        self.assertEqual(resumed["records"][0]["outputs"], {})
        self.assertEqual(resumed["meta"]["instrument"], cal.LIE_JUDGE_INSTRUMENT)

    def test_dry_run_never_rewrites_paid_outputs_after_instrument_change(self):
        sample = {"records": [{
            "record_id": "x", "episode_id": "e", "event_index": 1,
            "sender_seat": "P1", "recipient_seat": "P2", "text": "hello",
            "channel": "private", "gold_has_claims": False,
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "sample.json"
            out_path = Path(tmp) / "paid.json"
            sample_path.write_text(json.dumps(sample))
            existing = cal.initialise(sample, self.JUDGES, sample_path=sample_path,
                                      roster=["P1", "P2"])
            existing["meta"]["instrument"] = "older-instrument"
            existing["records"][0]["outputs"]["a"] = output()
            out_path.write_text(json.dumps(existing, indent=2))
            before = out_path.read_bytes()

            argv = ["calibrate_lie_judges", "--sample", str(sample_path),
                    "--out", str(out_path), "--judges", *self.JUDGES]
            with mock.patch.object(sys, "argv", argv):
                cal.main()

            self.assertEqual(out_path.read_bytes(), before)

    def test_replaced_paid_outputs_are_archived_before_spending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "paid.json"
            existing = {"meta": {"instrument": "older/instrument"},
                        "records": [{"outputs": {"a": output()}}]}
            out_path.write_text(json.dumps(existing, indent=2))

            archive = cal.archive_replaced_outputs(out_path, existing)

            self.assertTrue(archive.exists())
            self.assertEqual(json.loads(archive.read_text()), existing)
            self.assertEqual(json.loads(out_path.read_text()), existing)


class TestFillChannel(unittest.IsolatedAsyncioTestCase):
    async def test_public_record_uses_the_public_judge_prompt(self):
        seen = []

        async def judge(text, speaker, recipient, roster, channel="private"):
            seen.append(channel)
            return {"claims": [], "raw": "[]", "served_model": "m", "usage": {}}

        doc = {"records": [{
            "record_id": "x", "text": "public claim", "sender_seat": "P1",
            "recipient_seat": None, "channel": "public", "gold_has_claims": False,
            "outputs": {},
        }], "summary": {}}
        with (mock.patch.object(cal, "make_lie_judge", return_value=judge),
              mock.patch.object(cal, "checkpoint")):
            await cal.fill(doc, ["a"], Path("unused.json"), ["P1", "P2"])
        self.assertEqual(seen, ["public"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
